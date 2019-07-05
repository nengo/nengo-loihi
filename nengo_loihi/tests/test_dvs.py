import nengo
from nengo.utils.numpy import rms
import numpy as np
import pytest

import nengo_loihi
from nengo_loihi.dvs import get_dvs_reader
from nengo_loihi.inputs import DVSFileChipNode


def jittered_t(n, t, jitter, rng, dtype='<u4'):
    assert jitter >= 0
    assert t - jitter >= 0
    tt = (t - jitter) * np.ones(n, dtype=dtype)
    if jitter > 0:
        tt += rng.randint(0, 2*jitter + 1, size=tt.shape, dtype=dtype)
    return tt


def generate_sinusoidal_spikes(
        rng,
        t_length,
        period=30,
        max_rate=100,
        t_jitter=5,
        theta_fn=lambda t: 0,
        phase_fn=lambda t: 2*np.pi*t,
):
    dt_us = 1000
    assert t_jitter < dt_us // 2

    t_length_us = int(1e6 * t_length)

    X, Y = np.meshgrid(np.linspace(-1, 1, 240),
                       np.linspace(-1, 1, 180))

    max_prob = max_rate * 1e-6 * dt_us

    events = []
    for t_us in range(dt_us, t_length_us + 1, dt_us):
        t = t_us * 1e-6
        theta = theta_fn(t)
        phase = phase_fn(t)

        X1 = np.cos(theta)*X + np.sin(theta)*Y

        x = np.linspace(0, 1, 50)
        prob = np.sin((np.pi*180/period)*x + phase) * max_prob
        prob = np.interp(X1, x, prob, period=1)

        u = rng.rand(*prob.shape)
        s_on = u < prob
        s_off = u < -prob

        y, x = s_off.nonzero()
        tt = jittered_t(len(x), t_us, t_jitter, rng, dtype='<u4')
        events.append((tt, 0, x, y))

        y, x = s_on.nonzero()
        tt = jittered_t(len(x), t_us, t_jitter, rng, dtype='<u4')
        events.append((tt, 1, x, y))

    n_events = sum(len(xx) for _, _, xx, _ in events)
    event_data = np.zeros(
        n_events, dtype=[
            ('y', '<u2'), ('x', '<u2'), ('p', 'u1'), ('t', '<u4')])

    i = 0
    for tt, p, xx, yy in events:
        ee = event_data[i:i+len(xx)]
        ee['t'] = tt
        ee['p'] = p
        ee['x'] = xx
        ee['y'] = yy
        i += len(xx)

    return event_data


def write_events_file(filename, events):
    event_data = np.zeros(len(events),
                          dtype=[('y', '<u2'), ('x', '<u2'), ('t', '<u4')])
    event_data[:]['t'] = events[:]['t']
    event_data[:]['x'] = events[:]['x'] << 1
    event_data[:]['y'] = (events[:]['y'] << 2) + (events[:]['p'] << 1)

    with open(filename, 'wb') as fh:
        fh.write(event_data.tobytes())


@pytest.mark.parametrize(
    'pool, channels_last, use_cores', [
        ((6, 10), True, False),
        ((9, 12), False, False),
        ((6, 10), True, True),
    ])
def test_dvs_fileinput(pool, channels_last, use_cores,
                       Simulator, request, tmpdir, rng, plt):
    if use_cores and request.config.getoption("--target") != 'loihi':
        pytest.skip("use-cores not possible in emulator")

    t_length = 1

    events = generate_sinusoidal_spikes(
        rng=rng, t_length=t_length, period=120, max_rate=10, t_jitter=10,
        theta_fn=lambda t: t, phase_fn=lambda t: t)

    datafile = str(tmpdir.join("sinusoidal.events"))
    write_events_file(datafile, events)

    # --- test that file loads properly
    reader = get_dvs_reader(datafile)
    events2 = reader.read_events()
    assert np.array_equal(events[:]['t'], events2[:]['t'])
    assert np.array_equal(events[:]['x'], events2[:]['x'])
    assert np.array_equal(events[:]['y'], events2[:]['y'])
    assert np.array_equal(events[:]['p'], events2[:]['p'])
    del events2

    # --- test that file works in network
    height = 180 // pool[0]
    width = 240 // pool[1]
    gain = 100

    with nengo.Network() as net:
        u = DVSFileChipNode(filename=datafile, pool=pool,
                            channels_last=channels_last, use_cores=use_cores)
        assert u.height == height and u.width == width
        h = u.height
        w = u.width
        p = u.polarity

        ensembles = [
            nengo.Ensemble(h * w, 1,
                           neuron_type=nengo.SpikingRectifiedLinear(),
                           gain=nengo.dists.Choice([gain]),
                           bias=nengo.dists.Choice([0]))
            for _ in range(p)]

        for k, e in enumerate(ensembles):
            u_channel = u[k::p] if channels_last else u[k*h*w:(k+1)*h*w]
            nengo.Connection(u_channel, e.neurons, transform=1./np.prod(pool))

        probes = [nengo.Probe(e.neurons) for e in ensembles]

    with Simulator(net) as sim:
        sim.run(t_length)

    sim_t = sim.trange()
    et = events[:]['t'] * 1e-6
    ey = events[:]['y'] // pool[0]
    ex = events[:]['x'] // pool[1]
    ep = events[:]['p']

    images_ref = []
    images_out = []
    for k, t in enumerate(np.linspace(0.2, 1, 5)):
        t_window = 0.05

        m = (et > (t - t_window)) & (et <= t)
        image = np.zeros((h, w))
        m0 = m & (ep == 0)
        m1 = m & (ep == 1)
        np.add.at(image, (ey[m0], ex[m0]), -1)
        np.add.at(image, (ey[m1], ex[m1]), 1)
        # image /= np.prod(pool)
        image = image / np.abs(image).max()
        images_ref.append(image)

        p0, p1 = probes
        m = (sim_t > (t - t_window)) & (sim_t <= t)
        image = np.zeros((h, w))
        image -= sim.data[p0][m].sum(axis=0).reshape(h, w) * sim.dt
        image += sim.data[p1][m].sum(axis=0).reshape(h, w) * sim.dt
        # image /= gain
        image = image / np.abs(image).max()
        images_out.append(image)

    rows, cols = 3, len(images_ref)
    for k, (image_ref, image_out) in enumerate(zip(images_ref, images_out)):
        plt.subplot(rows, cols, k+1)
        plt.imshow(image_ref, vmin=-1, vmax=1)
        plt.xticks([])
        plt.yticks([])

        plt.subplot(rows, cols, cols+k+1)
        plt.imshow(image_out, vmin=-1, vmax=1)
        plt.xticks([])
        plt.yticks([])

        plt.subplot(rows, cols, 2*cols+k+1)
        plt.imshow(image_out - image_ref, vmin=-1, vmax=1)
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()

    for k, (image_ref, image_out) in enumerate(zip(images_ref, images_out)):
        assert np.allclose(image_out, image_ref, atol=0.25)
        assert rms(image_out - image_ref) / rms(image_ref) < 0.2
