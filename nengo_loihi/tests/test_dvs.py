import nengo
import numpy as np
import pytest
from nengo.utils.numpy import rms

import nengo_loihi
from nengo_loihi.dvs import AEDatFileIO, DVSEvents, DVSFileChipProcess


def jittered_t(n, t, jitter, rng, dtype="<u4"):
    assert jitter >= 0
    assert t - jitter >= 0
    tt = (t - jitter) * np.ones(n, dtype=dtype)
    if jitter > 0:
        tt += rng.randint(0, 2 * jitter + 1, size=tt.shape, dtype=dtype)
    return tt


def generate_sinusoidal_spikes(
    rng,
    t_length,
    period=30,
    max_rate=100,
    t_jitter=5,
    theta_fn=lambda t: 0,
    phase_fn=lambda t: 2 * np.pi * t,
    dvs_height=180,
    dvs_width=240,
):
    dt_us = 1000
    assert t_jitter < dt_us // 2

    t_length_us = int(1e6 * t_length)

    X, Y = np.meshgrid(np.linspace(-1, 1, dvs_width), np.linspace(-1, 1, dvs_height))

    max_prob = max_rate * 1e-6 * dt_us
    min_dimension = min(dvs_width, dvs_height)

    events = []
    for t_us in range(dt_us, t_length_us + 1, dt_us):
        t = t_us * 1e-6
        theta = theta_fn(t)
        phase = phase_fn(t)

        X1 = np.cos(theta) * X + np.sin(theta) * Y

        x = np.linspace(0, 1, 50)
        prob = np.sin((np.pi * min_dimension / period) * x + phase) * max_prob
        prob = np.interp(X1, x, prob, period=1)

        u = rng.rand(*prob.shape)
        s_on = u < prob
        s_off = u < -prob

        y, x = s_off.nonzero()
        tt = jittered_t(len(x), t_us, t_jitter, rng, dtype="<u4")
        events.append((tt, 0, x, y))

        y, x = s_on.nonzero()
        tt = jittered_t(len(x), t_us, t_jitter, rng, dtype="<u4")
        events.append((tt, 1, x, y))

    dvs_events = DVSEvents()
    dvs_events.init_events(n_events=sum(len(xx) for _, _, xx, _ in events))

    i = 0
    for tt, p, xx, yy in events:
        ee = dvs_events.events[i : i + len(xx)]
        ee["t"] = tt
        ee["p"] = p
        ee["x"] = xx
        ee["y"] = yy
        i += len(xx)

    return dvs_events


@pytest.mark.parametrize(
    "dvs_shape, pool, channels_last",
    [
        ((180, 240), (6, 10), True),
        ((147, 196), (7, 7), False),
    ],
)
def test_dvs_file_chip_node(
    dvs_shape, pool, channels_last, Simulator, request, tmpdir, rng, plt
):
    t_length = 1
    dvs_height, dvs_width = dvs_shape

    dvs_events = generate_sinusoidal_spikes(
        rng=rng,
        t_length=t_length,
        period=120,
        max_rate=10,
        t_jitter=10,
        theta_fn=lambda t: t,
        phase_fn=lambda t: t,
        dvs_height=dvs_height,
        dvs_width=dvs_width,
    )

    datafile = str(tmpdir.join("sinusoidal.events"))
    dvs_events.write_file(datafile)

    # --- test that file works in network
    height = dvs_height // pool[0]
    width = dvs_width // pool[1]
    gain = 101

    with nengo.Network() as net:
        dvs_process = DVSFileChipProcess(
            file_path=datafile,
            pool=pool,
            channels_last=channels_last,
            dvs_height=dvs_height,
            dvs_width=dvs_width,
        )
        u = nengo.Node(dvs_process)

        assert dvs_process.height == height and dvs_process.width == width
        h = dvs_process.height
        w = dvs_process.width
        p = dvs_process.polarity

        ensembles = [
            nengo.Ensemble(
                h * w,
                1,
                neuron_type=nengo_loihi.LoihiSpikingRectifiedLinear(),
                gain=nengo.dists.Choice([gain]),
                bias=nengo.dists.Choice([0]),
            )
            for _ in range(p)
        ]

        for k, e in enumerate(ensembles):
            u_channel = u[k::p] if channels_last else u[k * h * w : (k + 1) * h * w]
            nengo.Connection(
                u_channel, e.neurons, synapse=None, transform=1.0 / np.prod(pool)
            )

        probes = [nengo.Probe(e.neurons) for e in ensembles]

    with nengo.Simulator(net) as nengo_sim:
        nengo_sim.run(t_length)

    with Simulator(net) as loihi_sim:
        loihi_sim.run(t_length)

    et = dvs_events.events[:]["t"] * 1e-6
    ey = dvs_events.events[:]["y"] // pool[0]
    ex = dvs_events.events[:]["x"] // pool[1]
    ep = dvs_events.events[:]["p"]

    sims = {"nengo": nengo_sim, "loihi": loihi_sim}
    images_ref = []
    images_sim = {sim_name: [] for sim_name in sims}
    for k, t in enumerate(np.linspace(0.2, 1, 5)):
        t_window = 0.05

        m = (et > (t - t_window)) & (et <= t)
        image = np.zeros((h, w))
        m0 = m & (ep == 0)
        m1 = m & (ep == 1)
        np.add.at(image, (ey[m0], ex[m0]), -1)
        np.add.at(image, (ey[m1], ex[m1]), 1)
        images_ref.append(image)

        p0, p1 = probes
        for sim_name, sim in sims.items():
            sim_t = sim.trange()
            m = (sim_t > (t - t_window)) & (sim_t <= t)
            image = np.zeros((h, w))
            image -= sim.data[p0][m].sum(axis=0).reshape(h, w) * sim.dt
            image += sim.data[p1][m].sum(axis=0).reshape(h, w) * sim.dt
            images_sim[sim_name].append(image)

    normalized_image = lambda image: image / np.abs(image).max()

    rows, cols = 1 + len(sims), len(images_ref)
    for j in range(len(images_ref)):
        for i in range(rows):
            name = "ref" if i == 0 else list(sims)[i - 1]
            image = images_ref[j] if i == 0 else images_sim[name][j]

            plt.subplot(rows, cols, i * cols + j + 1)
            plt.imshow(normalized_image(image), vmin=-1, vmax=1)
            plt.xticks([])
            plt.yticks([])
            if j == 0:
                plt.title(name)

    plt.tight_layout()

    for j, image_ref in enumerate(images_ref):
        # assert that sims are vaguely close to reference, when normalized
        # (the reference does not account for neuron behaviour)
        image_ref = normalized_image(image_ref)
        for name in sims:
            image_sim = normalized_image(images_sim[name][j])
            rmse = rms(image_sim - image_ref) / rms(image_ref)
            assert rmse < 0.4, "Image %d for sim %s not close to reference" % (j, name)

        # assert that sims are close to each other (unnormalized)
        image_nengo = images_sim["nengo"][j]
        assert rms(images_sim["loihi"][j] - image_nengo) / rms(image_nengo) < 0.1


@pytest.mark.parametrize("ext", ["events", "aedat"])
@pytest.mark.parametrize("relative_time", [False, True])
def test_dvs_file_io(ext, relative_time, tmpdir, rng):
    t_length = 0.5
    dvs_events = generate_sinusoidal_spikes(
        rng=rng,
        t_length=t_length,
        period=120,
        max_rate=10,
        t_jitter=100,
        theta_fn=lambda t: t,
        phase_fn=lambda t: t,
    )

    datafile = str(tmpdir.join("sinusoidal." + ext))
    if ext == "aedat":
        write_aedat_file(datafile, dvs_events)
    else:
        dvs_events.write_file(datafile)

    # --- test that file loads properly
    # events should load sorted
    dvs_events.events.sort(order="t", kind="stable")
    if relative_time:
        dvs_events.events[:]["t"] -= dvs_events.events[0]["t"]

    dvs_events2 = DVSEvents.from_file(datafile, rel_time=relative_time)
    assert np.array_equal(dvs_events.events[:]["t"], dvs_events2.events[:]["t"])
    assert np.array_equal(dvs_events.events[:]["x"], dvs_events2.events[:]["x"])
    assert np.array_equal(dvs_events.events[:]["y"], dvs_events2.events[:]["y"])
    assert np.array_equal(dvs_events.events[:]["p"], dvs_events2.events[:]["p"])


def test_dvs_errors(tmpdir):
    def empty_file(path):
        with open(path, "w") as fh:
            fh.write(" ")

    no_ext_path = str(tmpdir.join("dvs"))
    empty_file(no_ext_path)
    with pytest.raises(ValueError, match="Events file .* has no extension"):
        DVSEvents.from_file(no_ext_path)

    bad_ext_path = str(tmpdir.join("dvs.what"))
    empty_file(bad_ext_path)
    with pytest.raises(ValueError, match="Unrecognized file format 'what'"):
        DVSEvents.from_file(bad_ext_path)

    # overwriting initialized events
    event_data = [(0, 0, 0, 0, 0), (1, 2, 1, 0, 3)]
    dvs_events = DVSEvents()
    dvs_events.init_events(event_data=event_data)
    assert dvs_events.n_events > 0
    with pytest.warns(UserWarning, match="`events` has already been initialized"):
        dvs_events.init_events(n_events=10)

    with pytest.raises(ValueError, match="The provided path.*has no extension"):
        dvs_events.write_file(no_ext_path)

    with pytest.raises(ValueError, match="Unsupported file format 'what'"):
        dvs_events.write_file(bad_ext_path)

    # mangled last event
    mangled_path = str(tmpdir.join("dvs.aedat"))
    write_aedat_file(mangled_path, dvs_events, mangle_last=True)
    with pytest.warns(UserWarning, match="Mangled event at end"):
        DVSEvents.from_file(mangled_path)


def write_aedat_file(file_path, dvs_events, random_header=True, mangle_last=False):
    """Write a basic AEDat file with the given events.

    This is not a valid AEDat file, but just the most basic that will work with
    ``AEDatFileIO.read_events``.
    """

    events = dvs_events.events

    with open(file_path, "wb") as fh:
        header = ["#!AER-DAT2.0\r\n"]
        if random_header:
            # add random lines to the header, to test reading long headers
            for _ in range(100):
                ints = np.random.randint(48, 127, size=50).astype("u1")
                line = "#" + bytes(ints).decode("ascii") + "\r\n"
                header.append(line)

        header.append("#End Of ASCII Header\r\n")
        fh.write("".join(header).encode("ascii"))

        sorted_inds = np.argsort(events[:]["t"])
        events = events[sorted_inds]

        for event in events:
            aedat_event = AEDatFileIO.AEDatEvent()
            aedat_event.type = 0  # 0 means DVS event
            aedat_event.y = event["y"]
            aedat_event.x = event["x"]
            aedat_event.polarity = event["p"]
            aedat_event.trigger = 0
            aedat_event.adc_sample = 0
            aedat_event.t = event["t"]

            fh.write(bytes(aedat_event))

        if mangle_last:
            fh.write(b"34")  # write some random extra bytes
