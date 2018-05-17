import nengo


def test_ens_ens(Simulator, plt):
    with nengo.Network(seed=1) as model:
        a = nengo.Ensemble(100, 1, label='a',
                           max_rates=nengo.dists.Uniform(100, 120),
                           intercepts=nengo.dists.Uniform(-0.5, 0.5))
        avp = nengo.Probe(a.neurons[:5], 'voltage')

        b = nengo.Ensemble(101, 1, label='b',
                           max_rates=nengo.dists.Uniform(100, 120),
                           intercepts=nengo.dists.Uniform(-0.5, 0.5))
        nengo.Connection(a, b, function=lambda x: x + 0.5)
        bvp = nengo.Probe(b.neurons[:5], 'voltage')

    with Simulator(model, target='sim') as sim:
        sim.run(0.1)

    print(sim.data[avp])
    print(sim.data[bvp])

    plt.subplot(211)
    plt.plot(sim.trange(), sim.data[avp][:, :10])
    plt.subplot(212)
    plt.plot(sim.trange(), sim.data[bvp][:, :10])


def test_node_ens_ens(Simulator, plt):
    with nengo.Network(seed=1) as model:
        u = nengo.Node(output=0.5)
        up = nengo.Probe(u)

        a = nengo.Ensemble(100, 1, label='a',
                           max_rates=nengo.dists.Uniform(100, 120),
                           intercepts=nengo.dists.Uniform(-0.5, 0.5))
        nengo.Connection(u, a, synapse=None)
        ap = nengo.Probe(a)

        b = nengo.Ensemble(101, 1, label='b',
                           max_rates=nengo.dists.Uniform(100, 120),
                           intercepts=nengo.dists.Uniform(-0.5, 0.5))
        nengo.Connection(a, b)
        bp = nengo.Probe(b)

    with Simulator(model, target='sim') as sim:
        sim.run(0.5)

    output_filter = nengo.synapses.Alpha(0.02)
    plt.plot(sim.trange(), output_filter.filtfilt(sim.data[up]))
    plt.plot(sim.trange(), output_filter.filtfilt(sim.data[ap]))
    plt.plot(sim.trange(), output_filter.filtfilt(sim.data[bp]))
