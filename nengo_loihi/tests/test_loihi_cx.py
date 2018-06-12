import numpy as np

from nengo_loihi.loihi_cx import CxGroup, CxProbe, CxModel

try:
    import nxsdk  # noqa pylint: disable=unused-import
    TARGET = 'loihi'
except ImportError:
    TARGET = 'sim'


def test_simulator_noise(plt):
    seed = 0

    model = CxModel()
    group = CxGroup(10)
    group.configure_relu()

    group.bias[:] = np.linspace(0, 0.01, group.n)

    group.enableNoise[:] = 1
    group.noiseExp0 = -2
    group.noiseMantOffset0 = 0
    group.noiseAtDendOrVm = 1

    probe = CxProbe(target=group, key='v')
    group.add_probe(probe)
    model.add_group(group)

    model.discretize()

    if TARGET == 'loihi':
        sim = model.get_loihi(seed=seed)
        sim.run_steps(1000)
        y = np.column_stack([
            p.timeSeries.data for p in sim.board.probe_map[probe]])
    else:
        sim = model.get_simulator(seed=seed)
        sim.run_steps(1000)
        y = sim.probe_outputs[probe]

    ax = plt.subplot(111)
    ax.plot(y)
    # ax.set_xticks([])
    ax.set_yticklabels([])
    plt.savefig('test_simulator_noise.pdf')


def run_tests():
    import os
    import matplotlib
    haveDisplay = "DISPLAY" in os.environ
    if not haveDisplay:
        matplotlib.use('Agg')

    import matplotlib.pyplot as plt

    test_simulator_noise(plt)
    if haveDisplay:
        plt.show()


if __name__ == '__main__':
    run_tests()
