import nengo
from nengo.exceptions import BuildError
import numpy as np
import pytest

from nengo_loihi.allocators import core_stdp_pre_cfgs
from nengo_loihi.loihi_api import Board
from nengo_loihi.loihi_cx import CxSynapses


def test_core_stdp_pre_cfgs():
    core = Board().new_chip().new_core()

    def new_syn(tracing_mag=None):
        syn = CxSynapses(n_axons=1)
        syn.set_full_weights(np.array([[1]]))
        if tracing_mag is not None:
            syn.set_learning(tracing_mag=tracing_mag)
        core.add_synapses(syn)
        return syn

    profile_idxs = {}
    # Do this one by one to guarantee order of created tracecfgs
    profile_idxs[new_syn(0.1)] = 0
    profile_idxs[new_syn(0.2)] = 1
    profile_idxs[new_syn(0.2)] = 1
    profile_idxs[new_syn(0.3)] = 2
    profile_idxs[new_syn(0.3)] = 2
    profile_idxs[new_syn()] = None

    profiles, ret_idxs = core_stdp_pre_cfgs(core)
    assert len(profiles) == 3
    assert ret_idxs == profile_idxs


def test_group_size(Simulator):
    with nengo.Network() as net:
        nengo.Ensemble(1024, 1)

    # n_neurons within limit, no problem
    with Simulator(net) as sim:
        sim.run_steps(5)

    with nengo.Network() as net:
        nengo.Ensemble(1025, 1)
    with pytest.raises(BuildError):
        with Simulator(net):
            pass
