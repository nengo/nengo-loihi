import nengo
import numpy as np
from nengo import Connection, Ensemble, Node
from nengo.connection import LearningRule
from nengo.ensemble import Neurons
from nengo.exceptions import BuildError

from nengo_loihi.builder.builder import Builder
from nengo_loihi.probe import LoihiProbe


def conn_probe(model, nengo_probe):
    # Connection probes create a connection from the target, and probe
    # the resulting signal (used when you want to probe the default
    # output of an object, which may not have a predefined signal)

    synapse = 0  # Removed internal filtering

    # get any extra arguments if this probe was created to send data
    #  to an off-chip Node via the splitter

    conn_label = None if nengo_probe.label is None else "%s_conn" % nengo_probe.label
    kwargs = model.chip2host_params.get(nengo_probe, None)
    if kwargs is not None:
        # this probe is for sending data to a Node
        kwargs.setdefault("label", conn_label)

        # determine the dimensionality
        input_dim = nengo_probe.target.size_out
        func = kwargs["function"]
        if func is not None:
            if callable(func):
                input_dim = np.asarray(func(np.zeros(input_dim, dtype=np.float64))).size
            else:
                input_dim = len(func[0])
        transform = np.asarray(kwargs["transform"], dtype=np.float64)
        if transform.ndim <= 1:
            output_dim = input_dim
        elif transform.ndim == 2:
            assert transform.shape[1] == input_dim
            output_dim = transform.shape[0]
        else:
            raise NotImplementedError()

        target = nengo.Node(size_in=output_dim, add_to_container=False)
        # TODO: This is a hack so that the builder can properly delegate the
        # connection build to the right method
        model.split.hostchip.chip_objs.add(target)

        conn = Connection(
            nengo_probe.target,
            target,
            synapse=synapse,
            solver=nengo_probe.solver,
            add_to_container=False,
            **kwargs,
        )
        model.nengo_probe_conns[nengo_probe] = conn
    else:
        conn = Connection(
            nengo_probe.target,
            nengo_probe,
            synapse=synapse,
            solver=nengo_probe.solver,
            add_to_container=False,
            label=conn_label,
        )
        target = nengo_probe

    # Set connection's seed to probe's (which isn't used elsewhere)
    model.seeded[conn] = model.seeded[nengo_probe]
    model.seeds[conn] = model.seeds[nengo_probe]

    d = conn.size_out
    if isinstance(nengo_probe.obj, Ensemble):
        # probed values are scaled by the target ensemble's radius
        scale = nengo_probe.obj.radius
        w = np.diag(scale * np.ones(d))
        weights = np.vstack([w, -w])
    else:
        raise NotImplementedError(
            "Nodes cannot be onchip, connections not yet probeable"
        )

    # probe target will be set when we build the connection below
    probe = LoihiProbe(
        target=[None], key="voltage", weights=[weights], synapse=nengo_probe.synapse
    )
    model.objs[target]["in"] = probe
    model.objs[target]["out"] = probe

    # add an extra entry for simulator.run_steps to read data out
    model.objs[nengo_probe]["out"] = probe

    # Build the connection (sets probe targets, adds probe)
    model.build(conn)


def signal_probe(model, key, probe):
    kwargs = model.chip2host_params.get(probe, None)
    weights = None
    if kwargs is not None:
        assert kwargs["function"] is None
        weights = kwargs["transform"].T / model.dt

    if isinstance(probe.obj, nengo.ensemble.Neurons):
        if probe.attr == "output":
            if weights is None:
                # spike probes should give values of 1.0/dt on spike events
                weights = 1.0 / model.dt

            if hasattr(probe.obj.ensemble.neuron_type, "amplitude"):
                weights *= probe.obj.ensemble.neuron_type.amplitude

    # Signal probes directly probe a target signal
    target = model.objs[probe.obj]["out"]

    loihi_probe = LoihiProbe(
        target=target,
        key=key,
        slice=probe.slice,
        synapse=probe.synapse,
        weights=weights,
    )
    model.add_probe(loihi_probe)
    model.objs[probe]["in"] = target
    model.objs[probe]["out"] = loihi_probe


probemap = {
    Ensemble: {"decoded_output": None, "input": "input"},
    Neurons: {
        "output": "spiked",
        "spikes": "spiked",
        "voltage": "voltage",
        "input": "current",
    },
    Node: {"output": None},
    Connection: {"output": "weighted", "input": "in"},
    LearningRule: {},  # make LR signals probeable, but no mapping required
}


@Builder.register(nengo.Probe)
def build_probe(model, probe):
    # This is a copy of Nengo's build_probe, but since conn_probe
    # and signal_probe are different, we have to include it here.

    # find the right parent class in `objtypes`, using `isinstance`
    for nengotype, probeables in probemap.items():
        if isinstance(probe.obj, nengotype):
            break
    else:
        raise BuildError("Type %r is not probeable" % type(probe.obj).__name__)

    key = probeables[probe.attr] if probe.attr in probeables else probe.attr
    if key is None:
        conn_probe(model, probe)
    else:
        signal_probe(model, key, probe)

    model.nengo_probes.append(probe)

    # Simulator will fill this list with probe data during simulation
    model.params[probe] = []
