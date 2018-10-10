from collections import OrderedDict
import logging
import traceback
import warnings

import numpy as np

import nengo
import nengo.utils.numpy as npext
from nengo.exceptions import (
    BuildError, ReadonlyError, SimulatorClosed, ValidationError)
from nengo.simulator import ProbeDict as NengoProbeDict

from nengo_loihi.builder import Model
from nengo_loihi.loihi_cx import CxSimulator
from nengo_loihi.loihi_interface import LoihiSimulator
from nengo_loihi.splitter import split
import nengo_loihi.config as config

logger = logging.getLogger(__name__)


class ProbeDict(NengoProbeDict):
    """Map from Probe -> ndarray

    This is more like a view on the dict that the simulator manipulates.
    However, for speed reasons, the simulator uses Python lists,
    and we want to return NumPy arrays. Additionally, this mapping
    is readonly, which is more appropriate for its purpose.
    """

    def __init__(self, raw):
        super(ProbeDict, self).__init__(raw=raw)
        self.fallbacks = []

    def add_fallback(self, fallback):
        assert isinstance(fallback, NengoProbeDict)
        self.fallbacks.append(fallback)

    def __getitem__(self, key):
        target = self.raw
        if key not in target:
            for fallback in self.fallbacks:
                if key in fallback:
                    target = fallback.raw
                    break
        assert key in target, "probed object not found"

        if (key not in self._cache
                or len(self._cache[key]) != len(target[key])):
            rval = target[key]
            if isinstance(rval, list):
                rval = np.asarray(rval)
                rval.setflags(write=False)
            self._cache[key] = rval
        return self._cache[key]

    def __iter__(self):
        for k in self.raw:
            yield k
        for fallback in self.fallbacks:
            for k in fallback:
                yield k

    def __len__(self):
        return len(self.raw) + sum(len(d) for d in self.fallbacks)

    # TODO: Should we override __repr__ and __str__?


class Simulator(object):
    """Nengo Loihi simulator for Loihi hardware and emulator.

    The simulator takes a `nengo.Network` and builds internal data structures
    to run the model defined by that network on Loihi emulator or hardware.
    Run the simulator with the `.Simulator.run` method, and access probed data
    through the ``data`` attribute.

    Building and running the simulation allocates resources. To properly free
    these resources, call the `.Simulator.close` method. Alternatively,
    `.Simulator.close` will automatically be called if you use
    ``with`` syntax::

        with nengo_loihi.Simulator(my_network) as sim:
            sim.run(0.1)
        print(sim.data[my_probe])

    Note that the ``data`` attribute is still accessible even when a simulator
    has been closed. Running the simulator, however, will raise an error.

    Parameters
    ----------
    network : Network or None
        A network object to be built and then simulated. If None,
        then the *model* parameter must be provided instead.
    dt : float, optional (Default: 0.001)
        The length of a simulator timestep, in seconds.
    seed : int, optional (Default: None)
        A seed for all stochastic operators used in this simulator.
        Will be set to ``network.seed + 1`` if not given.
    model : Model, optional (Default: None)
        A `.Model` that contains build artifacts to be simulated.
        Usually the simulator will build this model for you; however, if you
        want to build the network manually, or you want to inject build
        artifacts in the model before building the network, then you can
        pass in a `.Model` instance.
    precompute : bool, optional (Default: True)
        Whether model inputs should be precomputed to speed up simulation.
        When *precompute* is False, the simulator will be run one step
        at a time in order to use model outputs as inputs in other parts
        of the model.
    target : str, optional (Default: None)
        Whether the simulator should target the emulator (``'sim'``) or
        Loihi hardware (``'loihi'``). If None, *target* will default to
        ``'loihi'`` if NxSDK is installed, and the emulator if it is not.

    Attributes
    ----------
    closed : bool
        Whether the simulator has been closed.
        Once closed, it cannot be reopened.
    data : ProbeDict
        The dictionary mapping from Nengo objects to the data associated
        with those objects. In particular, each `nengo.Probe` maps to
        the data probed while running the simulation.
    model : Model
        The `.Model` containing the data structures necessary for
        simulating the network.
    precompute : bool
        Whether model inputs should be precomputed to speed up simulation.
        When *precompute* is False, the simulator will be run one step
        at a time in order to use model outputs as inputs in other parts
        of the model.

    """

    # 'unsupported' defines features unsupported by a simulator.
    # The format is a list of tuples of the form `(test, reason)` with `test`
    # being a string with wildcards (*, ?, [abc], [!abc]) matched against Nengo
    # test paths and names, and `reason` is a string describing why the feature
    # is not supported by the backend. For example:
    #     unsupported = [('test_pes*', 'PES rule not implemented')]
    # would skip all tests whose names start with 'test_pes'.
    unsupported = [
        # no ensembles on chip
        ('test_circularconv.py:*', "no ensembles onchip"),
        ('test_product.py:test_direct_mode_with_single_neuron',
         "no ensembles onchip"),
        ('test_connection.py:test_neuron_slicing', "no ensembles onchip"),
        ('test_connection.py:test_boolean_indexing', "no ensembles onchip"),
        ('test_learning_rules.py:test_pes_synapse*', "no ensembles onchip"),
        ('test_learning_rules.py:test_pes_recurrent_slice*',
         "no ensembles onchip"),
        ('test_neurons.py:test_amplitude[[]LIFRate[]]', "no ensembles onchip"),
        ('test_neurons.py:test_amplitude[[]RectifiedLinear[]]',
         "no ensembles onchip"),
        ('test_neurons.py:test_alif_rate', "no ensembles onchip"),
        ('test_neurons.py:test_izhikevich', "no ensembles onchip"),
        ('test_neurons.py:test_sigmoid_response_curves*',
         "no ensembles onchip"),
        ('test_node.py:test_[!i]*', "no ensembles onchip"),
        ('test_probe.py:test_multirun', "no ensembles onchip"),
        ('test_probe.py:test_dts', "no ensembles onchip"),
        ('test_probe.py:test_large', "no ensembles onchip"),
        ('test_probe.py:test_conn_output', "no ensembles onchip"),
        ('test_processes.py:test_time', "no ensembles onchip"),
        ('test_processes.py:test_brownnoise', "no ensembles onchip"),
        ('test_processes.py:test_gaussian_white*', "no ensembles onchip"),
        ('test_processes.py:test_whitesignal*', "no ensembles onchip"),
        ('test_processes.py:test_reset', "no ensembles onchip"),
        ('test_processes.py:test_seed', "no ensembles onchip"),
        ('test_processes.py:test_present_input', "no ensembles onchip"),
        ('test_processes.py:TestPiecewise*', "no ensembles onchip"),
        ('test_simulator.py:test_steps', "no ensembles onchip"),
        ('test_simulator.py:test_time_absolute', "no ensembles onchip"),
        ('test_simulator.py:test_trange*', "no ensembles onchip"),
        ('test_simulator.py:test_probe_cache', "no ensembles onchip"),
        ('test_simulator.py:test_invalid_run_time', "no ensembles onchip"),
        ('test_simulator.py:test_sample_every*', "no ensembles onchip"),
        ('test_synapses.py:test_lowpass', "no ensembles onchip"),
        ('test_synapses.py:test_alpha', "no ensembles onchip"),
        ('test_synapses.py:test_triangle', "no ensembles onchip"),
        ('test_synapses.py:test_linearfilter', "no ensembles onchip"),
        ('utils/*test_ensemble.py:test_*_curves_direct_mode*',
         "no ensembles onchip"),
        ('utils/*test_network.py:*direct_mode_learning[learning_rule[123]*',
         "no ensembles onchip"),
        ('utils/*test_neurons.py:test_rates_*', "no ensembles onchip"),

        # accuracy
        ('test_actionselection.py:test_basic', "inaccurate"),
        ('test_am_[!s]*', "integrator instability"),
        ('test_ensemblearray.py:test_matrix_mul', "inaccurate"),
        ('test_product.py:test_sine_waves', "inaccurate"),
        ('test_workingmemory.py:test_inputgatedmemory', "inaccurate"),
        ('test_cortical.py:test_convolution', "inaccurate"),
        ('test_thalamus.py:test_routing', "inaccurate"),
        ('test_thalamus.py:test_nondefault_routing', "inaccurate"),
        ('test_connection.py:test_node_to_ensemble*', "inaccurate"),
        ('test_connection.py:test_neurons_to_node*', "inaccurate"),
        ('test_connection.py:test_function_and_transform', "inaccurate"),
        ('test_connection.py:test_weights*', "inaccurate"),
        ('test_connection.py:test_vector*', "inaccurate"),
        ('test_connection.py:test_slicing*', "inaccurate"),
        ('test_connection.py:test_function_output_size', "inaccurate"),
        ('test_connection.py:test_function_points', "inaccurate"),
        ('test_ensemble.py:test_scalar*', "inaccurate"),
        ('test_ensemble.py:test_vector*', "inaccurate"),
        ('test_learning_rules.py:test_pes_transform', "inaccurate"),
        ('test_learning_rules.py:test_slicing', "inaccurate"),
        ('test_neurons.py:test_alif', "inaccurate"),
        ('test_neurons.py:test_amplitude[[]LIF[]]', "inaccurate"),
        ('test_neurons.py:test_amplitude[[]SpikingRectifiedLinear[]]',
         "inaccurate"),
        ('test_presets.py:test_thresholding_preset', "inaccurate"),
        ('test_synapses.py:test_decoders', "inaccurate"),
        ('test_actionselection.py:test_basic', "inaccurate"),
        ('test_actionselection.py:test_thalamus', "inaccurate"),

        # builder inconsistencies
        ('test_connection.py:test_neurons_to_ensemble*',
         "transform shape not implemented"),
        ('test_connection.py:test_transform_probe',
         "transform shape not implemented"),
        ('test_connection.py:test_list_indexing*', "indexing bug?"),
        ('test_connection.py:test_prepost_errors', "learning bug?"),
        ('test_ensemble.py:test_gain_bias_warning', "warning not raised"),
        ('test_ensemble.py:*invalid_intercepts*', "BuildError not raised"),
        ('test_learning_rules.py:test_pes_ens_*', "learning bug?"),
        ('test_learning_rules.py:test_pes_weight_solver', "learning bug?"),
        ('test_learning_rules.py:test_pes_neuron_*', "learning bug?"),
        ('test_learning_rules.py:test_pes_multidim_error',
         "dict of learning rules not handled"),
        ('test_learning_rules.py:test_reset*', "learning bug?"),
        ('test_neurons.py:test_lif_min_voltage*', "lif.min_voltage ignored"),
        ('test_neurons.py:test_lif_zero_tau_ref', "lif.tau_ref ignored"),
        ('test_probe.py:test_input_probe', "shape mismatch"),
        ('test_probe.py:test_slice', "ObjView not handled properly"),
        ('test_probe.py:test_update_timing', "probe bug?"),
        ('test_solvers.py:test_nosolver*', "NoSolver bug"),

        # reset bugs
        ('test_neurons.py:test_reset*', "sim.reset not working correctly"),

        # non-PES learning rules
        ('test_learning_rules.py:test_unsupervised*',
         "non-PES learning rules not implemented"),
        ('test_learning_rules.py:test_dt_dependence*',
         "non-PES learning rules not implemented"),
        ('*voja*', "voja not implemented"),
        ('test_learning_rules.py:test_custom_type',
         "non-PES learning rules not implemented"),

        # Nengo bug
        ('test_simulator.py:test_entry_point',
         "logic should be more flexible"),

        # ensemble noise
        ('test_ensemble.py:test_noise*', "ensemble.noise not implemented"),

        # probe types
        ('test_connection.py:test_dist_transform',
         "probe type not implemented"),
        ('test_connection.py:test_decoder_probe',
         "probe type not implemented"),
        ('test_probe.py:test_defaults', "probe type not implemented"),
        ('test_probe.py:test_ensemble_encoders', "probe type not implemented"),

        # probe.sample_every
        ('test_integrator.py:test_integrator',
         "probe.sample_every not implemented"),
        ('test_oscillator.py:test_oscillator',
         "probe.sample_every not implemented"),
        ('test_ensemble.py:test_product*',
         "probe.sample_every not implemented"),
        ('test_neurons.py:test_dt_dependence*',
         "probe.sample_every not implemented"),
        ('test_probe.py:test_multiple_probes',
         "probe.sample_every not implemented"),

        # needs better place and route
        ('test_ensemble.py:test_eval_points_heuristic*',
         "max number of compartments exceeded"),
        ('test_neurons.py:test_lif*', "idxBits out of range"),
        ('test_basalganglia.py:test_basal_ganglia',
         "output_axons exceedecd max"),
        ('test_cortical.py:test_connect', "total synapse bits exceeded max"),
        ('test_cortical.py:test_transform', "total synapse bits exceeded max"),
        ('test_cortical.py:test_translate', "total synapse bits exceeded max"),
        ('test_memory.py:test_run', "total synapse bits exceeded max"),
        ('test_memory.py:test_run_decay', "total synapse bits exceeded max"),
        ('test_state.py:test_memory_run', "total synapse bits exceeded max"),
        ('test_state.py:test_memory_run_decay',
         "total synapse bits exceeded max"),
        ('test_bind.py:test_run', "exceeded max cores per chip on loihi"),

        # serialization / deserialization
        ('test_cache.py:*', "model pickling not implemented"),
        ('test_copy.py:test_pickle_model', "model pickling not implemented"),
        ('test_simulator.py:test_signal_init_values',
         "nengo.builder.Model instances not handled"),

        # progress bars
        ('test_simulator.py:test_simulator_progress_bars',
         "progress bars not implemented"),

        # utils.connection.target_function (deprecated)
        ('utils/tests/test_connection.py*',
         "target_function (deprecated) not working"),

        # removing passthroughs changes test behaviour
        ('test_connection.py:test_zero_activities_error',
         "decoded connection optimized away"),
        ('test_connection.py:test_function_returns_none_error',
         "decoded connection optimized away"),
    ]

    def __init__(  # noqa: C901
            self,
            network,
            dt=0.001,
            seed=None,
            model=None,
            precompute=False,
            target=None,
            progress_bar=None,
            remove_passthrough=True
    ):
        self.closed = True  # Start closed in case constructor raises exception
        if progress_bar is not None:
            raise NotImplementedError("progress bars not implemented")

        if model is None:
            # Call the builder to make a model
            self.model = Model(dt=float(dt), label="%s, dt=%f" % (network, dt))
        else:
            assert isinstance(model, Model), (
                "model is not type 'nengo_loihi.builder.Model'")
            self.model = model
            assert self.model.dt == dt

        max_rate = self.model.inter_rate * self.model.inter_n
        rtol = 1e-8  # allow for floating point inaccuracies
        if max_rate > (1. / self.dt) * (1 + rtol):
            raise BuildError("Simulator `dt` must be <= %s (got %s)"
                             % (1. / max_rate, self.dt))
        self.precompute = precompute
        self.networks = None
        self.sims = OrderedDict()
        self._run_steps = None

        if network is not None:
            nengo.rc.set("decoder_cache", "enabled", "False")
            config.add_params(network)

            # split the host into one, two or three networks
            self.networks = split(
                network,
                precompute,
                max_rate,
                self.model.inter_tau,
                remove_passthrough=remove_passthrough,
            )
            network = self.networks.chip

            self.model.chip2host_params = self.networks.chip2host_params

            self.chip = self.networks.chip
            self.host = self.networks.host
            self.host_pre = self.networks.host_pre

            if len(self.host_pre.all_objects) > 0:
                self.sims["host_pre"] = nengo.Simulator(self.host_pre,
                                                        dt=self.dt,
                                                        progress_bar=False,
                                                        optimize=False)

            if len(self.host.all_objects) > 0:
                self.sims["host"] = nengo.Simulator(
                    self.host, dt=self.dt, progress_bar=False, optimize=False)
            elif not precompute:
                # If there is no host and precompute=False, then all objects
                # must be on the chip, which is precomputable in the sense that
                # no communication has to happen with the host.
                # We could warn about this, but we want to avoid people having
                # to specify `precompute` unless they absolutely have to.
                self.precompute = True

            # Build the network into the model
            self.model.build(network)

        self._probe_outputs = self.model.params
        self.data = ProbeDict(self._probe_outputs)
        for sim in self.sims.values():
            self.data.add_fallback(sim.data)

        if seed is None:
            if network is not None and network.seed is not None:
                seed = network.seed + 1
            else:
                seed = np.random.randint(npext.maxint)

        if target is None:
            try:
                import nxsdk
                target = 'loihi'
            except ImportError:
                target = 'sim'
        self.target = target

        logger.info("Simulator target is %r", target)
        logger.info("Simulator precompute is %r", self.precompute)

        if target != "simreal":
            self.model.discretize()

        if target in ("simreal", "sim"):
            self.sims["emulator"] = CxSimulator(self.model, seed=seed)
        elif target == 'loihi':
            self.sims["loihi"] = LoihiSimulator(
                self.model, use_snips=not self.precompute, seed=seed)
        else:
            raise ValidationError("Must be 'simreal', 'sim', or 'loihi'",
                                  attr="target")

        assert "emulator" in self.sims or "loihi" in self.sims

        self.closed = False
        self.reset(seed=seed)

    def __del__(self):
        """Raise a ResourceWarning if we are deallocated while open."""
        if not self.closed:
            warnings.warn(
                "Simulator with model=%s was deallocated while open. Please "
                "close simulators manually to ensure resources are properly "
                "freed." % self.model, ResourceWarning)

    def __enter__(self):
        for sim in self.sims.values():
            sim.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for sim in self.sims.values():
            sim.__exit__(exc_type, exc_value, traceback)
        self.close()

    @property
    def dt(self):
        """(float) The step time of the simulator."""
        return self.model.dt

    @dt.setter
    def dt(self, dummy):
        raise ReadonlyError(attr='dt', obj=self)

    @property
    def n_steps(self):
        """(int) The current time step of the simulator."""
        return self._n_steps

    @property
    def time(self):
        """(float) The current time of the simulator."""
        return self._time

    def close(self):
        """Closes the simulator.

        Any call to `.Simulator.run`, `.Simulator.run_steps`,
        `.Simulator.step`, and `.Simulator.reset` on a closed simulator raises
        a ``SimulatorClosed`` exception.
        """

        for sim in self.sims.values():
            if not sim.closed:
                sim.close()
        self.closed = True

    def _probe(self):
        """Copy all probed signals to buffers."""
        self._probe_step_time()

        for probe in self.model.probes:
            if probe in self.networks.chip2host_params:
                continue
            assert probe.sample_every is None, (
                "probe.sample_every not implemented")
            assert ("loihi" not in self.sims
                    or "emulator" not in self.sims)
            cx_probe = self.model.objs[probe]['out']
            if "loihi" in self.sims:
                data = self.sims["loihi"].get_probe_output(cx_probe)
            elif "emulator" in self.sims:
                data = self.sims["emulator"].get_probe_output(cx_probe)
            # TODO: stop recomputing this all the time
            del self._probe_outputs[probe][:]
            self._probe_outputs[probe].extend(data)
            assert len(self._probe_outputs[probe]) == self.n_steps, (
                len(self._probe_outputs[probe]), self.n_steps)

    def _probe_step_time(self):
        self._time = self._n_steps * self.dt

    def reset(self, seed=None):
        """Reset the simulator state.

        Parameters
        ----------
        seed : int, optional
            A seed for all stochastic operators used in the simulator.
            This will change the random sequences generated for noise
            or inputs (e.g. from processes), but not the built objects
            (e.g. ensembles, connections).
        """
        if self.closed:
            raise SimulatorClosed("Cannot reset closed Simulator.")

        if seed is not None:
            self.seed = seed

        self._n_steps = 0
        self._time = 0

        # clear probe data
        for probe in self.model.probes:
            self._probe_outputs[probe] = []
        self.data.reset()

    def run(self, time_in_seconds):
        """Simulate for the given length of time.

        If the given length of time is not a multiple of ``dt``,
        it will be rounded to the nearest ``dt``. For example, if ``dt``
        is 0.001 and ``run`` is called with ``time_in_seconds=0.0006``,
        the simulator will advance one timestep, resulting in the actual
        simulator time being 0.001.

        The given length of time must be positive. The simulator cannot
        be run backwards.

        Parameters
        ----------
        time_in_seconds : float
            Amount of time to run the simulation for. Must be positive.
        """
        if time_in_seconds < 0:
            raise ValidationError("Must be positive (got %g)"
                                  % (time_in_seconds,), attr="time_in_seconds")

        steps = int(np.round(float(time_in_seconds) / self.dt))

        if steps == 0:
            warnings.warn("%g results in running for 0 timesteps. Simulator "
                          "still at time %g." % (time_in_seconds, self.time))
        else:
            logger.info("Running %s for %f seconds, or %d steps",
                        self.model.label, time_in_seconds, steps)
            self.run_steps(steps)

    def step(self):
        """Advance the simulator by 1 step (``dt`` seconds)."""
        self.run_steps(1)

    def _collect_receiver_info(self):
        spikes = []
        errors = {}
        for sender, receiver in self.networks.host2chip_senders.items():
            receiver.clear()
            for t, x in sender.queue:
                receiver.receive(t, x)
            del sender.queue[:]

            if hasattr(receiver, 'collect_spikes'):
                for cx_spike_input, t, spike_idxs in receiver.collect_spikes():
                    ti = round(t / self.model.dt)
                    spikes.append((cx_spike_input, ti, spike_idxs))
            if hasattr(receiver, 'collect_errors'):
                for probe, t, e in receiver.collect_errors():
                    conn = self.model.probe_conns[probe]
                    synapses = self.model.objs[conn]['decoders']
                    assert synapses.tracing
                    ti = round(t / self.model.dt)
                    errors_ti = errors.setdefault(ti, {})
                    if synapses in errors_ti:
                        errors_ti[synapses] += e
                    else:
                        errors_ti[synapses] = e.copy()

        errors = [(synapses, ti, e) for ti, ee in errors.items()
                  for synapses, e in ee.items()]
        return spikes, errors

    def _host2chip(self, sim):
        spikes, errors = self._collect_receiver_info()
        sim.host2chip(spikes, errors)

    def _chip2host(self, sim):
        probes_receivers = {  # map cx_probes to receivers
            self.model.objs[probe]['out']: receiver
            for probe, receiver in self.networks.chip2host_receivers.items()}
        sim.chip2host(probes_receivers)

    def _make_run_steps(self):
        if self._run_steps is not None:
            return
        assert "emulator" not in self.sims or "loihi" not in self.sims
        if "emulator" in self.sims:
            self._make_emu_run_steps()
        else:
            self._make_loihi_run_steps()

    def _make_emu_run_steps(self):
        host_pre = self.sims.get("host_pre", None)
        emulator = self.sims["emulator"]
        host = self.sims.get("host", None)

        if self.precompute:
            if host_pre is not None and host is not None:

                def emu_precomputed_host_pre_and_host(steps):
                    host_pre.run_steps(steps)
                    self._host2chip(emulator)
                    emulator.run_steps(steps)
                    self._chip2host(emulator)
                    host.run_steps(steps)
                self._run_steps = emu_precomputed_host_pre_and_host

            elif host_pre is not None:

                def emu_precomputed_host_pre_only(steps):
                    host_pre.run_steps(steps)
                    self._host2chip(emulator)
                    emulator.run_steps(steps)
                self._run_steps = emu_precomputed_host_pre_only

            elif host is not None:

                def emu_precomputed_host_only(steps):
                    emulator.run_steps(steps)
                    self._chip2host(emulator)
                    host.run_steps(steps)
                self._run_steps = emu_precomputed_host_only

            else:
                self._run_steps = emulator.run_steps

        else:
            assert host is not None, "Model is precomputable"

            def emu_bidirectional_with_host(steps):
                for _ in range(steps):
                    host.step()
                    self._host2chip(emulator)
                    emulator.step()
                    self._chip2host(emulator)
            self._run_steps = emu_bidirectional_with_host

    def _make_loihi_run_steps(self):
        host_pre = self.sims.get("host_pre", None)
        loihi = self.sims["loihi"]
        host = self.sims.get("host", None)

        if self.precompute:
            if host_pre is not None and host is not None:

                def loihi_precomputed_host_pre_and_host(steps):
                    host_pre.run_steps(steps)
                    self._host2chip(loihi)
                    loihi.run_steps(steps, blocking=True)
                    self._chip2host(loihi)
                    host.run_steps(steps)
                self._run_steps = loihi_precomputed_host_pre_and_host

            elif host_pre is not None:

                def loihi_precomputed_host_pre_only(steps):
                    host_pre.run_steps(steps)
                    self._host2chip(loihi)
                    loihi.run_steps(steps, blocking=True)
                self._run_steps = loihi_precomputed_host_pre_only

            elif host is not None:

                def loihi_precomputed_host_only(steps):
                    loihi.run_steps(steps, blocking=True)
                    self._chip2host(loihi)
                    host.run_steps(steps)
                self._run_steps = loihi_precomputed_host_only

            else:
                self._run_steps = loihi.run_steps

        else:
            assert host is not None, "Model is precomputable"

            def loihi_bidirectional_with_host(steps):
                loihi.run_steps(steps, blocking=False)
                for _ in range(steps):
                    host.step()
                    self._host2chip(loihi)
                    self._chip2host(loihi)
                logger.info("Waiting for run_steps to complete...")
                loihi.wait_for_completion()
                logger.info("run_steps completed")
            self._run_steps = loihi_bidirectional_with_host

    def run_steps(self, steps):
        """Simulate for the given number of ``dt`` steps.

        Parameters
        ----------
        steps : int
            Number of steps to run the simulation for.
        """
        if self.closed:
            raise SimulatorClosed("Simulator cannot run because it is closed.")
        if self._run_steps is None:
            self._make_run_steps()
        try:
            self._run_steps(steps)
        except Exception:
            if "loihi" in self.sims and self.sims["loihi"].use_snips:
                # Need to write to board, otherwise it will wait indefinitely
                h2c = self.sims["loihi"].nengo_io_h2c
                c2h = self.sims["loihi"].nengo_io_c2h

                print(traceback.format_exc())
                print("\nAttempting to end simulation...")

                for _ in range(steps):
                    h2c.write(h2c.numElements, [0] * h2c.numElements)
                    c2h.read(c2h.numElements)
                self.sims["loihi"].wait_for_completion()
                self.sims["loihi"].n2board.nxDriver.stopExecution()
                self.sims["loihi"].n2board.nxDriver.stopDriver()
            raise

        self._n_steps += steps
        logger.info("Finished running for %d steps", steps)
        self._probe()

    def trange(self, sample_every=None, dt=None):
        """Create a vector of times matching probed data.

        Note that the range does not start at 0 as one might expect, but at
        the first timestep (i.e., ``dt``).

        Parameters
        ----------
        sample_every : float, optional (Default: None)
            The sampling period of the probe to create a range for.
            If None, a time value for every ``dt`` will be produced.
        """
        period = 1 if sample_every is None else sample_every / self.dt
        steps = np.arange(1, self.n_steps + 1)
        return self.dt * steps[steps % period < 1]
