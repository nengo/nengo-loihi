from collections import OrderedDict
import logging
import traceback
import warnings

import nengo
from nengo.exceptions import (
    ReadonlyError,
    SimulatorClosed,
    ValidationError,
)
from nengo.simulator import ProbeDict as NengoProbeDict
import nengo.utils.numpy as npext
import numpy as np

from nengo_loihi.builder import Model
from nengo_loihi.builder.nengo_dl import HAS_DL, install_dl_builders
from nengo_loihi.compat import seed_network
import nengo_loihi.config as config
from nengo_loihi.discretize import discretize_model
from nengo_loihi.emulator import EmulatorInterface
from nengo_loihi.hardware import HardwareInterface, HAS_NXSDK
from nengo_loihi.nxsdk_obfuscation import d_func
from nengo_loihi.splitter import Split

logger = logging.getLogger(__name__)


class Simulator:
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
    network : Network
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
    precompute : bool, optional (Default: False)
        Whether model inputs should be precomputed to speed up simulation.
        When *precompute* is False, the simulator will be run one step
        at a time in order to use model outputs as inputs in other parts
        of the model.
    target : str, optional (Default: None)
        Whether the simulator should target the emulator (``'sim'``) or
        Loihi hardware (``'loihi'``). If None, *target* will default to
        ``'loihi'`` if NxSDK is installed, and the emulator if it is not.
    hardware_options : dict, optional (Default: {})
        Dictionary of additional configuration for the hardware.
        See `.hardware.HardwareInterface` for possible parameters.

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

    def __init__(  # noqa: C901
            self,
            network,
            dt=0.001,
            seed=None,
            model=None,
            precompute=False,
            target=None,
            progress_bar=None,
            remove_passthrough=True,
            hardware_options=None,
    ):
        # initialize values used in __del__ and close() first
        self.closed = True
        self.precompute = precompute
        self.network = network
        self.sims = OrderedDict()
        self._run_steps = None

        hardware_options = {} if hardware_options is None else hardware_options

        if progress_bar:
            warnings.warn("nengo-loihi does not support progress bars")

        if HAS_DL:
            install_dl_builders()

        if model is None:
            # Call the builder to make a model
            self.model = Model(dt=float(dt), label="%s, dt=%f" % (network, dt))
        else:
            assert isinstance(model, Model), (
                "model is not type 'nengo_loihi.builder.Model'")
            self.model = model
            assert self.model.dt == dt

        if network is None:
            raise ValidationError("network parameter must not be None",
                                  attr="network")

        config.add_params(network)

        # ensure seeds are identical to nengo
        # this has no effect for nengo<=2.8.0
        seed_network(network, seeds=self.model.seeds,
                     seeded=self.model.seeded)

        # determine how to split the host into one, two or three models
        self.model.split = Split(network,
                                 precompute=precompute,
                                 remove_passthrough=remove_passthrough)

        # Build the network into the model
        self.model.build(network)

        # Build the extra passthrough connections into the model
        passthrough = self.model.split.passthrough
        for conn in passthrough.to_add:
            # Note: connections added by the passthrough splitter do not
            # respect seeds
            self.model.seeds[conn] = None
            self.model.seeded[conn] = False
            self.model.build(conn)

        if len(self.model.host_pre.params):
            assert precompute
            self.sims["host_pre"] = nengo.Simulator(
                network=None,
                dt=self.dt,
                model=self.model.host_pre,
                progress_bar=False,
                optimize=False)
        elif precompute:
            warnings.warn("No precomputable objects. Setting "
                          "precompute=True has no effect.")

        if len(self.model.host.params):
            self.sims["host"] = nengo.Simulator(
                network=None,
                dt=self.dt,
                model=self.model.host,
                progress_bar=False,
                optimize=False)
        elif not precompute:
            # If there is no host and precompute=False, then all objects
            # must be on the chip, which is precomputable in the sense that
            # no communication has to happen with the host.
            # We could warn about this, but we want to avoid people having
            # to specify `precompute` unless they absolutely have to.
            self.precompute = True

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
            target = 'loihi' if HAS_NXSDK else 'sim'
        self.target = target

        logger.info("Simulator target is %r", target)
        logger.info("Simulator precompute is %r", self.precompute)

        if target != "simreal":
            discretize_model(self.model)

        if target in ("simreal", "sim"):
            self.sims["emulator"] = EmulatorInterface(self.model, seed=seed)
        elif target == 'loihi':
            assert HAS_NXSDK, "Must have NxSDK installed to use Loihi hardware"
            self.sims["loihi"] = HardwareInterface(
                self.model, use_snips=not self.precompute, seed=seed,
                **hardware_options)
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
            if probe in self.model.chip2host_params:
                continue
            assert probe.sample_every is None, (
                "probe.sample_every not implemented")
            assert ("loihi" not in self.sims
                    or "emulator" not in self.sims)
            loihi_probe = self.model.objs[probe]['out']
            if "loihi" in self.sims:
                data = self.sims["loihi"].get_probe_output(loihi_probe)
            elif "emulator" in self.sims:
                data = self.sims["emulator"].get_probe_output(loihi_probe)
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
        errors = OrderedDict()
        for sender, receiver in self.model.host2chip_senders.items():
            receiver.clear()
            for t, x in sender.queue:
                receiver.receive(t, x)
            del sender.queue[:]

            if hasattr(receiver, 'collect_spikes'):
                for spike_input, t, spike_idxs in receiver.collect_spikes():
                    ti = round(t / self.model.dt)
                    spikes.append((spike_input, ti, spike_idxs))
            if hasattr(receiver, 'collect_errors'):
                for probe, t, e in receiver.collect_errors():
                    conn = self.model.probe_conns[probe]
                    synapse = self.model.objs[conn]['decoders']
                    assert synapse.learning
                    ti = round(t / self.model.dt)
                    errors_ti = errors.setdefault(ti, OrderedDict())
                    if synapse in errors_ti:
                        errors_ti[synapse] += e
                    else:
                        errors_ti[synapse] = e.copy()

        errors = [(synapse, ti, e) for ti, ee in errors.items()
                  for synapse, e in ee.items()]
        return spikes, errors

    def _host2chip(self, sim):
        spikes, errors = self._collect_receiver_info()
        sim.host2chip(spikes, errors)

    def _chip2host(self, sim):
        probes_receivers = OrderedDict(  # map probes to receivers
            (self.model.objs[probe]['out'], receiver)
            for probe, receiver in self.model.chip2host_receivers.items())
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


    def dvs_hook(self, pos):
        pass

    def _dvs_chip2host(self, loihi):
        x = loihi.nengo_dvs_c2h.read(2)
        self.dvs_hook(x)

    def _make_loihi_run_steps(self):
        host_pre = self.sims.get("host_pre", None)
        loihi = self.sims["loihi"]
        host = self.sims.get("host", None)

        do_DVS = True
        if do_DVS:
            def loihi_dvs(steps):
                loihi.run_steps(steps, blocking=False)
                for _ in range(steps):
                    self._dvs_chip2host(loihi)
                logger.info("Waiting for run_steps to complete...")
                loihi.wait_for_completion()
                logger.info("run_steps completed")
            self._run_steps = loihi_dvs
            return



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
                d_func(self.sims["loihi"].nxsdk_board, b'bnhEcml2ZXI=',
                       b'c3RvcEV4ZWN1dGlvbg==')
                d_func(self.sims["loihi"].nxsdk_board, b'bnhEcml2ZXI=',
                       b'c3RvcERyaXZlcg==')
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
