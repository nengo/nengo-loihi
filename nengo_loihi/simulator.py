import logging
import warnings

import numpy as np

import nengo
import nengo.utils.numpy as npext
from nengo.exceptions import (
    BuildError, ReadonlyError, SimulatorClosed, ValidationError)
from nengo.simulator import ProbeDict as NengoProbeDict
from nengo.utils.compat import ResourceWarning

from nengo_loihi.builder import Model
from nengo_loihi.loihi_cx import CxGroup
from nengo_loihi.splitter import PESModulatoryTarget, split
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
        assert key in target

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
    unsupported = []

    def __init__(self, network, dt=0.001, seed=None, model=None,  # noqa: C901
                 precompute=False, target=None):
        self.closed = True  # Start closed in case constructor raises exception

        if model is None:
            # Call the builder to make a model
            self.model = Model(dt=float(dt), label="%s, dt=%f" % (network, dt))
        else:
            self.model = model
            assert self.model.dt == dt

        max_rate = self.model.inter_rate * self.model.inter_n
        rtol = 1e-8  # allow for floating point inaccuracies
        if max_rate > (1. / self.dt) * (1 + rtol):
            raise BuildError("Simulator `dt` must be <= %s (got %s)"
                             % (1. / max_rate, self.dt))
        self.precompute = precompute
        self.networks = None

        self.chip2host_sent_steps = 0  # how many timesteps have been sent
        if network is not None:
            nengo.rc.set("decoder_cache", "enabled", "False")
            config.add_params(network)

            # split the host into two or three networks
            self.networks = split(
                network, precompute, max_rate, self.model.inter_tau)
            network = self.networks.chip

            self.chip2host_receivers = self.networks.chip2host_receivers
            self.host2chip_senders = self.networks.host2chip_senders
            self.model.chip2host_params = self.networks.chip2host_params

            self.chip = self.networks.chip
            self.host = self.networks.host
            self.host_pre = self.networks.host_pre

            if precompute:
                self.host_pre_sim = nengo.Simulator(
                    self.host_pre, dt=self.dt, progress_bar=False)
                self.host_post_sim = nengo.Simulator(
                    self.host, dt=self.dt, progress_bar=False)
            else:
                self.host_sim = nengo.Simulator(
                    self.host, dt=self.dt, progress_bar=False)

            # Build the network into the model
            self.model.build(network)

        self._probe_outputs = self.model.params
        self.data = ProbeDict(self._probe_outputs)
        if precompute:
            self.data.add_fallback(self.host_pre_sim.data)
            self.data.add_fallback(self.host_post_sim.data)
        elif self.host_sim is not None:
            self.data.add_fallback(self.host_sim.data)

        if seed is None:
            if network is not None and network.seed is not None:
                seed = network.seed + 1
            else:
                seed = np.random.randint(npext.maxint)

        self.loihi = None
        self.simulator = None

        if target is None:
            try:
                import nxsdk
                target = 'loihi'
            except ImportError:
                target = 'sim'

        if target == 'simreal':
            logger.info("Using real-valued simulator")
            self.simulator = self.model.get_simulator(seed=seed)
        elif target == 'sim':
            logger.info("Using discretized simulator")
            self.model.discretize()  # Make parameters fixed bit widths
            self.simulator = self.model.get_simulator(seed=seed)
        elif target == 'loihi':
            logger.info(
                "Using Loihi hardware with precompute=%s", self.precompute)
            self.model.discretize()  # Make parameters fixed bit widths
            if not precompute:
                # tag all probes as being snipbased
                #  (having normal probes at the same time as snips
                #   seems to cause problems)
                for group in self.model.cx_groups.keys():
                    for cx_probe in group.probes:
                        cx_probe.use_snip = True
                # create a place to store data from snip probes
                self.snip_probes = {}
                for probe in network.all_probes:
                    self.snip_probes[probe] = []

                # create a list of all the CxProbes and their nengo.Probes
                self.cx_probe2probe = {}
                for obj in self.model.objs.keys():
                    if isinstance(obj, nengo.Probe):
                        # actual nengo.Probes on chip objects
                        cx_probe = self.model.objs[obj]['out']
                        self.cx_probe2probe[cx_probe] = obj
                for probe in self.chip2host_receivers.keys():
                    # probes used for chip->host communication
                    cx_probe = self.model.objs[probe]['out']
                    self.cx_probe2probe[cx_probe] = probe

            self.loihi = self.model.get_loihi(seed=seed)
        else:
            raise ValidationError("Must be 'simreal', 'sim', or 'loihi'",
                                  attr="target")

        assert self.simulator or self.loihi

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
        if self.loihi is not None:
            self.loihi.__enter__()
        if self.simulator is not None:
            self.simulator.__enter__()
        if self.precompute:
            self.host_pre_sim.__enter__()
            self.host_post_sim.__enter__()
        else:
            self.host_sim.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.loihi is not None:
            self.loihi.__exit__(exc_type, exc_value, traceback)
        if self.simulator is not None:
            self.simulator.__exit__(exc_type, exc_value, traceback)
        if self.precompute:
            self.host_pre_sim.__exit__(exc_type, exc_value, traceback)
            self.host_post_sim.__exit__(exc_type, exc_value, traceback)
        else:
            self.host_sim.__exit__(exc_type, exc_value, traceback)
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
        if self.loihi is not None and not self.loihi.closed:
            self.loihi.close()
        if self.simulator is not None and not self.simulator.closed:
            self.simulator.close()
        if self.precompute:
            if not self.host_pre_sim.closed:
                self.host_pre_sim.close()
            if not self.host_post_sim.closed:
                self.host_post_sim.close()
        elif not self.host_sim.closed:
            self.host_sim.close()

        self.closed = True

    def _probe(self):
        """Copy all probed signals to buffers."""
        self._probe_step_time()

        for probe in self.model.probes:
            if probe in self.model.chip2host_params:
                continue
            assert probe.sample_every is None
            assert self.loihi is None or self.simulator is None
            if self.loihi is not None:
                cx_probe = self.loihi.model.objs[probe]['out']
                if cx_probe.use_snip:
                    data = self.snip_probes[probe]
                    if probe.synapse is not None:
                        data = probe.synapse.filt(data, dt=self.dt, y0=0)
                else:
                    data = self.loihi.get_probe_output(probe)
            elif self.simulator is not None:
                data = self.simulator.get_probe_output(probe)
            # TODO: stop recomputing this all the time
            del self._probe_outputs[probe][:]
            self._probe_outputs[probe].extend(data)
            assert len(self._probe_outputs[probe]) == self.n_steps

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

    def run_steps(self, steps):
        """Simulate for the given number of ``dt`` steps.

        Parameters
        ----------
        steps : int
            Number of steps to run the simulation for.
        """
        if self.closed:
            raise SimulatorClosed("Simulator cannot run because it is closed.")

        if self.simulator is not None:
            if self.precompute:
                self.host_pre_sim.run_steps(steps)
                self.handle_host2chip_communications()
                self.simulator.run_steps(steps)
                self.handle_chip2host_communications()
                self.host_post_sim.run_steps(steps)
            elif self.host_sim is None:
                self.simulator.run_steps(steps)
            else:
                for i in range(steps):
                    self.host_sim.step()
                    self.handle_host2chip_communications()
                    self.simulator.step()
                    self.handle_chip2host_communications()
        elif self.loihi is not None:
            if self.precompute:
                self.host_pre_sim.run_steps(steps)
                self.handle_host2chip_communications()
                self.loihi.run_steps(steps, blocking=True)
                self.handle_chip2host_communications()
                self.host_post_sim.run_steps(steps)
            elif self.host_sim is not None:
                self.loihi.create_io_snip()
                self.loihi.run_steps(steps, blocking=False)
                for i in range(steps):
                    self.host_sim.run_steps(1)
                    self.handle_host2chip_communications()
                    self.handle_chip2host_communications()

                logger.info("Waiting for completion")
                self.loihi.wait_for_completion()
                logger.info("done")
            else:
                self.loihi.run_steps(steps, blocking=True)

        self._n_steps += steps
        logger.info("Finished running for %d steps", steps)
        self._probe()

    def handle_host2chip_communications(self):  # noqa: C901
        if self.simulator is not None:
            if self.precompute or self.host_sim is not None:
                # go through the list of host2chip connections
                for sender, receiver in self.host2chip_senders.items():
                    learning_rate = 50  # This is set to match hardware
                    if isinstance(receiver, PESModulatoryTarget):
                        for t, x in sender.queue:
                            probe = receiver.target
                            conn = self.model.probe_conns[probe]
                            dec_syn = self.model.objs[conn]['decoders']
                            assert dec_syn.tracing

                            z = self.simulator.z[dec_syn]
                            x = np.hstack([-x, x])

                            delta_w = np.outer(z, x) * learning_rate

                            for i, w in enumerate(dec_syn.weights):
                                w += delta_w[i].astype('int32')
                    else:
                        for t, x in sender.queue:
                            receiver.receive(t, x)
                    del sender.queue[:]
        elif self.loihi is not None:
            if self.precompute:
                # go through the list of host2chip connections
                items = []
                for sender, receiver in self.host2chip_senders.items():
                    for t, x in sender.queue:
                        receiver.receive(t, x)
                    del sender.queue[:]
                    spike_input = receiver.cx_spike_input
                    sent_count = spike_input.sent_count
                    while sent_count < len(spike_input.spikes):
                        for j, s in enumerate(spike_input.spikes[sent_count]):
                            if s:
                                for output_axon in spike_input.axon_ids:
                                    items.append(
                                        (sent_count,) + output_axon[j])
                        sent_count += 1
                    spike_input.sent_count = sent_count
                if len(items) > 0:
                    for info in sorted(items):
                        spike_input.spike_gen.addSpike(*info)
            elif self.host_sim is not None:
                to_send = []
                errors = []
                # go through the list of host2chip connections
                for sender, receiver in self.host2chip_senders.items():
                    if isinstance(receiver, PESModulatoryTarget):
                        for t, x in sender.queue:
                            x = (100 * x).astype(int)
                            x = np.clip(x, -100, 100, out=x)
                            probe = receiver.target
                            conn = self.model.probe_conns[probe]
                            dec_cx = self.model.objs[conn]['decoded']
                            for core in self.loihi.board.chips[0].cores:
                                for group in core.groups:
                                    if group == dec_cx:
                                        # TODO: assumes one group per core
                                        coreid = core.learning_coreid
                                    break

                            assert coreid is not None

                            errors.append([coreid, len(x)] + x.tolist())
                        del sender.queue[:]

                    else:
                        for t, x in sender.queue:
                            receiver.receive(t, x)
                        del sender.queue[:]
                        spike_input = receiver.cx_spike_input
                        sent_count = spike_input.sent_count
                        axon_ids = spike_input.axon_ids
                        spikes = spike_input.spikes
                        while sent_count < len(spikes):
                            for j, s in enumerate(spikes[sent_count]):
                                if s:
                                    for output_axon in axon_ids:
                                        to_send.append(output_axon[j])
                            sent_count += 1
                        spike_input.sent_count = sent_count

                max_spikes = self.loihi.snip_max_spikes_per_step
                if len(to_send) > max_spikes:
                    warnings.warn("Too many spikes (%d) sent in one time "
                                  "step.  Increase the value of "
                                  "snip_max_spikes_per_step (currently "
                                  "set to %d)" % (len(to_send), max_spikes))
                    del to_send[max_spikes:]

                msg = [len(to_send)]
                for spike in to_send:
                    assert spike[0] == 0
                    msg.extend(spike[1:3])
                for error in errors:
                    msg.extend(error)
                self.loihi.nengo_io_h2c.write(len(msg), msg)

    def handle_chip2host_communications(self):  # noqa: C901
        if self.simulator is not None:
            if self.precompute or self.host_sim is not None:
                # go through the list of chip2host connections
                i = self.chip2host_sent_steps
                increment = None
                for probe, receiver in self.chip2host_receivers.items():
                    # extract the probe data from the simulator
                    cx_probe = self.simulator.model.objs[probe]['out']

                    x = self.simulator.probe_outputs[cx_probe][i:]
                    if len(x) > 0:
                        if increment is None:
                            increment = len(x)
                        else:
                            assert increment == len(x)
                        if cx_probe.weights is not None:
                            x = np.dot(x, cx_probe.weights)

                        for j in range(len(x)):
                            receiver.receive(self.dt * (i + j + 2), x[j])
                if increment is not None:
                    self.chip2host_sent_steps += increment
            else:
                raise NotImplementedError()
        elif self.loihi is not None:
            if self.precompute:
                # go through the list of chip2host connections
                increment = None
                for probe, receiver in self.chip2host_receivers.items():
                    # extract the probe data from the simulator
                    cx_probe = self.loihi.model.objs[probe]['out']
                    n2probe = self.loihi.board.probe_map[cx_probe]
                    x = np.column_stack([
                        p.timeSeries.data[self.chip2host_sent_steps:]
                        for p in n2probe])
                    if len(x) > 0:
                        if increment is None:
                            increment = len(x)
                        else:
                            assert increment == len(x)
                        if cx_probe.weights is not None:
                            x = np.dot(x, cx_probe.weights)
                        for j in range(len(x)):
                            receiver.receive(
                                self.dt * (self.chip2host_sent_steps + j + 2),
                                x[j])
                if increment is not None:
                    self.chip2host_sent_steps += increment
            elif self.host_sim is not None:
                count = self.loihi.nengo_io_c2h_count
                data = self.loihi.nengo_io_c2h.read(count)
                time_step, data = data[0], np.array(data[1:])
                snip_range = self.loihi.nengo_io_snip_range
                for cx_probe, probe in self.cx_probe2probe.items():
                    x = data[snip_range[cx_probe]]
                    if cx_probe.key == 's':
                        if isinstance(cx_probe.target, CxGroup):
                            refract_delays = cx_probe.target.refractDelay
                        else:
                            refract_delays = 1

                        # Loihi uses the voltage value to indicate where we
                        # are in the refractory period. We want to find neurons
                        # starting their refractory period.
                        x = (x == refract_delays * 128)
                    if cx_probe.weights is not None:
                        x = np.dot(x, cx_probe.weights)
                    receiver = self.chip2host_receivers.get(probe, None)
                    if receiver is not None:
                        # chip->host
                        receiver.receive(self.dt*(time_step), x)
                    else:
                        # onchip probes
                        self.snip_probes[probe].append(x)
            else:
                raise NotImplementedError()

    def trange(self, sample_every=None):
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
