import collections
import logging
import warnings

import numpy as np

import nengo
import nengo.utils.numpy as npext
from nengo.exceptions import ReadonlyError, SimulatorClosed, ValidationError
from nengo.utils.compat import ResourceWarning

from nengo_loihi.builder import Model, INTER_RATE, INTER_N
import nengo_loihi.splitter as splitter

logger = logging.getLogger(__name__)


class ProbeDict(collections.Mapping):
    """Map from Probe -> ndarray

    This is more like a view on the dict that the simulator manipulates.
    However, for speed reasons, the simulator uses Python lists,
    and we want to return NumPy arrays. Additionally, this mapping
    is readonly, which is more appropriate for its purpose.
    """

    def __init__(self, raw, host_probe_dict=None):
        super(ProbeDict, self).__init__()
        self.raw = raw
        self.fallbacks = []
        self._cache = {}

    def add_fallback_dict(self, fallback):
        self.fallbacks.append(fallback)

    def __getitem__(self, key):
        if (key not in self._cache or
                len(self._cache[key]) != len(self.raw[key])):
            if key in self.raw:
                rval = self.raw[key]
                if isinstance(rval, list):
                    rval = np.asarray(rval)
                    rval.setflags(write=False)
                self._cache[key] = rval
            else:
                for fallback in self.fallbacks:
                    if key in fallback:
                        return fallback[key]
                raise KeyError(key)
        return self._cache[key]

    def __iter__(self):
        # TODO: this should also include self.host_probe_dict
        return iter(self.raw)

    def __len__(self):
        # TODO: this should also include self.host_probe_dict
        return len(self.raw)

    def __repr__(self):
        # TODO: this should also include self.host_probe_dict
        return repr(self.raw)

    def __str__(self):
        # TODO: this should also include self.host_probe_dict
        return str(self.raw)

    def reset(self):
        # TODO: this should also include self.host_probe_dict
        self._cache.clear()


class Simulator(object):
    # 'unsupported' defines features unsupported by a simulator.
    # The format is a list of tuples of the form `(test, reason)` with `test`
    # being a string with wildcards (*, ?, [abc], [!abc]) matched against Nengo
    # test paths and names, and `reason` is a string describing why the feature
    # is not supported by the backend. For example:
    #     unsupported = [('test_pes*', 'PES rule not implemented')]
    # would skip all tests whose names start with 'test_pes'.
    unsupported = []

    def __init__(self, network, dt=0.001, seed=None, model=None,  # noqa: C901
                 precompute=True, target=None, max_time=None):
        self.closed = True  # Start closed in case constructor raises exception

        # can only use one of the precompute methods
        assert not precompute or max_time is None

        if model is None:
            # Call the builder to make a model
            self.model = Model(dt=float(dt), label="%s, dt=%f" % (network, dt),
                               max_time=max_time)
        else:
            assert max_time is None or model.max_time == max_time
            self.model = model

        self.precompute = precompute

        self.chip2host_sent_steps = 0  # how many timesteps have been sent
        if network is not None:
            nengo.rc.set("decoder_cache", "enabled", "False")
            if max_time is None and not precompute:
                # we don't have a max_time, so we need online communication
                host, chip, h2c, c2h_params, c2h = splitter.split(
                    network, INTER_RATE, INTER_N)
                network = chip
                self.chip2host_receivers = c2h
                self.host2chip_senders = h2c
                self.model.chip2host_params.update(c2h_params)
                self.host_sim = nengo.Simulator(host, progress_bar=False)
            elif max_time is None and precompute:
                # split the host into two networks, to allow precomputing
                host, chip, h2c, c2h_params, c2h = splitter.split(
                    network, INTER_RATE, INTER_N)
                host_pre = splitter.split_pre_from_host(host)
                network = chip
                self.chip2host_receivers = c2h
                self.host2chip_senders = h2c
                self.model.chip2host_params.update(c2h_params)
                self.host_pre_sim = nengo.Simulator(host_pre,
                                                    progress_bar=False)
                self.host_post_sim = nengo.Simulator(host,
                                                     progress_bar=False)
            else:
                self.host_sim = None
                self.chip2host_receivers = {}
            # Build the network into the model
            self.model.build(network)

        self._probe_outputs = self.model.params
        self.data = ProbeDict(self._probe_outputs)
        if precompute:
            self.data.add_fallback_dict(self.host_pre_sim.data)
            self.data.add_fallback_dict(self.host_post_sim.data)
        elif self.host_sim is not None:
            self.data.add_fallback_dict(self.host_sim.data)

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
            self.simulator = self.model.get_simulator(seed=seed)
        elif target == 'sim':
            self.model.discretize()  # Make parameters fixed bit widths
            self.simulator = self.model.get_simulator(seed=seed)
        elif target == 'loihi':
            self.model.discretize()  # Make parameters fixed bit widths
            if not precompute and max_time is None:
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
            raise ValueError("Unrecognized target")

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
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.loihi is not None:
            self.loihi.__exit__(exc_type, exc_value, traceback)
        self.close()

    @property
    def dt(self):
        """(float) The time step of the simulator."""
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
        a `.SimulatorClosed` exception.
        """
        self.closed = True
        self.signals = None  # signals may no longer exist on some backends

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
        # self._n_steps = self.signals[self.model.step].item()
        # self._time = self.signals[self.model.time].item()
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

        # reset signals
        # for key in self.signals:
        #     self.signals.reset(key)

        # rebuild steps (resets ops with their own state, like Processes)
        # self.rng = np.random.RandomState(self.seed)
        # self._steps = [op.make_step(self.signals, self.dt, self.rng)
        #                for op in self._step_order]

        # clear probe data
        for probe in self.model.probes:
            self._probe_outputs[probe] = []
        self.data.reset()

        # self._probe_step_time()

    def run(self, time_in_seconds):
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
                self.loihi.run_steps(steps)
                self.handle_chip2host_communications()
                self.host_post_sim.run_steps(steps)
            elif self.host_sim is not None:
                self.loihi.create_io_snip()
                self.loihi.run_steps(steps, async=True)
                for i in range(steps):
                    self.host_sim.run_steps(1)
                    self.handle_host2chip_communications()
                    self.handle_chip2host_communications()

                print('Waiting for completion')
                self.loihi.nengo_io_h2c.write(1, [0])
                self.loihi.wait_for_completion()
                print("done")
            else:
                self.loihi.run_steps(steps)

        self._n_steps += steps
        self._probe()

    def handle_host2chip_communications(self):  # noqa: C901
        if self.simulator is not None:
            if self.precompute or self.host_sim is not None:
                # go through the list of host2chip connections
                for sender, receiver in self.host2chip_senders.items():
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
                                    items.append((
                                        sent_count, *output_axon[j]))
                        sent_count += 1
                    spike_input.sent_count = sent_count
                if len(items) > 0:
                    for info in sorted(items):
                        spike_input.spike_gen.addSpike(*info)
            elif self.host_sim is not None:
                to_send = []
                # go through the list of host2chip connections
                for sender, receiver in self.host2chip_senders.items():
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
                self.loihi.nengo_io_h2c.write(1, [len(to_send)])
                for spike in to_send:
                    assert spike[0] == 0
                    self.loihi.nengo_io_h2c.write(2, spike[1:3])

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
                raise NotImplementedError
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
                # TODO: Why is this needed?  Without the sleep, we randomly
                #  get communication failures....
                import time
                time.sleep(0.005)
                count = self.loihi.nengo_io_c2h_count
                time_step = self.loihi.nengo_io_c2h.read(1)[0]
                while time_step == 0:
                    time_step = self.loihi.nengo_io_c2h.read(1)[0]
                data = self.loihi.nengo_io_c2h.read(count-1)
                data = np.array(data)
                snip_range = self.loihi.nengo_io_snip_range
                for cx_probe, probe in self.cx_probe2probe.items():
                    x = data[snip_range[cx_probe]]
                    if cx_probe.key == 's':
                        if isinstance(probe.target, nengo.ensemble.Neurons):
                            t_ref = probe.target.ensemble.neuron_type.tau_ref
                            assert t_ref == 0.002
                            # TODO: generalize this to other tau_ref values
                            x = (x == 384)
                        else:
                            # interneurons have a refractory period of 1 step
                            x = (x == 128)
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
                raise NotImplementedError

    def trange(self, dt=None):
        """Create a vector of times matching probed data.

        Note that the range does not start at 0 as one might expect, but at
        the first timestep (i.e., ``dt``).

        Parameters
        ----------
        dt : float, optional (Default: None)
            The sampling period of the probe to create a range for.
            If None, the simulator's ``dt`` will be used.
        """
        dt = self.dt if dt is None else dt
        n_steps = int(self.n_steps * (self.dt / dt))
        return dt * np.arange(1, n_steps + 1)


class NumpySimulator(Simulator):
    def __init__(self, *args, **kwargs):
        assert 'target' not in kwargs
        kwargs['target'] = 'sim'
        super(NumpySimulator, self).__init__(*args, **kwargs)
