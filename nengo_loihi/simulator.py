import logging
import warnings
from collections import Mapping

import numpy as np

# import nengo.utils.numpy as npext
from nengo.exceptions import ReadonlyError, SimulatorClosed, ValidationError
from nengo.utils.compat import range, ResourceWarning

logger = logging.getLogger(__name__)


class Simulator(object):

    # 'unsupported' defines features unsupported by a simulator.
    # The format is a list of tuples of the form `(test, reason)` with `test`
    # being a string with wildcards (*, ?, [abc], [!abc]) matched against Nengo
    # test paths and names, and `reason` is a string describing why the feature
    # is not supported by the backend. For example:
    #     unsupported = [('test_pes*', 'PES rule not implemented')]
    # would skip all test whose names start with 'test_pes'.
    unsupported = []

    def __init__(self, network, dt=0.001, seed=None, model=None):
        self.closed = True  # Start closed in case constructor raises exception

        if model is None:
            self.model = Model(dt=float(dt), label="%s, dt=%f" % (network, dt))
        else:
            self.model = model

        if network is not None:
            # Build the network into the model
            self.model.build(network)

        self.model.to_loihi()

        if seed is None:
            if network is not None and network.seed is not None:
                seed = network.seed + 1
            else:
                seed = np.random.randint(npext.maxint)

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
        return self

    def __exit__(self, exc_type, exc_value, traceback):
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
            period = (1 if probe.sample_every is None else
                      probe.sample_every / self.dt)
            if self.n_steps % period < 1:
                tmp = self.signals[self.model.sig[probe]['in']].copy()
                self._probe_outputs[probe].append(tmp)

    def _probe_step_time(self):
        self._n_steps = self.signals[self.model.step].item()
        self._time = self.signals[self.model.time].item()

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

        # reset signals
        # for key in self.signals:
        #     self.signals.reset(key)

        # rebuild steps (resets ops with their own state, like Processes)
        # self.rng = np.random.RandomState(self.seed)
        # self._steps = [op.make_step(self.signals, self.dt, self.rng)
        #                for op in self._step_order]

        # clear probe data
        # for probe in self.model.probes:
        #     self._probe_outputs[probe] = []
        # self.data.reset()

        # self._probe_step_time()

    def run(self, time_in_seconds):
        pass

        # if time_in_seconds < 0:
        #     raise ValidationError("Must be positive (got %g)"
        #                           % (time_in_seconds,), attr="time_in_seconds")

        # steps = int(np.round(float(time_in_seconds) / self.dt))

        # if steps == 0:
        #     warnings.warn("%g results in running for 0 timesteps. Simulator "
        #                   "still at time %g." % (time_in_seconds, self.time))
        # else:
        #     logger.info("Running %s for %f seconds, or %d steps",
        #                 self.model.label, time_in_seconds, steps)
        #     self.run_steps(steps, progress_bar=progress_bar)

    # def run_steps(self, steps):
    #     for i in range(steps):
    #         self.step()
    #         progress.step()

    def step(self):
        """Advance the simulator by 1 step (``dt`` seconds)."""
        if self.closed:
            raise SimulatorClosed("Simulator cannot run because it is closed.")

        self.model.cxSimulator.step()

        # self._probe()

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
