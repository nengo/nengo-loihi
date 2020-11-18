import numpy as np
from nengo import Ensemble, SpikingRectifiedLinear
from nengo.dists import Choice

from nengo_loihi.block import LoihiBlock, Synapse
from nengo_loihi.builder.sparse_matrix import scale_matrix, stack_matrices
from nengo_loihi.neurons import LoihiSpikingRectifiedLinear


class DecodeNeurons:
    """Defines parameters for a group of decode neurons.

    DecodeNeurons are used on the chip to facilitate NEF-style connections,
    where activities from a neural ensemble are first transformed into a
    decoded value (which is stored in the activities and synapses of the
    spiking decode neurons), before being passed on to another ensemble
    (via that ensemble's encoders).

    Parameters
    ----------
    dt : float
        Time step used by the simulator.
    """

    def __init__(self, dt=0.001):
        self.dt = dt

    def __str__(self):
        return "%s(dt=%0.3g)" % (type(self).__name__, self.dt)

    def get_block(self, weights, block_label=None, syn_label=None):
        """Get a LoihiBlock for implementing neurons on the chip.

        Parameters
        ----------
        weights : (n, d) ndarray
            Weights that project the ``n`` inputs to the ``d`` dimensions
            represented by these neurons. Typically, the inputs will be neurons
            belonging to an Ensemble, and these weights will be decoders.
        block_label : string (Default: None)
            Optional label for the LoihiBlock.
        syn_label : string (Default: None)
            Optional label for the Synapse.

        Returns
        -------
        block : LoihiBlock
            The neurons on the chip.
        syn : Synapse
            The synapses connecting into the chip neurons.
        """
        raise NotImplementedError()

    def get_ensemble(self, dim, add_to_container=True):
        """Get a Nengo Ensemble for implementing neurons on the host.

        Parameters
        ----------
        dim : int
            Number of dimensions to be represented by these neurons.
        add_to_container : bool, optional (Default: True)
            Whether to add the ensemble to the currently active network.

        Returns
        -------
        ens : Ensemble
            An Ensemble for implementing these neurons in a Nengo network.
        """
        raise NotImplementedError()

    def get_post_encoders(self, encoders):
        """Encoders for post population that these neurons connect in to.

        Parameters
        ----------
        encoders : (n, d) ndarray
            Regular scaled encoders for the ensemble, which map the ensemble's
            ``d`` input dimensions to its ``n`` neurons.

        Returns
        -------
        decode_neuron_encoders : (?, n) ndarray
            Encoders for mapping these neurons to the post-ensemble's neurons.
            The number of rows depends on how ``get_post_inds`` is being used
            (i.e. there could be one row per neuron in this block, or there
            could be fewer rows with ``get_post_inds`` mapping multiple neurons
            to each row).
        """
        raise NotImplementedError()

    def get_post_inds(self, inds, d):
        """Indices for mapping neurons to post-encoder dimensions.

        Parameters
        ----------
        inds : list of ints
            Indices for mapping decode neuron dimensions to post-ensemble
            dimensions. Usually, this will be determined by a slice on the
            post ensemble in a connection (which maps the output of the
            transform/function to select dimensions on the post ensemble).
        d : int
            Number of dimensions in the post-ensemble.
        """
        raise NotImplementedError()


class OnOffDecodeNeurons(DecodeNeurons):
    """One or more pairs of on/off neurons per dimension.

    In this class itself, all the pairs in a dimension are identical. It can
    still be advantageous to have more than one pair per dimension, though,
    since this can allow all neurons to have lower firing rates and thus
    act more linearly (due to period aliasing at high firing rates). Subclasses
    may use pairs that are not identical (by adding noise or heterogeneity).

    Parameters
    ----------
    pairs_per_dim : int
        Number of repeated neurons per dimension. Currently, all DecodeNeuron
        classes use separate on/off neuron pairs for each dimension. This is
        the number of such pairs per dimension.
    dt : float
        Time step used by the simulator.
    rate : float (Default: None)
        Max firing rate of each neuron. By default, this is chosen so that
        the sum of all repeated neuron rates is ``1. / dt``, and thus as a
        group the neurons average one spike per timestep.
    is_input : bool (Default: False)
        Whether these decode neurons are being used to provide input.
    """

    def __init__(self, pairs_per_dim=1, dt=0.001, rate=None, is_input=False):
        super().__init__(dt=dt)

        self.pairs_per_dim = pairs_per_dim
        self.is_input = is_input

        self.rate = 1.0 / (self.dt * self.pairs_per_dim) if rate is None else rate
        self.scale = 1.0 / (self.dt * self.rate * self.pairs_per_dim)
        self.neuron_type = LoihiSpikingRectifiedLinear()

        gain = 0.5 * self.rate * np.ones(self.pairs_per_dim)
        bias = gain  # intercept of -1
        self.gain = gain.repeat(2)
        self.bias = bias.repeat(2)
        # ^ repeat for on/off neurons

    def __str__(self):
        return "%s(pairs_per_dim=%d, dt=%0.3g, rate=%0.3g)" % (
            type(self).__name__,
            self.pairs_per_dim,
            self.dt,
            self.rate,
        )

    def get_block(self, weights, block_label=None, syn_label=None):
        gain = self.gain * self.dt
        bias = self.bias * self.dt

        n, d = weights.shape
        n_neurons = 2 * d * self.pairs_per_dim
        block = LoihiBlock(n_neurons, label=block_label)
        block.compartment.configure_relu(dt=self.dt)
        block.compartment.bias[:] = bias.repeat(d)

        syn = Synapse(n, label=syn_label)
        weights2 = []
        for ga, gb in gain.reshape(self.pairs_per_dim, 2):
            weights2.extend([scale_matrix(weights, ga), scale_matrix(weights, -gb)])
        weights2 = stack_matrices(weights2, order="h")
        syn.set_weights(weights2)
        block.add_synapse(syn)

        return block, syn

    def get_ensemble(self, dim, add_to_container=True):
        if self.is_input and self.pairs_per_dim != 1:
            # To support this, we need to figure out how to deal with the
            # `post_inds` that map neurons to axons. Either we can do this
            # on the host, in which case we'd have inputs going to the chip
            # where we can have multiple spikes per axon per timestep, or we
            # need to do it on the chip with one input axon per neuron.
            raise NotImplementedError(
                "Input neurons with more than one neuron per dimension"
            )

        n_neurons = 2 * dim * self.pairs_per_dim
        encoders = np.vstack([np.eye(dim), -np.eye(dim)] * self.pairs_per_dim)
        return Ensemble(
            n_neurons,
            dim,
            neuron_type=SpikingRectifiedLinear(initial_state={"voltage": Choice([0])}),
            encoders=encoders,
            gain=self.gain.repeat(dim),
            bias=self.bias.repeat(dim),
            add_to_container=add_to_container,
        )

    def get_post_encoders(self, encoders):
        encoders = encoders * self.scale
        return np.vstack([encoders.T, -encoders.T])

    def get_post_inds(self, inds, d):
        return np.concatenate([inds, inds + d] * self.pairs_per_dim)


class NoisyDecodeNeurons(OnOffDecodeNeurons):
    """Uses multiple on/off neuron pairs per dimension, plus noise.

    The noise allows each on-off neuron pair to do something different. The
    population average is a better representation of the encoded value
    than can be achieved with a single on/off neuron pair (if the magnitude
    of the noise is correctly calibrated).

    Parameters
    ----------
    pairs_per_dim : int
        Number of repeated neurons per dimension. Currently, all DecodeNeuron
        classes use separate on/off neuron pairs for each dimension. This is
        the number of such pairs per dimension.
    dt : float
        Time step used by the simulator.
    rate : float (Default: None)
        Max firing rate of each neuron. By default, this is chosen so that
        the sum of all repeated neuron rates is ``1. / dt``, and thus as a
        group the neurons average one spike per timestep.
    noise_exp : float, optional (Default: -2.)
        Base-10 exponent for noise added to neuron voltages.
    """

    def __init__(self, pairs_per_dim, dt=0.001, rate=None, noise_exp=-2.0):
        super().__init__(pairs_per_dim=pairs_per_dim, dt=dt, rate=rate)
        self.noise_exp = noise_exp  # noise exponent for added voltage noise

    def __str__(self):
        return "%s(pairs_per_dim=%d, dt=%0.3g, rate=%0.3g, noise_exp=%0.3g)" % (
            type(self).__name__,
            self.pairs_per_dim,
            self.dt,
            self.rate,
            self.noise_exp,
        )

    def get_block(self, weights, block_label=None, syn_label=None):
        block, syn = super().get_block(
            weights, block_label=block_label, syn_label=syn_label
        )

        if self.noise_exp > -30:
            block.compartment.enable_noise[:] = 1
            block.compartment.noise_exp = self.noise_exp
            block.compartment.noise_at_membrane = 1

        return block, syn


class Preset5DecodeNeurons(OnOffDecodeNeurons):
    """Uses five heterogeneous on/off pairs with pre-set values per dimension.

    The script for configuring these values can be found at:
        nengo-loihi-sandbox/utils/interneuron_unidecoder_design.py
    """

    def __init__(self, dt=0.001, rate=None):
        super().__init__(pairs_per_dim=5, dt=dt, rate=rate)

        assert self.pairs_per_dim == 5
        intercepts = np.linspace(-0.8, 0.8, self.pairs_per_dim)
        max_rates = np.linspace(160, 70, self.pairs_per_dim)
        gain, bias = self.neuron_type.gain_bias(max_rates, intercepts)

        target_point = 0.85
        target_rate = np.sum(self.neuron_type.rates(target_point, gain, bias))
        self.scale = 1.08 * target_point / (self.dt * target_rate)
        # ^ TODO: why does this 1.08 factor help? found it empirically in
        # test_decode_neurons.test_add_inputs

        self.gain = gain.repeat(2)
        self.bias = bias.repeat(2)
        # ^ repeat for on/off neurons

    def __str__(self):
        return "%s(dt=%0.3g, rate=%0.3g)" % (type(self).__name__, self.dt, self.rate)


class Preset10DecodeNeurons(OnOffDecodeNeurons):
    """Uses ten heterogeneous on/off pairs with pre-set values per dimension.

    The script for configuring these values can be found at:
        nengo-loihi-sandbox/utils/interneuron_unidecoder_design.py
    """

    def __init__(self, dt=0.001, rate=None):
        super().__init__(pairs_per_dim=10, dt=dt, rate=rate)

        # Parameters determined by hyperopt
        assert self.pairs_per_dim == 10
        intercepts = np.linspace(-1.171, 0.484, self.pairs_per_dim)
        max_rates = np.linspace(171.186, 74.620, self.pairs_per_dim)
        gain, bias = self.neuron_type.gain_bias(max_rates, intercepts)

        target_point = 1.0
        target_rate = np.sum(self.neuron_type.rates(target_point, gain, bias))
        self.scale = 1.05 * target_point / (self.dt * target_rate)
        # ^ TODO: why does this 1.05 factor help? found it empirically in
        # test_decode_neurons.test_add_inputs

        self.gain = gain.repeat(2)
        self.bias = bias.repeat(2)
        # ^ repeat for on/off neurons

    def __str__(self):
        return "%s(dt=%0.3g, rate=%0.3g)" % (type(self).__name__, self.dt, self.rate)
