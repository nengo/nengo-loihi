import pytest
import nengo
import numpy as np

from nengo_loihi.encoder import BinaryEncoder


@pytest.mark.parametrize("n_bits", [1, 2, 4, 8, 16, 32])
def test_binary_encoder(n_bits):
    encoder = BinaryEncoder(n_bits)
    weights = encoder.make_weights(np.ones((1, 1)))

    with nengo.Network() as model:
        u = nengo.Node(output=lambda t: np.sin(2*np.pi*t))
        spikes = nengo.Node(size_in=1, output=encoder)
        y = nengo.Node(size_in=1)

        nengo.Connection(u, spikes, synapse=None)
        nengo.Connection(spikes, y, transform=weights.T, synapse=None)

        p_y = nengo.Probe(y, synapse=None)
        p_ideal = nengo.Probe(u, synapse=None)

    with nengo.Simulator(model) as sim:
        sim.run(1.0)

    # ensure that the input does not differ from its
    # binary-encoded + weighted version by more than 2^(-n_bits)
    # (i.e., all of the error is from quantization)
    abstol = 2.**(-n_bits)
    error = sim.data[p_y] - sim.data[p_ideal]
    assert np.all(np.abs(error) <= abstol)
