import numpy as np
import pytest
from nengo.exceptions import BuildError

from nengo_loihi.block import Axon, LoihiBlock, Synapse
from nengo_loihi.builder.validate import validate_block


def test_validate_block():
    # too many compartments
    block = LoihiBlock(1200)
    assert block.compartment.n_compartments > 1024
    with pytest.raises(BuildError, match="Number of compartments"):
        validate_block(block)

    # too many input axons
    block = LoihiBlock(410)
    block.add_synapse(Synapse(5000))
    with pytest.raises(BuildError, match="Input axons"):
        validate_block(block)

    # too many output axons
    block = LoihiBlock(410)
    synapse = Synapse(2500)
    axon = Axon(5000)
    axon.target = synapse
    block.add_synapse(synapse)
    block.add_axon(axon)
    with pytest.raises(BuildError, match="Output axons"):
        validate_block(block)

    # too many synapse bits
    block = LoihiBlock(600)
    synapse = Synapse(500)
    synapse.set_weights(np.ones((500, 600)))
    axon = Axon(500)
    axon.target = synapse
    block.add_synapse(synapse)
    block.add_axon(axon)
    with pytest.raises(BuildError, match="synapse bits"):
        validate_block(block)
