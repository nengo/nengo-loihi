from nengo_loihi.hardware.nxsdk_objects import LoihiSpikeInput


def test_strings():
    axon = LoihiSpikeInput.LoihiAxon(
        axon_type=16, chip_id=3, core_id=5, axon_id=6, atom=8
    )
    assert str(axon) == (
        "LoihiAxon(axon_type=16, chip_id=3, core_id=5, axon_id=6, atom=8)"
    )

    spike = LoihiSpikeInput.LoihiSpike(time=4, axon=axon)
    assert str(spike) == (
        "LoihiSpike(time=4, axon_type=16, chip_id=3, core_id=5, axon_id=6, atom=8)"
    )
