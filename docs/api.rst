*************
API reference
*************

.. default-role:: obj

Main
====

.. autosummary::

   nengo_loihi.Simulator
   nengo_loihi.add_params
   nengo_loihi.BlockShape
   nengo_loihi.set_defaults

.. autoclass:: nengo_loihi.Simulator

.. autofunction:: nengo_loihi.add_params

.. autoclass:: nengo_loihi.BlockShape

.. autofunction:: nengo_loihi.set_defaults

Neurons
=======

.. autosummary::

   nengo_loihi.neurons.LoihiLIF
   nengo_loihi.neurons.LoihiSpikingRectifiedLinear
   nengo_loihi.neurons.NeuronOutputNoise
   nengo_loihi.neurons.LowpassRCNoise
   nengo_loihi.neurons.AlphaRCNoise

.. autoclass:: nengo_loihi.neurons.LoihiLIF

.. autoclass:: nengo_loihi.neurons.LoihiSpikingRectifiedLinear

.. autoclass:: nengo_loihi.neurons.NeuronOutputNoise

.. autoclass:: nengo_loihi.neurons.LowpassRCNoise

.. autoclass:: nengo_loihi.neurons.AlphaRCNoise

Decode neurons
==============

Decode neurons facilitate NEF-style connections on the chip.
The type of decode neurons used by a model can be set on `.builder.Model`.

.. autosummary::

   nengo_loihi.decode_neurons.DecodeNeurons
   nengo_loihi.decode_neurons.OnOffDecodeNeurons
   nengo_loihi.decode_neurons.NoisyDecodeNeurons
   nengo_loihi.decode_neurons.Preset5DecodeNeurons
   nengo_loihi.decode_neurons.Preset10DecodeNeurons

.. autoclass:: nengo_loihi.decode_neurons.DecodeNeurons

.. autoclass:: nengo_loihi.decode_neurons.OnOffDecodeNeurons

.. autoclass:: nengo_loihi.decode_neurons.NoisyDecodeNeurons

.. autoclass:: nengo_loihi.decode_neurons.Preset5DecodeNeurons

.. autoclass:: nengo_loihi.decode_neurons.Preset10DecodeNeurons

Builder
=======

The builder turns a Nengo network into a `.builder.Model`
appropriate for running on Loihi.
This model is mainly composed of `.LoihiBlock` objects,
which map parts of the network to Loihi compartments, axons, and synapses.
The model also contains `.LoihiInput` objects to provide input,
and `.LoihiProbe` objects to collect output.

.. autosummary::

   nengo_loihi.builder.Model
   nengo_loihi.builder.Builder
   nengo_loihi.block.LoihiBlock
   nengo_loihi.block.Compartment
   nengo_loihi.block.Axon
   nengo_loihi.block.Synapse
   nengo_loihi.inputs.LoihiInput
   nengo_loihi.inputs.SpikeInput
   nengo_loihi.probe.LoihiProbe

.. autoclass:: nengo_loihi.builder.Model

.. autoclass:: nengo_loihi.builder.Builder

.. autoclass:: nengo_loihi.block.LoihiBlock

.. autoclass:: nengo_loihi.block.Compartment

.. autoclass:: nengo_loihi.block.Axon

.. autoclass:: nengo_loihi.block.Synapse

.. autoclass:: nengo_loihi.inputs.LoihiInput

.. autoclass:: nengo_loihi.inputs.SpikeInput

.. autoclass:: nengo_loihi.probe.LoihiProbe

Discretization
--------------

.. autofunction:: nengo_loihi.builder.discretize.discretize_model

.. autofunction:: nengo_loihi.builder.discretize.discretize_block

.. autofunction:: nengo_loihi.builder.discretize.discretize_compartment

.. autofunction:: nengo_loihi.builder.discretize.discretize_synapse

.. autofunction:: nengo_loihi.builder.discretize.discretize_weights

.. autofunction:: nengo_loihi.builder.discretize.discretize_probe

Validation
----------

.. autofunction:: nengo_loihi.builder.validate.validate_model

.. autofunction:: nengo_loihi.builder.validate.validate_block

.. autofunction:: nengo_loihi.builder.validate.validate_compartment

.. autofunction:: nengo_loihi.builder.validate.validate_axon

.. autofunction:: nengo_loihi.builder.validate.validate_synapse

.. autofunction:: nengo_loihi.builder.validate.validate_synapse_cfg

.. autofunction:: nengo_loihi.builder.validate.validate_probe

Emulator
========

.. autosummary::

   nengo_loihi.emulator.EmulatorInterface
   nengo_loihi.emulator.interface.BlockInfo
   nengo_loihi.emulator.interface.IterableState
   nengo_loihi.emulator.interface.CompartmentState
   nengo_loihi.emulator.interface.NoiseState
   nengo_loihi.emulator.interface.SynapseState
   nengo_loihi.emulator.interface.AxonState
   nengo_loihi.emulator.interface.ProbeState

.. autoclass:: nengo_loihi.emulator.EmulatorInterface

.. autoclass:: nengo_loihi.emulator.interface.BlockInfo

.. autoclass:: nengo_loihi.emulator.interface.IterableState

.. autoclass:: nengo_loihi.emulator.interface.CompartmentState

.. autoclass:: nengo_loihi.emulator.interface.NoiseState

.. autoclass:: nengo_loihi.emulator.interface.SynapseState

.. autoclass:: nengo_loihi.emulator.interface.AxonState

.. autoclass:: nengo_loihi.emulator.interface.ProbeState

Hardware
========

.. autosummary::

   nengo_loihi.hardware.HardwareInterface

.. autoclass:: nengo_loihi.hardware.HardwareInterface

Allocators
----------

.. autosummary::

   nengo_loihi.hardware.allocators.Greedy
   nengo_loihi.hardware.allocators.RoundRobin

.. autoclass:: nengo_loihi.hardware.allocators.Greedy

.. autoclass:: nengo_loihi.hardware.allocators.RoundRobin

DVS
===

.. autosummary::

   nengo_loihi.dvs.DVSFileChipProcess
   nengo_loihi.dvs.DVSEvents

.. autoclass:: nengo_loihi.dvs.DVSFileChipProcess

.. autoclass:: nengo_loihi.dvs.DVSEvents
