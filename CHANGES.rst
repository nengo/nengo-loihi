***************
Release history
***************

.. Changelog entries should follow this format:

   version (release date)
   ======================

   **section**

   - One-line description of change (link to Github issue/PR)

.. Changes should be organized in one of several sections:

   - Added
   - Changed
   - Deprecated
   - Removed
   - Fixed

1.0.0 (January 20, 2021)
========================

*Compatible with Nengo 3.1.0*

*Compatible with NxSDK 0.9.0 - 0.9.9*

**Added**

- Added Legendre Memory Unit example.
  (`#267 <https://github.com/nengo/nengo-loihi/pull/267>`__)
- Added a ``timers`` attribute to ``Simulator`` that tracks the wall time
  taken by various parts of the model, including build time and run time.
  (`#260 <https://github.com/nengo/nengo-loihi/pull/260>`__)
- Added the ``pop_type`` configuration option to the ``Connection`` config.
  See `nengo_loihi.add_params
  <https://www.nengo.ai/nengo-loihi/api.html#nengo_loihi.add_params>`__
  for details. (`#261 <https://github.com/nengo/nengo-loihi/pull/261>`__)
- Added the ``block_shape`` configuration option to the ``Ensemble`` config,
  and added the ``nengo_loihi.BlockShape`` class to set that option.
  See `nengo_loihi.add_params
  <https://www.nengo.ai/nengo-loihi/api.html#nengo_loihi.add_params>`__
  for details. (`#264 <https://github.com/nengo/nengo-loihi/pull/264>`__)
- Added the ``Greedy`` allocator, which uses all cores on one chip before
  proceeding to the next chip. ``Greedy`` is now the default allocator.
  (`#266 <https://github.com/nengo/nengo-loihi/pull/266>`__)
- Added the ``n_chips`` parameter to ``HardwareInterface`` for specifying
  the number of chips on the board.
  (`#266 <https://github.com/nengo/nengo-loihi/pull/266>`__)
- Added ``Model.utilization_summary`` to provide a summary of how much of
  the various resources of each block are utilized.
  (`#279 <https://github.com/nengo/nengo-loihi/pull/279>`__)
- Added some `new documentation <https://www.nengo.ai/nengo-loihi/tips>`__ focused on
  mapping large models onto Loihi.
  (`#279 <https://github.com/nengo/nengo-loihi/pull/279>`__)
- Added a
  `new example <https://www.nengo.ai/nengo-loihi/examples/cifar10-convnet.html>`_
  showing how to map larger convolutional networks to Loihi (applied to CIFAR-10
  dataset). (`#282 <https://github.com/nengo/nengo-loihi/pull/282>`__)
- Added a
  `Keras example <https://www.nengo.ai/nengo-loihi/examples/keras-to-loihi.html>`_
  showing how to directly convert a Keras convolutional network to run on Loihi
  using the NengoDL Keras Converter.
  (`#281 <https://github.com/nengo/nengo-loihi/pull/281>`__)
- Added support for NxSDK 0.9.8 and 0.9.9.
  (`#296 <https://github.com/nengo/nengo-loihi/pull/296>`__)
- Added support for the ``nengo.RegularSpiking`` neuron type, when using ``LIFRate``
  or ``RectifiedLinear`` as the base type (these are equivalent to ``LIF`` and
  ``SpikingRectifiedLinear``, respectively).
  (`#296 <https://github.com/nengo/nengo-loihi/pull/296>`__)
- Added ``nengo_loihi.dvs.DVSFileChipProcess``, for getting input from a pre-recorded
  DVS file and sending it to the Loihi board.
  (`#306 <https://github.com/nengo/nengo-loihi/pull/306>`__)

**Changed**

- We improved performance when ``precompute=False`` through better spike packing,
  larger packets, and communicating to the host over a socket.
  (`#260 <https://github.com/nengo/nengo-loihi/pull/260>`__)
- The ``precompute`` argument of ``Simulator`` now defaults to ``None``
  and will be automatically set to ``True`` if the model can be precomputed.
  (`#260 <https://github.com/nengo/nengo-loihi/pull/260>`__)
- Added the ``add_to_container`` argument to ``DecodeNeurons.get_ensemble``,
  which makes it easier to add a decode neurons ensemble to a network.
  (`#260 <https://github.com/nengo/nengo-loihi/pull/260>`__)
- ``Convolution`` transforms with ``channels_last=True`` now work with outputs
  up to 1024 neurons.
  (`#261 <https://github.com/nengo/nengo-loihi/pull/261>`__)
- The ``Probe`` has been renamed to ``LoihiProbe`` to mirror the ``LoihiBlock``
  and ``LoihiInput`` classes, which are conceptually very similar.
  It has also been moved from ``nengo_loihi.block`` to ``nengo_loihi.probe``.
  (`#264 <https://github.com/nengo/nengo-loihi/pull/264>`__)
- We now raise a more informative error if connecting to Loihi hardware fails.
  (`#264 <https://github.com/nengo/nengo-loihi/pull/264>`__)
- It is now possible to build models with larger ensembles because
  the builder can now split large Loihi blocks into smaller ones.
  (`#264 <https://github.com/nengo/nengo-loihi/pull/264>`__)
- Modules for discretizing and validating models have been moved to the
  ``builder`` directory.
  (`#264 <https://github.com/nengo/nengo-loihi/pull/264>`__)
- It is now possible to use multi-chip allocators with all models,
  including those that cannot be precomputed.
  (`#266 <https://github.com/nengo/nengo-loihi/pull/266>`__)
- Allocators like ``RoundRobin`` no longer accept the ``n_chips`` parameter.
  Instead, the ``__call__`` method accepts ``n_chips``.
  (`#266 <https://github.com/nengo/nengo-loihi/pull/266>`__)
- NengoLoihi now supports NxSDK version 0.9.5.rc1.
  (`#272 <https://github.com/nengo/nengo-loihi/pull/272>`__)
- NengoLoihi now supports Nengo version 3.1. Support for Nengo 3.0 has been dropped.
  (`#296 <https://github.com/nengo/nengo-loihi/pull/296>`__)
- Minimum NengoDL version is now 3.4.0.
  (`#296 <https://github.com/nengo/nengo-loihi/pull/296>`__)

**Removed**

- Removed the ``OneToOne`` allocator, which only worked for one chip.
  The ``Greedy`` allocator is identical for models that fit on one chip.
  (`#266 <https://github.com/nengo/nengo-loihi/pull/266>`__)

**Fixed**

- We no longer create a spike generator if we are communicating through Snips.
  (`#260 <https://github.com/nengo/nengo-loihi/pull/260>`__)
- Fixed an issue in which ignored axons were still having an effect in
  convolutional networks where not all input pixels are used in the output.
  (`#261 <https://github.com/nengo/nengo-loihi/pull/261>`__)
- Fixed an issue that prevented population spikes to be sent to the chip when
  ``precompute=True``. (`#261 <https://github.com/nengo/nengo-loihi/pull/261>`__)
- Fixed a bug preventing making sparse connections to an ensemble.
  (`#245 <https://github.com/nengo/nengo-loihi/issues/245>`__,
  `#246 <https://github.com/nengo/nengo-loihi/pull/246>`__)
- We now ignore TensorFlow and NengoDL if an incompatible version is installed
  rather than exiting with an exception.
  (`#264 <https://github.com/nengo/nengo-loihi/pull/264>`__)
- We now shut down the connection to the board more reliably, which should
  reduce the number of cases in which a model hangs indefinitely.
  (`#266 <https://github.com/nengo/nengo-loihi/pull/266>`__)
- ``LoihiLIF`` neurons now round ``tau_rc`` to mimic the discretization that occurs on
  Loihi, for more accurate simulation in Nengo (this was already done in the rate
  equation and NengoDL implementation of this neuron).
  (`#275 <https://github.com/nengo/nengo-loihi/pull/275>`__)
- ``LoihiLIF`` and ``LoihiSpikingRectifiedLinear`` now add the appropriate NengoDL
  builders when instantiated, so they work properly if used in NengoDL without making
  a NengoLoihi simulator.
  (`#248 <https://github.com/nengo/nengo-loihi/issues/248>`__,
  `#275 <https://github.com/nengo/nengo-loihi/pull/275>`__)
- Fixed bug when probing sliced objects.
  (`#284 <https://github.com/nengo/nengo-loihi/pull/284>`__)
- Fixed bug when connecting to a single neuron ensemble with a single scalar
  weight. (`#287 <https://github.com/nengo/nengo-loihi/pull/287>`__)
- Added an error if more than 32 "populations" (e.g. convolutional filters) are used
  with ``pop_type=16`` axons, since this is not yet supported by NxSDK.
  (`#286 <https://github.com/nengo/nengo-loihi/pull/286>`__)

0.10.0 (November 25, 2019)
==========================

*Compatible with Nengo 3.0.0*

*Compatible with NxSDK 0.8.7 - 0.9.0*

**Changed**

- Nengo Loihi now requires NxSDK version 0.8.7 and supports NxSDK version 0.9.0.
  (`#255 <https://github.com/nengo/nengo-loihi/pull/255>`__)

0.9.0 (November 20, 2019)
=========================

*Compatible with Nengo 3.0.0*

*Compatible with NxSDK 0.8.5*

**Added**

- It is now possible to slice the ``pre`` neurons in a neuron->neuron
  connection.
  (`#226 <https://github.com/nengo/nengo-loihi/pull/226>`__)
- Connections now support ``Sparse`` transforms.
  (`#240 <https://github.com/nengo/nengo-loihi/pull/240>`__)
- A more informative error message is raised if any encoders contain NaNs.
  (`#251 <https://github.com/nengo/nengo-loihi/pull/251>`__)

**Changed**

- Connections from neurons with scalar transforms are now sparse internally.
  This allows much larger neuron->neuron connections with scalar transforms.
  (`#226 <https://github.com/nengo/nengo-loihi/pull/226>`__)
- The ``scipy`` package is now required to run Nengo Loihi.
  (`#240 <https://github.com/nengo/nengo-loihi/pull/240>`__)
- Increased minimum NengoDL version to 3.0 (and this transitively increases the minimum
  TensorFlow version to 2.0).
  (`#259 <https://github.com/nengo/nengo-loihi/pull/259>`__)
- Nengo Loihi is now compatible with Nengo version 3.0.0.
  (`#259 <https://github.com/nengo/nengo-loihi/pull/259>`__)

**Fixed**

- Fixed a bug in which ``scipy`` was not imported properly in some situations.
  (`#252 <https://github.com/nengo/nengo-loihi/issues/252>`__,
  `#258 <https://github.com/nengo/nengo-loihi/pull/258>`__)

0.8.0 (June 23, 2019)
=====================

*Compatible with Nengo 2.8.0*

*Compatible with NxSDK 0.8.5*

**Changed**

- Nengo Loihi now requires NxSDK version 0.8.5.
  (`#225 <https://github.com/nengo/nengo-loihi/pull/225>`__)

0.7.0 (June 21, 2019)
=====================

*Compatible with Nengo 2.8.0*

*Compatible with NxSDK 0.8.0 - 0.8.1*

**Added**

- Added ``RoundRobin`` allocator, which allows networks to be run across
  multiple chips (multi-chip) by assigning each ensemble to a different chip
  in a round-robin format. This allocator can be selected using the
  ``hardware_options`` argument when creating ``nengo_loihi.Simulator``.
  (`#197 <https://github.com/nengo/nengo-loihi/pull/197>`__)
- Added support for ``Ensemble.neurons -> Ensemble`` connections.
  (`#156 <https://github.com/nengo/nengo-loihi/pull/156>`__)

**Changed**

- Switched to nengo-bones templating system for TravisCI config/scripts.
  (`#204 <https://github.com/nengo/nengo-loihi/pull/204>`__)
- It is no longer possible to pass ``network=None`` to ``Simulator``.
  Previously this was possible, but unlikely to work as expected.
  (`#202 <https://github.com/nengo/nengo-loihi/pull/202>`__)
- Better error messages are raised when attempting to simulate networks
  in which certain objects participating in a learning rule are on-chip.
  (`#202 <https://github.com/nengo/nengo-loihi/pull/202>`__,
  `#208 <https://github.com/nengo/nengo-loihi/issues/208>`__,
  `#209 <https://github.com/nengo/nengo-loihi/issues/209>`__)
- Nengo Loihi now requires at least NxSDK version 0.8.0.
  (`#218 <https://github.com/nengo/nengo-loihi/pull/218>`__)
- The default intercept range set by ``nengo_loihi.set_defaults()`` is now
  (-1, 0.5), instead of (-0.5, 0.5).
  (`#126 <https://github.com/nengo/nengo-loihi/pull/126>`__)
- Obfuscated non-public information related to Intel's NxSDK.
  (`#228 <https://github.com/nengo/nengo-loihi/pull/228>`__)

**Fixed**

- The splitting and passthrough removal procedures were significantly
  refactored, which fixed an issue in which networks could be modified
  in the splitting process.
  (`#202 <https://github.com/nengo/nengo-loihi/pull/202>`__,
  `#211 <https://github.com/nengo/nengo-loihi/issues/211>`__)
- It is now possible to make connections and probes with object slices
  (e.g., ``nengo.Probe(my_ensemble[0])``).
  (`#202 <https://github.com/nengo/nengo-loihi/pull/202>`__,
  `#205 <https://github.com/nengo/nengo-loihi/issues/205>`__,
  `#206 <https://github.com/nengo/nengo-loihi/issues/206>`__)
- We no longer disable the Nengo decoder cache for all models.
  (`#202 <https://github.com/nengo/nengo-loihi/pull/202>`__,
  `#207 <https://github.com/nengo/nengo-loihi/issues/207>`__)
- Transforms to on-chip neurons are now applied on-chip,
  which avoids scaling issues and large off-chip transforms.
  (`#126 <https://github.com/nengo/nengo-loihi/pull/126>`__)

0.6.0 (February 22, 2019)
=========================

*Compatible with NxSDK 0.7.0 - 0.8.0*

**Changed**

- New Nengo transforms are supported, including ``nengo.Convolution``. Many of
  the classes previously in ``conv.py`` have been moved to Nengo as part of
  this transition. The MNIST convnet example demonstrates the new syntax.
  (`#142 <https://github.com/nengo/nengo-loihi/pull/142>`__)
- Emulator now fails for any cx_base < 0, except -1 which indicates
  an unused axon.
  (`#185 <https://github.com/nengo/nengo-loihi/pull/185>`__)
- Noise now works correctly with small exponents on both the chip and
  emulator. Previously, the emulator did not allow very small exponents, and
  such exponents produced noise with the wrong magnitude on the chip.
  (`#185 <https://github.com/nengo/nengo-loihi/pull/185>`__)
- Models trained using NengoDL use tuning curves more similar to those
  of neuron on the chip, improving the accuracy of these model.
  (`#140 <https://github.com/nengo/nengo-loihi/pull/140>`__)

**Removed**

- Removed the ``NIF`` and ``NIFRate`` neuron types. These types were only used
  for encoding node values in spikes to send to the chip, which can be done
  just as well with ``nengo.SpikingRectifiedLinear`` neurons.
  (`#185 <https://github.com/nengo/nengo-loihi/pull/185>`__)
- Removed the unused/untested ``Synapse.set_diagonal_weights``.
  (`#185 <https://github.com/nengo/nengo-loihi/pull/185>`__)

**Fixed**

- Objects in nengo-loihi will have the same random seeds as in
  nengo core (and therefore any randomly generated parameters, such as
  ensemble encoders, will be generated in the same way).
  (`#70 <https://github.com/nengo/nengo-loihi/pull/70>`_)
- Seeded networks that have learning are now deterministic on both
  emulator and hardware.
  (`#140 <https://github.com/nengo/nengo-loihi/pull/140>`__)

0.5.0 (February 12, 2019)
=========================

*Compatible with NxSDK 0.7.0 - 0.8.0*

**Added**

- Allow ``LIF.min_voltage`` to have effect. The exact minimum voltage on the
  chip is highly affected by discritization (since the chip only allows
  minimum voltages in powers of two), but this will at least provide something
  in the ballpark.
  (`#169 <https://github.com/nengo/nengo-loihi/pull/169>`__)
- Population spikes can now be used to send information more efficiently
  to the chip. Population spikes are necessary for larger models
  like those using CIFAR-10 data.
  (`#161 <https://github.com/nengo/nengo-loihi/pull/161>`__)

**Changed**

- PES learning in Nengo Loihi more closely matches learning in core Nengo.
  (`#139 <https://github.com/nengo/nengo-loihi/pull/139>`__)
- Learning in the emulator more closely matches learning on hardware.
  (`#139 <https://github.com/nengo/nengo-loihi/pull/139>`__)
- The neurons used to transmit decoded values on-chip can be configured.
  By default, we use ten pairs of heterogeneous neurons per dimension.
  (`#132 <https://github.com/nengo/nengo-loihi/pull/132>`_)
- Internal classes and functions have been reorganized and refactored.
  See the pull request for more details.
  (`#159 <https://github.com/nengo/nengo-loihi/pull/159>`_)
- Simulator now gives a warning if the user requests a progress bar, instead
  of an error. This avoids potential problems in ``nengo_gui`` and elsewhere.
  (`#187 <https://github.com/nengo/nengo-loihi/pull/187>`_)
- Nengo Loihi now supports NxSDK version 0.8.0.
  Versions 0.7.0 and 0.7.5 are still supported.
  (`#188 <https://github.com/nengo/nengo-loihi/pull/188>`__)

**Fixed**

- We integrate current (U) and voltage (V) more accurately now by accounting
  for rounding during the decay process. This integral is used when
  discretizing weights and firing thresholds. This change significantly
  improves accuracy for many networks, but in particular dynamical systems
  like integrators.
  (`#124 <https://github.com/nengo/nengo-loihi/pull/124>`_,
  `#114 <https://github.com/nengo/nengo-loihi/issues/114>`_)
- Ensure things in the build and execution happen in a consistent order from
  one build/run to the next (by using ``OrderedDict``, which is deterministic,
  instead of ``dict``, which is not). This makes debugging easier and seeding
  consistent.
  (`#151 <https://github.com/nengo/nengo-loihi/pull/151>`_)
- Probes that use snips on the chip (when running with ``precompute=False``)
  now deal with negative values correctly.
  (`#169 <https://github.com/nengo/nengo-loihi/pull/124>`_,
  `#141 <https://github.com/nengo/nengo-loihi/issues/141>`_)
- Filtering for probes on the chip
  is guaranteed to use floating-point now (so that the filtered output
  is correct, even if the underlying values are integers).
  (`#169 <https://github.com/nengo/nengo-loihi/pull/124>`_,
  `#141 <https://github.com/nengo/nengo-loihi/issues/141>`_)
- Neuron (spike) probes can now be filtered with ``synapse`` objects.
  (`#182 <https://github.com/nengo/nengo-loihi/issues/182>`__,
  `#183 <https://github.com/nengo/nengo-loihi/pull/180>`__)

0.4.0 (December 6, 2018)
========================

*Compatible with NxSDK 0.7.0*

**Added**

- Added version tracking to documentation.

**Changed**

- An error is now raised if
  a learning rule is applied to a non-decoded connection.
  (`#103 <https://github.com/nengo/nengo-loihi/pull/103>`_)
- Switched documentation to new
  `nengo-sphinx-theme <https://github.com/nengo/nengo-sphinx-theme>`_.
  (`#143 <https://github.com/nengo/nengo-loihi/pull/143>`__)

**Fixed**

- Snips directory included when pip installing nengo-loihi.
  (`#134 <https://github.com/nengo/nengo-loihi/pull/134>`__)
- Closing ``nengo_loihi.Simulator`` will now close all the inner
  sub-simulators as well.
  (`#102 <https://github.com/nengo/nengo-loihi/issues/102>`_)

0.3.0 (September 28, 2018)
==========================

*Compatible with NxSDK 0.7.0*

**Added**

- Models can now use the ``nengo.SpikingRectifiedLinear`` neuron model
  on both the emulator and hardware backends.
- Models can now run with different ``dt`` values
  (the default is 0.001, or 1 millisecond).
- Added support for Distributions on Connection transforms.

**Changed**

- Now compatible with NxSDK 0.7. We are currently not supporting
  older versions of NxSDK, but may in the future.
- Models will not be precomputed by default. To precompute models,
  you must explicitly pass ``precompute=True`` to ``nengo_loihi.Simulator``.
- Models that do not run any objects on Loihi will raise an error.
- Ensemble intercept values are capped to 0.95 to fix issues with
  the current discretization method.

**Fixed**

- Tuning curves now take into account the Loihi discretization,
  improving accuracy on most models.
- PES learning can now be done with multidimensional error signals.
- Manually reset spike probes when Simulator is initialized.
- Several fixes to filtering and connecting
  between objects on and off chip.

0.2.0 (August 27, 2018)
=======================

First public alpha release of Nengo Loihi!
If you have any questions,
please `ask on our forum <https://forum.nengo.ai/c/backends/loihi>`_
and if you run into any issues
`let us know <https://github.com/nengo/nengo-loihi/issues>`_.

0.1.0 (July 4, 2018)
====================

Pre-alpha release of Nengo Loihi for testing at the
2018 Telluride neuromorphic engineering conference.
Thanks to all participants who tried out
this early version of Nengo Loihi
and provided feedback.
