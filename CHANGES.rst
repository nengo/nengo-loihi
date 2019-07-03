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

0.10.0 (unreleased)
===================

**Changed**

- Nengo Loihi now requires NxSDK version 0.8.7.
  (`#255 <https://github.com/nengo/nengo-loihi/pull/255>`__)

0.9.0 (November 20, 2019)
=========================

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

**Changed**

- Nengo Loihi now requires NxSDK version 0.8.5.
  (`#225 <https://github.com/nengo/nengo-loihi/pull/225>`__)

0.7.0 (June 21, 2019)
=====================

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
