***************
Troubleshooting
***************

Increasing snip_max_spikes_per_step
===================================

**Symptom**:
You see the following warning:

.. code-block:: none

   Too many spikes (140) sent in one timestep. Increase the value
   of `snip_max_spikes_per_step` (currently set to 50).

**Cause**:
We send spikes to the Loihi chip through
a channel that has a fixed size.
Models that spike more than we expect
need to have that fixed size changed.

**Solution**:
You can increase the ``snip_max_spikes_per_step`` value
after creating a ``nengo_loihi.Simulator`` object.
Assuming your instance is called ``sim``, do:

.. code-block:: python

   sim.sims["loihi"].snip_max_spikes_per_step = 200

Do this within the ``with`` block
that we normally use when constructing ``Simulator`` objects.

.. note:: You must set ``snip_max_spikes_per_step``
          before calling ``sim.run``.
