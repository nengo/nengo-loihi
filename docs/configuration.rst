*************
Configuration
*************

Config system parameters
========================

Some Loihi-specific configuration options
are exposed through Nengo's config system.

To configure these parameters,
first call `.add_params` on your network.
Then, you can use the added configuration options in your network::

  with nengo.Network() as net:
      nengo_loihi.add_params(net)

      a = nengo.Ensemble(10, 1)
      net.config[a].on_chip = False  # run the ensemble off-chip

For a full list of available config options, see `.add_params`.

Build parameters
================

The builder contains a number of configurable parameters
specific to Loihi models.
Most of these parameters are located on the `.builder.Model` object.

To configure these parameters,
create an instance of `.builder.Model` and pass it in to `.Simulator`
along with your network::

    model = nengo_loihi.builder.Model(dt=0.001)
    model.pes_error_scale = 50.

    with nengo_loihi.Simulator(network, model=model) as sim:
        sim.run(1.0)

See `.builder.Model` for a list of build parameters.

Loihi parameters
================

There are parameters specific to the Loihi board itself
that are only exposed through the `.HardwareInterface`.

To set these parameters, use the ``hardware_options`` argument.

.. code-block:: python

   with nengo_loihi.Simulator(network, target='loihi', hardware_options={
       "snip_max_spikes_per_step": 300,
       "allocator": RoundRobin(),
       "n_chips": 32,
   }) as sim:
       ...

See `.HardwareInterface` for details on what parameters are available
and what they do.
