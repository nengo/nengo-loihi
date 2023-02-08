********
Overview
********

Hardware
========

Intel's neuromorphic Loihi chip
is made accessible through an FPGA board.
We will refer to the devices involved in
a Loihi model using the following terms.

**Board**
  The Loihi board, which contains one or more Loihi chips.
**Chip**
  A Loihi chip, which contains several cores.
**Core**
  A computational unit on a chip.
  Each chip has several neuron cores, which simulate compartments,
  synapses, etc. and several Lakemont cores, which are general purpose
  CPUs for handling input/output and other general tasks.
**Host**
  The FPGA board that the Loihi board is connected to.
  The host runs a Linux-based operating system to allow programs
  to interact with the board using drivers provided by Intel.
**Superhost**
  The PC physically connected to the FPGA board.
  Typically the superhost and host communicate over ethernet,
  but it is also possible to communicate over serial USB.
**INRC**
  A superhost provided to members of the
  Intel Neuromorphic Research Community.
  Whenever we refer to the superhost, you can use the INRC.
**Local machine**
  The computer you are currently using.
  We usually assume that your local machine is not the superhost,
  though you can work directly on the superhost.

NengoLoihi runs on the superhost
and will automatically handle the communication
with the host and board.
Unless you are setting up a new host and board,
you will only need to interact with
your local machine and the superhost.

.. note:: If you are setting up a new host or board,
          see the :doc:`setup/host-board` page.

Software
========

NengoLoihi is a Python package for running
Nengo models on Loihi boards.
It contains a Loihi **emulator backend**
for rapid model development and easier debugging,
and a Loihi **hardware backend**
for running models on a Loihi board.

NengoLoihi requires the `Nengo <https://www.nengo.ai/nengo/>`_
Python package to define large-scale neural models.
Please refer to the `Nengo documentation <https://www.nengo.ai/nengo/>`_
for example models and instructions
for building your own models.

Nengo and NengoLoihi's emulator backend
are pure Python packages that use
`NumPy <http://www.numpy.org/>`_
to simulate neural models quickly.
On your local machine,
you only need to install
NengoLoihi and its dependencies,
which include Nengo and NumPy.
See :doc:`installation` for details.

NengoLoihi's hardware backend
uses Intel's NxSDK API
to interact with the host
and configure the board.
On the superhost,
you need to install NengoLoihi and its dependencies,
as well as NxSDK.
See :doc:`installation` for details.

Running models
==============

While you can use most models constructed
in Nengo with NengoLoihi,
some models will see degraded performance
due to the discretization process used to
convert float values to integers
for processing on the Loihi chip.

We can recover some of this performance
by choosing parameters better suited
to the range of values used by the chip.
Before you create any Nengo objects, call::

  nengo_loihi.set_defaults()

This will change the default parameters
for the core Nengo objects,
resulting in better performance.

After creating the model,
running it on NengoLoihi is done by replacing::

  nengo.Simulator(model)

with::

  nengo_loihi.Simulator(model)

By default, NengoLoihi will use the
hardware backend if it is available.
You can choose to use the emulator
even when the hardware backend is installed
by doing::

  nengo_loihi.Simulator(model, target='sim')

See :doc:`configuration` for advanced configuration options.
See :doc:`api` for additional options
and other functions and classes available
in NengoLoihi.
