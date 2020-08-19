***************
Tips and tricks
***************

Making models fit on Loihi
==========================

Splitting large Ensembles
-------------------------

By default, NengoLoihi will split ``Ensemble`` objects
that are too large to fit on a single Loihi core
into smaller pieces to distribute across multiple cores.
For some networks (e.g. most densely-connected networks),
this can happen by itself without any guidance from the user.

For networks that use `nengo.Convolution` transforms, such as image processing networks,
some assistance is usually required to tell NengoLoihi *how* to split an ensemble.
This is because grouping the neurons sequentially is rarely ideal for such networks.
For example, if an ``Ensemble`` is representing a 32 x 32 x 4 image
(that is 32 rows, 32 columns, and 4 channels),
we might want to split that ensemble into four 32 x 32 x 1 groups,
or four 16 x 16 x 4 groups.
In the first case,
each Loihi core will contain information from all spatial locations in the image,
but each will only contain one of the channels.
In the second case,
each Loihi core will represent a different spacial quadrant of the image
(i.e., top-left, top-right, bottom-left, bottom-right),
but each will contain all channels for its respective location.
In neither case, though, will all cores contain solely consecutive pixels,
assuming our pixel array is ordered by rows, then columns, then channels.
Since the default behaviour of the system is to split into consecutive groups,
we need to override that behaviour.

To do this, we use the `.BlockShape` class
with the ``block_shape`` configuration option:

.. testcode::

    import numpy as np

    image_shape = (32, 32, 4)

    with nengo.Network() as net:
         nengo_loihi.add_params(net)  # this gives us access to block_shape

         # first case: splitting across channels
         ens1 = nengo.Ensemble(np.prod(image_shape), 1)
         net.config[ens1].block_shape = nengo_loihi.BlockShape((32, 32, 1), image_shape)

         # second case: splitting spatially
         ens2 = nengo.Ensemble(np.prod(image_shape), 1)
         net.config[ens2].block_shape = nengo_loihi.BlockShape((16, 16, 4), image_shape)

We are not limited to splitting only along the spatial or channel axes.
For example, with a 32 x 32 x 4 image we could choose a block shape of 16 x 16 x 2,
which would result in 8 cores tiling the image both in the spatial and channel
dimensions.
We could also use shapes that are uneven in the spatial dimensions,
for example 16 x 32 x 2.

Furthermore, the block shape does not have to fit evenly into the image shape
in all (or even any) of the dimensions.
For example, with a 4 x 4 image, we could choose a block shape of 3 x 3;
this would result in 4 blocks:
a 3 x 3 block for the top-left of the image,
a 3 x 1 block for the top-right,
a 1 x 3 block for the bottom-left,
and a 1 x 1 block for the bottom-right.
In this case, it would be better to use a 2 x 2 block shape,
which also results in 4 blocks,
but uses resources more equally across all cores.
(This assumes that resource constraints are preventing us from using
e.g. a 2 x 4 or 4 x 4 block shape that would simply use fewer cores.)

The constraints on `.BlockShape` are that each block has to fit on one Loihi core.
The most basic resource limitation is that the number of neurons (the product of the
shape), must be less than or equal to 1024 (the maximum number of neurons per core).
Our two original block shapes of 32 x 32 x 1 and 16 x 16 x 4
both equal exactly 1024 neurons per core.
However, there are other limiting resources on Loihi cores,
such as the numbers of input and output axons, and the amount of synapse memory.
We therefore may not always be able to use block shapes
that fully utilize the number of compartments, if other resources are in short supply.

Measuring utilization of chip resources
---------------------------------------

The `.Model.utilization_summary` command can be used
to get more information on the resources used by each block (i.e. Loihi core).
This can help you to judge whether cores are being optimally utilized.
When using this command, it is best to give all your ensembles unique labels
(like ``nengo.Ensemble(..., label="my_ensemble")``);
these names will show up in the summary, allowing you to identify problematic blocks.

Reducing axons by changing ``pop_type``
---------------------------------------

The ``pop_type`` configuration option can be used
to set the type of population axons used on convolutional connections.
Setting this to 16 instead of 32 (the default)
reduces the number of axons required by the model,
but also adds some restrictions on how convolutional connections are set up.
See `.add_params` for more details.

Reducing synapses and axons by changing block shape
---------------------------------------------------

In networks with convolutional connections, an inefficient parameterization can
cause some connection weights to be copied up to four times (to work around limitations
when mapping these connections onto Loihi).
If you are running out of synapse memory or axons
on convolutional connections,
use the following guidelines to see whether restructuring could help.

First, it is important to understand the difference between
the spatial dimensions of a shape and the channel dimension.
The channel dimension will be the first dimension of the shape
if ``channels_last=False`` or the last dimension if ``channels_last=True``.
All other dimensions of the shape will be part of the spatial shape.
For example, if ``channels_last=True`` and we have the shape 32 x 32 x 4,
the *spatial shape* is 32 x 32, and the *spatial size* is ``32 * 32 = 1024``.

When choosing our block shape, we can trade off between spatial size and channel size.
For example, if our image shape is 32 x 32 x 8,
an ``Ensemble`` representing this will need at least 8 cores
(since ``32 * 32 * 8 / 1024 == 8``).
Two potential block shapes to achieve these 8 cores are
32 x 32 x 1, which has a spatial size of 1024,
and 16 x 16 x 4, which has a spatial size of 256.
If we wish to reduce the spatial size of our block shape,
decreasing the spatial size per block
while simultaneously increasing the number of channels per block
will often let us keep the same number of neurons per core,
but decrease other resources such as synapse memory or axon usage.

If the problem is output axon usage,
try to increase the channel size of the blocks
on any Ensemble *targeted* by connections from the problematic Ensemble.
Axons can be reused across channels,
so the more channels per core,
the fewer output axons are required to send the information to all target cores.

If the problem is input axon usage,
try to reduce the spatial size of the problematic Ensemble.
Again, since axons are reused across channels,
changing the channel size will have no effect
(potentially, it can even be increased to compensate for the drop in spatial size,
and keep the same number of compartments per core).

If the problem is synapse memory usage,
then the problem is caused by incoming Connections to the problematic Ensemble.
The solution depends on the value of ``channels_last`` on the ``Convolution`` transform,
and the value of ``pop_type`` on the Connection
(if you have not set ``pop_type``, 32 is the default value).
The following can be applied to any or all of the incoming Connections:

- If the connection is using ``channels_last=False`` and ``pop_type=32``,
  extra weights are created if the spatial size is greater than 256
  (the factor by which the size of the weights is multiplied is approximately the
  spatial size divided by 256).
  Decrease the spatial size.
- If the connection is using ``channels_last=False`` and ``pop_type=16``,
  extra weights are always created.
  Consider using ``channels_last=True``,
  or not using ``pop_type=16`` if you are using less than 50% of the available axons.
- If the connection is using ``channels_last=True`` and ``pop_type=32``,
  extra weights are created if there are more than 256 neurons per core.
  Consider using ``channels_last=False``.
- If the connection is using ``channels_last=True`` and ``pop_type=16``,
  extra weights are created if the number of channels per block is not a multiple of 4,
  and if there are more than 256 neurons per core.
  Consider making the channels per block a multiple of 4.

In all cases, decreasing the number of channels per block
will decrease the amount of synapse memory used,
since there is one set of weights per channel.

Local machine
=============

SSH hosts
---------

Adding ``ssh hosts`` to your SSH configuration
will make working with remote superhosts, hosts, and boards
much quicker and easier.
After setting them up,
you will be able to connect to any machine
through a single ``ssh <machine>`` command.

To begin, make a ``~/.ssh/config`` file.

.. code-block:: bash

   touch ~/.ssh/config

Then open that file in a text editor
and add a ``Host`` entry
for each machine that you want to interact with remotely.

Typically machines that you can connect to directly
will have a configuration like this:

.. code-block:: text

   Host <short name>
     User <username>
     HostName <host name or IP address>

For security, the port on which ssh connections are accepted
is often changed. To specify a port, add the following
to the ``Host`` entry.

.. code-block:: text

   Host <short name>
     ...
     Port 1234

Finally, many machines (especially hosts and boards)
are not accessible through the open internet
and must instead be accessed through another machine,
like a superhost.
To access these with one command,
add the following to the ``Host`` entry.
``<tunnel short name>`` refers to the ``<short name>``
of the ``Host`` entry through which
you access the machine
(e.g., the ``<host short name>`` entry uses
the superhost's short name for ``<tunnel short name>``).

.. code-block:: text

   Host <short name>
     ...
     ProxyCommand ssh <tunnel short name> -W %h:%p

Once host entries are defined, you can access those machine with:

.. code-block:: bash

   ssh <short name>

You can also use the short name in ``rsync``, ``scp``,
and other commands that use ``ssh`` under the hood.

For more details and options, see `this tutorial
<https://www.digitalocean.com/community/tutorials/how-to-configure-custom-connection-options-for-your-ssh-client>`_.

We recommend that Loihi system administrators
make specific host entries for their system
available to all users.

SSH keys
--------

SSH keys allow you to log in to remote machines
without providing your password.
This is especially useful when accessing
a board through a host and superhost,
each of which require authentication.

You may already have created
an SSH key for another purpose.
By default, SSH keys are stored as

* ``~/.ssh/id_rsa`` (private key)
* ``~/.ssh/id_rsa.pub`` (public key)

If these files exist when you do ``ls ~/.ssh``,
then you already have an SSH key.

If you do not have an SSH key,
you can create one with

.. code-block:: bash

   ssh-keygen

Follow the prompts,
using the default values when unsure.
We recommend setting a passphrase
in case someone obtains
your SSH key pair.

Once you have an SSH key pair,
you will copy your public key
to each machine you want to
log into without a password.

.. code-block:: bash

   ssh-copy-id <host short name>

``<host short name>`` is the name you specified
in your SSH config file for that host
(e.g., ``ssh-copy-id loihi-host``).
You will be prompted for your password
in order to copy the key.
Once it is copied, try ``ssh <host short name>``
to confirm that you can log in
without providing a password.

Remote port tunneling
---------------------

Tunneling a remote port to your local machine
allows you to run the Jupyter notebook server
or the NengoGUI server on the superhost or host,
but access the web-based interface
on your local machine.

To do this, we will
create a new terminal window on the local machine
that we will keep open while the tunnel is active.
In this terminal, do

.. code-block:: bash

   ssh -L <local port>:localhost:<remote port>

You will then enter an SSH session
in which you can start the process
that will communicate over ``<remote port>``.

**Example 1**:
Starting a NengoGUI server on port 8000
of ``superhost-1``,
which has a ``loihi`` conda environment.

.. code-block:: bash

   # In a new terminal window on your local machine
   ssh -L 8000:localhost:8000 superhost-1
   # We are now on superhost-1
   source activate loihi
   cd ~/nengo-loihi/docs/examples
   nengo --port 8000 --no-browser --auto-shutdown 0 --backend nengo_loihi

On your local machine,
open ``http://localhost:8000/``
and you should see the NengoGUI interface.

**Example 2**:
Starting a Jupyter notebook server on port 8080
of ``superhost-2``,
which has a ``loihi`` virtualenv environment.

.. code-block:: bash

   # In a new terminal window on your local machine
   ssh -L 8080:localhost:8080 superhost-2
   # We are now on superhost-2
   workon loihi
   cd ~/nengo-loihi/docs/examples
   jupyter notebook --no-browser --port 8080

The ``jupyter`` command should print out a URL of the form
``http://localhost:8888/?token=<long-strong>``,
which you can open on your local machine.

Syncing with rsync
------------------

If you work on your local machine
and push changes to multiple remote superhosts,
it is worth spending some time to set up
a robust solution for syncing files
between your local machine and the superhosts.

``rsync`` is a good option because it is fast
(it detects what has changed and only sends changes)
and can be configured to ensure that
the files on your local machine are the canonical files
and are not overwritten by changes made on remotes.
``rsync`` also uses SSH under the hood,
so the SSH hosts you set up previously can be used.

``rsync`` is available from most package managers
(e.g. ``apt``, ``brew``)
and in many cases
will already be installed
on your system.

The basic command that is most useful is

.. code-block:: bash

   rsync -rtuv --exclude=*.pyc /src/folder /dst/folder

* ``-r`` recurses into subdirectories
* ``-t`` copies and updates file modifications times
* ``-u`` replaces files with the most up-to-date version
  as determined by modification time
* ``-v`` adds more console output to see what has changed
* ``--exclude=*.pyc`` ensures that ``*.pyc`` files are not copied

See also `more details and options
<https://ss64.com/bash/rsync_options.html>`_.

When sending files to a remote host,
you may also want to use the ``--delete`` option
to delete files in the destination folder
that have been removed from the source folder.

To simplify ``rsync`` usage,
you can make small ``bash`` functions
to make your workflow explicit.

For example, the following
bash functions will sync the ``NxSDK``
and ``nengo-loihi`` folders
between the local machine
and the user's home directory on ``host-1``.
In this example, the ``--delete`` flag
is only used on pushing so that files
are never deleted from the local machine.
The ``--exclude=*.pyc`` flag
is only used for ``nengo-loihi`` because
``*.pyc`` files are an important
part of the NxSDK source tree.
These and other options can be adapted
based on your personal workflow.

.. code-block:: bash

   LOIHI="/path/to/nengo-loihi/"
   NXSDK="/path/to/NxSDK/"
   push_host1() {
       rsync -rtuv --exclude=*.pyc --delete "$LOIHI" "host-1:nengo-loihi"
       rsync -rtuv --delete "$NXSDK" "host-1:NxSDK"
   }
   pull_host1() {
       rsync -rtuv --exclude=*.pyc "host-1:nengo-loihi/" "$LOIHI"
       rsync -rtuv "host-1:NxSDK" "$NXSDK"
   }

These functions are placed in the ``~/.bashrc`` file
and executed at a terminal with

.. code-block:: bash

   push_host1
   pull_host1

Remote editing with SSHFS
-------------------------

If you primarily work with a single remote superhost,
SSHFS is a good option that allows you
to mount a remote filesystem to your local machine,
meaning that you manipulate files as you
normally would on your local machine,
but those files will actually exist
on the remote machine.
SSHFS ensures that change you make locally
are efficiently sent to the remote.

SSHFS is available from most package managers,
including ``apt`` and ``brew``.

To mount a remote directory to your local machine,
create a directory to mount to,
then call ``sshfs`` to mount it.

.. code-block:: bash

   mkdir -p <mount point>
   sshfs -o allow_other,defer_permissions <host short name>:<remote directory> <mount point>

When you are done using the remote files,
unmount the mount point.

.. code-block:: bash

   fusermount -u <mount point>

.. note::
   If ``fusermount`` is not available
   and you have ``sudo`` access, you can also unmount with

   .. code-block:: bash

      sudo umount <mount point>

As with ``rsync``, since you may do these commands frequently,
it can save time to make a short bash function.
The following example functions mount and unmount
the ``host-2`` ``~/loihi`` directory
to the local machine's ``~/remote/host-2`` directory.

.. code-block:: bash

   mount_host2() {
       mkdir -p ~/remote/host-2
       sshfs host-2:loihi ~/remote/host-2
   }
   unmount_host2() {
       fusermount -u ~/remote/host-2
   }

Superhost
=========

Plotting
--------

If you are generating plots with Matplotlib
on the superhost or host,
you may run into issues due to there being
no monitor attached to those machines
(i.e., they are "headless").
Rather than plotting to a screen,
you can instead save plots as files
with ``plt.savefig``.
You will also need to configure
Matplotlib to use a headless backend by default.

The easiest way to do this is with a ``matplotlibrc`` file.

.. code-block:: bash

   mkdir -p ~/.config/matplotlib
   echo "backend: Agg" >> ~/.config/matplotlib/matplotlibrc

IPython / Jupyter
-----------------

If you want to use the IPython interpreter
or the Jupyter notebook on a superhost
(e.g., the INRC superhost),
you may run into issues due to the
network file system (NFS),
which does not work well
with how IPython and Jupyter track command history.
You can configure IPython and Jupyter
to instead store command history to memory only.

To do this, start by generating the configuration files.

.. code-block:: bash

   jupyter notebook --generate-config
   ipython profile create

Then add a line to three files to
configure the command history for NFS.

.. code-block:: bash

   echo "c.NotebookNotary.db_file = ':memory:'" >> ~/.jupyter/jupyter_notebook_config.py
   echo "c.HistoryAccessor.hist_file = ':memory:'" >> ~/.ipython/profile_default/ipython_config.py
   echo "c.HistoryAccessor.hist_file = ':memory:'" >> ~/.ipython/profile_default/ipython_kernel_config.py

Slurm cheatsheet
----------------

Most Loihi superhosts use `Slurm <https://slurm.schedmd.com/>`_
to schedule and distribute jobs to Loihi hosts.
Below are the commands that Slurm makes available
and what they do.

``sinfo``
  Check the status (availability) of connected hosts.
``squeue``
  Check the status of your jobs.
``scancel <jobid>``
  Kill one of your jobs.
``scancel --user=<username>``
  Kill all of your jobs.
``sudo scontrol update nodename="<nodename>" state="idle"``
  Mark a Loihi host as "idle",
  which places it in the pool of available hosts to be used.
  Use this when a Loihi host that was down comes back up.

  .. note:: This should only be done by a system administrator.

Use Slurm by default
--------------------

Most superhosts use Slurm to run models on the host.
Normally you can opt in to executing a command with

.. code-block:: bash

   SLURM=1 my-command

However, you will usually want to use Slurm,
so to switch to an opt-out setup,
open your shell configuration file
in a text editor (usually ``~/.bashrc``),
and add the following line to the end of the file.

.. code-block:: bash

   export SLURM=1

Once making this change you can opt out of using Slurm
by executing a command with

.. code-block:: bash

   SLURM=0 my-command

Running large models
--------------------

Normally you do not need to do anything
other than setting the ``SLURM`` environment variable
to run a model on Slurm.
However, in some situation Slurm may kill your job
due to long run times or other factors.

Custom Slurm partitions can be used to run
your job with different sets of restrictions.
Your system administrator will have to set up the partition.
You can see a list of all partitions and nodes with ``sinfo``.

To run a job with the ``loihiinf`` partition,
set the environment variable ``PARTITION``.
For example, you can run ``bigmodel.py``
using this partition with

.. code-block:: bash

   PARTITION=loihiinf python bigmodel.py

Similarly, if you wish to use
a particular board (called a "node" in Slurm),
set the ``BOARD`` environment variable.
For example, to run ``model.py`` on the
``loihimh`` board, do

.. code-block:: bash

   BOARD=loihimh python model.py
