***************
Tips and tricks
***************

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
or the Nengo GUI server on the superhost or host,
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
Starting a Nengo GUI server on port 8000
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
and you should see the Nengo GUI interface.

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
