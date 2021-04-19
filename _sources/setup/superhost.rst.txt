*********
Superhost
*********

.. todo:: Add missing sections (setting up Slurm, etc)

Adding a new user
=================

1. Log in as a user who can use ``sudo``.

2. Change to the super user.

   .. code-block:: bash

      sudo -s

3. Add the user.

   .. code-block:: bash

      adduser <username>

4. *(Optional)*: Enable the user to use ``sudo``.

   .. code-block:: bash

      usermod -aG sudo <username>

5. Add the user to the ``loihi_sudo`` group.

   This is necessary for allowing the user
   to run models on Loihi boards.

   .. code-block:: bash

      usermod -aG loihi_sudo <username>

6. Propagate the new user information to connected Loihi boards.

   .. code-block:: bash

      make -C /var/yp

You can then run ``exit`` to exit the superuser session.

Note that the final step copies user information
to the Loihi boards.
You therefore do not have to make a new user account
on the hosts or boards that are connected to the superhost.

To be sure that the user information has been copied correctly,
once finishing the above steps,
you should test by logging into all connected hosts and boards.

For example, on the superhost try

.. code-block:: bash

   ssh <username>@host-1
   ssh <username>@board-1

Connecting to a host
====================

The host and superhost communicate through
a hardwired Ethernet connection.
The superhost therefore must have
at least two networks interfaces,
one for an external internet connection
and one to connect to the FPGA host.

The host only has one network interface,
which is connected to the superhost.
In order to access the internet,
the superhost must share
its external connection with the host.

To do this, assuming that you are running Ubuntu:

1. Open "Network Connections".

2. Identify the Ethernet connection being used
   to connect to the Loihi system.
   Clicking the network icon in the task bar
   will inform you which network interfaces are available.

3. Select "Wired connection <x>" and click "Edit".

4. Navigate to "IPv4 Settings" and change
   "Method" to "Shared to other computers".

5. Click "Save".

6. Check that the network interface has been assigned the correct IP.

   When the Ethernet cable between the host and superhost is connected, do:

   .. code-block:: bash

      sudo ifconfig -a

   to display the information for each network interface.
   The network interface being used to connect to the Loihi system
   should be assigned the IP ``10.42.0.1``.
