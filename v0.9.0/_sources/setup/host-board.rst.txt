**************
Board and host
**************

Two variants of the Loihi boards exist:
a single-chip board codenamed "Wolf Mountain"
and an eight-chip board codenamed "Nahuku."

Currently, both boards use an
Altera ARRIA 10 SoC FPGA board
as the host.
The Wolf Mountain board is paired with
an ARRIA 10 based on
the "Meridian Hill" (MH) architecture.
The Nahuku board is paired with
an ARRIA 10 based on
the "Golden Hardware Reference Design" (GHRD) architecture.

The remainder of this page explains
how to set up a host-board pair.
We will use the terminology
introduced in the :doc:`../overview`
(board, host, superhost).

Wolf Mountain / Meridian Hill
=============================

The Wolf Mountain board comes pre-connected to its host SoC.
They are both contained within a single plastic box.
The images below show the enclosure
as well as where ports can be found.

.. todo:: Add images

To set up the Wolf Mountain / Meridian Hill system:

1. Configure the power supply units.
   Two bench power supplies are needed to supply 5.3V and 12V respectively.
   The bench power supplies should be dialed
   to the right voltages **before** they are connected
   to the Loihi system.
   The bench power supplies should also be **off** before
   they are wired to the Loihi system.

2. Connect the bench power supplies
   to the appropriate "5V" and "12V" ports on the box.
   Take care identifying the correct ports
   before connecting the bench power supplies.
   Do not plug the 12V power supply to the 5V port or vice versa!

3. Connect the USB tty cable (USB-A male to USB-A male cable)
   to the "TTY" port on the Loihi box,
   and connect the other end of the cable to the superhost.

4. Connect the Ethernet cable to the "eth" port on the Loihi box,
   and connect the other end of the cable to the superhost.

5. If a microSD card is present in the microSD card slot,
   and it has not yet been set up (see `below <#sd-card-image>`__),
   remove the microSD card from its slot.
   This is done by using a pair of tweezers to push the card in,
   and then releasing it (the card slot is spring loaded).
   Next, use the tweezers to grab on to and gently remove the card.

6. If necessary, set up the microSD card as described
   `below <#sd-card-image>`__.
   Then reinsert the microSD card into the microSD card slot.
   Be sure to push the card into the slot
   far enough to engage the spring-loaded latch.

7. Turn on the bench power supplies (in any order)
   and check that the system boots properly.

Nahuku / Golden Hardware Reference Design
=========================================

Useful links:

- `Altera ARRIA 10 SoC GHRD information
  <https://www.intel.com/content/www/us/en/programmable/products/boards_and_kits/dev-kits/altera/arria-10-soc-development-kit.html>`_
- `Altera ARRIA 10 SoC GHRD user guide
  <https://www.intel.com/content/dam/altera-www/global/en_US/support/boards-kits/arria10/soc/es2_files/A10-SoC-DK-UG_2.pdf>`_

The user guide is especially useful
for reading status LEDs on the host
(see section 5-3).
The image below shows the location of components
important to the Nahuku / GHRD Loihi system.

.. image:: https://i.imgur.com/l5hsHM1.png
   :width: 100%
   :target: https://i.imgur.com/l5hsHM1.png

To set up the Nahuku / GHRD system:

1. Install the two FPGA RAM modules on the host
   (see image above for where they should be installed).

2. Connect the Nahuku board to the "Nahuku board connection" indicated above.

   .. warning:: The pins in the connector can be quite fragile.
                Ensure that the two sides of the connectors are lined up
                before applying pressure to mate the two connectors.

3. Connect the USB tty cable (microUSB male to USB-A male cable)
   to the "TTY" port on the host,
   and connect the other end of the cable to the superhost.

4. Connect the Ethernet cable to the ethernet port on the host,
   and connect the other end to the superhost.

5. If a microSD card is present in the microSD card slot,
   and it has not yet been set up (see `below <#sd-card-image>`__),
   remove the microSD card from its slot.
   The card slot has a latch that is spring loaded.
   To remove the microSD card, push it into the card slot, then release.
   Once the microSD card is unlatched from the card slot,
   it can then be removed by sliding it out of the card slot.

6. If necessary, set up the microSD card as described
   `below <#sd-card-image>`__).
   Then reinsert the microSD card into the microSD card slot.
   Be sure to push the card into the slot
   far enough to engage the spring-loaded latch.

7. Connect the power brick to the power port of the host.
   Plug the power brick into the wall socket.

8. Turn on the power switch on the host
   and check that the system boots properly.

Creating an SD card image
=========================

The microSD card on the host
contains its operating system.
Creating an SD card image
requires you to:

1. compile Ubuntu 16.04 for the ARM processor,
2. add Loihi specific configuration files, and
3. run a Python script to create the SD card image.

Instructions for each step follow.

Compiling Ubuntu
----------------

These steps are based on `this guide
<https://gnu-linux.org/building-ubuntu-rootfs-for-arm.html>`_.
These steps should be performed on the superhost.
You will need root access.

For simplicity,
begin these steps in a new empty directory
on a partition with several GB of free space.

Begin by switching to the root user.

.. code-block:: bash

   sudo -s

Then, as ``root``:

1. Create and navigate to a new folder for storing Ubuntu files.

   .. code-block:: bash

      mkdir ubuntu-rootfs
      cd ubuntu-rootfs

2. Download the latest Ubuntu 16.04 release compiled for ARM.

   .. code-block:: bash

      wget http://cdimage.ubuntu.com/ubuntu-base/releases/16.04/release/ubuntu-base-16.04.4-base-armhf.tar.gz -o ubuntu-base.tar.gz

3. Untar the files from the downloaded tarball.

   .. code-block:: bash

      tar -xpf ubuntu-base.tar.gz

4. Install ``qemu-user-static`` and copy it to ``ubuntu-rootfs``.

   .. code-block:: bash

      apt install qemu-user-static
      cp /usr/bin/qemu-arm-static ./usr/bin/

5. Copy the superhost's ``/etc/resolv.conf`` file to ``ubuntu-rootfs``.
   This will allow us to access repositories on the internet in later steps.

   .. code-block:: bash

      cp /etc/resolv.conf ./etc/resolv.conf

6. Return to the parent directory.

   .. code-block:: bash

      cd ..

   If you do ``ls``, you should see the ``ubuntu-rootfs`` directory
   that you were working on earlier.

The ``ubuntu-rootfs`` directory you set up
contains operating system files.
We will now use ``chroot`` to
act as though we are using those files
rather than the actual superhost OS.
Note that we are still running as the ``root`` user.

Begin by mounting system components and running ``chroot``.

.. code-block:: bash

   mount -t proc /proc ./ubuntu-rootfs/proc
   mount -t sysfs /sys ./ubuntu-rootfs/sys
   mount -o bind /dev ./ubuntu-rootfs/dev
   mount -o bind /dev/pts ./ubuntu-rootfs/dev/pts
   chroot ./ubuntu-rootfs

Then, within the ``chroot`` environment:

1. Update ``apt`` sources.

   .. code-block:: bash

      apt update

2. Install a minimal set of general packages.
   Since you are in the ``chroot`` environment,
   these will be installed inside ``ubuntu-rootfs``,
   not the superhost's OS files.

   .. code-block:: bash

      apt install --no-install-recommends \
          language-pack-en-base sudo ssh rsyslog \
          net-tools ethtool network-manager wireless-tools iputils-ping \
          lxde xfce4-power-manager \
          xinit xorg lightdm lightdm-gtk-greeter \
          alsa-utils gnome-mplayer bash-completion \
          lxtask htop python-gobject-2 python-gtk2 \
          synaptic resolvconf

3. Install packages needed to run Loihi models.

   .. code-block:: bash

      apt install libffi6 python3-pip python3-dev fake-hwclock

4. Add a user to the OS, and give it admin privileges.

   We will call our user ``abr-user``,
   but you can use a different name if desired.

   .. code-block:: bash

      adduser abr-user
      addgroup abr-user adm && addgroup abr-user sudo

5. Set a unique hostname.

   We use ``loihi-mh`` for our Wolf Mountain / Meridian Hill system
   and ``loihi-ghrd`` for our Nahuku / GHRD system.
   If you have more than one of the same type of system,
   use a more detailed naming scheme.

   .. code-block:: bash

      echo 'loihi-xxx' > /etc/hostname

6. Add host entries.

   .. code-block:: bash

      echo '127.0.0.1 localhost' >> /etc/hosts
      echo '127.0.1.1 loihi-xxx' >> /etc/hosts

7. Assign a static IP to the board.

   Begin by opening ``/etc/network/interfaces``
   your text editor of choice.
   If you are not sure, try

   .. code-block:: bash

      nano /etc/network/interfaces

   Add the following text to the end of the ``interfaces`` file.
   Replace ``<address>`` with:

   * ``10.42.0.34`` for Wolf Mountain / Meridian Hill systems
   * ``10.42.0.100`` for Nahuku / GHRD systems

   .. code-block:: text

      auto lo
      iface lo inet loopback

      auto eth0
      iface eth0 inet static
          address <address>
          netmask 255.255.255.0
          gateway 10.42.0.1

      dns-nameserver 10.42.0.1

8. Update DNS configuration based on the network connection.
   This will modify the ``/etc/resolv.conf`` we changed previously.

   When prompted, select "Yes" to the dialog box
   because we want to allow dynamic updates.

   .. code-block:: bash

      dpkg-reconfigure resolvconf

9. (Optional) Set up NFS.

   .. todo:: Add instructions for setting up NFS.

We can now exit the ``chroot`` environment

.. code-block:: bash

   exit

And unmount the environment files

.. code-block:: bash

   umount ubuntu-rootfs/proc
   umount ubuntu-rootfs/sys
   umount ubuntu-rootfs/dev/pts
   umount ubuntu-rootfs/dev

But stay as the root user for the remaining steps.

Adding Loihi-specific FPGA configuration files
----------------------------------------------

The Loihi specific configuration files
can be obtained from Intel's cloud server.
Download all of the files below to the directory
that contains the ``ubuntu-rootfs`` directory.

As of August 2018,
the latest files for the
two Loihi boards are located in:

* *Wolf Mountain*: ``/nfs/ncl/ext/boot/mh_2018-07-04/``
* *Nahuku*: ``/nfs/ncl/ext/boot/ghrd_2018-07-04/``

Download the following files:

* ``zImage``: A linux kernel compiled for the host.
* ``u-boot.scr``: The ``uboot`` script for configuring the FPGA.
* ``socfpga.rbf``: The FPGA configuration file.

and one of the following FPGA device tree blob files,
depending on the system:

* *Wolf Mountain*: ``meridian_hill_fab1b.dtb``
* *Nahuku*: ``socfpga_arria10_socdk.dtb``

Additionally, you need the u-boot preloader image,
``uboot_w_dtb-mkpimage.bin``.
The location of this file is also system dependent.

* *Wolf Mountain*:
  Download ``NxRuntime_01_05_17.tar.gz`` from the Intel sharepoint site
  and extract it. ``uboot_w_dtb-mkpimage.bin`` is in the ``board`` folder.
* *Nahuku*:
  Located in the ``/nfs/ncl/ext/boot/ghrd_2018-05-17`` folder
  on the Intel cloud server.

Your folder should now contain the following files
if you are setting up a Wolf Mountain system:

* ``ubuntu-rootfs/``
* ``meridian_hill_fab1b.dtb``
* ``socfpga.rbf``
* ``u-boot.scr``
* ``uboot_w_dtb-mkpimage.bin``
* ``zImage``

And the following files
if you are setting up a Nahuku system.

* ``ubuntu-rootfs/``
* ``socfpga.rbf``
* ``socfpga_arria10_socdk.dtb``
* ``u-boot.scr``
* ``uboot_w_dtb-mkpimage.bin``
* ``zImage``

Making the SD card image
------------------------

The easiest way to make the SD card image
is to use a Python script provided by RocketBoards.org.

We assume in the following steps that you are
in the directory containing ``ubuntu-rootfs``
and the Loihi FPGA files,
and that you are still acting as the root user
(if not, do ``sudo -s``).

1. Download the SD card image script.

   .. code-block:: bash

      wget http://releases.rocketboards.org/release/2017.10/gsrd/tools/make_sdimage.py

2. Run the script with to create the SD card image.

   .. note:: Replace ``<device-dtb>.dtb`` below with the appropriate
             ``*.dtb`` file from the previous step.

   .. code-block:: bash

      python ./make_sdimage.py -f \
          -P uboot_w_dtb-mkpimage.bin,num=3,format=raw,size=10M,type=A2 \
          -P ubuntu-rootfs/*,num=2,format=ext3,size=3000M \
          -P zImage,socfpga.rbf,<device-dtb>.dtb,u-boot.scr,num=1,format=vfat,size=500M \
          -s 3550M \
          -n sdimage_small.img

   After running this command,
   you should have a ``sdimage_small.img`` in the current directory.

   This image file contains three partitions:

   * Partition 1 (500 MB): contains the ``/boot`` partition,
     which contains ``zImage``, ``socfpga.rbf``, ``<device-dtb>.dtb``,
     and ``u-boot.scr``.
   * Partition 2 (3 GB): contains the Ubuntu OS file system.
   * Partition 3 (10 MB): contains the u-boot preloader image.

   .. note:: The partition sizes should not be changed from the values above.

   .. note:: When making an SD card for the Nahuku system,
             the Python script may throw an error
             when finalizing the third partition.
             This error can be safely ignored.
             It occurs because the ``uboot_w_dtb-mkpimage.bin`` image
             for Nahuku is 1 byte larger than the 10 MB partition size.
             However, this does not seem to impact the functionality
             of the SD card image.

3. Connect an SD card to the superhost.
   Determine the identifier assigned to it by Linux with

   .. code-block:: bash

      lsblk

   You should be able to determine which device (e.g. ``sdc``)
   is the SD card via the size and mountpoint.

4. Write the SD card image to the physical SD card.

   .. warning:: Be sure to use the correct device
                in the ``dd`` command below.
                Using the wrong device will destroy
                existing data on that device.

   .. code-block:: bash

      dd if=sdimage_small.img | pv -s 3550M | dd of=/dev/<device>

   where ``<device>`` is the device determined with ``lsblk``.

5. Remove the SD card from the superhost
   and insert it into the host SD card slot.
