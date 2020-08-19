************
Installation
************

Local machine
=============

On a local machine *not* connected to a Loihi host,
you can use any version of Python
that has ``pip``.

.. code-block:: bash

   pip install nengo-loihi

``pip`` will do its best to install
NengoLoihi's requirements.
If anything goes wrong during this process,
it is likely related to installing NumPy.
Follow `our NumPy install instructions
<https://www.nengo.ai/nengo/getting-started.html#installing-numpy>`_,
then try again.

INRC/Superhost
==============

These steps will take you through
setting up a Python environment
for running NengoLoihi,
as well as for running models
using the NxSDK directly.

Note, you *must* use Python 3.5.2 when working with NxSDK.
The easiest way to satisfy those constraints is to use `Miniconda
<https://conda.io/projects/conda/en/latest/user-guide/install/index.html>`_
to set up an isolated environment
for running Loihi models.

1. Ensure that ``conda`` is available.

   To see if it is available, run

   .. code-block:: bash

      conda -V

   If conda is available, the conda version should be printed
   to the console.

   If it is not available:

   a. Ask your superhost administrator if conda is installed.
      If it is, you need to add the ``bin`` directory of
      the conda installation to your path.

      .. code-block:: bash

         export PATH="/path/to/conda/bin:$PATH"

      Running this once will change your path for the current session.
      Adding it to a shell configuration file
      (e.g., ``~/.profile``, ``~/.bashrc``)
      will change your path for all future terminal sessions.

   b. If conda is not installed, install Miniconda.

      .. code-block:: bash

         wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
         bash miniconda.sh

      Follow the prompts to set up Miniconda as desired.

2. Create a new ``conda`` environment.

   .. warning:: You *must* use Python 3.5.2 when working with NxSDK.
                Python 3.5.2 is not available in the default
                ``conda`` channels, so you must pass
                ``--channel conda-forge`` to the command below.

   .. code-block:: bash

      conda create --channel conda-forge --name loihi python=3.5.2

3. Activate your new environment.

   .. code-block:: bash

      source activate loihi

   Sometimes the environment can have issues when first created.
   Before continuing, run ``which pip`` and ensure that the path
   to ``pip`` is in your conda environment.

   .. note:: You will need to run ``source activate loihi`` every time
             you log onto the superhost.

4. Install NumPy and Cython with conda.

   .. code-block:: bash

      conda install numpy cython

   The NumPy provided by conda is usually faster
   than those installed by other means.

5. Copy the latest NxSDK release to your current directory.

   .. note:: The location of NxSDK may have changed.
             Refer to Intel's documentation to be sure.
             The most recent release and NxSDK location
             are current as of November, 2019.

   If you are logged into INRC:

   .. code-block:: bash

      cp /nfs/ncl/releases/0.9.0/nxsdk-0.9.0.tar.gz .

   If you are setting up a non-INRC superhost:

   .. code-block:: bash

      scp <inrc-host>:/nfs/ncl/releases/0.9.0/nxsdk-0.9.0.tar.gz .

6. Install NxSDK.

   .. code-block:: bash

      pip install nxsdk-0.9.0.tar.gz

7. Install NengoLoihi.

   .. code-block:: bash

      pip install nengo-loihi

   ``pip`` will install other requirements like Nengo automatically.

8. Test that both packages installed correctly.

   Start Python by running the ``python`` command.
   If everything is installed correctly, you should
   be able to import ``nxsdk`` and ``nengo_loihi``.

   .. code-block:: pycon

      Python 3.5.2 |Anaconda, Inc.| (default, May 13 2018, 21:12:35)
      [GCC 7.2.0] on linux
      Type "help", "copyright", "credits" or "license" for more information.
      >>> import nxsdk
      >>> import nengo_loihi

Developer install
=================

If you plan to make changes to NengoLoihi,
you should perform a developer install.
All of the steps above are the same
with a developer install,
except that instead of doing ``pip install nengo-loihi``,
you should do

.. code-block:: bash

   git clone https://github.com/nengo/nengo-loihi.git
   pip install -e nengo-loihi
   cd nengo-loihi
   pre-commit install
