************
Installation
************

Local machine
=============

On a local machine *not* connected to a Loihi host,
you can use any version of Python
that has ``pip``.

.. code-block:: bash

   git clone https://github.com/nengo/nengo-loihi.git
   cd nengo-loihi
   pip install -e .

Note that the ``.`` at the end of ``pip`` command is required.

``pip`` will do its best to install
Nengo Loihi's requirements.
If anything goes wrong during this process,
it is likely related to installing NumPy.
Follow `our NumPy install instructions
<https://www.nengo.ai/nengo/getting_started.html#installing-numpy>`_,
then try again.

INRC
====

.. note:: These instructions will only work on the INRC superhost.
          If you have set up your own superhost
          as per the :doc:`setup/superhost` page,
          then see the next section.

These steps will take you through
setting up a Python environment
for running Nengo Loihi,
as well as for running models
using the NxSDK directly.

1. Create a new virtual environment.
   Note, you *must* use the ``python_nx`` executable
   provided by Intel when working with NxSDK.

   .. code-block:: bash

      mkvirtualenv loihi --python=python_nx

2. Activate your new environment.

   .. code-block:: bash

      workon loihi

   Sometimes the environment can have issues when first created.
   Before continuing, run ``which pip`` and ensure that the path
   to ``pip`` is in your virtual environment.

   .. note:: You will need to run ``workon loihi`` every time
             you log onto the INRC superhost.

3. Clone the NxSDK git repository.

   The location of ``NxSDK.git`` may have changed.
   Refer to Intel's documentation to be sure.

   .. code-block:: bash

      git clone /nfs/ncl/git/NxSDK.git

4. Check out a release tag.

   As of August 2018, the most recent release is 0.5.5,
   which is compatible with Nengo Loihi.

   .. code-block:: bash

      cd NxSDK
      git checkout 0.5.5

5. Add a ``setup.py`` file to NxSDK.

   As of August 2018, NxSDK does not have a ``setup.py`` file,
   which is necessary for installing NxSDK in a virtual environment.

   To add it, execute the following command.

   .. code-block:: bash

      cat > setup.py << 'EOL'
      import sys
      from setuptools import setup

      if not ((3, 5, 2) <= sys.version_info[:3] < (3, 6, 0)):
          pyversion = ".".join("%d" % v for v in sys.version_info[:3])
          raise EnvironmentError(
              "NxSDK has .pyc files that only work on Python 3.5.2 through 3.5.5. "
              "You are running version %s." % pyversion)

      setup(
          name='nxsdk',
          version='0.5.5',
          install_requires=[
              "numpy",
              "pandas",
              "matplotlib",
              "teamcity-messages",
              "rpyc<4",
          ]
      )
      EOL

   Or you may paste the text above (excluding the first and last lines)
   into a text editor and save as ``setup.py`` in the NxSDK folder.

6. Install NxSDK.

   .. code-block:: bash

      pip install -e .

7. Install Nengo Loihi.

   .. code-block:: bash

      cd ..
      git clone https://github.com/nengo/nengo-loihi.git
      cd nengo-loihi
      pip install -e .

   ``pip`` will install other requirements like Nengo automatically.

8. Test that both packages installed correctly.

   Start Python by running the ``python`` command.
   If everything is installed correctly, you should
   be able to import ``nxsdk`` and ``nengo_loihi``.

   .. code-block:: pycon

      Python 3.5.5 (default, Mar 15 2018, 11:03:27)
      [GCC 5.4.0 20160609] on linux
      Type "help", "copyright", "credits" or "license" for more information.
      >>> import nxsdk
      >>> import nengo_loihi


Superhost
=========

.. note:: These instructions assume that you are working
          on a superhost that has already been configured
          as per the :doc:`setup/superhost` page.
          Those instructions only need to be run once
          for each superhost,
          while these instructions need to be run
          by every user that is using the superhost.

If you are installing Nengo Loihi on a superhost,
there are several additional constraints
due to needing to install NxSDK.
The easiest way to satisfy
all of those constraints is to use
`Miniconda <https://conda.io/docs/user-guide/install/index.html>`_
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
   Note, you *must* use Python 3.5.5 when working with NxSDK.

   .. code-block:: bash

      conda create --name loihi python=3.5.5

3. Activate your new environment.

   .. code-block:: bash

      source activate loihi

   Sometimes the environment can have issues when first created.
   Before continuing, run ``which pip`` and ensure that the path
   to ``pip`` is in your conda environment.

   .. note:: You will need to run ``source activate loihi`` every time
             you log onto the superhost.

4. Install NumPy with conda.

   .. code-block:: bash

      conda install numpy

   The NumPy provided by conda is usually faster
   than those installed by other means.

5. Clone the NxSDK git repository.

   As of August 2018, NxSDK is not publicly available,
   but is available through the INRC.
   Refer to Intel's documentation for the details
   on how to clone the repository,
   but the command will look something like

   .. code-block:: bash

      git clone ssh://inrc/nfs/ncl/git/NxSDK.git

6. Check out a release tag.

   As of August 2018, the most recent release is 0.5.5,
   which is compatible with Nengo Loihi.

   .. code-block:: bash

      cd NxSDK
      git checkout 0.5.5

7. Add a ``setup.py`` file to NxSDK.

   As of August 2018, NxSDK does not have a ``setup.py`` file,
   which is necessary for installing NxSDK in a conda environment.

   To add it, execute the following command.

   .. code-block:: bash

      cat > setup.py << 'EOL'
      import sys
      from setuptools import setup

      if not ((3, 5, 2) <= sys.version_info[:3] < (3, 6, 0)):
          pyversion = ".".join("%d" % v for v in sys.version_info[:3])
          raise EnvironmentError(
              "NxSDK has .pyc files that only work on Python 3.5.2 through 3.5.5. "
              "You are running version %s." % pyversion)

      setup(
          name='nxsdk',
          version='0.5.5',
          install_requires=[
              "numpy",
              "pandas",
              "matplotlib",
              "teamcity-messages",
              "rpyc<4",
          ]
      )
      EOL

   Or you may paste the text above (excluding the first and last lines)
   into a text editor and save as ``setup.py`` in the NxSDK folder.

8. Install NxSDK.

   .. code-block:: bash

      pip install -e .

9. Install Nengo Loihi.

   .. code-block:: bash

      cd ..
      git clone https://github.com/nengo/nengo-loihi.git
      cd nengo-loihi
      pip install -e .

   ``pip`` will install other requirements like Nengo automatically.

10. Test that both packages installed correctly.

    Start Python by running the ``python`` command.
    If everything is installed correctly, you should
    be able to import ``nxsdk`` and ``nengo_loihi``.

    .. code-block:: pycon

       Python 3.5.5 |Anaconda, Inc.| (default, May 13 2018, 21:12:35)
       [GCC 7.2.0] on linux
       Type "help", "copyright", "credits" or "license" for more information.
       >>> import nxsdk
       >>> import nengo_loihi
