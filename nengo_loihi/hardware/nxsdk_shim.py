from distutils.version import LooseVersion
import os
import shutil
import sys
import tempfile

try:
    import nxsdk
    nxsdk_dir = os.path.realpath(
        os.path.join(os.path.dirname(nxsdk.__file__), "..")
    )
    nxsdk_version = LooseVersion(getattr(nxsdk, "__version__", "0.0.0"))
    HAS_NXSDK = True

    def assert_nxsdk():
        pass

    from nxsdk.graph import graph
    from nxsdk.driver.hwdriver import driver

    class PatchedGraph(graph.Graph):
        """Patched version of NxSDK Graph that is multiprocess safe."""

        def __init__(self, *args, **kwargs):
            super(PatchedGraph, self).__init__(*args, **kwargs)

            # We need to store references to the temporary directories so
            # that they don't get cleaned up until the graph is closed
            self.nengo_tmp_dirs = []

        def createProcess(self, name, cFilePath, includeDir, *args, **kwargs):
            # Copy the c file to a temporary directory (so that multiple
            # simulations can use the same snip files without running into
            # problems)
            tmp = tempfile.TemporaryDirectory()
            self.nengo_tmp_dirs.append(tmp)

            os.mkdir(os.path.join(tmp.name, name))

            tmp_path = os.path.join(
                tmp.name, name, os.path.basename(cFilePath))
            shutil.copyfile(cFilePath, tmp_path)

            # Also copy all the include files
            include_path = os.path.join(tmp.name, name, "include")
            shutil.copytree(includeDir, include_path)

            return super(PatchedGraph, self).createProcess(
                name, tmp_path, include_path, *args, **kwargs)

    graph.Graph = PatchedGraph

    class PatchedDriver(driver.N2Driver):
        """Patched version of NxSDK N2Driver that is multiprocess safe."""

        def startDriver(self, *args, **kwargs):
            super().startDriver(*args, **kwargs)

            # NxSDK tries to make a temporary directory for compiledir, but
            # this does it in a more secure way.
            # Note: we use mkdtemp rather than TemporaryDirectory because
            # NxSDK is already taking care of cleaning up the directory.
            self.compileDir = tempfile.mkdtemp()

    driver.N2Driver = PatchedDriver

except ImportError:
    HAS_NXSDK = False
    nxsdk_dir = None
    nxsdk_version = None
    nxsdk = None

    exception = sys.exc_info()[1]

    def assert_nxsdk(exception=exception):
        raise exception


if HAS_NXSDK:
    import nxsdk.compiler.microcodegen.interface as micro_gen
    from nxsdk.compiler.tracecfggen.tracecfggen import (
        TraceCfgGen as TraceConfigGenerator)
    from nxsdk.graph.nxboard import N2Board as NxsdkBoard
    from nxsdk.graph.nxinputgen import BasicSpikeGenerator as SpikeGen
    from nxsdk.graph.nxprobes import N2SpikeProbe as SpikeProbe
else:
    micro_gen = None
    TraceConfigGenerator = None
    NxsdkBoard = None
    SpikeGen = None
    SpikeProbe = None
