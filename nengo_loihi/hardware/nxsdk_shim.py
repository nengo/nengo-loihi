from distutils.version import LooseVersion
import os
import sys

try:
    import nxsdk
    nxsdk_dir = os.path.realpath(
        os.path.join(os.path.dirname(nxsdk.__file__), "..")
    )
    nxsdk_version = LooseVersion(getattr(nxsdk, "__version__", "0.0.0"))
    HAS_NXSDK = True

    def assert_nxsdk():
        pass

except ImportError:
    HAS_NXSDK = False
    nxsdk_dir = None
    nxsdk_version = None

    exception = sys.exc_info()[1]

    def assert_nxsdk(exception=exception):
        raise exception


if HAS_NXSDK:
    import nxsdk.compiler.microcodegen.interface as microcodegen_uci
    from nxsdk.compiler.tracecfggen.tracecfggen import TraceCfgGen
    from nxsdk.graph.nxboard import N2Board
    from nxsdk.graph.nxinputgen import BasicSpikeGenerator
    from nxsdk.graph.nxprobes import N2SpikeProbe
else:
    BasicSpikeGenerator = None
    microcodegen_uci = None
    N2Board = None
    N2SpikeProbe = None
    TraceCfgGen = None
    nxsdk = None
    nxsdk_dir = None
