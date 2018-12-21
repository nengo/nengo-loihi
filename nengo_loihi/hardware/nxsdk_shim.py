import os
import sys

try:
    import nxsdk
    nxsdk_dir = os.path.realpath(
        os.path.join(os.path.dirname(nxsdk.__file__), "..")
    )
    import nxsdk.arch.n2a.compiler.microcodegen.interface as microcodegen_uci
    from nxsdk.arch.n2a.compiler.tracecfggen.tracecfggen import TraceCfgGen
    from nxsdk.arch.n2a.graph.graph import N2Board
    from nxsdk.arch.n2a.graph.inputgen import BasicSpikeGenerator
    from nxsdk.arch.n2a.graph.probes import N2SpikeProbe
    HAS_NXSDK = True

    def assert_nxsdk():
        pass

except ImportError:
    BasicSpikeGenerator = None
    microcodegen_uci = None
    N2Board = None
    N2SpikeProbe = None
    TraceCfgGen = None
    nxsdk = None
    nxsdk_dir = None
    HAS_NXSDK = False

    exception = sys.exc_info()[1]

    def assert_nxsdk(exception=exception):
        raise exception
