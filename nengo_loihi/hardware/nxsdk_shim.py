import os

try:
    import nxsdk
    nxsdk_dir = os.path.realpath(
        os.path.join(os.path.dirname(nxsdk.__file__), "..")
    )
    from nxsdk.arch.n2a.compiler.tracecfggen.tracecfggen import TraceCfgGen
    from nxsdk.arch.n2a.graph.graph import N2Board
    from nxsdk.arch.n2a.graph.inputgen import BasicSpikeGenerator
    HAS_NXSDK = True

except ImportError:
    BasicSpikeGenerator = None
    N2Board = None
    TraceCfgGen = None
    nxsdk_dir = None
    HAS_NXSDK = False
