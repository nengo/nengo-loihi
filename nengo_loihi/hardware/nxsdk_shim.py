from distutils.version import LooseVersion
import os
import shutil
import sys
import tempfile

from nengo_loihi.nxsdk_obfuscation import d_get, d_import, d_set

try:
    import nxsdk
    nxsdk_dir = os.path.realpath(
        os.path.join(os.path.dirname(nxsdk.__file__), "..")
    )
    nxsdk_version = LooseVersion(getattr(nxsdk, "__version__", "0.0.0"))
    HAS_NXSDK = True

    def assert_nxsdk():
        pass

    snip_maker = d_import(b'bnhzZGsuZ3JhcGguZ3JhcGg=')

    class SnipMaker(d_get(snip_maker, b"R3JhcGg=")):
        """Patch of the snip process manager that is multiprocess safe."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            # We need to store references to the temporary directories so
            # that they don't get cleaned up
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
            with open(cFilePath) as f0, open(tmp_path) as f1:
                src = f0.read()
                dst = f1.read()
                if src != dst:
                    print("=== SOURCE: %s" % (cFilePath,))
                    print(src)
                    print("\n=== DEST: %s" % (tmp_path,))
                    print(dst)
                    raise ValueError("Snip file not copied correctly")

            # Also copy all the include files
            include_path = os.path.join(tmp.name, name, "include")
            shutil.copytree(includeDir, include_path)
            assert os.path.isdir(include_path), "Copy failed %s" % include_path

            return super().createProcess(
                name, tmp_path, include_path, *args, **kwargs)

    d_set(snip_maker, b"R3JhcGg=", val=SnipMaker)

except ImportError:
    HAS_NXSDK = False
    nxsdk_dir = None
    nxsdk_version = None
    nxsdk = None

    exception = sys.exc_info()[1]

    def assert_nxsdk(exception=exception):
        raise exception


if HAS_NXSDK:
    micro_gen = d_import(
        b'bnhzZGsuY29tcGlsZXIubWljcm9jb2RlZ2VuLmludGVyZmFjZQ==')
    TraceConfigGenerator = d_import(
        b'bnhzZGsuY29tcGlsZXIudHJhY2VjZmdnZW4udHJhY2VjZmdnZW4=',
        b'VHJhY2VDZmdHZW4=')
    NxsdkBoard = d_import(
        b'bnhzZGsuZ3JhcGgubnhib2FyZA==',
        b'TjJCb2FyZA==')
    SpikeGen = d_import(
        b'bnhzZGsuZ3JhcGgubnhpbnB1dGdlbi5ueGlucHV0Z2Vu',
        b'QmFzaWNTcGlrZUdlbmVyYXRvcg==')
    DVSSpikeGen = d_import(
        b'bnhzZGsuZ3JhcGgubnhpbnB1dGdlbi5kdnNpbnB1dGdlbg==',
        b'RFZTU3Bpa2VHZW5lcmF0b3I=')
    SpikeProbe = d_import(
        b'bnhzZGsuZ3JhcGgubnhwcm9iZXM=',
        b'TjJTcGlrZVByb2Jl')
    from nxsdk_modules.dvs.src.dvs import DVS
else:
    micro_gen = None
    TraceConfigGenerator = None
    NxsdkBoard = None
    SpikeGen = None
    DVSSpikeGen = None
    SpikeProbe = None
    DVS = None
