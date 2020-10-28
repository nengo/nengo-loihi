import os
import shutil
import sys
import tempfile

from packaging.version import parse as parse_version

from nengo_loihi.nxsdk_obfuscation import d_get, d_import, d_set


def parse_nxsdk_version(nxsdk):
    """
    Modify nxsdk versions to be PEP 440 compliant.

    NxSDK uses the `daily` suffix for some versions, which is not part of the PEP 440
    specification and so does not compare correctly with other version strings.
    """

    v = nxsdk if isinstance(nxsdk, str) else getattr(nxsdk, "__version__", "0.0.0")
    v = v.replace("daily", "dev")
    return parse_version(v)


try:
    import nxsdk

    HAS_NXSDK = True

    nxsdk_dir = os.path.realpath(os.path.join(os.path.dirname(nxsdk.__file__), ".."))
    nxsdk_version = parse_nxsdk_version(nxsdk)

    snip_maker = d_import(b"bnhzZGsuZ3JhcGguZ3JhcGg=")

    def assert_nxsdk():
        pass


except ImportError:
    HAS_NXSDK = False
    nxsdk = None
    nxsdk_dir = None
    nxsdk_version = None
    snip_maker = None

    exception = sys.exc_info()[1]

    def assert_nxsdk(exception=exception):
        raise exception


if HAS_NXSDK:  # noqa: C901

    class SnipMaker(d_get(snip_maker, b"R3JhcGg=")):
        """Patch of the snip process manager that is multiprocess safe."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            # We need to store references to the temporary directories so
            # that they don't get cleaned up
            self.nengo_tmp_dirs = []

        def _make_tmp_files(self, name, c_file, include_dir=None):
            """Copy C/C++ file, and optionally header files, to a temporary directory.

            So that multiple simulations can use the same snip files without running
            into problems.
            """
            tmp = tempfile.TemporaryDirectory()
            self.nengo_tmp_dirs.append(tmp)

            os.mkdir(os.path.join(tmp.name, name))

            new_c_file = os.path.join(tmp.name, name, os.path.basename(c_file))
            shutil.copyfile(c_file, new_c_file)
            with open(c_file) as f0, open(new_c_file) as f1:
                src = f0.read()
                dst = f1.read()
                if src != dst:  # pragma: no cover
                    print("=== SOURCE: %s" % (c_file,))
                    print(src)
                    print("\n=== DEST: %s" % (new_c_file,))
                    print(dst)
                    raise ValueError("Snip file not copied correctly")

            new_include_dir = None
            if include_dir is not None:
                # Also copy all the include files
                new_include_dir = os.path.join(tmp.name, name, "include")
                shutil.copytree(include_dir, new_include_dir)
                assert os.path.isdir(new_include_dir), (
                    "Copy failed %s" % new_include_dir
                )

            return new_c_file, new_include_dir

        def createProcess(
            self, name, cFilePath, includeDir, *args, **kwargs
        ):  # pragma: no cover (only used with older NxSDK)
            cFilePath, includeDir = self._make_tmp_files(name, cFilePath, includeDir)
            return super().createProcess(name, cFilePath, includeDir, *args, **kwargs)

        def createSnip(self, phase, *args, **kwargs):
            cppFile = kwargs.get("cFilePath", kwargs.get("cppFile", None))
            includeDir = kwargs.get("includeDir", None)
            name = kwargs.get("name", "generic_nengo_snip")
            if cppFile is not None:
                cppFile, includeDir = self._make_tmp_files(name, cppFile, includeDir)
                kwargs["cFilePath" if "cFilePath" in kwargs else "cppFile"] = cppFile
                if "includeDir" in kwargs:
                    kwargs["includeDir"] = includeDir

            return super().createSnip(phase, *args, **kwargs)

    d_set(snip_maker, b"R3JhcGg=", val=SnipMaker)

    micro_gen = d_import(b"bnhzZGsuY29tcGlsZXIubWljcm9jb2RlZ2VuLmludGVyZmFjZQ==")
    try:
        # try new location (nxsdk > 0.9.0)
        TraceConfigGenerator = d_import(
            b"bnhzZGsuYXJjaC5uMmEuY29tcGlsZXIudHJhY2VjZmdnZW4udHJhY2VjZmdnZW4=",
            b"VHJhY2VDZmdHZW4=",
        )
    except ImportError:  # pragma: no cover
        # try old location (nxsdk <= 0.9.0)
        TraceConfigGenerator = d_import(
            b"bnhzZGsuY29tcGlsZXIudHJhY2VjZmdnZW4udHJhY2VjZmdnZW4=", b"VHJhY2VDZmdHZW4="
        )

    NxsdkBoard = d_import(b"bnhzZGsuZ3JhcGgubnhib2FyZA==", b"TjJCb2FyZA==")
    SpikeGen = d_import(
        b"bnhzZGsuZ3JhcGgubnhpbnB1dGdlbi5ueGlucHV0Z2Vu", b"QmFzaWNTcGlrZUdlbmVyYXRvcg=="
    )
    SpikeProbe = d_import(b"bnhzZGsuZ3JhcGgubnhwcm9iZXM=", b"TjJTcGlrZVByb2Jl")
    SnipPhase = d_import(b"bnhzZGsuZ3JhcGgucHJvY2Vzc2VzLnBoYXNlX2VudW1z", b"UGhhc2U=")
else:
    SnipMaker = None
    micro_gen = None
    TraceConfigGenerator = None
    NxsdkBoard = None
    SpikeGen = None
    SpikeProbe = None
    SnipPhase = None
