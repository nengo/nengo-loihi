"""NengoLoihi version information.

We use semantic versioning (see http://semver.org/).
and conform to PEP440 (see https://www.python.org/dev/peps/pep-0440/).
'.devN' will be added to the version unless the code base represents
a release version. Release versions are git tagged with the version.
"""
import warnings


def check_nengo_version():
    import nengo  # pylint: disable=import-outside-toplevel

    if nengo.version.version_info < minimum_nengo_version_info:
        raise ValueError(
            "nengo-loihi does not support Nengo version %s. "
            "Upgrade with 'pip install --upgrade --no-deps nengo'." % nengo.__version__
        )
    elif nengo.version.version_info > latest_nengo_version_info:
        warnings.warn(
            "This version of `nengo_loihi` has not been tested "
            "with your `nengo` version (%s). The latest fully "
            "supported version is %s" % (nengo.__version__, latest_nengo_version)
        )


def info2string(info):
    return ".".join(str(v) for v in info)


name = "nengo_loihi"
version_info = (1, 0, 0)  # (major, minor, patch)
dev = None

version = "{v}{dev}".format(
    v=info2string(version_info), dev=(".dev%d" % dev) if dev is not None else ""
)

# --- Nengo version compatibility
# oldest nengo version we are compatible with (test on release)
minimum_nengo_version_info = (3, 1, 0)
minimum_nengo_version = info2string(minimum_nengo_version_info)

# newest nengo version we are compatible with (set to latest released nengo
# version when releasing this repository)
latest_nengo_version_info = (3, 1, 0)
latest_nengo_version = info2string(latest_nengo_version_info)
