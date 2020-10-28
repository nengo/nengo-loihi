import os


def set_os_environ(key, value):
    if value is not None:
        os.environ[key] = value
    elif key in os.environ:
        del os.environ[key]


def set_partition_env(
    partition=os.environ.get("PARTITION", None),
    lmt_options=os.environ.get("LMTOPTIONS", None),
):
    set_os_environ("PARTITION", partition)
    set_os_environ("LMTOPTIONS", lmt_options)


def has_partition(partition):
    return os.popen("sinfo -h --partition=%s" % (partition,)).read().find("idle") > 0


def require_partition(partition, request, action="return", **kwargs):
    assert action in ("return", "skip", "fail")
    import pytest  # pylint: disable=import-outside-toplevel

    if request.config.getoption("--target") == "loihi":
        if has_partition(partition):
            request.addfinalizer(set_partition_env)
            set_partition_env(partition=partition, **kwargs)
        elif action == "return":  # pragma: no cover
            return False
        else:  # pragma: no cover
            (pytest.fail if action == "fail" else pytest.skip)(
                "Partition %r is unavailable" % (partition,)
            )

    return True
