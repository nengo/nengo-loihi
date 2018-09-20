import hashlib
import logging
import os
from functools import partial

import nengo.utils.numpy as npext
import matplotlib as mpl
import numpy as np
import pytest

import nengo.conftest
from nengo.conftest import plt, TestConfig  # noqa: F401
from nengo.utils.compat import ensure_bytes

import nengo_loihi

from nengo_loihi.loihi_cx import CxSimulator


def pytest_configure(config):
    mpl.use("Agg")
    CxSimulator.strict = True

    TestConfig.RefSimulator = TestConfig.Simulator = nengo_loihi.Simulator
    if config.getoption('seed_offset'):
        TestConfig.test_seed = config.getoption('seed_offset')[0]
    nengo_loihi.set_defaults()
    # Only log warnings from Nengo
    logging.getLogger("nengo").setLevel(logging.WARNING)


def pytest_addoption(parser):
    parser.addoption("--target", type=str, default="sim",
                     help="Platform on which to run tests ('sim' or 'loihi')")
    parser.addoption("--no-hang", action="store_true", default=False,
                     help="Skip tests that hang")


def pytest_report_header(config, startdir):
    target = config.getoption("--target")
    return "Nengo Loihi is using {}".format(
        "Loihi hardware" if target == "loihi" else "the Numpy simulator")


def pytest_terminal_summary(terminalreporter):
    tr = terminalreporter
    all_rmses = []
    for passed_test in tr.stats.get("passed", []):
        for name, val in passed_test.user_properties:
            if name == "rmse_relative":
                all_rmses.append(val)

    if len(all_rmses) > 0:
        tr.write_sep(
            "=", "relative root mean squared error for allclose checks")
        tr.write_line("mean relative rmse: %.5f +/- %.4f" % (
            np.mean(all_rmses), np.std(all_rmses)))


def pytest_runtest_setup(item):
    if (getattr(item.obj, "hang", False) and
            item.config.getvalue("--target") == "loihi" and
            item.config.getvalue("--no-hang")):
        pytest.xfail("This test causes Loihi to hang indefinitely")


@pytest.fixture(scope="session")
def Simulator(request):
    """Simulator class to be used in tests"""
    target = request.config.getoption("--target")
    Sim = partial(nengo_loihi.Simulator, target=target)
    Sim.__module__ = "nengo_loihi.simulator"
    return Sim


def function_seed(function, mod=0):
    """Generates a unique seed for the given test function.

    The seed should be the same across all machines/platforms.
    """
    c = function.__code__

    # get function file path relative to Nengo directory root
    nengo_path = os.path.abspath(os.path.dirname(nengo_loihi.__file__))
    path = os.path.relpath(c.co_filename, start=nengo_path)

    # take start of md5 hash of function file and name, should be unique
    hash_list = os.path.normpath(path).split(os.path.sep) + [c.co_name]
    hash_string = ensure_bytes('/'.join(hash_list))
    i = int(hashlib.md5(hash_string).hexdigest()[:15], 16)
    s = (i + mod) % npext.maxint
    int_s = int(s)  # numpy 1.8.0 bug when RandomState on long type inputs
    assert type(int_s) == int  # should not still be a long because < maxint
    return int_s


@pytest.fixture
def rng(request):
    """A seeded random number generator.

    This should be used in lieu of np.random because we control its seed.
    """
    # add 1 to seed to be different from `seed` fixture
    seed = function_seed(request.function, mod=TestConfig.test_seed + 1)
    return np.random.RandomState(seed)


@pytest.fixture
def seed(request):
    """A seed for seeding Networks.

    This should be used in lieu of an integer seed so that we can ensure that
    tests are not dependent on specific seeds.
    """
    return function_seed(request.function, mod=TestConfig.test_seed)


@pytest.fixture
def allclose(request):
    def safe_rms(x):
        x = np.asarray(x)
        return npext.rms(x) if x.size > 0 else np.nan

    def _allclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False):
        rmse = safe_rms(a - b)
        if not np.any(np.isnan(rmse)):
            request.node.user_properties.append(("rmse", rmse))

            ab_rms = safe_rms(a) + safe_rms(b)
            rmse_relative = (2 * rmse / ab_rms) if ab_rms > 0 else np.nan
            if not np.any(np.isnan(rmse_relative)):
                request.node.user_properties.append(
                    ("rmse_relative", rmse_relative))

        return np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)
    return _allclose
