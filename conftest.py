from distutils.version import LooseVersion
from functools import partial
import shlex

import matplotlib as mpl
import nengo
import pytest
from pytest_allclose import report_rmses

import nengo_loihi
from nengo_loihi.emulator import EmulatorInterface
from nengo_loihi.tests import make_test_sim


def pytest_configure(config):
    mpl.use("Agg")
    EmulatorInterface.strict = True

    nengo_loihi.set_defaults()

    config.addinivalue_line(
        "markers", "target_sim: mark test as only running on emulator"
    )
    config.addinivalue_line(
        "markers", "target_loihi: mark test as only running on hardware"
    )
    config.addinivalue_line(
        "markers", "hang: mark test as hanging indefinitely on hardware"
    )

    # add unsupported attribute to Simulator (for compatibility with nengo<3.0)
    # join all the lines and then split (preserving quoted strings)
    if nengo.version.version_info <= (2, 8, 0):
        unsupported = shlex.split(" ".join(config.getini("nengo_test_unsupported")))
        # group pairs (representing testname + reason)
        unsupported = [unsupported[i : i + 2] for i in range(0, len(unsupported), 2)]
        # wrap square brackets to interpret them literally
        # (see https://docs.python.org/3/library/fnmatch.html)
        for i, (testname, _) in enumerate(unsupported):
            unsupported[i][0] = "".join(
                "[%s]" % c if c in ("[", "]") else c for c in testname
            ).replace("::", ":")

        nengo_loihi.Simulator.unsupported = unsupported


def pytest_addoption(parser):
    parser.addoption(
        "--target",
        type=str,
        default="sim",
        help="Platform on which to run tests ('sim' or 'loihi')",
    )
    parser.addoption(
        "--no-hang", action="store_true", default=False, help="Skip tests that hang"
    )

    if nengo.version.version_info <= (2, 8, 0):
        # add the pytest option from future nengo versions
        parser.addini(
            "nengo_test_unsupported",
            type="linelist",
            help="List of unsupported unit tests with reason for exclusion",
        )


def pytest_report_header(config, startdir):
    target = config.getoption("--target")
    return "Nengo Loihi is using Loihi {}".format(
        "hardware" if target == "loihi" else "emulator"
    )


def pytest_terminal_summary(terminalreporter):
    report_rmses(terminalreporter)


@pytest.fixture(scope="session")
def Simulator(request):
    """Simulator class to be used in tests"""
    return make_test_sim(request)


def pytest_collection_modifyitems(session, config, items):
    target = config.getoption("--target")

    # Skip tests specific to emulator/hardware
    if target == "loihi":
        skip_sim = pytest.mark.skip(reason="test only runs on emulator")
        for item in items:
            if "target_sim" in item.keywords:
                item.add_marker(skip_sim)
    elif target == "sim":
        skip_loihi = pytest.mark.skip(reason="test only runs on hardware")
        for item in items:
            if "target_loihi" in item.keywords:
                item.add_marker(skip_loihi)

    # Skip hanging tests when running on hardware
    if target != "loihi":
        return

    hanging_nengo_tests = [
        "nengo/tests/test_learning_rules.py::test_slicing",
        "nengo/tests/test_neurons.py::test_direct_mode_nonfinite_value",
        "nengo/tests/test_neurons.py::test_lif_min_voltage[-inf]",
        "nengo/tests/test_neurons.py::test_lif_min_voltage[-1]",
        "nengo/tests/test_neurons.py::test_lif_min_voltage[0]",
        "nengo/tests/test_neurons.py::test_lif_zero_tau_ref",
        "nengo/tests/test_node.py::test_none",
        "nengo/tests/test_node.py::test_invalid_values[inf]",
        "nengo/tests/test_node.py::test_invalid_values[nan]",
        "nengo/tests/test_node.py::test_invalid_values[string]",
        "nengo/utils/tests/test_network.py::"  # no comma
        "test_activate_direct_mode_learning[learning_rule0-False]",
        "nengo/utils/tests/test_network.py::"  # no comma
        "test_activate_direct_mode_learning[learning_rule1-True]",
        "nengo/utils/tests/test_network.py::"  # no comma
        "test_activate_direct_mode_learning[learning_rule2-True]",
        "nengo/utils/tests/test_network.py::"  # no comma
        "test_activate_direct_mode_learning[learning_rule3-False]",
    ]

    skip_hanging = pytest.mark.skip(reason="test hangs on hardware")
    for item in items:
        if "hang" in item.keywords or item.nodeid in hanging_nengo_tests:
            # pragma: no cover, because we may find no hanging tests
            item.add_marker(skip_hanging)


if LooseVersion(nengo.__version__) < "3.0.0":
    from nengo import conftest

    conftest.Simulator = Simulator
    conftest.RefSimulator = Simulator
    plt = conftest.plt
