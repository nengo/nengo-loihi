def pytest_addoption(parser):
    parser.addoption("--target", type=str, default="sim",
                     help="Platform on which to run tests ('sim' or 'loihi')")


def pytest_report_header(config, startdir):
    target = config.getoption("--target")
    return "Nengo Loihi is using {}".format(
        "Loihi hardware" if target == "loihi" else "the Numpy simulator")
