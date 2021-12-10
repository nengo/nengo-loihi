import nengo_loihi


def make_test_sim(request):
    """A Simulator factory to be used in tests.

    The factory allows simulator arguments to be controlled via pytest command line
    arguments.

    This is used in the ``conftest.Simulator`` fixture, or can be be passed
    to the ``nengo_simloader`` option when running the Nengo core tests.
    """

    target = request.config.getoption("--target")

    def TestSimulator(net, *args, **kwargs):
        """Simulator constructor to be used in tests"""
        kwargs.setdefault("target", target)
        return nengo_loihi.Simulator(net, *args, **kwargs)

    return TestSimulator
