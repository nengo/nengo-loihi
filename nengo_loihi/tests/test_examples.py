import os
import warnings

import nengo
from nengo.utils.stdlib import execfile

try:
    from nengo.utils.ipython import iter_cells, load_notebook
except ImportError:

    def iter_cells(nb, cell_type="code"):
        return (cell for cell in nb.cells if cell.cell_type == cell_type)

    def load_notebook(nb_path):
        import io  # pylint: disable=import-outside-toplevel

        from nengo.utils.ipython import (  # pylint: disable=import-outside-toplevel
            nbformat,
        )

        with io.open(nb_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        return nb


import _pytest.capture
import numpy as np
import pytest

# Monkeypatch _pytest.capture.DontReadFromInput
#  If we don't do this, importing IPython will choke as it reads the current
#  sys.stdin to figure out the encoding it will use; pytest installs
#  DontReadFromInput as sys.stdin to capture output.
#  Running with -s option doesn't have this issue, but this monkeypatch
#  doesn't have any side effects, so it's fine.
_pytest.capture.DontReadFromInput.encoding = "utf-8"
_pytest.capture.DontReadFromInput.write = lambda: None
_pytest.capture.DontReadFromInput.flush = lambda: None

examples_dir = os.path.realpath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "docs", "examples")
)

all_examples = []
for subdir, _, files in os.walk(examples_dir):
    if (os.path.sep + ".") in subdir:
        continue
    files = [f for f in files if f.endswith(".ipynb")]
    examples = [os.path.join(subdir, os.path.splitext(f)[0]) for f in files]
    all_examples.extend(examples)

# os.walk goes in arbitrary order, so sort after the fact to keep pytest happy
all_examples.sort()


def execexample(fname):
    example = os.path.join(examples_dir, fname)
    if not os.path.exists(example):
        msg = "Cannot find examples/{}".format(fname)
        warnings.warn(msg)
        pytest.skip(msg)
    example_ns = {}
    execfile(example, example_ns, example_ns)
    return example_ns


@pytest.mark.parametrize("nb_file", all_examples)
def test_no_outputs(nb_file):
    """Ensure that no cells have output."""
    pytest.importorskip("IPython", minversion="3.0")
    nb = load_notebook(os.path.join(examples_dir, "%s.ipynb" % nb_file))
    for cell in iter_cells(nb):
        assert cell.outputs == [], "Cell outputs not cleared"
        assert cell.execution_count is None, "Execution count not cleared"


@pytest.mark.parametrize("nb_file", all_examples)
def test_version_4(nb_file):
    pytest.importorskip("IPython", minversion="3.0")
    nb = load_notebook(os.path.join(examples_dir, "%s.ipynb" % nb_file))
    assert nb.nbformat == 4


@pytest.mark.parametrize("nb_file", all_examples)
def test_minimal_metadata(nb_file):
    pytest.importorskip("IPython", minversion="3.0")
    nb = load_notebook(os.path.join(examples_dir, "%s.ipynb" % nb_file))

    assert "kernelspec" not in nb.metadata
    assert "signature" not in nb.metadata

    badinfo = (
        "codemirror_mode",
        "file_extension",
        "mimetype",
        "nbconvert_exporter",
        "version",
    )
    for info in badinfo:
        assert info not in nb.metadata.language_info


def test_ens_ens(allclose, plt):
    ns = execexample("ens_ens.py")
    sim = ns["sim"]
    ap = ns["ap"]
    bp = ns["bp"]

    plt.figure()
    output_filter = nengo.synapses.Alpha(0.02)
    a = output_filter.filtfilt(sim.data[ap])
    b = output_filter.filtfilt(sim.data[bp])
    t = sim.trange()
    plt.plot(t, a)
    plt.plot(t, b)

    assert allclose(a, 0.0, atol=0.03)
    assert allclose(b[t > 0.1], 0.5, atol=0.075)


def test_ens_ens_slice(allclose, plt):
    ns = execexample("ens_ens_slice.py")
    sim = ns["sim"]
    b = ns["b"]
    b_vals = ns["b_vals"]
    bp = ns["bp"]
    c = ns["c"]
    cp = ns["cp"]

    output_filter = nengo.synapses.Alpha(0.02)
    t = sim.trange()
    b = output_filter.filtfilt(sim.data[bp])
    c = output_filter.filtfilt(sim.data[cp])
    plt.plot(t, b)
    plt.plot(t, c)
    plt.legend(
        ["b%d" % d for d in range(b.shape[1])] + ["c%d" % d for d in range(c.shape[1])]
    )

    assert allclose(b[t > 0.15, 0], b_vals[0], atol=0.15)
    assert allclose(b[t > 0.15, 1], b_vals[1], atol=0.2)
    assert allclose(c[t > 0.15, 0], b_vals[1], atol=0.2)
    assert allclose(c[t > 0.15, 1], b_vals[0], atol=0.2)


def test_node_ens_ens(allclose, plt):
    ns = execexample("node_ens_ens.py")
    sim = ns["sim"]
    up = ns["up"]
    ap = ns["ap"]
    bp = ns["bp"]

    output_filter = nengo.synapses.Alpha(0.02)
    u = output_filter.filtfilt(sim.data[up])
    a = output_filter.filtfilt(sim.data[ap])
    b = output_filter.filtfilt(sim.data[bp])

    plt.figure(figsize=(8, 6))
    t = sim.trange()
    plt.subplot(411)
    plt.plot(t, u[:, 0], "b", label="u[0]")
    plt.plot(t, a[:, 0], "g", label="a[0]")
    plt.ylim([-1, 1])
    plt.legend(loc=0)

    plt.subplot(412)
    plt.plot(t, u[:, 1], "b", label="u[1]")
    plt.plot(t, a[:, 1], "g", label="a[1]")
    plt.ylim([-1, 1])
    plt.legend(loc=0)

    plt.subplot(413)
    plt.plot(t, a[:, 0] ** 2, c="b", label="a[0]**2")
    plt.plot(t, b[:, 0], c="g", label="b[0]")
    plt.ylim([-0.05, 1])
    plt.legend(loc=0)

    plt.subplot(414)
    plt.plot(t, a[:, 1] ** 2, c="b", label="a[1]**2")
    plt.plot(t, b[:, 1], c="g", label="b[1]")
    plt.ylim([-0.05, 1])
    plt.legend(loc=0)

    tmask = t > 0.1  # ignore transients at the beginning
    assert allclose(a[tmask], np.clip(u[tmask], -1, 1), atol=0.1, rtol=0.1)
    assert allclose(b[tmask], a[tmask] ** 2, atol=0.15, rtol=0.2)
