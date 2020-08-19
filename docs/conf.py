# -*- coding: utf-8 -*-
#
# Automatically generated by nengo-bones, do not edit this file directly

import os

import nengo_loihi

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "nbsphinx",
    "nengo_sphinx_theme",
    "nengo_sphinx_theme.ext.backoff",
    "nengo_sphinx_theme.ext.redirects",
    "nengo_sphinx_theme.ext.sourcelinks",
    "notfound.extension",
    "numpydoc",
]

# -- sphinx.ext.autodoc
autoclass_content = "both"  # class and __init__ docstrings are concatenated
autodoc_default_options = {"members": None}
autodoc_member_order = "bysource"  # default is alphabetical

# -- sphinx.ext.doctest
doctest_global_setup = """
import nengo_loihi
import nengo
"""

# -- sphinx.ext.intersphinx
intersphinx_mapping = {
    "nengo": ("https://www.nengo.ai/nengo/", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "python": ("https://docs.python.org/3", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
}

# -- sphinx.ext.todo
todo_include_todos = True

# -- nbsphinx
nbsphinx_timeout = -1

# -- notfound.extension
notfound_template = "404.html"
notfound_urls_prefix = "/nengo-loihi/"

# -- numpydoc config
numpydoc_show_class_members = False

# -- nengo_sphinx_theme.ext.sourcelinks
sourcelinks_module = "nengo_loihi"
sourcelinks_url = "https://github.com/nengo/nengo-loihi"

# -- sphinx
nitpicky = True
exclude_patterns = [
    "_build",
    "**/.ipynb_checkpoints",
]
linkcheck_timeout = 30
source_suffix = ".rst"
source_encoding = "utf-8"
master_doc = "index"
linkcheck_ignore = [r"http://localhost:\d+"]
linkcheck_anchors = True
default_role = "py:obj"
pygments_style = "sphinx"
user_agent = "nengo_loihi"

project = "Nengo Loihi"
authors = "Applied Brain Research"
copyright = "2018-2020 Applied Brain Research"
version = ".".join(nengo_loihi.__version__.split(".")[:2])  # Short X.Y version
release = nengo_loihi.__version__  # Full version, with tags

# -- HTML output
templates_path = ["_templates"]
html_static_path = ["_static"]
html_theme = "nengo_sphinx_theme"
html_title = "Nengo Loihi {0} docs".format(release)
htmlhelp_basename = "Nengo Loihi"
html_last_updated_fmt = ""  # Default output format (suppressed)
html_show_sphinx = False
html_favicon = os.path.join("_static", "favicon.ico")
html_theme_options = {
    "nengo_logo": "nengo-loihi-full-light.svg",
    "nengo_logo_color": "#127bc1",
    "tagmanager_id": "GTM-KWCR2HN",
}
html_redirects = [
    ("examples/adaptive_motor_control.html", "examples/adaptive-motor-control.html"),
    ("examples/communication_channel.html", "examples/communication-channel.html"),
    ("examples/integrator_multi_d.html", "examples/integrator-multi-d.html"),
    ("examples/keyword_spotting.html", "examples/keyword-spotting.html"),
    (
        "examples/learn_communication_channel.html",
        "examples/learn-communication-channel.html",
    ),
    ("examples/mnist_convnet.html", "examples/mnist-convnet.html"),
    ("examples/neuron_to_neuron.html", "examples/neuron-to-neuron.html"),
    ("examples/oscillator_nonlinear.html", "examples/oscillator-nonlinear.html"),
]
