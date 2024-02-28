# -*- coding: utf-8 -*-
#
# Automatically generated by nengo-bones, do not edit this file directly

import pathlib

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
    "nengo_extras": ("https://www.nengo.ai/nengo-extras", None),
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

project = "NengoLoihi"
authors = "Applied Brain Research"
copyright = "2018-2024 Applied Brain Research"
version = ".".join(nengo_loihi.__version__.split(".")[:2])  # Short X.Y version
release = nengo_loihi.__version__  # Full version, with tags

# -- HTML output
templates_path = ["_templates"]
html_static_path = ["_static"]
html_theme = "nengo_sphinx_theme"
html_title = f"NengoLoihi {release} docs"
htmlhelp_basename = "NengoLoihi"
html_last_updated_fmt = ""  # Default output format (suppressed)
html_show_sphinx = False
html_favicon = str(pathlib.Path("_static", "favicon.ico"))
html_theme_options = {
    "nengo_logo": "nengo-loihi-full-light.svg",
    "nengo_logo_color": "#127bc1",
    "analytics": """
        <!-- Google tag (gtag.js) -->
        <script async src="https://www.googletagmanager.com/gtag/js?id=G-GT8XEDLTMJ"></script>
        <script>
         window.dataLayer = window.dataLayer || [];
         function gtag(){dataLayer.push(arguments);}
         gtag('js', new Date());
         gtag('config', 'G-GT8XEDLTMJ');
        </script>
        <!-- End Google tag (gtag.js) -->
        <!-- Matomo -->
        <script>
         var _paq = window._paq = window._paq || [];
         _paq.push(["setDocumentTitle", document.domain + "/" + document.title]);
         _paq.push(["setCookieDomain", "*.appliedbrainresearch.com"]);
         _paq.push(["setDomains", ["*.appliedbrainresearch.com","*.edge.nengo.ai","*.forum.nengo.ai","*.nengo.ai"]]);
         _paq.push(["enableCrossDomainLinking"]);
         _paq.push(["setDoNotTrack", true]);
         _paq.push(['trackPageView']);
         _paq.push(['enableLinkTracking']);
         (function() {
           var u="https://appliedbrainresearch.matomo.cloud/";
           _paq.push(['setTrackerUrl', u+'matomo.php']);
           _paq.push(['setSiteId', '3']);
           var d=document, g=d.createElement('script'), s=d.getElementsByTagName('script')[0];
           g.async=true; g.src='//cdn.matomo.cloud/appliedbrainresearch.matomo.cloud/matomo.js'; s.parentNode.insertBefore(g,s);
         })();
        </script>
        <!-- End Matomo Code -->
    """,
}
html_redirects = [
    ("examples/index.html", "https://www.nengo.ai/nengo-examples/loihi/"),
    (
        "examples/adaptive_motor_control.html",
        "https://www.nengo.ai/nengo-examples/loihi/adaptive-motor-control.html",
    ),
    (
        "examples/adaptive-motor-control.html",
        "https://www.nengo.ai/nengo-examples/loihi/adaptive-motor-control.html",
    ),
    (
        "examples/cifar10-convnet.html",
        "https://www.nengo.ai/nengo-examples/loihi/cifar10-convnet.html",
    ),
    (
        "examples/communication_channel.html",
        "https://www.nengo.ai/nengo-examples/loihi/communication-channel.html",
    ),
    (
        "examples/communication-channel.html",
        "https://www.nengo.ai/nengo-examples/loihi/communication-channel.html",
    ),
    (
        "examples/dvs-from-file.html",
        "https://www.nengo.ai/nengo-examples/loihi/dvs-from-file.html",
    ),
    (
        "examples/integrator.html",
        "https://www.nengo.ai/nengo-examples/loihi/integrator.html",
    ),
    (
        "examples/integrator_multi_d.html",
        "https://www.nengo.ai/nengo-examples/loihi/integrator-multi-d.html",
    ),
    (
        "examples/integrator-multi-d.html",
        "https://www.nengo.ai/nengo-examples/loihi/integrator-multi-d.html",
    ),
    (
        "examples/keras-to-loihi.html",
        "https://www.nengo.ai/nengo-examples/loihi/keras-to-loihi.html",
    ),
    (
        "examples/keyword_spotting.html",
        "https://www.nengo.ai/nengo-examples/loihi/keyword-spotting.html",
    ),
    (
        "examples/keyword-spotting.html",
        "https://www.nengo.ai/nengo-examples/loihi/keyword-spotting.html",
    ),
    (
        "examples/learn_communication_channel.html",
        "https://www.nengo.ai/nengo-examples/loihi/learn-communication-channel.html",
    ),
    (
        "examples/learn-communication-channel.html",
        "https://www.nengo.ai/nengo-examples/loihi/learn-communication-channel.html",
    ),
    ("examples/lmu.html", "https://www.nengo.ai/nengo-examples/loihi/lmu.html"),
    (
        "examples/mnist_convnet.html",
        "https://www.nengo.ai/nengo-examples/loihi/mnist-convnet.html",
    ),
    (
        "examples/mnist-convnet.html",
        "https://www.nengo.ai/nengo-examples/loihi/mnist-convnet.html",
    ),
    (
        "examples/neuron_to_neuron.html",
        "https://www.nengo.ai/nengo-examples/loihi/neuron-to-neuron.html",
    ),
    (
        "examples/neuron-to-neuron.html",
        "https://www.nengo.ai/nengo-examples/loihi/neuron-to-neuron.html",
    ),
    (
        "examples/oscillator.html",
        "https://www.nengo.ai/nengo-examples/loihi/oscillator.html",
    ),
    (
        "examples/oscillator_nonlinear.html",
        "https://www.nengo.ai/nengo-examples/loihi/oscillator-nonlinear.html",
    ),
    (
        "examples/oscillator-nonlinear.html",
        "https://www.nengo.ai/nengo-examples/loihi/oscillator-nonlinear.html",
    ),
]
