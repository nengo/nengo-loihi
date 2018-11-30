# -*- coding: utf-8 -*-
#
# This file is execfile()d with the current directory set
# to its containing dir.

import os
import sys

try:
    import nengo_loihi
    import nengo_sphinx_theme  # noqa: F401 pylint: disable=unused-import
except ImportError:
    print("To build the documentation, nengo_loihi and nengo_sphinx_theme "
          "must be installed in the current environment. Please install these "
          "and their requirements first. A virtualenv is recommended!")
    sys.exit(1)

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.githubpages',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'nengo_sphinx_theme',
    'numpydoc',
    'nbsphinx',
]

default_role = 'py:obj'

# -- sphinx.ext.autodoc
autoclass_content = 'both'  # class and __init__ docstrings are concatenated
autodoc_default_options = {"members": None}
autodoc_member_order = 'bysource'  # default is alphabetical

# -- sphinx.ext.intersphinx
intersphinx_mapping = {
    'nengo': ('https://www.nengo.ai/nengo/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
}

# -- sphinx.ext.todo
todo_include_todos = True

# -- numpydoc
numpydoc_show_class_members = False

# -- nbsphinx
nbsphinx_timeout = -1

# -- sphinx
nitpicky = True
exclude_patterns = ['_build']
source_suffix = '.rst'
source_encoding = 'utf-8'
master_doc = 'index'

project = u'Nengo Loihi'
authors = u'Applied Brain Research'
copyright = nengo_loihi.__copyright__
version = '.'.join(nengo_loihi.__version__.split('.')[:2])
release = nengo_loihi.__version__  # Full version, with tags
pygments_style = 'default'

# -- Options for HTML output --------------------------------------------------

pygments_style = "sphinx"
templates_path = ["_templates"]
html_static_path = ["_static"]

html_theme = "nengo_sphinx_theme"

html_title = "Nengo Loihi {0} docs".format(release)
htmlhelp_basename = 'Nengo Loihi'
html_last_updated_fmt = ''  # Suppress 'Last updated on:' timestamp
html_show_sphinx = False
html_favicon = os.path.join("_static", "favicon.ico")
html_theme_options = {
    "sidebar_logo_width": 200,
    "nengo_logo": "nengo-loihi-full-light.svg",
}

# -- Options for LaTeX output -------------------------------------------------

latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '11pt',
    # 'preamble': '',
}

latex_documents = [
    # (source start file, target, title, author, documentclass [howto/manual])
    ('index', 'nengo_loihi.tex', html_title, authors, 'manual'),
]

# -- Options for manual page output -------------------------------------------

man_pages = [
    # (source start file, name, description, authors, manual section).
    ('index', 'nengo_loihi', html_title, [authors], 1)
]

# -- Options for Texinfo output -----------------------------------------------

texinfo_documents = [
    # (source start file, target, title, author, dir menu entry,
    #  description, category)
    ('index', 'nengo_loihi', html_title, authors, 'Nengo',
     'Loihi backend for Nengo', 'Miscellaneous'),
]
