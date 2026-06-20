# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

import dare2d

# -- Project information -----------------------------------------------------

project = 'dare2d'
copyright = '2023, H.Courteille, R. Karpinski'
author = 'H.Courteille, R. Karpinski'

# The full version, including alpha/beta/rc tags
release = '1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    # 'sphinx_rtd_dark_mode', https://github.com/MrDogeBro/sphinx_rtd_dark_mode/issues/28
    # 'sphinx_rtd_theme',
    'sphinx_mdinclude',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'nbsphinx',
]

autodoc_default_options = {
    'members': True,
    'inherited-members': False
}

utosummary_generate = True

autodoc_member_order = 'bysource'

napoleon_use_ivar = True

numpydoc_class_members_toctree = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    '**.ipynb_checkpoints',
]

source_suffix = '.rst'


# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# With this enabled, navigation entries are not expandable – the [+] icons next to each entry are removed.
collapse_navigation = True


# The master toctree document.
master_doc = 'index'

# user starts in dark mode
# default_dark_mode = True


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_theme = 'sphinx_rtd_theme'
nbsphinx_execute = 'always'
nbsphinx_kernel_name = 'python3'
nbsphinx_execute = 'never'