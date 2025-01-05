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

import tomlkit

from sed import __version__

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------


def _get_project_meta():
    with open("../pyproject.toml") as pyproject:
        file_contents = pyproject.read()

    return tomlkit.parse(file_contents)["project"]


# -- Project information -----------------------------------------------------
pkg_meta = _get_project_meta()
project = "SED"
copyright = "2024, OpenCOMPES team"
author = "OpenCOMPES team"

# The short X.Y version
version = __version__
# The full version, including alpha/beta/rc tags
release = version

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.coverage",
    "sphinx_autodoc_typehints",
    "bokeh.sphinxext.bokeh_autodoc",
    "bokeh.sphinxext.bokeh_plot",
    "nbsphinx",
    "myst_parser",
]


exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

autoclass_content = "class"
autodoc_member_order = "bysource"

autodoc_mock_imports = [
    "astor",
    "pep8ext_naming",
    "flake8_builtins",
    "flake8_quotes",
]

autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "exclude-members": "__dict__,__weakref__",
    "show-inheritance": True,
}


# Set `typing.TYPE_CHECKING` to `True`:
# https://pypi.org/project/sphinx-autodoc-typehints/
napoleon_use_param = True
always_document_param_types = True
typehints_use_rtype = False
typehints_fully_qualified = True
typehints_defaults = "comma"


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "github_url": "https://github.com/OpenCOMPES/sed",
    "primary_sidebar_end": ["indices.html"],
    "navbar_center": ["version-switcher", "navbar-nav"],
    "show_nav_level": 2,
    "show_version_warning_banner": True,
    # maybe better to use _static/switcher.json on github pages link instead of the following
    "switcher": {
        "json_url": "https://raw.githubusercontent.com/OpenCOMPES/docs/main/sed/switcher.json",
        "version_match": version,
    },
    "content_footer_items": ["last-updated"],
}

html_context = {
    "github_user": "OpenCOMPES",
    "github_repo": "sed",
    "github_version": "main",
    "doc_path": "docs",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
