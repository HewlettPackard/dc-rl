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
sys.path.insert(0, os.path.abspath('../'))


# -- Project information -----------------------------------------------------

project = 'SustainDC'
copyright = '2024, HPE'
author = 'HPE'

# The full version, including alpha/beta/rc tags
release = '2.1.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autodoc',
    'sphinxcontrib.bibtex', # for citations
    'sphinxemoji.sphinxemoji', # for emojis
    'sphinx_copybutton', # to copy code block
    #'sphinx_panels', # for backgrounds
    #'sphinx.ext.autosectionlabel', #for reference sections using its title
    'sphinx_multitoc_numbering', #numbering sections
    'sphinxcontrib.globalsubs', #global substitutions
    'myst_parser',
]

#define global substitutions
global_substitutions = {
    'F': 'SustainDC' #define framework name 
}

latex_elements = {
    'sphinxsetup': 'verbatimwithframe=false',
}

# source for bib references
bibtex_bibfiles = ['references.bib']

# citation style
bibtex_default_style = 'plain'
bibtex_reference_style = 'label'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

source_suffix = '.rst'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo' #'sphinx_rtd_theme'

# html_logo = "images/logo.jpg"



# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["images"]

html_theme_options = {
    "light_logo": "logo.png",
    "dark_logo": "logo_darkmode.png",
}

#numbered figures
numfig = True
