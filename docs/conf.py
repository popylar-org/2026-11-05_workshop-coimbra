"""Sphinx configuration for the Popylar pRF modelling workshop website."""

import sys
from pathlib import Path

from sphinx_polyversion.api import LoadError, load
from sphinx_polyversion.git import GitRef  # noqa: F401 (registers GitRef for load())

sys.path.insert(0, str(Path(__file__).parent / "_ext"))

project = "Popylar Population Receptive Field Modeling Workshop"
copyright = "2026, Spinoza Centre for Neuroimaging and Netherlands eScience Center"
author = "The Popylar Team"

extensions = [
    "myst_nb",
    "sphinx_copybutton",
    "solutions",
    "colab_notebooks",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "_ext",
    "_polyversion",
    "_templates",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    # The MyST .md files are the sources; local .ipynb copies would be found as duplicate documents.
    "**.ipynb",
]

# -- MyST / MyST-NB ----------------------------------------------------------

myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "html_image"]
# Generate anchors for headings so that the notebook table of contents can link to sections.
myst_heading_anchors = 3
# Code is not executed at build time; participants run the notebooks on Colab.
nb_execution_mode = "off"
# Shell commands such as `!pip install` are not valid Python and cannot be highlighted.
suppress_warnings = ["misc.highlighting_failure"]

# -- HTML output -------------------------------------------------------------

html_theme = "sphinx_book_theme"
html_title = "Popylar Population Receptive Field Modeling Workshop"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "repository_url": "https://github.com/popylar-org/2026-11-05_workshop-coimbra",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_download_button": False,
    "home_page_in_toc": True,
    "show_toc_level": 2,
    # No search bar in the header; search stays available in the sidebar.
    "navbar_persistent": [],
}

# -- Solutions (see _ext/solutions.py) ---------------------------------------

# Every document in solutions_dir except the index renders the practical of the same name in solutions_practicals_dir
# with the solutions inserted below its exercises.
solutions_dir = "solutions"
solutions_practicals_dir = "workshop"

# -- Colab notebooks (see _ext/colab_notebooks.py) ---------------------------

colab_repo = "popylar-org/2026-11-05_workshop-coimbra"
colab_branch = "gh-pages"
colab_notebook_dir = "notebooks"
# Modules that the notebooks import; copied next to the notebooks and downloaded by a cell that is inserted before
# the first import of the module in every notebook in the same directory.
colab_modules = ["workshop/plotting_helpers.py"]

# -- Versions (see poly.py) --------------------------------------------------

try:
    # Adds the revision being built ("current"), all "revisions", their output
    # "paths", and the "root_branch" to html_context for the version switcher.
    load(globals())
except LoadError:
    # Plain sphinx-build: a single version without version switcher.
    pass
else:
    _revision = html_context["current"].name
    # The root branch is built into the site root, every other version into a
    # directory named after its branch or tag.
    _path = html_context["paths"].get(_revision, _revision)
    html_context["polyversion_root"] = "../" * (_path.count("/") + 1) if _path else ""
    colab_path_prefix = _path
    if _revision != "local":
        html_theme_options["repository_branch"] = _revision
