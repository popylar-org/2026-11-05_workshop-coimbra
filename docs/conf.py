"""Sphinx configuration for the Popylar pRF modelling workshop website."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "_ext"))

project = "Popylar Population Receptive Field Modeling Workshop"
copyright = "2026, Spinoza Centre for Neuroimaging and Netherlands eScience Center"
author = "The Popylar Team"

extensions = [
    "myst_nb",
    "sphinx_copybutton",
    "colab_notebooks",
]

exclude_patterns = ["_build", "_ext", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# -- MyST / MyST-NB ----------------------------------------------------------

myst_enable_extensions = ["colon_fence", "deflist", "html_image"]
# Code is not executed at build time; participants run the notebooks on Colab.
nb_execution_mode = "off"

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
}

# -- Colab notebooks (see _ext/colab_notebooks.py) ---------------------------

colab_repo = "popylar-org/2026-11-05_workshop-coimbra"
colab_branch = "gh-pages"
colab_notebook_dir = "notebooks"
# Prepended as the first code cell of every generated notebook.
colab_setup_cell = "%pip install git+https://github.com/popylar-org/prfmodel.git"
