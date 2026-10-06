"""Convert MyST notebooks to .ipynb files and link them to Google Colab.

A MyST document counts as a notebook when its jupytext front matter defines a
``kernelspec``. For each notebook, this extension

* adds an "Open in Colab" badge and a download link below the page title, and
* writes ``<outdir>/<colab_notebook_dir>/<docname>.ipynb`` after an HTML build.

Colab can only open notebooks hosted on GitHub, so the badge points to the
``colab_branch`` of ``colab_repo``, where the deployed HTML output lives.
"""

from pathlib import Path

import jupytext
import nbformat
from docutils import nodes
from sphinx.application import Sphinx
from sphinx.util import logging

logger = logging.getLogger(__name__)

COLAB_BADGE = "https://colab.research.google.com/assets/colab-badge.svg"


def _is_notebook(path: Path) -> bool:
    if path.suffix != ".md":
        return False
    return "kernelspec" in jupytext.read(path).metadata


def _add_colab_links(app: Sphinx, doctree: nodes.document) -> None:
    env = app.env
    docname = env.docname
    if not _is_notebook(Path(env.doc2path(docname))):
        return
    env.colab_notebooks.add(docname)

    config = app.config
    notebook_path = f"{config.colab_notebook_dir}/{docname}.ipynb"
    colab_url = (
        f"https://colab.research.google.com/github/{config.colab_repo}"
        f"/blob/{config.colab_branch}/{notebook_path}"
    )
    download_url = "../" * docname.count("/") + notebook_path
    html = (
        '<p class="colab-links">'
        f'<a href="{colab_url}" target="_blank" rel="noopener">'
        f'<img src="{COLAB_BADGE}" alt="Open in Colab"></a> '
        f'<a href="{download_url}" download>Download notebook</a>'
        "</p>"
    )
    node = nodes.raw("", html, format="html")

    # Place the links directly below the page title if there is one.
    section = next(iter(doctree.findall(nodes.section)), None)
    if section is not None and section.children and isinstance(section[0], nodes.title):
        section.insert(1, node)
    else:
        doctree.insert(0, node)


def _write_notebooks(app: Sphinx, exception: Exception | None) -> None:
    if exception is not None or app.builder.format != "html":
        return
    config = app.config
    for docname in sorted(app.env.colab_notebooks):
        notebook = jupytext.read(Path(app.env.doc2path(docname)))
        if config.colab_setup_cell:
            notebook.cells.insert(0, nbformat.v4.new_code_cell(config.colab_setup_cell))
        out = Path(app.outdir) / config.colab_notebook_dir / f"{docname}.ipynb"
        out.parent.mkdir(parents=True, exist_ok=True)
        jupytext.write(notebook, out, fmt="ipynb")
        logger.info("colab_notebooks: wrote %s", out.relative_to(app.outdir))


def _init_env(app: Sphinx, env) -> None:
    if not hasattr(env, "colab_notebooks"):
        env.colab_notebooks = set()


def _purge_doc(app: Sphinx, env, docname: str) -> None:
    _init_env(app, env)
    env.colab_notebooks.discard(docname)


def _merge_info(app: Sphinx, env, docnames, other) -> None:
    env.colab_notebooks |= other.colab_notebooks


def setup(app: Sphinx) -> dict:
    app.add_config_value("colab_repo", "", "html")
    app.add_config_value("colab_branch", "gh-pages", "html")
    app.add_config_value("colab_notebook_dir", "notebooks", "html")
    app.add_config_value("colab_setup_cell", "", "html")

    app.connect("builder-inited", lambda app: _init_env(app, app.env))
    app.connect("env-purge-doc", _purge_doc)
    app.connect("env-merge-info", _merge_info)
    app.connect("doctree-read", _add_colab_links)
    app.connect("build-finished", _write_notebooks)

    return {"version": "0.1", "parallel_read_safe": True, "parallel_write_safe": True}
