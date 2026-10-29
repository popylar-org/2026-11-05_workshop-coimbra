"""Convert MyST notebooks to .ipynb files and link them to Google Colab.

A MyST document counts as a notebook when its jupytext front matter defines a
``kernelspec``. For each notebook, this extension

* adds an "Open in Colab" badge and a download link below the page title, and
* writes ``<outdir>/<colab_notebook_dir>/<docname>.ipynb`` after an HTML build.

The notebooks are created from the source as Sphinx reads it, that is, after
other extensions changed it in the ``source-read`` event (e.g., solutions.py
inserts the solutions into the practicals).

On every page, blockquotes that start with a bold "Exercise ..." label are
rendered as exercise admonitions.

Colab can only open notebooks hosted on GitHub, so the badge points to the
``colab_branch`` of ``colab_repo``, where the deployed HTML output lives. If
the output is deployed to a subdirectory of that branch (e.g. one per version),
set ``colab_path_prefix`` to that subdirectory.

Python modules that notebooks import (``colab_modules``, paths relative to the
source directory) are copied next to the generated notebooks. In every
notebook that imports a module, a code cell that downloads it from the
``colab_branch`` is inserted before the first cell that imports it, so that the
import also works on Colab and in notebooks outside the module's directory.
"""

import re
import shutil
from pathlib import Path

import jupytext
import nbformat
from docutils import nodes
from sphinx.application import Sphinx
from sphinx.util import logging

logger = logging.getLogger(__name__)

COLAB_BADGE = "https://colab.research.google.com/assets/colab-badge.svg"


def _collect_notebook(app: Sphinx, docname: str, source: list[str]) -> None:
    """Keep the source of notebook pages to convert it to a notebook after the build."""
    if Path(app.env.doc2path(docname)).suffix != ".md":
        return
    if "kernelspec" in jupytext.reads(source[0], fmt="md").metadata:
        app.env.colab_notebooks[docname] = source[0]


def _add_colab_links(app: Sphinx, doctree: nodes.document) -> None:
    env = app.env
    docname = env.docname
    if docname not in env.colab_notebooks:
        return

    config = app.config
    notebook_path = f"{config.colab_notebook_dir}/{docname}.ipynb"
    deployed_path = "/".join(filter(None, [config.colab_path_prefix, notebook_path]))
    colab_url = (
        f"https://colab.research.google.com/github/{config.colab_repo}"
        f"/blob/{config.colab_branch}/{deployed_path}"
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


def _highlight_exercises(app: Sphinx, doctree: nodes.document) -> None:
    """Turn blockquotes starting with a bold "Exercise ..." label into admonitions.

    Notebook pages write exercises as ``> **Exercise 1:** ...`` so that they
    render on Colab; on the website they get the theme's admonition styling.
    """
    for quote in list(doctree.findall(nodes.block_quote)):
        if not quote.children or not isinstance(quote[0], nodes.paragraph):
            continue
        paragraph = quote[0]
        # MyST may emit empty text nodes before the label.
        for child in [c for c in paragraph.children if isinstance(c, nodes.Text) and not c.astext()]:
            paragraph.remove(child)
        label = paragraph[0] if paragraph.children else None
        if not isinstance(label, nodes.strong) or not label.astext().startswith("Exercise"):
            continue
        paragraph.remove(label)
        if paragraph.children and isinstance(paragraph[0], nodes.Text):
            paragraph[0] = nodes.Text(paragraph[0].astext().lstrip())
        admonition = nodes.admonition("", classes=["exercise"])
        admonition += nodes.title("", label.astext().rstrip(":"))
        admonition.extend(quote.children)
        quote.replace_self(admonition)


def _insert_module_downloads(config, notebook: nbformat.NotebookNode) -> None:
    """Insert a cell that downloads a module before the first cell of the notebook that imports it."""
    for module in config.colab_modules:
        name = Path(module).stem
        imports = re.compile(rf"^\s*(from\s+{name}\s+import|import\s+{name}\b)", re.MULTILINE)
        index = next(
            (i for i, cell in enumerate(notebook.cells) if cell.cell_type == "code" and imports.search(cell.source)),
            None,
        )
        if index is None:
            continue
        deployed_path = "/".join(filter(None, [config.colab_path_prefix, config.colab_notebook_dir, module]))
        url = f"https://raw.githubusercontent.com/{config.colab_repo}/{config.colab_branch}/{deployed_path}"
        source = "\n".join([
            f"# Download the module {Path(module).name} with helper functions next to this notebook",
            "import urllib.request",
            "",
            f'urllib.request.urlretrieve("{url}", "{Path(module).name}")',
        ])
        notebook.cells.insert(index, nbformat.v4.new_code_cell(source))


def _write_notebooks(app: Sphinx, exception: Exception | None) -> None:
    if exception is not None or app.builder.format != "html":
        return
    config = app.config
    for module in config.colab_modules:
        out = Path(app.outdir) / config.colab_notebook_dir / module
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(app.srcdir) / module, out)
        logger.info("colab_notebooks: copied %s", out.relative_to(app.outdir))
    for docname, source in sorted(app.env.colab_notebooks.items()):
        notebook = jupytext.reads(source, fmt="md")
        _insert_module_downloads(config, notebook)
        if config.colab_setup_cell:
            notebook.cells.insert(0, nbformat.v4.new_code_cell(config.colab_setup_cell))
        out = Path(app.outdir) / config.colab_notebook_dir / f"{docname}.ipynb"
        out.parent.mkdir(parents=True, exist_ok=True)
        jupytext.write(notebook, out, fmt="ipynb")
        logger.info("colab_notebooks: wrote %s", out.relative_to(app.outdir))


def _init_env(app: Sphinx, env) -> None:
    # Maps the docname of every notebook page to its source (a set of docnames in older environments).
    if not isinstance(getattr(env, "colab_notebooks", None), dict):
        env.colab_notebooks = {}


def _purge_doc(app: Sphinx, env, docname: str) -> None:
    _init_env(app, env)
    env.colab_notebooks.pop(docname, None)


def _merge_info(app: Sphinx, env, docnames, other) -> None:
    env.colab_notebooks |= other.colab_notebooks


def setup(app: Sphinx) -> dict:
    app.add_config_value("colab_repo", "", "html")
    app.add_config_value("colab_branch", "gh-pages", "html")
    app.add_config_value("colab_notebook_dir", "notebooks", "html")
    app.add_config_value("colab_path_prefix", "", "html")
    app.add_config_value("colab_setup_cell", "", "html")
    app.add_config_value("colab_modules", [], "html")

    app.connect("builder-inited", lambda app: _init_env(app, app.env))
    app.connect("env-purge-doc", _purge_doc)
    app.connect("env-merge-info", _merge_info)
    # Run after other extensions changed the source (default priority 500).
    app.connect("source-read", _collect_notebook, priority=900)
    app.connect("doctree-read", _add_colab_links)
    app.connect("doctree-read", _highlight_exercises)
    app.connect("build-finished", _write_notebooks)

    return {"version": "0.1", "parallel_read_safe": True, "parallel_write_safe": True}
