# 2026_11_05-workshop_coimbra
Website and materials for the Popylar pRF modelling workshop at the University of Coimbra (5–6 November 2026).

## Building the website

The site is built with [Sphinx](https://www.sphinx-doc.org) and [MyST-NB](https://myst-nb.readthedocs.io); dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
uv run sphinx-build -b html docs _build/html
```

### Versions

The deployed website contains one version per git branch and tag, built with
[sphinx-polyversion](https://github.com/real-yfprojects/sphinx-polyversion)
(configured in `docs/poly.py`). `main` is built into the site root, so its URLs stay
the same; every other version lives in a subdirectory named after its branch or tag.
A version switcher in the top bar, left of the GitHub button, links the versions. Only branches and tags that
contain `docs/poly.py` are built, and names that clash with paths of `main`
(e.g. `workshop` or `_static`) are skipped.

```bash
# Build main into _build/html/ and all other local branches and tags into _build/html/<name>/
uv run sphinx-polyversion docs/poly.py
# Build only the working tree into _build/html/local/
uv run sphinx-polyversion docs/poly.py --local
```

Pushes to `main` and `develop`, and new tags, rebuild all versions and deploy them to
the `gh-pages` branch with GitHub Actions. Use `develop` to preview changes on the
website before merging them into `main`.

## Adding workshop content

Add pages to `docs/workshop/` and list them in the toctree in `docs/workshop/index.md`.

To make a page a notebook that participants can run on Google Colab, write it as a
[MyST notebook](https://myst-nb.readthedocs.io/en/latest/authoring/text-notebooks.html):
add a jupytext header with a `kernelspec` and put code in `{code-cell}` blocks
(see `docs/workshop/01-getting-started.md`). During the build, such pages are converted to
`.ipynb` files under `notebooks/` and get an "Open in Colab" badge. Code is not executed
during the build. Prefer plain Markdown in notebook pages, since MyST directives such as
admonitions are not rendered on Colab.
