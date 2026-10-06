# 2026_11_05-workshop_coimbra
Website and materials for the Popylar pRF modelling workshop at the University of Coimbra (5–6 November 2026).

## Building the website

The site is built with [Sphinx](https://www.sphinx-doc.org) and [MyST-NB](https://myst-nb.readthedocs.io); dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
uv run sphinx-build -b html docs _build/html
```

Pushes to `main` are built and deployed to the `gh-pages` branch by GitHub Actions.

## Adding workshop content

Add pages to `docs/workshop/` and list them in the toctree in `docs/workshop/index.md`.

To make a page a notebook that participants can run on Google Colab, write it as a
[MyST notebook](https://myst-nb.readthedocs.io/en/latest/authoring/text-notebooks.html):
add a jupytext header with a `kernelspec` and put code in `{code-cell}` blocks
(see `docs/workshop/01-getting-started.md`). During the build, such pages are converted to
`.ipynb` files under `notebooks/` and get an "Open in Colab" badge. Code is not executed
during the build. Prefer plain Markdown in notebook pages, since MyST directives such as
admonitions are not rendered on Colab.
