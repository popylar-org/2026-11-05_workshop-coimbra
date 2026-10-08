"""sphinx-polyversion configuration: build every branch and tag of the website.

Run from the repository root:

    uv run sphinx-polyversion docs/poly.py          # all branches and tags
    uv run sphinx-polyversion docs/poly.py --local  # working tree only

``ROOT_BRANCH`` is built into ``OUTPUT_DIR`` itself, so its URLs do not change.
Every other revision is built into ``<OUTPUT_DIR>/<branch or tag name>/``.
Each revision is built in its own virtual environment, synced from that
revision's ``uv.lock``.
"""

import asyncio
import os
import shutil
from asyncio.subprocess import PIPE
from pathlib import Path
from subprocess import CalledProcessError

import jinja2
from sphinx_polyversion import logger
from sphinx_polyversion.api import apply_overrides
from sphinx_polyversion.builder import BuildError
from sphinx_polyversion.driver import DefaultDriver
from sphinx_polyversion.git import Git, GitRef, GitRefType, file_predicate, refs_by_type
from sphinx_polyversion.pyvenv import VirtualPythonEnvironment
from sphinx_polyversion.sphinx import SphinxBuilder

# Options below can be overridden on the command line with `-o KEY=VALUE`.

#: Branches to build (everything except the deployment branch).
BRANCH_REGEX = r"(?!gh-pages$).+"
#: Tags to build.
TAG_REGEX = r".+"
#: Branch built into the root of the website.
ROOT_BRANCH = "main"
#: Output directory, relative to the repository root.
OUTPUT_DIR = "_build/html"
#: Arguments passed to `sphinx-build`.
SPHINX_ARGS = "-b html"
#: Arguments passed to `uv sync` when creating the environment of a revision.
UV_SYNC_ARGS = "--frozen --only-group docs"

#: Data used for `--local` builds of the working tree.
LOCAL_REF = GitRef("local", "", "", GitRefType.BRANCH, None)
MOCK_DATA = {"revisions": [LOCAL_REF], "current": LOCAL_REF}

apply_overrides(globals())

root = Git.root(Path(__file__).parent)
docs = Path(__file__).parent.relative_to(root)
output_dir = root / OUTPUT_DIR

#: Top-level names in the output of `ROOT_BRANCH`. Revisions with these names
#: would be built into its directories, so they are skipped.
RESERVED_NAMES = {path.stem for path in (root / docs).iterdir()} | {
    "notebooks",
    "genindex",
    "search",
    "searchindex",
    "objects",
}


class UvEnvironment(VirtualPythonEnvironment):
    """Virtual environment synced with `uv sync` from the revision's lock file."""

    def __init__(self, path: Path, name: str, *, args: list[str]):
        super().__init__(path, name, path / ".venv")
        self.args = args

    async def create_venv(self) -> None:
        cmd = ["uv", "sync", *self.args]
        env = os.environ.copy()
        # Ignore the environment polyversion itself runs in.
        env.pop("VIRTUAL_ENV", None)
        env["UV_PROJECT_ENVIRONMENT"] = str(self.venv)
        process = await asyncio.create_subprocess_exec(
            *cmd, cwd=self.path, env=env, stdout=PIPE, stderr=PIPE
        )
        out, err = await process.communicate()
        if process.returncode:
            self.logger.error("uv sync failed:\n%s", err.decode(errors="ignore"))
            raise BuildError from CalledProcessError(process.returncode, " ".join(cmd), out, err)


async def predicate(repo: Path, rev: GitRef) -> bool:
    """Only build revisions that support polyversion and do not clash with `ROOT_BRANCH`."""
    first = rev.name.split("/")[0]
    if rev.name != ROOT_BRANCH and (first in RESERVED_NAMES or first.startswith(("_", "."))):
        logger.warning("Skipping %s: its name clashes with a path of %s", rev.name, ROOT_BRANCH)
        return False
    return await file_predicate([docs / "poly.py"])(repo, rev)


def output_path(rev: GitRef) -> str:
    """Directory of a revision, relative to the output directory."""
    return "" if rev.name == ROOT_BRANCH else rev.name


def sort_revisions(revisions):
    """Order revisions as: root branch, other branches, then tags (newest first)."""
    # Deduplicate: `--local` builds list the local revision twice.
    revisions = {rev.name: rev for rev in revisions}.values()
    branches, tags = refs_by_type(revisions)
    branches.sort(key=lambda rev: (rev.name != ROOT_BRANCH, rev.name))
    tags.sort(key=lambda rev: rev.date, reverse=True)
    return branches + tags


def data(driver, rev, env):
    """Data passed to `conf.py` of each revision (see `sphinx_polyversion.api.load`)."""
    return {
        "current": rev,
        "revisions": sort_revisions(driver.targets),
        "paths": {rev.name: output_path(rev) for rev in driver.targets},
        "root_branch": ROOT_BRANCH,
    }


driver = DefaultDriver(
    root,
    output_dir,
    vcs=Git(branch_regex=BRANCH_REGEX, tag_regex=TAG_REGEX, predicate=predicate),
    builder=SphinxBuilder(docs, args=SPHINX_ARGS.split()),
    env=UvEnvironment.factory(args=UV_SYNC_ARGS.split()),
    namer=output_path,
    data_factory=data,
    mock=MOCK_DATA,
)
driver.run(MOCK, SEQUENTIAL)

# MyST-NB writes a `jupyter_execute` folder next to each revision's output
# directory; it is not part of the website.
for rev in driver.builds:
    path = "local" if MOCK else output_path(rev)
    shutil.rmtree((output_dir / path).parent / "jupyter_execute", ignore_errors=True)

# Without `ROOT_BRANCH` (e.g. in `--local` builds), the root lists the versions.
if not (output_dir / "index.html").exists():
    template = jinja2.Environment(
        loader=jinja2.FileSystemLoader(root / docs / "_polyversion"), autoescape=True
    ).get_template("index.html")
    revisions = sort_revisions(driver.builds)
    paths = ["local" if MOCK else output_path(rev) for rev in revisions]
    (output_dir / "index.html").write_text(template.render(versions=zip(revisions, paths)))
