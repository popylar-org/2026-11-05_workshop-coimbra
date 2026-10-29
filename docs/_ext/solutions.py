"""Render the practicals with the solutions to their exercises.

A document ``<solutions_dir>/<name>.md`` contains the solutions to the
exercises of the practical ``<solutions_practicals_dir>/<name>.md``, one
``## Exercise <number>`` section per exercise. When Sphinx reads the document,
its source is replaced by the source of the practical with each solution
inserted below its exercise as a collapsed ``<details class="solution">``
block, which also collapses on Colab. Thus, the solutions pages show the
practicals without a copy of them.

Exercises are blockquotes that start with a bold "Exercise <number>" label. A
solution is inserted below the blockquote and the code cells that directly
follow it, but above a ``# Your code goes here`` placeholder cell.
"""

import re
from pathlib import Path

from sphinx.application import Sphinx
from sphinx.util import logging

logger = logging.getLogger(__name__)

EXERCISE = re.compile(r"^> \*\*Exercise (\d+)\b")
SOLUTION_HEADING = re.compile(r"^## Exercise (\d+)\s*$")
FENCE = re.compile(r"^(`{3,}|~{3,})")
CODE_CELL = re.compile(r"^(`{3,})\{code-cell\}")
PLACEHOLDER = "# Your code goes here"


def _closes_fence(line: str, fence: str) -> bool:
    stripped = line.strip()
    return stripped.startswith(fence) and stripped == stripped[0] * len(stripped)


def parse_solutions(text: str, location: str | None = None) -> dict[str, str]:
    """Split the text of a solutions document into the solutions by exercise number."""
    solutions: dict[str, list[str]] = {}
    current = None  # Lines of the current solution
    has_preamble = False
    fence = None  # Headings inside code blocks do not start a solution
    for line in text.splitlines():
        heading = SOLUTION_HEADING.match(line) if fence is None else None
        if heading:
            number = heading.group(1)
            if number in solutions:
                logger.warning("duplicate solution for Exercise %s", number, location=location)
            current = solutions[number] = []
            continue
        if fence is None:
            if opening := FENCE.match(line):
                fence = opening.group(1)
        elif _closes_fence(line, fence):
            fence = None
        if current is not None:
            current.append(line)
        elif line.strip():
            has_preamble = True
    if has_preamble:
        logger.warning("text before the first '## Exercise' heading is ignored", location=location)
    return {number: "\n".join(lines).strip("\n") for number, lines in solutions.items()}


def _solution_position(lines: list[str], index: int) -> int:
    """Return the index of the line above which to insert a solution, starting below its exercise."""
    while True:
        while index < len(lines) and not lines[index].strip():
            index += 1
        cell = CODE_CELL.match(lines[index]) if index < len(lines) else None
        if cell is None:
            return index
        end = next((i for i in range(index + 1, len(lines)) if _closes_fence(lines[i], cell.group(1))), len(lines))
        if PLACEHOLDER in (line.strip() for line in lines[index + 1 : end]):
            return index
        index = end + 1


def insert_solutions(practical: str, solutions: dict[str, str], location: str | None = None) -> str:
    """Insert the solutions into the text of a practical below their exercises."""
    lines = practical.splitlines()
    insertions: dict[int, list[str]] = {}
    exercises = []
    index = 0
    while index < len(lines):
        exercise = EXERCISE.match(lines[index])
        index += 1
        if exercise is None:
            continue
        number = exercise.group(1)
        if number in exercises:
            logger.warning("duplicate Exercise %s in the practical", number, location=location)
        exercises.append(number)
        if number not in solutions:
            logger.warning("no solution for Exercise %s", number, location=location)
            continue
        while index < len(lines) and lines[index].startswith(">"):
            index += 1
        position = _solution_position(lines, index)
        block = ['<details class="solution">', "<summary>Solution</summary>", "", solutions[number], "", "</details>", ""]
        if position > 0 and lines[position - 1].strip():
            block.insert(0, "")
        insertions.setdefault(position, []).extend(block)
    for number in sorted(solutions.keys() - set(exercises), key=int):
        logger.warning("solution for Exercise %s, which is not in the practical", number, location=location)

    merged = []
    for index, line in enumerate(lines):
        merged.extend(insertions.get(index, []))
        merged.append(line)
    merged.extend(insertions.get(len(lines), []))
    return "\n".join(merged) + ("\n" if practical.endswith("\n") else "")


def _insert_solutions(app: Sphinx, docname: str, source: list[str]) -> None:
    config = app.config
    path = Path(docname)
    if path.parent != Path(config.solutions_dir) or path.name == "index":
        return
    practical = Path(app.srcdir) / config.solutions_practicals_dir / f"{path.name}.md"
    if not practical.is_file():
        logger.warning("no practical %s for these solutions", practical.relative_to(app.srcdir), location=docname)
        return
    # Rebuild the solutions page when the practical changes.
    app.env.note_dependency(practical)
    solutions = parse_solutions(source[0], location=docname)
    source[0] = insert_solutions(practical.read_text(encoding="utf-8"), solutions, location=docname)


def setup(app: Sphinx) -> dict:
    app.add_config_value("solutions_dir", "solutions", "env")
    app.add_config_value("solutions_practicals_dir", "workshop", "env")

    app.connect("source-read", _insert_solutions)

    return {"version": "0.1", "parallel_read_safe": True, "parallel_write_safe": True}
