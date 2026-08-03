#!/usr/bin/env python3
"""Resolve the next actionable task for a project (flat or phased).

Reads the live Tasks status table under ``projects/<slug>/`` and emits a
single-line JSON object describing either the next unfinished task, the
requested task, or a done/error state.

Layouts handled:

* Flat project: ``projects/<slug>/task-notes/README.md`` ``## Tasks`` table.
* Phased project: the current phase is the lowest-numbered file under
  ``projects/<slug>/phases/`` (ignoring ``_template.md``) whose frontmatter
  ``status:`` is not ``Complete``; its table lives at
  ``projects/<slug>/task-notes/phase-XX/README.md``. When every phase is
  ``Complete`` the project itself is done.

Without ``--task`` the lowest-numbered row whose Status is not terminal
(Complete / Superseded / Dropped / Skipped — the full status vocabulary
observed across ``projects/*/task-notes``) is chosen; if that row is
``Blocked`` the script reports it rather than skipping ahead (a later task
may depend on it) or starting it. With ``--task`` that exact row is
returned — IDs compare as full normalized strings, so ``3`` and ``3a`` are
distinct tasks.

Output JSON keys: ``status`` (ready|done|blocked|error), ``task_id``,
``task_title``, ``task_slug``, ``phase`` (or null), ``reason``.

Exit codes: 0 when a task is resolved (status ready) or the project/phase
is done (status done); 1 on a blocked next task (status blocked) or a
resolvable-input error (status error); 2 on a usage/argument error
(argparse). Run ``--self-test`` to exercise the parsing rules against
representative task-table rows.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Status tokens that are terminal ("no longer actionable"). Derived from
# the full status vocabulary in projects/*/task-notes tables: Complete,
# Superseded, Dropped, Skipped are terminal; Not started / In Progress are
# actionable; Blocked is non-actionable but NOT terminal (see BLOCKED_TOKENS).
DONE_TOKENS = {"complete", "superseded", "dropped", "skipped"}

# Non-terminal, non-startable: surfaced as status "blocked" instead of
# being auto-selected or silently skipped (later tasks may depend on it).
BLOCKED_TOKENS = {"blocked"}

# Task-id shape: numeric core (1, 12, 1.2) + optional letter suffix (6c).
# Used for SORTING only — exact --task matching compares normalized full
# ids, so "3" and "3a" stay distinct.
_ID_RE = re.compile(r"(\d+(?:\.\d+)*)\s*([a-z]*)", re.IGNORECASE)


def repo_root() -> Path:
    """Repo root, derived from this script's location (scripts/agents/x.py)."""
    return Path(__file__).resolve().parents[2]


def die(reason: str, *, phase: str | None = None) -> None:
    """Emit an error JSON to stdout, a message to stderr, and exit 1."""
    print(
        json.dumps(
            {
                "status": "error",
                "task_id": None,
                "task_title": None,
                "task_slug": None,
                "phase": phase,
                "reason": reason,
            }
        )
    )
    print(f"resolve_task: error: {reason}", file=sys.stderr)
    raise SystemExit(1)


def strip_md(text: str) -> str:
    """Strip inline markdown emphasis/code/strikethrough from a cell."""
    return text.replace("`", "").replace("*", "").replace("~", "").strip()


def _status_head(status: str) -> str:
    """First word of a status cell, sans annotations like ``(PR #537)``."""
    cleaned = strip_md(status)
    head = re.split(r"[(;]", cleaned, maxsplit=1)[0].strip()
    return head.split()[0].lower() if head.split() else ""


def status_is_done(status: str) -> bool:
    """True when the status cell's leading word is a terminal token."""
    return _status_head(status) in DONE_TOKENS


def status_is_blocked(status: str) -> bool:
    """True when the status cell's leading word marks the task Blocked."""
    return _status_head(status) in BLOCKED_TOKENS


def normalize_id(task_id: str) -> str:
    """Normalize a task id for exact comparison.

    ``Task 3`` → ``3``; ``**6c**`` → ``6c``; ``2.4`` → ``2.4``. Keeps the
    letter suffix, so ``3`` and ``3a`` do NOT compare equal.
    """
    cleaned = strip_md(task_id).lower()
    cleaned = re.sub(r"^task\s+", "", cleaned)
    return cleaned.strip().rstrip(".")


def parse_id_key(task_id: str) -> tuple[tuple[int, ...], str]:
    """Sort key for a task id: (numeric tuple, lowercase letter suffix).

    ``3`` → ``((3,), "")``; ``3a`` → ``((3,), "a")``; ``2.4`` →
    ``((2, 4), "")``. Sorting only — never used for exact matching.
    """
    match = _ID_RE.search(task_id)
    if not match:
        return ((), "")
    nums = tuple(int(part) for part in match.group(1).split("."))
    return (nums, match.group(2).lower())


def kebab(text: str) -> str:
    """Lowercase kebab-case slug from arbitrary text."""
    text = strip_md(text).lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def split_row(line: str) -> list[str]:
    """Split a markdown table row into stripped cells (drop edge pipes)."""
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [cell.strip() for cell in line.split("|")]


def is_separator(cells: list[str]) -> bool:
    """A separator row is all cells like ``---`` / ``:--:``."""
    return bool(cells) and all(
        re.fullmatch(r":?-{1,}:?", cell) for cell in cells if cell != ""
    ) and any(cell for cell in cells)


class Task:
    def __init__(self, task_id: str, title: str, status: str, note: str) -> None:
        self.id = task_id
        self.title = title
        self.status = status
        self.note = note

    @property
    def sort_key(self) -> tuple[tuple[int, ...], str]:
        return parse_id_key(self.id)

    @property
    def done(self) -> bool:
        return status_is_done(self.status)

    @property
    def blocked(self) -> bool:
        return status_is_blocked(self.status)

    @property
    def slug(self) -> str:
        note = strip_md(self.note)
        if note.endswith(".md"):
            note = note[: -len(".md")]
        # Note filename (e.g. task-1-group-taxonomy) already is the slug.
        if note.startswith("task-"):
            return note
        # Fallback: synthesize from id + title.
        return f"task-{self.id}-{kebab(self.title)}".rstrip("-")


def find_tasks_table(text: str, source: Path) -> list[Task]:
    """Parse the ``## Tasks`` markdown table from a task-notes README."""
    lines = text.splitlines()

    # Locate the "## Tasks" heading.
    start = None
    for i, line in enumerate(lines):
        if re.match(r"^#{1,6}\s+Tasks\s*$", line.strip()):
            start = i + 1
            break
    if start is None:
        die(f"no '## Tasks' heading in {source}")

    # Find the header row (first table-looking line after the heading).
    header_idx = None
    for i in range(start, len(lines)):
        stripped = lines[i].strip()
        if stripped.startswith("|"):
            cells = split_row(stripped)
            if not is_separator(cells):
                header_idx = i
                break
        # Stop if we hit the next section before any table.
        if stripped.startswith("#") and i > start:
            break
    if header_idx is None:
        die(f"no task table under '## Tasks' in {source}")

    header = split_row(lines[header_idx])
    col = {name.lower(): idx for idx, name in enumerate(header)}

    def find_col(*candidates: str) -> int | None:
        for cand in candidates:
            for name, idx in col.items():
                if name == cand or cand in name:
                    return idx
        return None

    id_col = find_col("#", "id")
    status_col = find_col("status")
    title_col = find_col("task")
    note_col = find_col("task note", "note")

    if id_col is None or status_col is None:
        die(f"task table missing '#' or 'Status' column in {source}")

    tasks: list[Task] = []
    for i in range(header_idx + 1, len(lines)):
        stripped = lines[i].strip()
        if not stripped.startswith("|"):
            # Blank line or prose ends the table.
            if stripped == "":
                break
            continue
        cells = split_row(lines[i])
        if is_separator(cells):
            continue

        def cell(idx: int | None) -> str:
            if idx is None or idx >= len(cells):
                return ""
            return cells[idx]

        raw_id = strip_md(cell(id_col))
        if not parse_id_key(raw_id)[0]:
            # Not a real task row (e.g. a continuation / stray line).
            continue
        tasks.append(
            Task(
                task_id=raw_id,
                title=strip_md(cell(title_col)),
                status=cell(status_col),
                note=cell(note_col),
            )
        )

    return tasks


def phase_status(phase_file: Path) -> str:
    """Read the ``status:`` value from a phase file's YAML frontmatter."""
    text = phase_file.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return ""
    for line in lines[1:]:
        if line.strip() == "---":
            break
        m = re.match(r"\s*status\s*:\s*(.+?)\s*$", line)
        if m:
            return m.group(1).strip()
    return ""


def resolve_phase(project_dir: Path) -> tuple[str | None, Path | None]:
    """Return (phase_number, task_notes_readme) for the current phase.

    ``(None, None)`` means every phase is Complete (project done).
    """
    phases_dir = project_dir / "phases"
    phase_files = sorted(
        p
        for p in phases_dir.glob("phase-*.md")
        if p.name != "_template.md"
    )
    if not phase_files:
        die(f"phased project has no phase files under {phases_dir}")

    def phase_num(p: Path) -> tuple[int, ...]:
        m = re.search(r"phase-(\d+)", p.name)
        return (int(m.group(1)),) if m else ()

    phase_files.sort(key=phase_num)

    for phase_file in phase_files:
        status = phase_status(phase_file)
        if status.lower() != "complete":
            m = re.search(r"phase-(\d+)", phase_file.name)
            num = m.group(1) if m else phase_file.stem
            readme = project_dir / "task-notes" / f"phase-{num}" / "README.md"
            if not readme.is_file():
                die(
                    f"phase {num} task-notes README not found: {readme}",
                    phase=num,
                )
            return num, readme

    # All phases complete.
    return None, None


def emit(task: Task, phase: str | None) -> None:
    if task.done:
        status = "done"
        reason = f"task {task.id} already {strip_md(task.status).lower()}"
    elif task.blocked:
        status = "blocked"
        reason = (
            f"task {task.id} is Blocked — resolve the blocker (see its "
            f"task note) or pick a different task explicitly"
        )
    else:
        status = "ready"
        reason = f"next actionable task: {task.id}"
    print(
        json.dumps(
            {
                "status": status,
                "task_id": task.id,
                "task_title": task.title,
                "task_slug": task.slug,
                "phase": phase,
                "reason": reason,
            }
        )
    )
    if status == "blocked":
        print(f"resolve_task: blocked: {reason}", file=sys.stderr)
        raise SystemExit(1)
    raise SystemExit(0)


def emit_done(phase: str | None, reason: str) -> None:
    print(
        json.dumps(
            {
                "status": "done",
                "task_id": None,
                "task_title": None,
                "task_slug": None,
                "phase": phase,
                "reason": reason,
            }
        )
    )
    raise SystemExit(0)


def self_test() -> None:
    """Pin parsing rules against representative task-table row shapes.

    The rows mirror real tables (lsp-code-actions' struck-through Dropped
    row, lsp-assist-rules' 3a-before-3 ordering, annotated Complete cells)
    so regressions in status/id handling fail here, not mid-pipeline.
    """
    # Status classification — full observed vocabulary.
    assert status_is_done("Complete")
    assert status_is_done("**Complete** (2026-06-06; PR #537)")
    assert status_is_done("Superseded")
    assert status_is_done("Dropped")
    assert status_is_done("~~Skipped~~")
    assert not status_is_done("Not started")
    assert not status_is_done("In Progress")
    assert not status_is_done("Blocked")
    assert status_is_blocked("Blocked (waiting on ADR-0002)")
    assert not status_is_blocked("Not started")

    # Exact id matching — suffixes are significant.
    assert normalize_id("3") != normalize_id("3a")
    assert normalize_id("Task 3") == normalize_id("3")
    assert normalize_id("**6c**") == "6c"
    assert normalize_id("2.4") == "2.4"

    # Sorting — numeric core first, then letter suffix.
    ids = ["3a", "1", "2.4", "3", "10", "2"]
    ordered = sorted(ids, key=parse_id_key)
    assert ordered == ["1", "2", "2.4", "3", "3a", "10"], ordered

    # Full-table parse over representative row shapes.
    table = """\
## Tasks

| #  | Task                                  | Deps | Status      | Task note |
|----|---------------------------------------|------|-------------|-----------|
| 1  | Group taxonomy                        | —    | Complete (PR #631) | task-1-group-taxonomy.md |
| 2  | Rule scaffolding                      | 1    | Blocked     | task-2-rule-scaffolding.md |
| 3a | `resolve_source_member_set` API       | —    | Not started | task-3a-resolve-source-members.md |
| 3  | Collapse-to-wildcard rule             | 2,3a | Not started | task-3-collapse-to-wildcard.md |
| 6c | ~~Measure perf gate~~ (DROPPED)       | 6b   | Dropped     | (folded into 6b note) |
"""
    tasks = find_tasks_table(table, Path("<self-test>"))
    by_id = {t.id: t for t in tasks}
    assert len(tasks) == 5, [t.id for t in tasks]
    assert by_id["6c"].done, "Dropped must be terminal"
    assert by_id["2"].blocked and not by_id["2"].done
    # Exact matching keeps 3 and 3a distinct.
    assert [t.id for t in tasks if normalize_id(t.id) == "3"] == ["3"]
    # Sorted order is by id (3 before 3a), regardless of table order; the
    # first non-done task is the Blocked task 2 — surfaced, never skipped.
    ordered_tasks = sorted(tasks, key=lambda t: t.sort_key)
    assert [t.id for t in ordered_tasks] == ["1", "2", "3", "3a", "6c"]
    first_open = next(t for t in ordered_tasks if not t.done)
    assert first_open.id == "2" and first_open.blocked

    print("self-test: all assertions passed")
    raise SystemExit(0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve the next actionable task for a project.",
    )
    parser.add_argument("--project", help="project slug")
    parser.add_argument("--task", help="specific task id (e.g. 5, 1.2, 3a)")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run the built-in parsing assertions and exit",
    )
    args = parser.parse_args()

    if args.self_test:
        self_test()
    if not args.project:
        parser.error("--project is required (unless running --self-test)")

    project_dir = repo_root() / "projects" / args.project
    if not project_dir.is_dir():
        die(f"project not found: {project_dir}")

    phases_dir = project_dir / "phases"
    phased = phases_dir.is_dir() and any(
        p.name != "_template.md" for p in phases_dir.glob("phase-*.md")
    )

    if phased:
        phase_num, readme = resolve_phase(project_dir)
        if phase_num is None or readme is None:
            emit_done(None, "all phases complete")
    else:
        phase_num = None
        readme = project_dir / "task-notes" / "README.md"
        if not readme.is_file():
            die(f"task-notes README not found: {readme}")

    tasks = find_tasks_table(readme.read_text(encoding="utf-8"), readme)
    if not tasks:
        die(f"no task rows parsed from {readme}", phase=phase_num)

    if args.task is not None:
        want = normalize_id(args.task)
        if not want:
            die(f"unparseable task id: '{args.task}'", phase=phase_num)
        matches = [t for t in tasks if normalize_id(t.id) == want]
        if not matches:
            die(f"task '{args.task}' not found in {readme}", phase=phase_num)
        emit(matches[0], phase_num)

    tasks.sort(key=lambda t: t.sort_key)
    for task in tasks:
        if not task.done:
            emit(task, phase_num)

    emit_done(phase_num, "all tasks complete")


if __name__ == "__main__":
    main()
