#!/usr/bin/env python3
"""Report the lint findings a working tree adds relative to its base commit.

`isort` and `ruff` both answer "is this whole tree clean?", and hazma's is
not: the rule set configured in `pyproject.toml` reports thousands of
findings on unmodified trunk code. Asked absolutely, those two gates fail
on every branch, so their verdict says nothing about the branch that ran
them — and a genuine regression is indistinguishable from the backlog it
lands next to.

This asks the narrower question instead. The linter runs twice, once over
the working tree and once over the same paths as they stand at the merge
base, and only what the working tree *adds* is reported. A branch that
touches no Python inherits nothing, and a branch that introduces a finding
fails on that finding alone.

Findings are compared as a multiset of `(path, rule, message)`, with line
and column numbers dropped, so inserting a line above an existing finding
does not re-report it as new. The cost of dropping position is that
removing one finding and adding an identical one elsewhere in the same
file cancels out; the alternative, matching on position, re-reports every
finding below an inserted line and buries the real one.

`isort --check-only` is file-granular — it reports that a file's imports
are unsorted, not which import is at fault — so its unit of comparison is
the file. A file already unsorted at the base stays exempt until someone
sorts it.

Exit status is 0 when the tree adds nothing, 1 when it adds findings
(which are printed), and 2 when the comparison itself could not be made.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

# The linters read their rule sets from here, so a branch that edits it is
# changing the question being asked and must be compared under both
# answers: the base tree is linted with the base's copy.
CONFIG = "pyproject.toml"

# Suffixes whose contents can move a finding. `pyproject.toml` is handled
# separately, since it sits outside the linted paths.
PYTHON_SUFFIXES = frozenset({".py", ".pyi"})

# One finding, reduced to what survives an unrelated edit elsewhere in the
# file: which file, which rule, and what the rule said.
Finding = tuple[str, str, str]


class LintDeltaError(RuntimeError):
    """The comparison could not be made, so no verdict is available."""


def git(*args: str, cwd: Path) -> str:
    """Run a read-only git command and return its stdout."""
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise LintDeltaError(
            f"git {' '.join(args)} failed ({proc.returncode}): "
            f"{proc.stderr.strip()}"
        )
    return proc.stdout


def git_lines(*args: str, cwd: Path) -> list[str]:
    """Run a read-only git command and return its non-empty stdout lines."""
    return [line for line in git(*args, cwd=cwd).splitlines() if line]


def resolve_base(base_ref: str, *, root: Path) -> str:
    """Resolve the commit the working tree should be compared against.

    The merge base rather than the tip of the trunk: a branch is
    responsible for what it added to the tree it forked from, not for
    findings that landed on the trunk afterwards. When the two share no
    history — a shallow clone, most often — the ref itself is the closest
    honest answer available.
    """
    try:
        return git("merge-base", "HEAD", base_ref, cwd=root).strip()
    except LintDeltaError:
        return git("rev-parse", "--verify", f"{base_ref}^{{commit}}", cwd=root).strip()


def exists_at(rev: str, path: str, *, root: Path) -> bool:
    """Report whether `path` names a blob or tree in `rev`."""
    return (
        subprocess.run(
            ["git", "cat-file", "-e", f"{rev}:{path}"],
            cwd=root,
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )


def diff_touches_python(rev: str, paths: list[str], *, root: Path) -> bool:
    """Report whether the working tree differs from `rev` in a linted file.

    Both halves of "the working tree" count: files tracked and modified,
    and files not yet added at all. A new `.py` that git has never seen is
    exactly the case a gate must not miss.
    """
    if git_lines("diff", "--name-only", rev, "--", CONFIG, cwd=root):
        return True
    changed = git_lines("diff", "--name-only", rev, "--", *paths, cwd=root)
    changed += git_lines(
        "ls-files", "--others", "--exclude-standard", "--", *paths, cwd=root
    )
    return any(Path(name).suffix in PYTHON_SUFFIXES for name in changed)


def materialize(rev: str, paths: list[str], *, root: Path, dest: Path) -> list[str]:
    """Extract `rev`'s copy of `paths` into `dest`, plus the linter config.

    Returns the subset of `paths` that exists in `rev`; a path added by
    this branch is absent there, and every finding in it is new by
    definition.
    """
    present = [path for path in paths if exists_at(rev, path, root=root)]
    wanted = present + ([CONFIG] if exists_at(rev, CONFIG, root=root) else [])
    if not wanted:
        return []

    archive = dest / "base.tar"
    with archive.open("wb") as handle:
        proc = subprocess.run(
            ["git", "archive", "--format=tar", rev, "--", *wanted],
            cwd=root,
            stdout=handle,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    if proc.returncode != 0:
        raise LintDeltaError(f"git archive failed: {proc.stderr.strip()}")

    tree = dest / "tree"
    tree.mkdir()
    # The `tar` binary rather than `tarfile.extractall`, whose extraction
    # filter is the caller's responsibility on some supported interpreters
    # and the default on others; the shell-out behaves the same on all of
    # them.
    extract = subprocess.run(
        ["tar", "-xf", str(archive), "-C", str(tree)],
        capture_output=True,
        text=True,
        check=False,
    )
    if extract.returncode != 0:
        raise LintDeltaError(f"tar failed: {extract.stderr.strip()}")
    archive.unlink()
    return present


def relative_to(path: str, tree: Path) -> str:
    """Express an absolute path a linter printed as a repo-relative one."""
    try:
        return str(Path(path).resolve().relative_to(tree.resolve()))
    except ValueError:
        return path


def run_ruff(tree: Path, paths: list[str]) -> Counter[Finding]:
    """Collect ruff's findings for `paths`, keyed by file, rule, and message."""
    proc = subprocess.run(
        [
            "ruff",
            "check",
            "--no-cache",
            "--output-format",
            "json",
            "--config",
            str(tree / CONFIG),
            *paths,
        ],
        cwd=tree,
        capture_output=True,
        text=True,
        check=False,
    )
    # 0 is clean and 1 is "found something"; anything else means ruff could
    # not answer, which is not the same as a clean tree.
    if proc.returncode not in (0, 1):
        raise LintDeltaError(
            f"ruff exited {proc.returncode}: {proc.stderr.strip()[:500]}"
        )
    try:
        report = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError as exc:
        raise LintDeltaError(f"could not parse ruff JSON output: {exc}") from exc

    findings: Counter[Finding] = Counter()
    for item in report:
        # `code` is null for syntax errors, which carry a `name` instead.
        rule = item.get("code") or item.get("name") or "error"
        findings[(relative_to(item["filename"], tree), rule, item["message"])] += 1
    return findings


ISORT_ERROR = re.compile(r"^ERROR:\s+(?P<rest>.+)$")


def split_isort_error(rest: str, tree: Path) -> tuple[str, str]:
    """Split an isort `ERROR:` line into the file it names and its message.

    isort separates the two with a space and neither is quoted, so the
    split point is found by taking the longest leading run of words that
    still names a file on disk. Matching the message text instead would
    tie this to isort's wording.
    """
    words = rest.split(" ")
    for count in range(len(words), 0, -1):
        candidate = " ".join(words[:count])
        if (tree / candidate).exists():
            return relative_to(candidate, tree), " ".join(words[count:])
    # Unreachable while isort names a file it has just read. Strip the
    # tree root regardless: a key still holding an absolute path would
    # differ between the two trees and invent a finding in the diff.
    return rest.removeprefix(f"{tree}/"), ""


def run_isort(tree: Path, paths: list[str]) -> Counter[Finding]:
    """Collect the files isort reports as unsorted under `paths`."""
    proc = subprocess.run(
        ["isort", "--check-only", "--settings-path", str(tree / CONFIG), *paths],
        cwd=tree,
        capture_output=True,
        text=True,
        check=False,
    )
    findings: Counter[Finding] = Counter()
    # isort reports on stderr, but has moved the stream before; read both
    # rather than depending on which one this version chose.
    output = proc.stderr + proc.stdout
    for line in output.splitlines():
        match = ISORT_ERROR.match(line.strip())
        if match is None:
            continue
        path, message = split_isort_error(match.group("rest"), tree)
        findings[(path, "isort", message)] += 1

    if proc.returncode not in (0, 1):
        raise LintDeltaError(f"isort exited {proc.returncode}: {output.strip()[:500]}")
    # Exit 1 means "something here would be re-sorted", so isort has to
    # have named the file. It also exits 1 when it cannot run at all --
    # a traceback for an unreadable settings file, `Broken N paths` for a
    # file it could not open -- and those leave no ERROR line behind. The
    # exit code cannot separate the two, but the empty result can: a run
    # that reports changes while naming nothing has not answered the
    # question, and returning its empty counter would be read as a clean
    # tree and pass the gate.
    if proc.returncode == 1 and not findings:
        raise LintDeltaError(
            f"isort reported changes but named no file: {output.strip()[:500]}"
        )
    return findings


LINTERS = {"isort": run_isort, "ruff": run_ruff}


def format_finding(finding: Finding, count: int) -> str:
    """Render one new finding as a single line, with its multiplicity."""
    path, rule, message = finding
    suffix = f" (x{count})" if count > 1 else ""
    return f"{path}: {rule} {message}{suffix}".rstrip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Report the isort or ruff findings the working tree adds "
            "relative to its merge base."
        )
    )
    parser.add_argument("--linter", required=True, choices=sorted(LINTERS))
    parser.add_argument(
        "--base",
        default="origin/master",
        help="ref to compare against; the merge base with HEAD is used",
    )
    parser.add_argument("paths", nargs="+", help="files or directories to lint")
    args = parser.parse_args(argv)

    try:
        root = Path(git("rev-parse", "--show-toplevel", cwd=Path.cwd()).strip())
        base = resolve_base(args.base, root=root)
        short = base[:12]

        if not diff_touches_python(base, args.paths, root=root):
            print(
                f"SUMMARY: no Python changed against {short} — "
                f"nothing for {args.linter} to compare"
            )
            return 0

        run = LINTERS[args.linter]
        head = run(root, args.paths)
        with tempfile.TemporaryDirectory(prefix="lint-delta-") as workdir:
            dest = Path(workdir)
            present = materialize(base, args.paths, root=root, dest=dest)
            baseline = run(dest / "tree", present) if present else Counter()
    except LintDeltaError as exc:
        print(f"SUMMARY: could not compare against {args.base}: {exc}")
        return 2

    added = head - baseline
    removed = baseline - head
    carried = sum((head & baseline).values())
    trailer = f", {sum(removed.values())} fixed" if removed else ""
    print(
        f"SUMMARY: {sum(added.values())} new{trailer} "
        f"({carried} pre-existing at {short})"
    )
    if not added:
        return 0
    for finding, count in sorted(added.items()):
        print(format_finding(finding, count))
    return 1


if __name__ == "__main__":
    sys.exit(main())
