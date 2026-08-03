#!/usr/bin/env python3
"""Bounds-check every `<file>.py:<line>` citation in a set of markdown docs.

Durable docs (PLANs, ADRs, task notes, follow-ups) cite source lines as
`gamma_ray.py:142` or `hazma/spectra/_photon/_muon.py:73-88`. Those citations
rot silently: the cited file keeps importing while the line number drifts to
something unrelated. This checker resolves each citation to a tracked file and
asserts the cited line is inside it, so the doc-consistency "line-number
citation sweep" (`docs/agents/doc-consistency.md`) is a command anyone can
re-run instead of a hand-audit.

It bounds-checks; it cannot know whether line 142 still says what the doc
claims. Load-bearing citations still deserve a read.

Source extensions checked: `.py`, `.pyx`, `.pxd`.

Resolution order for each cited path:
  1. exact  — the citation is already a repo-relative path that exists
  2. suffix — exactly one tracked source file ends with `/<citation>`
  3. context — the citation is a bare basename and the *same doc* pins it
     via a longer form (e.g. `spectra/_muon.py:83` fixes `_muon.py`)
  4. map    — an explicit `--map <repo/relative/path.py>` override
  5. otherwise: AMBIGUOUS (candidates exist, none chosen) → failure,
     or EXTERNAL (no candidate at all, e.g. a numpy/scipy citation)
     → skipped and reported by name so the skip stays visible

Usage:
    # every changed markdown doc on this branch
    check_doc_citations.py --changed-vs origin/master

    # explicit docs, with short-form overrides
    check_doc_citations.py docs/foo.md \
        --map hazma/spectra/_photon/_muon.py

    # dump each citation and how it resolved
    check_doc_citations.py --changed-vs origin/master --list

Exit code: 0 when every citation is in range, 1 on any out-of-range or
ambiguous citation, 2 on usage error.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

SOURCE_GLOBS = ("*.py", "*.pyx", "*.pxd")

# `gamma_ray.py:83`, `spectra/_muon.py:993-995`, `hazma/utils.py:1`, and
# the run form `utils.py:71/76/85/104-110` (every element is bounds-checked).
CITATION_RE = re.compile(
    r"([A-Za-z0-9_][A-Za-z0-9_./-]*\.(?:py|pyx|pxd)):"
    r"(\d+(?:-\d+)?(?:/\d+(?:-\d+)?)*)"
)


def run_git(repo_root: Path, *args: str) -> str:
    """Run a git command in `repo_root` and return its stdout."""
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def repo_root() -> Path:
    """Absolute path of the enclosing git worktree."""
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return Path(out.strip())


def tracked_source_files(root: Path) -> list[str]:
    """Repo-relative paths of every tracked Python/Cython source file."""
    listing = run_git(root, "ls-files", "--", *SOURCE_GLOBS)
    return [line for line in listing.splitlines() if line]


def changed_docs(root: Path, ref: str) -> list[str]:
    """Repo-relative paths of markdown docs changed since `ref`.

    Uses the merge-base (`ref...HEAD`) so an out-of-date branch does not
    pull unrelated files into the set, and drops deletions.
    """
    listing = run_git(
        root, "diff", "--name-only", "--diff-filter=d", f"{ref}...HEAD"
    )
    return [line for line in listing.splitlines() if line.endswith(".md")]


def line_count(path: Path) -> int:
    """Number of lines in `path` (a trailing newline does not add one)."""
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


class Resolver:
    """Maps a cited source path onto a tracked repo file."""

    def __init__(self, root: Path, overrides: list[str]) -> None:
        self.root = root
        self.files = tracked_source_files(root)
        self.by_suffix: dict[str, list[str]] = defaultdict(list)
        for path in self.files:
            parts = path.split("/")
            # Index every path suffix so `utils.py`, `hazma/utils.py`, … all hit.
            for start in range(len(parts)):
                self.by_suffix["/".join(parts[start:])].append(path)
        self.overrides: dict[str, str] = {}
        for override in overrides:
            self.overrides[Path(override).name] = override

    def candidates(self, citation: str) -> list[str]:
        return self.by_suffix.get(citation, [])

    def resolve(
        self, citation: str, context: dict[str, set[str]]
    ) -> tuple[str | None, str]:
        """Resolve `citation` to a repo path, returning (path, how)."""
        if (self.root / citation).is_file() and citation in self.files:
            return citation, "exact"

        hits = self.candidates(citation)
        if len(hits) == 1:
            return hits[0], "suffix"

        if "/" not in citation:
            pinned = context.get(citation, set())
            if len(pinned) == 1:
                return next(iter(pinned)), "context"

        override = self.overrides.get(Path(citation).name)
        if override is not None and override in self.files:
            return override, "map"

        return None, "ambiguous" if hits else "external"


def doc_context(
    resolver: Resolver, citations: list[str]
) -> dict[str, set[str]]:
    """Basename → paths the doc itself pins via exact/unique-suffix forms."""
    context: dict[str, set[str]] = defaultdict(set)
    for citation in citations:
        if "/" not in citation:
            continue
        hits = resolver.candidates(citation)
        if len(hits) == 1:
            context[Path(citation).name].add(hits[0])
    return context


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Bounds-check `<file>.py:<line>` citations in markdown docs."
    )
    parser.add_argument("docs", nargs="*", help="markdown docs to check")
    parser.add_argument(
        "--changed-vs",
        metavar="REF",
        help="also check every markdown doc changed since REF (merge-base)",
    )
    parser.add_argument(
        "--map",
        action="append",
        default=[],
        metavar="PATH",
        help="repo-relative path pinning an otherwise ambiguous basename; "
        "repeatable",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="print every citation and how it resolved",
    )
    args = parser.parse_args(argv[1:])

    root = repo_root()
    docs = list(args.docs)
    if args.changed_vs:
        docs.extend(doc for doc in changed_docs(root, args.changed_vs)
                    if doc not in docs)
    if not docs:
        print(
            "error: no docs to check (pass paths or --changed-vs REF)",
            file=sys.stderr,
        )
        return 2

    resolver = Resolver(root, args.map)
    how_counts: Counter[str] = Counter()
    external_names: Counter[str] = Counter()
    failures: list[str] = []
    checked = 0

    for doc in sorted(docs):
        doc_path = root / doc
        if not doc_path.is_file():
            failures.append(f"{doc}: no such file")
            continue
        text = doc_path.read_text(encoding="utf-8")
        matches = list(CITATION_RE.finditer(text))
        context = doc_context(resolver, [m.group(1) for m in matches])

        for match in matches:
            citation = match.group(1)
            spans = match.group(2).split("/")
            line_no = text.count("\n", 0, match.start()) + 1

            path, how = resolver.resolve(citation, context)
            if path is None:
                how_counts[how] += len(spans)
                if how == "external":
                    external_names[citation] += len(spans)
                else:
                    hits = resolver.candidates(citation)
                    failures.append(
                        f"{doc}:{line_no}: AMBIGUOUS {citation} — "
                        f"{len(hits)} candidates, pin one with --map "
                        f"(e.g. --map {hits[0]})"
                    )
                continue

            total = line_count(root / path)
            for span in spans:
                how_counts[how] += 1
                checked += 1
                bounds = [int(part) for part in span.split("-")]
                if args.list:
                    print(f"{doc}:{line_no}\t{citation}:{span}\t{how}\t{path}")
                if min(bounds) < 1 or max(bounds) > total:
                    failures.append(
                        f"{doc}:{line_no}: OUT OF RANGE {citation}:{span} — "
                        f"{path} has {total} lines"
                    )

    print(f"docs scanned: {len(docs)}")
    print(f"in-repo citations checked: {checked}")
    for how in ("exact", "suffix", "context", "map"):
        if how_counts[how]:
            print(f"  resolved by {how}: {how_counts[how]}")
    external_total = sum(external_names.values())
    print(f"external citations skipped: {external_total}")
    for name, count in sorted(external_names.items()):
        print(f"  {name} ({count})")

    if failures:
        print(f"FAIL: {len(failures)} problem(s)")
        for failure in failures:
            print(f"  {failure}")
        return 1

    print("out-of-range or ambiguous: NONE")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
