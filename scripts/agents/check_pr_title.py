#!/usr/bin/env python3
"""Validate a Conventional Commits PR-title / header string.

A **local** validator, so an agent can deterministically pre-check a title
instead of eyeballing a character count. The rules it enforces are the ones
written down in docs/PR_GUIDELINES.md -- that document is authoritative and
this script is its executable form. If the two ever disagree, the document
wins and this script is the bug.

There is **no CI job enforcing PR titles in this repo.** Nothing rejects a
malformed title automatically; the convention is upheld by this checker and
by review. Do not read a green CI run as a title that passed.

Rules (see docs/PR_GUIDELINES.md "Title format"):
  - type   ∈ {feat fix chore ci docs test refactor perf style build revert}
  - scope  required; matches ^[a-z0-9-]+$; ≤10 chars; no leading/trailing
           hyphen; not equal to a type
  - header total length ≤69 chars
  - subject starts with an alphanumeric char; no trailing "." or space

The "scope must not equal a type" rule is stricter than Conventional Commits
itself requires; it catches accidental `docs(docs):`-style titles.

Usage:
    check_pr_title.py "feat(lint): add a rule"   # title as argv
    echo "feat(lint): add a rule" | check_pr_title.py   # title on stdin

Output: a single "OK: <title>" line on success, or one line per violation.
Exit code: 0 when valid, 1 when any violation is found, 2 on usage error.
"""

from __future__ import annotations

import re
import sys

TYPES = frozenset(
    {
        "feat",
        "fix",
        "chore",
        "ci",
        "docs",
        "test",
        "refactor",
        "perf",
        "style",
        "build",
        "revert",
    }
)

MAX_HEADER_LEN = 69
MAX_SCOPE_LEN = 10
SCOPE_RE = re.compile(r"^[a-z0-9-]+$")

# type, optional (scope), optional breaking-change "!", ": ", subject.
HEADER_RE = re.compile(
    r"^(?P<type>[a-zA-Z]+)"
    r"(?:\((?P<scope>[^)]*)\))?"
    r"(?P<bang>!)?"
    r": (?P<subject>.*)$"
)


def check(header: str) -> list[str]:
    """Return a list of violation messages (empty means the title is valid)."""
    violations: list[str] = []

    if len(header) > MAX_HEADER_LEN:
        violations.append(
            f"header is {len(header)} chars; must be ≤{MAX_HEADER_LEN}"
        )

    match = HEADER_RE.match(header)
    if match is None:
        violations.append(
            'not a conventional-commit header: expected "type(scope): subject"'
        )
        return violations

    type_ = match.group("type")
    scope = match.group("scope")
    subject = match.group("subject")

    if type_ not in TYPES:
        allowed = " ".join(sorted(TYPES))
        violations.append(f'type "{type_}" is not allowed; use one of: {allowed}')

    if scope is None:
        violations.append("scope is required, e.g. type(scope): subject")
    else:
        if scope == "":
            violations.append("scope is empty; put a name inside the ()")
        else:
            if not SCOPE_RE.match(scope):
                violations.append(
                    f'scope "{scope}" must match ^[a-z0-9-]+$ '
                    "(lowercase alphanumeric and hyphens)"
                )
            if len(scope) > MAX_SCOPE_LEN:
                violations.append(
                    f'scope "{scope}" is {len(scope)} chars; must be '
                    f"≤{MAX_SCOPE_LEN}"
                )
            if scope.startswith("-") or scope.endswith("-"):
                violations.append(
                    f'scope "{scope}" must not start or end with a hyphen'
                )
            if scope in TYPES:
                violations.append(
                    f'scope "{scope}" must not be a commit type'
                )

    if subject == "":
        violations.append("subject is empty")
    else:
        if not subject[0].isalnum():
            violations.append(
                f'subject must start with an alphanumeric char, not "{subject[0]}"'
            )
        if subject.endswith("."):
            violations.append('subject must not end with a "."')
        elif subject.endswith(" "):
            violations.append("subject must not end with a space")

    return violations


def read_title(argv: list[str]) -> str:
    """Get the title from argv (joined) or, if none given, from stdin."""
    args = argv[1:]
    if args:
        return " ".join(args)
    data = sys.stdin.read()
    # Take the first non-empty line; a PR title is a single line.
    for line in data.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return data.strip()


def main(argv: list[str]) -> int:
    title = read_title(argv)
    if title == "":
        print("error: no title provided (pass as an argument or on stdin)",
              file=sys.stderr)
        return 2

    violations = check(title)
    if not violations:
        print(f"OK: {title}")
        return 0

    print(f"INVALID: {title}", file=sys.stderr)
    for message in violations:
        print(f"  - {message}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
