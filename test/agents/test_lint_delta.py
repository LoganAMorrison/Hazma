"""Regression tests for `scripts/agents/lint_delta.py`.

The agent-workflow helpers are not importable as a package (they live under
`scripts/`, outside `hazma`), so the module is loaded by path and the
command-line behavior is exercised through a subprocess.

Each test builds a throwaway git repository whose base commit already
carries lint findings, then edits the working tree without committing —
the state `scripts/agents/preflight.sh` actually runs against. What the
gate must get right is the split between the two: findings the base commit
already had are the trunk's, and only what the working tree adds belongs
to the branch being checked.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(
    subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).parent,
    ).stdout.strip()
)

SCRIPT = REPO_ROOT / "scripts/agents/lint_delta.py"

# The script's exit codes, named so the assertions below do not compare
# against bare integers.
NO_NEW_FINDINGS = 0
NEW_FINDINGS = 1

FIXTURE_PYPROJECT = """\
[project]
name = "fixture"
version = "0.0.0"

[tool.isort]
profile = "black"

[tool.ruff]
target-version = "py310"

[tool.ruff.lint]
select = ["F"]
"""

# Two findings on purpose, at different depths: `os` is unused at the top
# of the file and never moves, while `unused` sits inside the function and
# shifts whenever a line is inserted above it. The drift test needs a
# finding that moves.
BASE_MODULE = """\
import os


def compute():
    unused = 1
    return 2
"""

# isort wants these two lines swapped, so the file is already unsorted at
# the base commit.
UNSORTED_MODULE = """\
import sys
import os
"""

requires_ruff = pytest.mark.skipif(
    shutil.which("ruff") is None, reason="ruff is not installed"
)
requires_isort = pytest.mark.skipif(
    shutil.which("isort") is None, reason="isort is not installed"
)


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("lint_delta", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(root: Path, *args: str) -> str:
    """Run git in `root`, isolated from the developer's own configuration."""
    proc = subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout


def _run(root: Path, linter: str, *paths: str) -> subprocess.CompletedProcess[str]:
    """Compare the working tree in `root` against its own base commit."""
    base = _git(root, "rev-parse", "HEAD").strip()
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--linter", linter, "--base", base, *paths],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A git repository whose base commit already carries lint findings."""
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "pyproject.toml").write_text(FIXTURE_PYPROJECT)
    (root / "pkg" / "module.py").write_text(BASE_MODULE)
    (root / "pkg" / "unsorted.py").write_text(UNSORTED_MODULE)
    (root / "notes.md").write_text("# Notes\n")
    _git(root, "init", "--quiet")
    _git(root, "config", "user.email", "gate@example.invalid")
    _git(root, "config", "user.name", "Preflight Fixture")
    _git(root, "add", "-A")
    _git(root, "commit", "--quiet", "-m", "base")
    return root


@requires_ruff
def test_pre_existing_findings_are_not_reported(repo: Path) -> None:
    """The trunk's backlog stays the trunk's, even on a file you edited."""
    module = repo / "pkg" / "module.py"
    module.write_text(module.read_text() + "\n\nOTHER = 3\n")

    result = _run(repo, "ruff", "pkg")

    assert result.returncode == NO_NEW_FINDINGS, result.stdout
    assert "0 new" in result.stdout
    # Two findings in each of the fixture's two modules, all four carried
    # rather than added.
    assert "4 pre-existing" in result.stdout


@requires_ruff
def test_new_finding_is_reported(repo: Path) -> None:
    """A finding the working tree introduces fails the gate, alone."""
    module = repo / "pkg" / "module.py"
    module.write_text("import sys\n" + module.read_text())

    result = _run(repo, "ruff", "pkg")

    assert result.returncode == NEW_FINDINGS, result.stdout
    assert "1 new" in result.stdout
    assert "`sys`" in result.stdout
    # The pre-existing unused import is carried, so it must not be listed
    # among the findings this tree is being asked to fix.
    assert "`os`" not in result.stdout


@requires_ruff
def test_line_drift_does_not_resurface_existing_findings(repo: Path) -> None:
    """Inserting a line above a finding must not present it as new.

    This is the reason findings are compared without their position. A
    positional comparison reports every finding below an inserted line,
    which is the whole file for an edit near the top.
    """
    module = repo / "pkg" / "module.py"
    shifted = module.read_text().replace(
        "def compute():", "# A comment that shifts the body down.\ndef compute():"
    )
    module.write_text(shifted)

    result = _run(repo, "ruff", "pkg")

    assert result.returncode == NO_NEW_FINDINGS, result.stdout
    assert "0 new" in result.stdout


@requires_ruff
def test_findings_in_a_new_file_are_all_reported(repo: Path) -> None:
    """An untracked file has no baseline, so everything in it is added."""
    (repo / "pkg" / "added.py").write_text("import json\n")

    result = _run(repo, "ruff", "pkg")

    assert result.returncode == NEW_FINDINGS, result.stdout
    assert "pkg/added.py" in result.stdout
    assert "`json`" in result.stdout


@requires_ruff
def test_fixing_a_finding_is_credited_and_passes(repo: Path) -> None:
    """Removing trunk debt is a pass, and is reported as such."""
    module = repo / "pkg" / "module.py"
    module.write_text(module.read_text().replace("import os\n", ""))

    result = _run(repo, "ruff", "pkg")

    assert result.returncode == NO_NEW_FINDINGS, result.stdout
    assert "0 new, 1 fixed" in result.stdout


@requires_ruff
def test_a_diff_without_python_is_not_compared(repo: Path) -> None:
    """A docs-only change needs no flag to avoid the trunk's backlog.

    `preflight.sh` defaults `--paths` to the package when the caller names
    none, so before this the cheapest honest answer for a markdown-only
    commit was a tree-wide lint run that could only fail.
    """
    (repo / "notes.md").write_text("# Notes\n\nA second line.\n")

    result = _run(repo, "ruff", "pkg")

    assert result.returncode == NO_NEW_FINDINGS, result.stdout
    assert "no Python changed" in result.stdout


@requires_ruff
def test_changing_the_rule_set_forces_a_comparison(repo: Path) -> None:
    """Editing the config is a change of question, not a docs-only change.

    The base tree is linted with the base's own `pyproject.toml`, so
    tightening the rule set surfaces the findings the new rules add rather
    than silently skipping the comparison.
    """
    config = repo / "pyproject.toml"
    config.write_text(
        FIXTURE_PYPROJECT.replace('select = ["F"]', 'select = ["F", "I"]')
    )

    result = _run(repo, "ruff", "pkg")

    assert result.returncode == NEW_FINDINGS, result.stdout
    assert "pkg/unsorted.py" in result.stdout
    assert "I001" in result.stdout


@requires_isort
def test_isort_reports_only_newly_unsorted_files(repo: Path) -> None:
    """A file unsorted before you arrived is not yours; a new one is."""
    (repo / "pkg" / "fresh.py").write_text(UNSORTED_MODULE)

    result = _run(repo, "isort", "pkg")

    assert result.returncode == NEW_FINDINGS, result.stdout
    assert "pkg/fresh.py" in result.stdout
    assert "pkg/unsorted.py" not in result.stdout


@requires_isort
def test_isort_failing_to_run_is_not_a_clean_result(repo: Path) -> None:
    """A linter that crashed must fail the gate, not report an empty diff.

    isort exits 1 both when it would re-sort a file and when it cannot run
    at all — an unreadable settings file raises and prints a traceback,
    and a path it cannot open prints `Broken N paths`. Neither leaves an
    ERROR line behind, so the exit code alone cannot separate them from a
    clean tree, and an empty result would be reported as a pass.
    """
    (repo / "pyproject.toml").write_text("[tool.isort]\nprofile =\n")

    result = _run(repo, "isort", "pkg")

    assert result.returncode not in (NO_NEW_FINDINGS, NEW_FINDINGS), result.stdout
    assert "could not compare" in result.stdout


def test_a_linter_exiting_unexpectedly_is_not_a_clean_result(repo: Path) -> None:
    """An exit code neither linter defines means no verdict is available."""
    # Without a Python change the comparison short-circuits before any
    # linter runs, and the shim below would never be reached.
    module = repo / "pkg" / "module.py"
    module.write_text(module.read_text() + "\nOTHER = 3\n")

    shim = repo / "shim"
    shim.mkdir()
    (shim / "isort").write_text("#!/bin/sh\nexit 3\n")
    (shim / "isort").chmod(0o755)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--linter", "isort", "--base", "HEAD", "pkg"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PATH": f"{shim}:{os.environ['PATH']}"},
    )

    assert result.returncode not in (NO_NEW_FINDINGS, NEW_FINDINGS), result.stdout
    assert "exited 3" in result.stdout


def test_isort_error_lines_split_on_paths_containing_spaces(tmp_path: Path) -> None:
    """The path/message split survives a directory name with a space.

    isort separates the file from its message with a space and quotes
    neither, so the split is found by taking the longest leading run of
    words that still names a file.
    """
    module = _load_module()
    target = tmp_path / "a dir" / "mod.py"
    target.parent.mkdir()
    target.write_text("import os\n")

    path, message = module.split_isort_error(
        f"{target} Imports are incorrectly sorted and/or formatted.", tmp_path
    )

    assert path == "a dir/mod.py"
    assert message == "Imports are incorrectly sorted and/or formatted."


def test_a_missing_baseline_never_reports_success(repo: Path) -> None:
    """No comparison means no verdict, so the gate must not go green."""
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--linter",
            "ruff",
            "--base",
            "refs/heads/no-such-branch",
            "pkg",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode not in (NO_NEW_FINDINGS, NEW_FINDINGS)
    assert "could not compare" in result.stdout
