"""Regression tests for `.claude/skills/begin-phase/scripts/resolve_phase.py`.

The agent-workflow helpers are not importable as a package (they live under
`.claude/` and `scripts/`, outside `hazma`), so they are loaded by path.

The bug these guard: `build_prompt` used to interpolate the *canonical* phase
id into filesystem paths. `phase_id_from_prefix("02")` normalizes to `"2"`, so
a `phase-02-*.md` phase file produced a kickoff prompt pointing at
`task-notes/phase-2/README.md` while the template and
`scripts/agents/resolve_task.py` both use `task-notes/phase-02/README.md`.
The next agent was sent to a directory that does not exist.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

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

SCRIPT = REPO_ROOT / ".claude/skills/begin-phase/scripts/resolve_phase.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("resolve_phase", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["resolve_phase"] = module
    spec.loader.exec_module(module)
    return module


pytestmark = pytest.mark.skipif(not SCRIPT.is_file(), reason=f"{SCRIPT} not present")


@pytest.fixture(scope="module")
def resolve_phase():
    return _load_module()


def _scaffold(tmp_path: Path, prefix: str, slug: str = "demo") -> Path:
    """Create a minimal phased project whose phase file uses `prefix`."""
    project = tmp_path / "projects" / slug
    (project / "phases").mkdir(parents=True)
    (project / "task-notes" / f"phase-{prefix}").mkdir(parents=True)
    (project / "learnings").mkdir(parents=True)

    (project / "PLAN.md").write_text(
        "---\nstatus: In Progress\nphased: true\nversion_bump: patch\n---\n\n"
        "# Project: Demo\n"
    )
    (project / "phases" / f"phase-{prefix}-kernels.md").write_text(
        f"---\nphase: {prefix}\ntitle: Kernels\nstatus: Not started\n---\n\n"
        f"# Phase {prefix}: Kernels\n\n"
        "## Prerequisites\n\n- None.\n\n"
        "## Tasks\n\n### Task 1.1: Build it\n\n"
        "**Exit criteria:**\n\n- green\n"
    )
    (project / "task-notes" / f"phase-{prefix}" / "README.md").write_text(
        "# Working Memory\n\n## Tasks\n\n"
        "| # | Task | Depends on | Status | Task Note |\n"
        "|---|------|------------|--------|-----------|\n"
        "| 1.1 | Build it | — | Not started | `task-1.1-build.md` |\n"
    )
    return project


def test_phase_id_normalization_drops_zero_padding(resolve_phase):
    """The canonical id is unpadded -- this is what makes the bug possible."""
    assert resolve_phase.phase_id_from_prefix("02") == "2"
    assert resolve_phase.phase_id_from_prefix("2") == "2"
    assert resolve_phase.phase_id_from_prefix("10") == "10"


@pytest.mark.parametrize("prefix", ["01", "02", "09", "10"])
def test_prompt_paths_use_the_filename_prefix(resolve_phase, tmp_path, prefix):
    """Kickoff-prompt paths must match the on-disk directory, zero-padding
    included -- not the normalized display id."""
    project = _scaffold(tmp_path, prefix)
    phases = resolve_phase.load_phase_records(project)
    phase_id = resolve_phase.phase_id_from_prefix(prefix)

    prompt = resolve_phase.build_prompt(
        project_slug="demo",
        project_dir=project,
        repo_root=tmp_path,
        phases=phases,
        phase_id=phase_id,
    )

    expected = f"task-notes/phase-{prefix}/README.md"
    assert prompt.count(expected) == 2, (
        f"expected both per-phase README references to use {expected!r}\n"
        f"prompt was:\n{prompt}"
    )

    # And the directory the prompt names must actually exist on disk.
    assert (project / "task-notes" / f"phase-{prefix}" / "README.md").is_file()

    # The normalized id must never appear as a path segment when it differs
    # from the padded prefix (this is the exact regression).
    if phase_id != prefix:
        assert f"task-notes/phase-{phase_id}/" not in prompt


def test_prompt_still_uses_the_canonical_id_for_display(resolve_phase, tmp_path):
    """Display text keeps the unpadded id -- 'Phase 2', not 'Phase 02'."""
    project = _scaffold(tmp_path, "02")
    phases = resolve_phase.load_phase_records(project)

    prompt = resolve_phase.build_prompt(
        project_slug="demo",
        project_dir=project,
        repo_root=tmp_path,
        phases=phases,
        phase_id="2",
    )

    assert "You are starting Phase 2: Kernels" in prompt
    assert "Work only within Phase 2." in prompt


@pytest.mark.parametrize("prefix", ["01", "02", "09", "10"])
def test_resolve_task_agrees_on_the_phase_directory(tmp_path, prefix):
    """The two helpers must derive the same per-phase directory, or a
    begin-phase handoff points somewhere resolve_task cannot read.

    `resolve_task.resolve_phase()` is called directly rather than through the
    CLI: the CLI locates the project via `git rev-parse`, so it would look
    under the real repo rather than `tmp_path`.
    """
    resolver_path = REPO_ROOT / "scripts/agents/resolve_task.py"
    if not resolver_path.is_file():
        pytest.skip(f"{resolver_path} not present")

    spec = importlib.util.spec_from_file_location("resolve_task", resolver_path)
    assert spec is not None and spec.loader is not None
    resolve_task = importlib.util.module_from_spec(spec)
    sys.modules["resolve_task"] = resolve_task
    spec.loader.exec_module(resolve_task)

    project = _scaffold(tmp_path, prefix)
    num, readme = resolve_task.resolve_phase(project)

    # This is the shared convention: the directory name carries the phase
    # filename's digits, zero-padding included.
    assert num == prefix
    assert readme == project / "task-notes" / f"phase-{prefix}" / "README.md"
    assert readme.is_file()
