#!/usr/bin/env python3
"""Resolve which phase of a phased project is eligible to start next.

Build a kickoff prompt for the next agent. This script is the readiness oracle
for the agent-specific ``begin-phase`` skills. It inspects
the frontmatter of each ``projects/<slug>/phases/phase-XX-*.md`` file plus the
project's ``PLAN.md`` frontmatter, then emits JSON with one of three states:

- ``ready`` -- exactly one phase is eligible and a kickoff prompt is included
- ``choose`` -- multiple phases are eligible; the caller must pick one
- ``blocked`` -- the requested (or next) phase has unmet prerequisites

Prerequisite handling is intentionally conservative. Each phase file's
``## Prerequisites`` section is prose (human-readable). This script scans it
for machine-readable patterns like ``Phase 0 complete`` or ``Phase 3.1
complete`` and maps them to phase IDs. Phases whose prerequisites do not match
any such pattern are flagged as ``prereqs unknown (manual check required)``
rather than being optimistically marked eligible.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Phase-id helpers


def normalize_phase_id(value: str) -> str:
    """Normalize a user-provided phase ID.

    Accept ``phase-03``, ``3.1``, and ``8A``; canonical forms are:
      - ``"3"`` for major-only phases
      - ``"3.1"`` for decimal sub-phases
      - ``"8A"`` for letter-suffixed parallel phases
    """
    token = value.strip().lower()
    token = re.sub(r"^phase[\s_-]*", "", token)
    token = token.replace("_", ".").replace(" ", "")
    if not token:
        raise ValueError("phase id cannot be empty")

    letter_match = re.fullmatch(r"0*(\d+)([a-z])", token)
    if letter_match:
        return f"{int(letter_match.group(1))}{letter_match.group(2).upper()}"

    numeric_match = re.fullmatch(r"0*(\d+)(?:\.0*(\d+))?", token)
    if numeric_match:
        major = str(int(numeric_match.group(1)))
        minor = numeric_match.group(2)
        if minor is None:
            return major
        return f"{major}.{int(minor)}"

    raise ValueError(f"unsupported phase id: {value!r}")


def phase_id_from_prefix(prefix: str) -> str:
    """Return the canonical phase ID from a filename prefix.

    Accept ``XX``, ``XX_Y``, and ``XXa`` prefixes.
    """
    lower = prefix.lower()
    if lower and lower[-1].isalpha():
        return f"{int(lower[:-1])}{lower[-1].upper()}"
    if "_" in lower:
        major, minor = lower.split("_", 1)
        return f"{int(major)}.{int(minor)}"
    return str(int(lower))


def phase_sort_key(phase_id: str) -> tuple[int, int, int, str]:
    """Sort canonical phase ids deterministically.

    Order: major phases < decimal sub-phases within a major < lettered
    parallel phases within a major.
    """
    if phase_id and phase_id[-1].isalpha():
        return (int(phase_id[:-1]), 2, 0, phase_id[-1])
    if "." in phase_id:
        major, minor = phase_id.split(".", 1)
        return (int(major), 1, int(minor), "")
    return (int(phase_id), 0, 0, "")


# ---------------------------------------------------------------------------
# Frontmatter + prerequisite parsing


_FRONTMATTER_RE = re.compile(r"\A---\n(?P<body>.*?)\n---\n?", re.S)
_KEY_VALUE_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_-]*)\s*:\s*(.+?)\s*$")
_PHASE_NUMBER_RE = re.compile(
    r"Phase\s+(?P<id>[0-9]+(?:\.[0-9]+)?|[0-9]+[A-Za-z])",
    re.I,
)
_COMPLETE_MENTION_RE = re.compile(
    r"Phase\s+(?P<id>[0-9]+(?:\.[0-9]+)?|[0-9]+[A-Za-z])" r"\s+(?:is\s+)?complete",
    re.I,
)
_MIN_QUOTED_VALUE_LENGTH = 2
_REPO_ROOT_PARENT_DEPTH = 2


def parse_frontmatter(text: str) -> dict[str, str]:
    """Parse the YAML-ish frontmatter block at the top of a Markdown file.

    We do not depend on a YAML library -- frontmatter in this repo is always
    simple ``key: value`` lines. Quoted values are unquoted.
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}

    data: dict[str, str] = {}
    for line in match.group("body").splitlines():
        kv = _KEY_VALUE_RE.match(line)
        if not kv:
            continue
        value = kv.group(2).strip()
        if (
            len(value) >= _MIN_QUOTED_VALUE_LENGTH
            and value[0] == value[-1]
            and value[0] in ("'", '"')
        ):
            value = value[1:-1]
        data[kv.group(1)] = value
    return data


def status_is_complete(status: str | None) -> bool:
    return status is not None and status.strip().lower().startswith("complete")


def status_is_not_started(status: str | None) -> bool:
    if status is None:
        return False
    return status.strip().lower().startswith("not started")


def extract_phase_title(text: str, default: str) -> str:
    match = re.search(r"^#\s+Phase\s+[^\n:]+:\s+(?P<title>.+)$", text, re.M)
    if match:
        return match.group("title").strip()
    return default


def extract_prerequisite_section(text: str) -> str:
    """Return the body of the ``## Prerequisites`` block.

    Return ``""`` if the section is missing.
    """
    match = re.search(
        r"^## Prerequisites\s*\n(?P<body>.*?)(?=^## |\Z)",
        text,
        re.M | re.S,
    )
    if not match:
        return ""
    return match.group("body")


def extract_prereq_phase_ids(prereq_text: str) -> tuple[tuple[str, ...], bool]:
    """Extract machine-resolvable phase-ID prerequisites from prose.

    Returns ``(ids, machine_resolvable)`` where ``machine_resolvable`` is:
      - ``True`` if the section is empty, or mentions no phases, or mentions
        phases with explicit ``complete`` wording for every reference.
      - ``False`` if the prose references phases but does not use the
        ``Phase N complete`` phrasing we understand, so a human should check.

    We're intentionally strict: ambiguous prose is flagged rather than
    optimistically resolved.
    """
    stripped = prereq_text.strip()
    if not stripped:
        return (tuple(), True)

    raw_mentions = list(_PHASE_NUMBER_RE.finditer(stripped))
    complete_mentions = list(_COMPLETE_MENTION_RE.finditer(stripped))

    if not raw_mentions:
        # Prose has no phase references -- treat as no phase-level prereqs.
        return (tuple(), True)

    if len(complete_mentions) < len(raw_mentions):
        # Some phase references don't have matching "complete" wording.
        # Return what we found but flag as not machine-resolvable.
        ids = tuple(
            sorted({normalize_phase_id(m.group("id")) for m in complete_mentions})
        )
        return (ids, False)

    ids = tuple(
        sorted(
            {normalize_phase_id(m.group("id")) for m in complete_mentions},
            key=phase_sort_key,
        )
    )
    return (ids, True)


# ---------------------------------------------------------------------------
# Phase records


@dataclass(frozen=True)
class PhaseRecord:
    """A single phase file, parsed and normalized."""

    phase_id: str
    title: str
    prefix: str
    phase_file: Path
    status: str | None
    prerequisite_ids: tuple[str, ...]
    prereqs_machine_resolvable: bool
    learnings_paths: tuple[Path, ...] = field(default_factory=tuple)


def load_phase_records(project_dir: Path) -> dict[str, PhaseRecord]:
    phases_dir = project_dir / "phases"
    if not phases_dir.is_dir():
        return {}

    learnings_dir = project_dir / "learnings"

    records: dict[str, PhaseRecord] = {}
    for phase_file in sorted(phases_dir.glob("phase-*.md")):
        # Skip the template file used as a reference artifact.
        if phase_file.name.startswith("_"):
            continue
        prefix_match = re.match(r"phase-(?P<prefix>[^-]+)-", phase_file.name)
        if not prefix_match:
            continue
        prefix = prefix_match.group("prefix")

        text = phase_file.read_text()
        frontmatter = parse_frontmatter(text)
        try:
            phase_id = phase_id_from_prefix(prefix)
        except ValueError:
            continue
        # Frontmatter `phase:` wins if present and consistent.
        if frontmatter.get("phase"):
            try:
                phase_id = normalize_phase_id(frontmatter["phase"])
            except ValueError:
                pass
        title = frontmatter.get("title") or extract_phase_title(text, phase_id)
        status = frontmatter.get("status")

        prereq_section = extract_prerequisite_section(text)
        prereq_ids, machine_resolvable = extract_prereq_phase_ids(prereq_section)

        learnings_paths: tuple[Path, ...] = tuple()
        if learnings_dir.is_dir():
            pattern = f"phase-{prefix.lower()}-*.md"
            learnings_paths = tuple(
                sorted(
                    p for p in learnings_dir.glob(pattern) if not p.name.startswith("_")
                )
            )

        records[phase_id] = PhaseRecord(
            phase_id=phase_id,
            title=title,
            prefix=prefix,
            phase_file=phase_file,
            status=status,
            prerequisite_ids=prereq_ids,
            prereqs_machine_resolvable=machine_resolvable,
            learnings_paths=learnings_paths,
        )
    return records


# ---------------------------------------------------------------------------
# Eligibility resolution


def relpath(repo_root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def phase_closeout_issues(repo_root: Path, record: PhaseRecord) -> list[str]:
    """Return reasons a phase is not fully closed out.

    An empty list allows downstream phases to proceed.
    """
    issues: list[str] = []
    if not status_is_complete(record.status):
        issues.append(
            f"{relpath(repo_root, record.phase_file)} frontmatter "
            f"status is {record.status or 'unknown'}, not Complete"
        )
    if not record.learnings_paths:
        issues.append(
            f"missing learnings document matching "
            f"{relpath(repo_root, record.phase_file.parent.parent)}/"
            f"learnings/phase-{record.prefix.lower()}-*.md"
        )
    return issues


def prerequisite_blockers(
    repo_root: Path,
    phases: dict[str, PhaseRecord],
    phase_id: str,
) -> tuple[list[dict[str, object]], list[str]]:
    """Return ``(blockers, unknown_prereqs)`` for the given phase.

    - ``blockers`` -- prereq phases that are known and not yet closed out.
    - ``unknown_prereqs`` -- raw prereq ids referenced but not present as
      phase files in the project.
    """
    record = phases[phase_id]
    blockers: list[dict[str, object]] = []
    unknown: list[str] = []
    for prereq_id in sorted(record.prerequisite_ids, key=phase_sort_key):
        prereq = phases.get(prereq_id)
        if prereq is None:
            unknown.append(prereq_id)
            continue
        issues = phase_closeout_issues(repo_root, prereq)
        if issues:
            blockers.append(
                {
                    "phase_id": prereq.phase_id,
                    "title": prereq.title,
                    "issues": issues,
                }
            )
    return blockers, unknown


def eligible_frontier(repo_root: Path, phases: dict[str, PhaseRecord]) -> list[str]:
    """Return eligible not-started phases.

    Every prerequisite must be closed out and machine-resolvable.
    """
    frontier: list[str] = []
    for phase_id, record in sorted(
        phases.items(), key=lambda item: phase_sort_key(item[0])
    ):
        if not status_is_not_started(record.status):
            continue
        if not record.prereqs_machine_resolvable:
            # Don't silently promote prereqs-unknown phases to the frontier.
            continue
        blockers, unknown = prerequisite_blockers(repo_root, phases, phase_id)
        if blockers or unknown:
            continue
        frontier.append(phase_id)
    return frontier


def blocked_frontier_candidates(
    repo_root: Path, phases: dict[str, PhaseRecord]
) -> list[dict[str, object]]:
    """Return not-started phases that are blocked.

    A phase makes the blocked-candidate list when its own status is
    ``Not started`` but one or more of the following is true:
      - its prerequisites are not machine-resolvable (prose doesn't match
        the ``Phase N complete`` pattern); or
      - at least one prerequisite is incomplete; or
      - at least one prerequisite references a phase that has no file in
        the project.
    """
    candidates: list[dict[str, object]] = []
    for phase_id, record in sorted(
        phases.items(), key=lambda item: phase_sort_key(item[0])
    ):
        if not status_is_not_started(record.status):
            continue

        issues: list[str] = []
        if not record.prereqs_machine_resolvable:
            issues.append(
                "prose in `## Prerequisites` does not match the "
                "`Phase N complete` pattern; manual check required"
            )
            # Continue to also list any prereqs we *did* recognize so
            # the operator sees concrete context.

        blockers, unknown = prerequisite_blockers(repo_root, phases, phase_id)
        for prereq_id in unknown:
            issues.append(
                f"Phase {prereq_id} referenced as a prerequisite but "
                "no matching phase file exists in the project"
            )
        for blocker in blockers:
            prereq_label = f"Phase {blocker['phase_id']}"
            for issue in blocker["issues"]:  # type: ignore[index]
                issues.append(f"{prereq_label}: {issue}")

        if not issues:
            # No machine-visible blockers. It should have been on the
            # eligible frontier; skip.
            continue

        candidates.append(
            {
                "phase_id": record.phase_id,
                "title": record.title,
                "phase_file": relpath(repo_root, record.phase_file),
                "issues": issues,
            }
        )
    return candidates


def phase_summary(
    repo_root: Path,
    phases: dict[str, PhaseRecord],
    phase_id: str,
) -> dict[str, object]:
    record = phases[phase_id]
    payload: dict[str, object] = {
        "phase_id": record.phase_id,
        "title": record.title,
        "status": record.status,
        "phase_file": relpath(repo_root, record.phase_file),
        "prerequisites": list(record.prerequisite_ids),
        "prereqs_machine_resolvable": record.prereqs_machine_resolvable,
    }
    prereqs = list(record.prerequisite_ids)
    if not prereqs:
        payload["reason"] = "No phase-level prerequisites."
    else:
        labels = [f"Phase {p}" for p in prereqs]
        payload["reason"] = f"Prerequisites satisfied: {', '.join(labels)}."
    return payload


def build_prompt(  # noqa: PLR0913, PLR0917
    project_slug: str,
    project_dir: Path,
    repo_root: Path,
    phases: dict[str, PhaseRecord],
    phase_id: str,
    agent: str = "claude",
) -> str:
    record = phases[phase_id]

    # `phase_id` is the *canonical display* id ("2"); `record.prefix` is the
    # filename prefix ("02"). Working-memory directories are named after the
    # filename prefix -- `projects/<slug>/task-notes/phase-02/` -- matching
    # both the template and `scripts/agents/resolve_task.py`, which derives
    # the directory from the phase filename. Interpolating `phase_id` into a
    # path yields `phase-2/` and sends the next agent to a directory that
    # does not exist, so every filesystem path below MUST use `dir_prefix`.
    # Display text keeps `phase_id`.
    dir_prefix = record.prefix.lower()

    prereq_learnings: list[str] = []
    for prereq_id in sorted(record.prerequisite_ids, key=phase_sort_key):
        prereq = phases.get(prereq_id)
        if prereq is None:
            continue
        for learning_path in prereq.learnings_paths:
            prereq_learnings.append(relpath(repo_root, learning_path))
    learning_block = "\n".join(f"- `{p}`" for p in prereq_learnings)
    if not learning_block:
        learning_block = "- None yet."

    plan_path = relpath(repo_root, project_dir / "PLAN.md")
    rules_path = relpath(repo_root, project_dir / "rules.md")
    phase_path = relpath(repo_root, record.phase_file)
    recheck = (
        f"python3 scripts/agents/resolve_phase.py --project "
        f"{project_slug} --agent {agent} --target-phase {phase_id}"
    )

    return (
        f"Use the `$execute-single-task` skill.\n\n"
        f"You are starting Phase {phase_id}: {record.title} for project "
        f"`{project_slug}` (root `{repo_root}`).\n\n"
        "Before editing:\n"
        f"1. Re-run `{recheck}` and refuse to proceed unless the result "
        f"is still `ready`.\n"
        "2. Create your worktree as Step 3 of `$execute-single-task` "
        f"instructs -- path `.{agent}/worktrees/{project_slug}/"
        f"<task-slug>/`, branch `{agent}/{project_slug}/<task-slug>`.\n\n"
        "Scope:\n"
        f"- Work only within Phase {phase_id}.\n"
        "- Do not start sibling or downstream phases.\n"
        f"- Start with the next lowest-numbered unfinished task in "
        f"Phase {phase_id} -- read it from the live Tasks table in "
        f"`projects/{project_slug}/task-notes/phase-"
        f"{dir_prefix}/README.md` (not `PLAN.md` or the phase file).\n\n"
        "Read only:\n"
        f"1. `{plan_path}`\n"
        f"2. `projects/{project_slug}/task-notes/README.md` "
        "(project-level working memory)\n"
        f"3. `{rules_path}` (optional -- skip if missing)\n"
        f"4. `{phase_path}`\n"
        f"5. `projects/{project_slug}/task-notes/phase-"
        f"{dir_prefix}/README.md` (per-phase working memory -- the live "
        f"Tasks status table for Phase {phase_id})\n"
        f"6. The files listed under `## Prerequisites` in `{phase_path}`\n"
        "7. Relevant upstream learnings already closed out:\n"
        f"{learning_block}\n"
        "8. Active ADRs touching the same subsystem (project-scoped "
        f"under `projects/{project_slug}/adrs/` and repo-wide under "
        "`docs/adrs/`)\n"
        f"9. The existing task note for the first unfinished Phase "
        f"{phase_id} task, if one exists\n\n"
        "Guardrails:\n"
        "- Avoid reading unrelated phase files.\n"
        "- Follow `$execute-single-task` to keep scope to one task or "
        "tightly related task cluster.\n"
        "- If prerequisite closeout or repo state has drifted since this "
        "prompt was generated, stop and report it instead of beginning "
        "the phase.\n"
    )


# ---------------------------------------------------------------------------
# Top-level resolution


def resolve_project_dir(
    repo_root: Path, project_arg: str | None, project_dir_arg: Path | None
) -> Path:
    if project_dir_arg is not None:
        return project_dir_arg.resolve()
    if project_arg is not None:
        return (repo_root / "projects" / project_arg).resolve()
    raise ValueError("one of --project <slug> or --project-dir <path> is required")


def resolve_phase(  # noqa: PLR0911, PLR0912, PLR0913, PLR0917
    repo_root: Path,
    project_slug: str,
    project_dir: Path,
    completed_phase: str | None = None,
    target_phase: str | None = None,
    agent: str = "claude",
) -> dict[str, object]:
    if agent not in {"claude", "codex"}:
        raise ValueError(f"unsupported agent: {agent!r}")
    plan_path = project_dir / "PLAN.md"
    if not plan_path.exists():
        raise ValueError(
            f"project PLAN not found at " f"{relpath(repo_root, plan_path)}"
        )
    plan_frontmatter = parse_frontmatter(plan_path.read_text())
    phased_flag = plan_frontmatter.get("phased", "").strip().lower()
    if phased_flag != "true":
        raise ValueError(
            f"project `{project_slug}` is not phased "
            f"(PLAN.md frontmatter `phased: {plan_frontmatter.get('phased') or 'missing'}`). "
            "begin-phase only works on phased projects."
        )

    phases = load_phase_records(project_dir)
    if not phases:
        raise ValueError(
            f"project `{project_slug}` has no phase files under "
            f"{relpath(repo_root, project_dir / 'phases')}"
        )

    notes: list[str] = []

    # Flag any phases whose prereqs couldn't be machine-resolved so the
    # operator sees them in the output.
    prereq_unknown = [
        phase_id
        for phase_id, record in phases.items()
        if not record.prereqs_machine_resolvable
    ]
    for phase_id in sorted(prereq_unknown, key=phase_sort_key):
        notes.append(
            f"Phase {phase_id} prerequisites are not machine-resolvable "
            "(prose did not match `Phase N complete`). Manual check "
            "required before starting that phase."
        )

    frontier = eligible_frontier(repo_root, phases)
    frontier_payload = [
        phase_summary(repo_root, phases, phase_id) for phase_id in frontier
    ]

    normalized_completed = (
        normalize_phase_id(completed_phase) if completed_phase else None
    )
    normalized_target = normalize_phase_id(target_phase) if target_phase else None
    if normalized_completed and normalized_completed not in phases:
        raise ValueError(f"unknown completed phase: {completed_phase}")
    if normalized_target and normalized_target not in phases:
        raise ValueError(f"unknown target phase: {target_phase}")

    # --- Explicit target -----------------------------------------------------
    if normalized_target:
        record = phases[normalized_target]
        if status_is_complete(record.status):
            return {
                "status": "blocked",
                "project": project_slug,
                "message": f"Phase {normalized_target} is already complete.",
                "eligible_frontier": frontier_payload,
                "blockers": [
                    {
                        "phase_id": normalized_target,
                        "title": record.title,
                        "issues": [
                            f"{relpath(repo_root, record.phase_file)} "
                            "frontmatter status is Complete"
                        ],
                    }
                ],
                "notes": notes,
                "prompt": None,
            }
        if not status_is_not_started(record.status):
            return {
                "status": "blocked",
                "project": project_slug,
                "message": (
                    f"Phase {normalized_target} is not in the " "`Not started` state."
                ),
                "eligible_frontier": frontier_payload,
                "blockers": [
                    {
                        "phase_id": normalized_target,
                        "title": record.title,
                        "issues": [
                            f"{relpath(repo_root, record.phase_file)} "
                            f"frontmatter status is "
                            f"{record.status or 'unknown'}"
                        ],
                    }
                ],
                "notes": notes,
                "prompt": None,
            }
        if not record.prereqs_machine_resolvable:
            return {
                "status": "blocked",
                "project": project_slug,
                "message": (
                    f"Phase {normalized_target} prerequisites are not "
                    "machine-resolvable; cannot verify readiness."
                ),
                "eligible_frontier": frontier_payload,
                "blockers": [
                    {
                        "phase_id": normalized_target,
                        "title": record.title,
                        "issues": [
                            "prose in `## Prerequisites` does not match the "
                            "`Phase N complete` pattern; manual check "
                            "required"
                        ],
                    }
                ],
                "notes": notes,
                "prompt": None,
            }

        blockers, unknown = prerequisite_blockers(repo_root, phases, normalized_target)
        for prereq_id in unknown:
            blockers.append(
                {
                    "phase_id": prereq_id,
                    "title": "(unknown -- no phase file found)",
                    "issues": [
                        f"Phase {prereq_id} referenced as a prerequisite "
                        "but no matching phase file exists in the project"
                    ],
                }
            )
        if blockers:
            return {
                "status": "blocked",
                "project": project_slug,
                "message": (
                    f"Phase {normalized_target} is blocked by incomplete "
                    "prerequisite closeout."
                ),
                "eligible_frontier": frontier_payload,
                "blockers": blockers,
                "notes": notes,
                "prompt": None,
            }

        return {
            "status": "ready",
            "project": project_slug,
            "message": f"Phase {normalized_target} is eligible to begin.",
            "eligible_frontier": frontier_payload,
            "phase": phase_summary(repo_root, phases, normalized_target),
            "notes": notes,
            "prompt": build_prompt(
                project_slug,
                project_dir,
                repo_root,
                phases,
                normalized_target,
                agent,
            ),
        }

    # --- No explicit target; inspect the current frontier --------------------
    candidates = frontier[:]
    if normalized_completed:
        # Prefer phases this newly-completed phase unlocks.
        direct_unlocks = [
            phase_id
            for phase_id in frontier
            if normalized_completed in phases[phase_id].prerequisite_ids
        ]
        if direct_unlocks:
            candidates = direct_unlocks
        else:
            notes.append(
                f"Phase {normalized_completed} does not directly unlock a "
                "new phase by itself. Showing the current eligible "
                "frontier instead."
            )

    if not candidates:
        blockers = blocked_frontier_candidates(repo_root, phases)
        if not blockers:
            # No "Not started" phase at all -- every phase is either
            # complete or in progress. Surface that explicitly so the
            # caller doesn't get an empty list with no context.
            in_progress_or_complete = [
                {
                    "phase_id": record.phase_id,
                    "title": record.title,
                    "issues": [
                        f"{relpath(repo_root, record.phase_file)} "
                        f"frontmatter status is "
                        f"{record.status or 'unknown'}; no phase is in "
                        "the `Not started` state"
                    ],
                }
                for _, record in sorted(
                    phases.items(),
                    key=lambda item: phase_sort_key(item[0]),
                )
            ]
            blockers = [
                {
                    "phase_id": "(none)",
                    "title": "No phase is `Not started`",
                    "issues": [
                        "Every phase file is either Complete or In "
                        "Progress; there is no next phase to begin. "
                        "Manual check required to confirm the project "
                        "status or whether a new phase file is needed.",
                    ],
                    "phases": in_progress_or_complete,
                }
            ]
        return {
            "status": "blocked",
            "project": project_slug,
            "message": "No project phase is currently eligible to begin.",
            "eligible_frontier": frontier_payload,
            "blockers": blockers,
            "notes": notes,
            "prompt": None,
        }

    if len(candidates) > 1:
        return {
            "status": "choose",
            "project": project_slug,
            "message": (
                "Multiple phases are eligible. Ask the user to choose one "
                "before generating a kickoff prompt."
            ),
            "eligible_frontier": frontier_payload,
            "choices": [
                phase_summary(repo_root, phases, phase_id) for phase_id in candidates
            ],
            "notes": notes,
            "prompt": None,
        }

    phase_id = candidates[0]
    return {
        "status": "ready",
        "project": project_slug,
        "message": f"Phase {phase_id} is eligible to begin.",
        "eligible_frontier": frontier_payload,
        "phase": phase_summary(repo_root, phases, phase_id),
        "notes": notes,
        "prompt": build_prompt(
            project_slug, project_dir, repo_root, phases, phase_id, agent
        ),
    }


# ---------------------------------------------------------------------------
# CLI


def default_repo_root() -> Path:
    """Walk up from the script looking for a checkout root.

    This script lives at ``<repo>/scripts/agents/resolve_phase.py``, so the
    repo root is two ``parents`` up. If that doesn't look like a checkout (no
    ``projects/`` and no ``.git``), fall back to CWD.
    """
    here = Path(__file__).resolve()
    candidate = (
        here.parents[_REPO_ROOT_PARENT_DEPTH]
        if len(here.parents) > _REPO_ROOT_PARENT_DEPTH
        else here.parent
    )
    if (candidate / "projects").is_dir() or (candidate / ".git").exists():
        return candidate
    return Path.cwd()


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve which phase of a phased project is eligible to "
            "start next, and emit a kickoff-prompt payload as JSON."
        )
    )
    parser.add_argument(
        "--project",
        help=(
            "Project slug under `projects/`. Example: "
            "--project overhaul-test-infra. Either --project or "
            "--project-dir is required."
        ),
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        help=(
            "Absolute or repo-relative path to the project directory. "
            "Overrides --project."
        ),
    )
    parser.add_argument(
        "--completed-phase",
        help=(
            "Phase that just completed. Example: --completed-phase 2. "
            "When set, the helper prefers downstream phases this one "
            "unlocks if any are now eligible."
        ),
    )
    parser.add_argument(
        "--target-phase",
        help=(
            "Ask about a specific phase. Example: --target-phase 3.1. "
            "The helper returns `ready`, `blocked`, or an error."
        ),
    )
    parser.add_argument(
        "--agent",
        choices=("claude", "codex"),
        default="claude",
        help=(
            "Agent identity to embed in the generated kickoff prompt. "
            "Defaults to claude for backward compatibility."
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root. Defaults to the containing checkout.",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    if args.project is None and args.project_dir is None:
        sys.stderr.write("error: one of --project or --project-dir is required\n")
        return 2

    repo_root = (args.repo_root or default_repo_root()).resolve()

    try:
        project_dir = resolve_project_dir(repo_root, args.project, args.project_dir)
        # Infer the slug if only --project-dir was supplied.
        project_slug = args.project or project_dir.name
        result = resolve_phase(
            repo_root=repo_root,
            project_slug=project_slug,
            project_dir=project_dir,
            completed_phase=args.completed_phase,
            target_phase=args.target_phase,
            agent=args.agent,
        )
    except (OSError, ValueError) as exc:
        payload = {
            "status": "error",
            "project": args.project,
            "message": str(exc),
        }
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 1

    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
