#!/usr/bin/env bash
#
# setup_task_worktree.sh — create a fresh task worktree branched off the
# trunk branch for an agent-driven task.
#
# This is the deterministic worktree-creation step for the task-pipeline
# and execute-single-task skills. It always fetches origin, always
# branches from origin/<trunk> (never from the ambient HEAD, which may be a
# stale or unrelated checkout), picks a collision-free branch/worktree
# name, and verifies the new worktree's HEAD matches origin/<trunk> before
# reporting success.
#
# Usage:
#   scripts/agents/setup_task_worktree.sh \
#     --project <slug> --task-slug <slug> [--agent claude|codex] \
#     [--trunk <branch>]
#
#   --project     project slug (e.g. lsp-inlay-hints)
#   --task-slug   task slug (e.g. task-3-refresh)
#   --agent       claude (default) or codex; selects the branch prefix
#                 and the .{claude,codex}/worktrees/ base directory
#   --trunk       trunk branch to branch from; defaults to origin's
#                 default branch, falling back to 'master'
#
# On success prints a single line of JSON to stdout:
#   {"branch":"…","wt_path":"<absolute>","head_sha":"…"}
#
# All diagnostics go to stderr; a non-zero exit indicates failure and no
# JSON is printed.

set -uo pipefail

die() {
    echo "error: $*" >&2
    exit 1
}

# --- Argument parsing -------------------------------------------------

PROJECT=""
TASK_SLUG=""
AGENT="claude"
TRUNK=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --project)
            [[ $# -ge 2 ]] || die "--project requires a value"
            PROJECT="$2"
            shift 2
            ;;
        --task-slug)
            [[ $# -ge 2 ]] || die "--task-slug requires a value"
            TASK_SLUG="$2"
            shift 2
            ;;
        --agent)
            [[ $# -ge 2 ]] || die "--agent requires a value"
            AGENT="$2"
            shift 2
            ;;
        --trunk)
            [[ $# -ge 2 ]] || die "--trunk requires a value"
            TRUNK="$2"
            shift 2
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done

[[ -n "$PROJECT" ]] || die "--project is required"
[[ -n "$TASK_SLUG" ]] || die "--task-slug is required"

case "$AGENT" in
    claude|codex) ;;
    *) die "--agent must be 'claude' or 'codex' (got: $AGENT)" ;;
esac

# --- Repo root --------------------------------------------------------

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" \
    || die "not inside a git repository"

# --- Fetch ------------------------------------------------------------

git -C "$REPO_ROOT" fetch origin \
    || die "git fetch origin failed"

# --- Trunk resolution -------------------------------------------------

if [[ -z "$TRUNK" ]]; then
    # origin/HEAD -> refs/remotes/origin/<trunk>, when the symref exists.
    TRUNK="$(git -C "$REPO_ROOT" symbolic-ref --quiet --short \
        refs/remotes/origin/HEAD 2>/dev/null | sed 's#^origin/##')"
fi
[[ -n "$TRUNK" ]] || TRUNK="master"

git -C "$REPO_ROOT" rev-parse --verify --quiet "origin/${TRUNK}" >/dev/null \
    || die "origin/${TRUNK} not found after fetch"

# --- Collision-free branch / worktree name ----------------------------

# Does a branch (local ref or remote ref) already exist?
branch_exists() {
    local branch="$1"
    if git -C "$REPO_ROOT" show-ref --verify --quiet \
        "refs/heads/${branch}"; then
        return 0
    fi
    if git -C "$REPO_ROOT" ls-remote --exit-code --heads origin \
        "$branch" >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

BASE_BRANCH="${AGENT}/${PROJECT}/${TASK_SLUG}"
WT_BASE="${REPO_ROOT}/.${AGENT}/worktrees/${PROJECT}"

FINAL_SLUG="$TASK_SLUG"
BRANCH="$BASE_BRANCH"
WT_PATH="${WT_BASE}/${FINAL_SLUG}"

# The initial candidate has no numeric suffix; collisions try -2..-9.
if branch_exists "$BRANCH" || [[ -e "$WT_PATH" ]]; then
    found=""
    for n in 2 3 4 5 6 7 8 9; do
        FINAL_SLUG="${TASK_SLUG}-${n}"
        BRANCH="${AGENT}/${PROJECT}/${FINAL_SLUG}"
        WT_PATH="${WT_BASE}/${FINAL_SLUG}"
        if ! branch_exists "$BRANCH" && [[ ! -e "$WT_PATH" ]]; then
            found="yes"
            break
        fi
    done
    [[ -n "$found" ]] \
        || die "no free branch/worktree name for ${BASE_BRANCH} (-2..-9 all taken)"
fi

# --- Create the worktree ----------------------------------------------

mkdir -p "$WT_BASE" || die "could not create ${WT_BASE}"

# Redirect git's progress/tracking chatter to stderr so stdout carries
# only the final JSON line.
git -C "$REPO_ROOT" worktree add "$WT_PATH" -b "$BRANCH" "origin/${TRUNK}" 1>&2 \
    || die "git worktree add failed for ${WT_PATH}"

# Absolute, normalized worktree path.
WT_PATH_ABS="$(cd "$WT_PATH" && pwd)" \
    || die "created worktree ${WT_PATH} is not accessible"

# --- Verify HEAD == origin/<trunk> ------------------------------------

HEAD_SHA="$(git -C "$WT_PATH_ABS" rev-parse HEAD)" \
    || die "could not read HEAD of ${WT_PATH_ABS}"
TRUNK_SHA="$(git -C "$REPO_ROOT" rev-parse "origin/${TRUNK}")" \
    || die "could not read origin/${TRUNK}"

if [[ "$HEAD_SHA" != "$TRUNK_SHA" ]]; then
    die "worktree HEAD (${HEAD_SHA}) != origin/${TRUNK} (${TRUNK_SHA})"
fi

# --- Report -----------------------------------------------------------

printf '{"branch":"%s","wt_path":"%s","head_sha":"%s"}\n' \
    "$BRANCH" "$WT_PATH_ABS" "$HEAD_SHA"
