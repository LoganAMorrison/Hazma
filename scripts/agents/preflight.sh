#!/usr/bin/env bash
#
# preflight.sh — the one-command pre-commit / pre-PR gate for hazma.
#
# This is the executable form of docs/agents/preflight.md. It runs the
# mandatory gates in order, prints one PASS / FAIL / WARN / SKIP row per
# gate, and exits non-zero if any first-class gate FAILs. WARN rows are
# advisory (they never fail the run); SKIP rows mean the gate did not
# apply to this invocation.
#
# Usage:
#   scripts/agents/preflight.sh [--paths "hazma/spectra test/spectra"] \
#       [--tests "test/spectra"] [--md "a.md b.md"] [--closing]
#
#   --paths "a b"    Space-separated files/dirs your diff touched. The
#                    formatters and linters run against these. Defaults
#                    to `hazma test` when omitted.
#   --tests "a b"    Space-separated pytest targets. Defaults to `test`.
#                    A run that collects ZERO tests FAILs the gate — a
#                    green-but-nothing-ran run is a false pass.
#   --md "a.md"      Also run `markdownlint --dot` on the given files
#                    (pass the curated docs your diff changed).
#   --closing        Also run the version-bump gate; pass this on a PR
#                    that flips a project PLAN status to Complete.
#
# Gates, in order: black --check, isort --check-only, ruff check, pytest,
# import smoke, markdownlint (with --md), version bump (with --closing),
# and a forbidden-token scan of the diff against the trunk (WARN only).
#
# Design notes:
#   - `set -uo pipefail` (NOT `-e`): every gate runs so the table is
#     complete; the exit code is derived from the collected results.
#   - Never pipe a gate through head/tail/grep in a way that masks its
#     exit code — each gate's output is captured to a temp file and the
#     bare exit status is read.
#   - The repo root is anchored to this script's own location so the gate
#     runs against the worktree that contains it, from any cwd.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}" || {
    echo "error: cannot cd to repo root ${REPO_ROOT}" >&2
    exit 2
}

# --------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------

PATHS=""
TESTS=""
MD_FILES=""
CLOSING=0

usage() {
    sed -n '3,32p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --paths)
            [[ $# -ge 2 ]] || { echo "error: --paths needs a value" >&2; exit 2; }
            PATHS="$2"; shift 2 ;;
        --tests)
            [[ $# -ge 2 ]] || { echo "error: --tests needs a value" >&2; exit 2; }
            TESTS="$2"; shift 2 ;;
        --md)
            [[ $# -ge 2 ]] || { echo "error: --md needs a value" >&2; exit 2; }
            MD_FILES="$2"; shift 2 ;;
        --closing)
            CLOSING=1; shift ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "${PATHS}" ]] || PATHS="hazma test"
[[ -n "${TESTS}" ]] || TESTS="test"

# --------------------------------------------------------------------------
# Result table
# --------------------------------------------------------------------------

ROWS=()
HARD_FAIL=0
TMPDIR_PF="$(mktemp -d)"
trap 'rm -rf "${TMPDIR_PF}"' EXIT

row() {
    # row <status> <gate> <detail>
    ROWS+=("$(printf '%-5s  %-22s  %s' "$1" "$2" "$3")")
    [[ "$1" == "FAIL" ]] && HARD_FAIL=1
    return 0
}

have() { command -v "$1" >/dev/null 2>&1; }

# Run a command, capture combined output, return its bare exit status.
# Usage: capture <outfile> <cmd...>
capture() {
    local out="$1"; shift
    "$@" >"${out}" 2>&1
    return $?
}

tail_of() { tail -n 4 "$1" | sed 's/^/        /'; }

# --------------------------------------------------------------------------
# Trunk resolution (for diff-scoped gates)
# --------------------------------------------------------------------------

TRUNK="$(git symbolic-ref --quiet --short refs/remotes/origin/HEAD 2>/dev/null \
    | sed 's#^origin/##')"
[[ -n "${TRUNK}" ]] || TRUNK="master"
BASE_REF="origin/${TRUNK}"
git rev-parse --verify --quiet "${BASE_REF}" >/dev/null 2>&1 || BASE_REF="${TRUNK}"

# --------------------------------------------------------------------------
# Gate 1: black --check
# --------------------------------------------------------------------------

if have black; then
    OUT="${TMPDIR_PF}/black.log"
    # shellcheck disable=SC2086
    capture "${OUT}" black --check ${PATHS}
    if [[ $? -eq 0 ]]; then
        row PASS "black --check" "${PATHS}"
    else
        row FAIL "black --check" "run \`black ${PATHS}\` and re-check"
        tail_of "${OUT}"
    fi
else
    row FAIL "black --check" "black not installed (pip install black)"
fi

# --------------------------------------------------------------------------
# Gate 2: isort --check-only
# --------------------------------------------------------------------------

if have isort; then
    OUT="${TMPDIR_PF}/isort.log"
    # shellcheck disable=SC2086
    capture "${OUT}" isort --check-only ${PATHS}
    if [[ $? -eq 0 ]]; then
        row PASS "isort --check-only" "${PATHS}"
    else
        row FAIL "isort --check-only" "run \`isort ${PATHS}\` and re-check"
        tail_of "${OUT}"
    fi
else
    row WARN "isort --check-only" "isort not installed — import order unchecked"
fi

# --------------------------------------------------------------------------
# Gate 3: ruff check
# --------------------------------------------------------------------------

if have ruff; then
    OUT="${TMPDIR_PF}/ruff.log"
    # shellcheck disable=SC2086
    capture "${OUT}" ruff check ${PATHS}
    if [[ $? -eq 0 ]]; then
        row PASS "ruff check" "${PATHS}"
    else
        row FAIL "ruff check" "see output below"
        tail_of "${OUT}"
    fi
else
    row WARN "ruff check" "ruff not installed — lint unchecked"
fi

# --------------------------------------------------------------------------
# Gate 4: pytest (with the zero-collection guard)
# --------------------------------------------------------------------------

if have pytest; then
    OUT="${TMPDIR_PF}/pytest.log"
    # shellcheck disable=SC2086
    capture "${OUT}" pytest -q ${TESTS}
    STATUS=$?
    SUMMARY="$(grep -E '^(=+ )?[0-9]+ (passed|failed|error)' "${OUT}" | tail -n 1)"
    [[ -n "${SUMMARY}" ]] || SUMMARY="$(tail -n 1 "${OUT}")"
    if [[ ${STATUS} -eq 5 ]]; then
        # pytest exit 5 == no tests collected. Green-looking, ran nothing.
        row FAIL "pytest" "ZERO tests collected for '${TESTS}' — false green"
    elif [[ ${STATUS} -eq 0 ]]; then
        row PASS "pytest" "${SUMMARY}"
    else
        row FAIL "pytest" "${SUMMARY}"
        tail_of "${OUT}"
    fi
else
    row FAIL "pytest" "pytest not installed (pip install pytest)"
fi

# --------------------------------------------------------------------------
# Gate 5: import smoke — the compiled extensions still load
# --------------------------------------------------------------------------

OUT="${TMPDIR_PF}/import.log"
capture "${OUT}" python -c "import hazma; print(hazma.__version__)"
if [[ $? -eq 0 ]]; then
    row PASS "import hazma" "version $(tr -d '\n' <"${OUT}")"
else
    row FAIL "import hazma" "package does not import — rebuild (pip install -e .)"
    tail_of "${OUT}"
fi

# --------------------------------------------------------------------------
# Gate 6: markdownlint (only with --md)
# --------------------------------------------------------------------------

if [[ -n "${MD_FILES}" ]]; then
    if have markdownlint; then
        OUT="${TMPDIR_PF}/md.log"
        # shellcheck disable=SC2086
        capture "${OUT}" markdownlint --dot ${MD_FILES}
        if [[ $? -eq 0 ]]; then
            row PASS "markdownlint" "${MD_FILES}"
        else
            row FAIL "markdownlint" "see output below"
            tail_of "${OUT}"
        fi
    else
        row WARN "markdownlint" "markdownlint not installed — docs unchecked"
    fi
else
    row SKIP "markdownlint" "no --md files given"
fi

# --------------------------------------------------------------------------
# Gate 7: version bump (only with --closing)
# --------------------------------------------------------------------------

if [[ ${CLOSING} -eq 1 ]]; then
    # Read the version out of a file. Written as a standalone script (not a
    # heredoc) because a heredoc claims stdin, which silently swallows a
    # piped `git show` and makes the whole gate pass on an empty old value.
    read_version() {
        python -c '
import re, sys
try:
    text = open(sys.argv[1], encoding="utf-8").read()
except OSError:
    sys.exit(1)
m = re.search(r"VERSION\s*:\s*Final\[str\]\s*=\s*\"([^\"]+)\"", text)
if not m:
    sys.exit(1)
print(m.group(1))
' "$1"
    }

    OLD_FILE="${TMPDIR_PF}/old_init.py"
    NEW_VER="$(read_version hazma/__init__.py)" || NEW_VER=""
    if git show "${BASE_REF}:hazma/__init__.py" >"${OLD_FILE}" 2>/dev/null; then
        OLD_VER="$(read_version "${OLD_FILE}")" || OLD_VER=""
    else
        OLD_VER=""
    fi

    if [[ -z "${NEW_VER}" ]]; then
        row FAIL "version bump" "could not read VERSION from hazma/__init__.py"
    elif [[ -z "${OLD_VER}" ]]; then
        # No baseline means the comparison is vacuous — never report PASS.
        row FAIL "version bump" \
            "could not read VERSION from ${BASE_REF}:hazma/__init__.py — bump unverifiable"
    elif [[ "${NEW_VER}" == "${OLD_VER}" ]]; then
        row FAIL "version bump" "VERSION still ${OLD_VER} — closing PRs must bump it"
    elif ! grep -q "\[${NEW_VER}\]" CHANGELOG.md 2>/dev/null; then
        row FAIL "version bump" "no '## [${NEW_VER}]' section in CHANGELOG.md"
    else
        row PASS "version bump" "${OLD_VER} → ${NEW_VER} + CHANGELOG entry"
    fi
else
    row SKIP "version bump" "not a closing PR (pass --closing)"
fi

# --------------------------------------------------------------------------
# Gate 8: forbidden tokens in the diff (advisory)
# --------------------------------------------------------------------------

OUT="${TMPDIR_PF}/forbidden.log"
git diff "${BASE_REF}" -- '*.py' '*.pyx' 2>/dev/null \
    | grep -E '^\+' \
    | grep -Ev '^\+\+\+' \
    | grep -nE 'breakpoint\(\)|pdb\.set_trace|import pdb|XXX:|FIXME' \
    >"${OUT}" 2>/dev/null
if [[ -s "${OUT}" ]]; then
    row WARN "forbidden tokens" "$(wc -l <"${OUT}" | tr -d ' ') hit(s) — resolve or justify"
    tail_of "${OUT}"
else
    row PASS "forbidden tokens" "none added"
fi

# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

echo
echo "preflight — ${REPO_ROOT} (base ${BASE_REF})"
echo "-------------------------------------------------------------------"
for line in "${ROWS[@]}"; do
    echo "${line}"
done
echo "-------------------------------------------------------------------"

if [[ ${HARD_FAIL} -eq 1 ]]; then
    echo "RESULT: FAIL — blocked commit. Fix the red gates and re-run."
    exit 1
fi
echo "RESULT: PASS"
exit 0
