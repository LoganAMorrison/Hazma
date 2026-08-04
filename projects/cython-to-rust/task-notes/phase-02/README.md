# Working Memory: Phase 02 — Rust scaffold

**Date:** 2026-08-03 (created)
**Project:** cython-to-rust
**Phase:** 02
**Status:** Not started
**Plan References:** `../../phases/phase-02-rust-scaffold.md`
**Related ADRs:** ADR-0001 (accepted)
**Depends On:** Phase 01 complete

## Objective

Track live per-task status and phase-scoped findings for the Rust
scaffold.

## Tasks

| # | Task | Depends on | Status | Task Note |
| --- | ------ | ------------ | -------- | ----------- |
| 2.1 | Crate + setuptools-rust integration | — | Not started | [task-2.1-crate-skeleton.md](task-2.1-crate-skeleton.md) |
| 2.2 | CI, preflight, dev-loop docs | 2.1 | Not started | [task-2.2-ci-devloop.md](task-2.2-ci-devloop.md) |
| 2.3 | Cross-language plumbing test | 2.1 | Not started | [task-2.3-plumbing-test.md](task-2.3-plumbing-test.md) |

## Exit Criteria

- All rows Complete; phase file frontmatter `status: Complete`.
- Phase learnings at `../../learnings/phase-02-rust-scaffold.md`.

## Inputs Reviewed

- `../../phases/phase-02-rust-scaffold.md`; `../README.md`;
  `../../rules.md` rules 6–8.

## Findings

_None yet._

## Decisions and Implementation Notes

_None yet._

## Files Changed

_None yet — phase not started._

## Verification

- `pip install -e .` builds hybrid; `python -c "import hazma._core"`;
  wheel job produces abi3-tagged wheels; preflight incl. cargo gates.

## Open Questions

_None yet._

## Plan Impact

**Impact Level:** None (this file is metadata, not a canonical change).

## Handoff to Next Task

**For the next agent working in Phase 02:** read `../../PLAN.md`,
`../README.md`, this file, then the phase file. The extension's import
path is final from day one: `hazma._core`.

**Currently safe to assume:** rustc/cargo 1.96 available locally
(Homebrew); CI must install its own toolchain.

**Currently risky / unknown:** setuptools-rust + editable-install
rebuild ergonomics under uv — document whatever loop actually works in
Task 2.2.
