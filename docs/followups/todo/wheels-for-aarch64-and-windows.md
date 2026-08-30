# Wheels for linux-aarch64 and Windows

- **Added:** 2026-08-29
- **Source:** `projects/cython-to-rust/learnings/project-retrospective.md` §5
- **Scope:** commit
- **Status:** open
- **Triggers / blockers:** a user asking for either platform. There is no
  technical blocker and there has not been one since the maturin cutover.

## Why

Hazma publishes wheels for macOS arm64 and manylinux x86_64 and nothing
else. Users on Windows or on 64-bit ARM Linux must build from the sdist,
which since 3.0.0 means having a Rust toolchain on `PATH`.

The cython-to-rust project put both platforms out of scope, and Task 7.2
made that a deliberate decision rather than an oversight when it rebuilt
`release.yml` on `maturin-action`: the two shipped platforms are the two
CI tests on, and no user had asked. It declined to file this entry at the
time because `projects/cython-to-rust/PLAN.md` §Scope already recorded
it. Closing that project turned the record archival, and
`docs/followups/todo/` is the repo's live backlog by construction — so
the entry lands here now, with the decision unchanged.

What *has* changed is the price. Under cibuildwheel each new platform
meant a full CPython matrix; under abi3 and `maturin-action` it is one
matrix row producing one wheel, which is why the deliberate "no" is worth
writing down rather than leaving implicit.

## What

Add matrix rows to `.github/workflows/release.yml`'s wheel job — one for
`aarch64` manylinux, one for Windows x86_64 — and extend the existing
in-workflow assertions to cover them. Two things the existing assertions
already teach:

- **The wheel-filename platform field is a compressed tag *set*, not a
  tag.** The manylinux wheel's field is
  `manylinux_2_17_x86_64.manylinux2014_x86_64`, standing for two `Tag:`
  lines, while the macOS wheel is one tag per field. An assertion that
  compares field against `Tag:` line as a single string passes on macOS
  and rejects a correct Linux wheel — so a new platform's assertion is
  not believed until that platform's job has actually run.
- **The sole-extension assertion hard-codes a POSIX suffix.**
  `release.yml` requires the wheel's compiled objects to be exactly
  `["hazma/_core.abi3.so"]`. Windows names the same extension
  `hazma/_core.pyd`, so that check needs a platform-aware suffix before a
  Windows wheel can pass it.

## Entry points

- `.github/workflows/release.yml` — the `maturin-action` wheel matrix,
  the `Tag:` cross-product check and the sole-extension assertion
- `projects/cython-to-rust/task-notes/phase-07/task-7.2-release-pipeline.md`
  §Decisions — where the "no" was decided, and the wheel-tag finding
- `projects/cython-to-rust/PLAN.md` §Scope — the original exclusion

## Risks / open questions

- **Windows is not just a matrix row.** The extension suffix differs, and
  nothing in the suite has ever run there, so the first Windows job is
  likely to find assumptions rather than confirm them. aarch64 Linux is
  the genuinely cheap half of this entry.
- **Adding a platform is a support commitment.** A wheel that builds is
  not the same as a platform the numerical suite is known to pass on;
  the parity corpus is scoped to its capturing platform by construction.
