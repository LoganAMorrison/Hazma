# Task 3.1: Constants module

**Date:** 2026-08-09
**Project:** cython-to-rust
**Status:** Complete
**Plan References:** `../../phases/phase-03-numerics-foundation.md` (Task 3.1);
`../../rules.md` rule 4 (Constants 1), rules 6–8 (Rust conventions 1–3)
**Related ADRs:** none
**Depends On:** Phase 02 complete

## Objective

Give the Rust crate the constant tables every later kernel port stands
on: both of hazma's divergent Cython tables, in two namespaces that keep
the divergence verbatim, plus the module-local `DEF`s individual `.pyx`
files declare — with the transcription checked by machine rather than by
reading.

## Exit Criteria

From the phase file, verbatim:

- `rust/src/constants.rs` carries every `DEF` from `_utils/constants.pxd`
  and every value from `_utils/legacy_parameters.pxd`, in **two distinct
  namespaces** preserving the known divergences verbatim (rules.md
  rule 4).
- A test extracts values from the `.pxd` sources (script or generated
  fixture) and asserts bit-equality — no hand-transcription trust.
- Derived module-local `DEF`s (e.g. `eng_mu_pi_rf`) become `const fn` or
  literal consts with the same float semantics.

## Inputs Reviewed

- `../../PLAN.md` (all sections, incl. Numerical impact);
  `../README.md`; `../../rules.md`; the phase file's Task 3.1 block.
- `hazma/_utils/constants.pxd` (151 `DEF`s) and
  `hazma/_utils/legacy_parameters.pxd` (48 `cdef double`s), in full.
- Every `.pyx` under `hazma/` carrying module-local `DEF`s — five files,
  found by `rg -n '^\s*DEF\s' hazma/` rather than from the inventory.
- `rust/src/{lib,kernels,dispatch}.rs` for the crate's layering and
  doc-comment conventions; `test/parity/cases.py` for the `REPO_ROOT`
  idiom.
- `docs/agents/lessons.md`, `docs/agents/environment.md`.
- External, for the module header's `# Sources` block: the PDG constants
  review index
  (<https://pdg.lbl.gov/2025/reviews/constants_atomic_and_related.html>);
  CODATA 2022, Mohr, Newell, Taylor & Tiesinga,
  [arXiv:2409.03787](https://arxiv.org/abs/2409.03787), cross-checked
  against NIST's current α⁻¹ = 137.035999177(21)
  (<https://physics.nist.gov/cgi-bin/cuu/Value?alphinv>).

## Findings

- **The two tables share 19 names and disagree on 12 of them.** Ten
  masses (`MASS_E`, `MASS_MU`, `MASS_PI0`, `MASS_PI`, `MASS_K0`,
  `MASS_K`, `MASS_ETA`, `MASS_ETAP`, `MASS_RHO`, `MASS_OMEGA`) plus
  `ALPHA_EM` and the derived `RATIO_E_MU_MASS_SQ`. The seven that agree
  are the form factors and decay constants (`F_A_PI`, `F_V_PI`,
  `F_V_PI_SLOPE`, `F_A_K`, `F_V_K`, `DECAY_CONST_PI`, `DECAY_CONST_K`).
  That partition is now a literal roster in
  `test/test_core_constants.py`, because rule 4's whole content is that
  it does not move — a computed partition would accept any partition.
- **`hazma/spectra/_photon/_pion.pyx` mixes the two tables inside one
  module, and nothing said so before now.** It `include`s
  `constants.pxd`, so its `MPI` / `ME` / `MMU` aliases are PDG values —
  but its five hard-coded kinematic literals reproduce **bit-exactly**
  from the *legacy* masses and from no other table:

  | `DEF` | Formula (legacy masses) |
  | --- | --- |
  | `ENG_MU_PIRF` | `0.5 (mπ² + mμ²) / mπ` |
  | `GAMMA_MU_PIRF` | `ENG_MU_PIRF / mμ` |
  | `BETA_MU_PIRF` | `sqrt(1 − 1/γ²)` |
  | `ENG_GAM_MAX_MURF` | `(mμ² − mₑ²) / (2 mμ)` |
  | `ENG_GAM_MAX_PIRG` | `ENG_GAM_MAX_MURF · γ (1 + β)` |

  Recomputing them from the header the file actually includes moves
  `ENG_MU_PIRF` by 4.7e-5 MeV and every charged-pion photon spectrum
  with it. **Phase 04 must not tidy this up.** Both halves of the mix
  are pinned, in Python and in Rust.
- **`R_FACTOR`'s comment has an exponent typo.** Both muon kernels
  (`spectra/_positron/_muon.pyx:14`, `spectra/_neutrino/_muon.pyx:22`)
  annotate the literal `1.0001870858234163` as
  `1 / (1 - 8 r^2 + 8 r^6 - r^8 - 12 r^2 ln(r^2))`. Only `r^4` on the
  log term reproduces the digits: the commented form evaluates to
  `0.9972020119096803`, 3.0e-3 relative from the published
  `1.0001870858234163`, so the number settles it. Unlike the
  `_photon/_pion.pyx` literals this one is frozen against the **PDG**
  table, so it agrees with the `R` beside it. Pinned both ways; the
  Cython comment is left alone (untouched file, and the number is
  unambiguous).
- **Hazma holds three different fine-structure constants, and only the
  masses agree with themselves.** `constants.pxd` has
  `1/137.035999084(21)` — a pre-2022 CODATA adjustment; CODATA 2022
  revised α⁻¹ to 137.035999177(21) (arXiv:2409.03787, cross-checked
  against NIST). `legacy_parameters.pxd` has `1/137`.
  `hazma/parameters.py:205` has a third, `1/137.04`. By contrast **every
  one of the fourteen masses in `constants.pxd` is bit-equal to its
  `hazma/parameters.py` counterpart** (checked, not assumed). All three
  αs are kept; the third is pure Python and outside this project's scope
  entirely, but it belongs in the same conversation as the table merge.
- **`constants.pxd` is `include`d by twelve spectra extensions, not
  eleven** — counted at execution time
  (`rg -c 'include "../../_utils/constants.pxd"' -g '*.pyx' hazma`),
  because the header's own summary sentence is the sort of number that
  goes stale as Phases 04-06 delete files.
- **`clippy::excessive_precision` is on by default and fires on verbatim
  transcription.** Trailing zeros the `.pxd` writes for significant
  figures (`0.9998770`, `0.0023900`, `1.760e-2`) are digits clippy wants
  dropped. Dropping them would break character-for-character diffability
  against the Cython without changing one f64, so the module carries
  `#![allow(clippy::excessive_precision)]` with that reason stated.
- **A fresh env silently puts the parity corpus into budget mode, over a
  NumPy patch release.** The first bare-suite run on this branch came
  back `1087 passed, 14 skipped` — 14, where Phase 02 closed at 13, and
  the working memory records an unchanged 13 as the tell that the corpus
  is in bit-equality mode. It was not this task's diff:
  `tolerances.provenance` compares the whole numerics environment, and
  `uv pip install -e .` had resolved NumPy **2.5.2** against the
  manifest's **2.5.1** (`exact: False`, detail
  `numpy '2.5.1' -> '2.5.2'`). The kernel digest was identical
  (`f5e6e269be47`) and `cases.rust_core_kernels()` was empty, so both
  port-specific predicates were clean. Pinning `numpy==2.5.1` and
  rebuilding `--no-build-isolation` restored `exact: True`, and the
  numbers reported below are from that tree. Recipe and a one-second
  provenance check are now in `../README.md` — **this will recur for
  every Phase 03–06 task that builds a fresh env.**
- **The `.pyx` `DEF`s only ever reference the PDG header.** All five
  files with module-local `DEF`s `include` `constants.pxd`; the four
  mediator extensions that `include` the legacy header declare no `DEF`s
  of their own. Asserted, not assumed
  (`test_every_derived_pyx_includes_the_table_it_is_scored_against`).

## Decisions and Implementation Notes

- **Namespaces are `pdg` and `legacy`**, mapped to the two `.pxd` in the
  module header rather than named for the files (`constants::constants`
  would have been the alternative). A third namespace, `derived`, holds
  the module-local `DEF`s in one submodule per source `.pyx`
  (`derived::photon_pion`, …), so a Phase 04 port has one place to look
  and the coverage check can be total.
- **Every module-local `DEF` is carried, not only the derived ones.**
  The exit criterion asks for the derived ones; including the pure
  aliases (`MPI = MASS_PI`) costs three lines per module and buys a
  *total* coverage assertion —
  `test_no_pyx_declares_constants_this_module_ignores` rescans the tree
  and fails if any `.pyx` grows or loses a `DEF`. For
  `derived::photon_pion` it is the point: PDG aliases and legacy
  literals sit side by side, which is the documentation Phase 04 needs.
- **Hard-coded `DEF`s stay literals; computed ones become `const`
  expressions** in the same association order as the Cython. That is
  both what the sources have and what Rust allows — `sqrt` and `ln` are
  not `const`, so the six frozen literals could not be expressions even
  if we wanted them to be. `const fn` was not needed: nothing here takes
  an argument.
- **The test compares source text to source text, and says so.** It
  parses the `.pxd`, the `.pyx` and `constants.rs`, evaluates all three
  through one restricted AST evaluator, and compares IEEE payloads. Both
  CPython's `float()` and rustc's literal parsing are correctly rounded,
  so this is a sound bit-equality claim about what the two files
  *denote*. It deliberately does **not** import `hazma._core`: it needs
  no build, runs in 0.03s, and holds on every platform — unlike the
  parity corpus. The compiled side is `cargo test`'s job, and
  `rust/src/constants.rs` carries five unit tests that recheck the
  derived values against runtime arithmetic and pin both tables' split.
- **Rust name resolution is reimplemented, `use` parsing is not.** The
  parser resolves `pdg::MASS_E` from inside `derived::positron_muon` by
  walking outward from the current module, exactly as Rust does. That
  keeps the Rust source idiomatic (short qualified paths that show
  provenance at the point of use) without the test needing to understand
  `use`.
- **`pub mod constants;` in `lib.rs`, where its neighbours are private.**
  Nothing in the crate reads the tables yet, and a private module of
  ~220 unread `const`s is a wall of `dead_code` under `-D warnings`. The
  reason is recorded in `lib.rs`'s header alongside the existing
  layering note.
- **The 199 `.pxd` literals were transcribed by a throwaway script**
  (`.pxd` → `pub const`, comments carried through) rather than retyped.
  The script is not committed: it is worth nothing after one run, and
  the permanent guarantee is the test, which is independent of how the
  file was produced.
- **`test/test_core_constants.py` is dated by construction and says so.**
  It reads Cython, so it dies with it; the module docstring and
  `require()`'s failure message tell Phases 04–06 to drop each row as its
  `.pyx` goes and delete the module with the last `.pxd`. That is a
  `pytest.fail` with instructions, not a `FileNotFoundError`.

## Files Changed

- `rust/src/constants.rs` — **new.** 224 `pub const`s in three
  namespaces (`pdg` 151, `legacy` 48, `derived::*` 25), the `# Sources`
  provenance header, and five `cargo test` unit tests.
- `rust/src/lib.rs` — `pub mod constants;` plus a header paragraph on
  why it is `pub`.
- `test/test_core_constants.py` — **new.** 25 tests: parser
  self-checks, name/bit-equality against both `.pxd` and all five
  `.pyx`, the frozen-literal provenance reconstructions, and the rule-4
  divergence roster.
- `projects/cython-to-rust/task-notes/phase-03/task-3.1-constants.md` —
  this note.
- `projects/cython-to-rust/task-notes/phase-03/README.md` — Task 3.1 row
  and phase-scoped findings.
- `projects/cython-to-rust/task-notes/README.md` — Findings, Numerical
  impact so far, Decisions, Files Changed, Handoff.

## Verification

Commands and their real output, all from the task worktree on the
capturing environment (CPython 3.12.12, macOS/arm64):

- `.venv/bin/python -m pytest test/test_core_constants.py -q` →
  **`25 passed in 0.03s`**. Coverage by category:
  - *Parser self-checks* (4): module set, non-trivial table sizes,
    rejection of unsupported syntax and undefined names, injectivity of
    the `.upper()` name fold (`etap_BR_pi0_pi0_eta` is the one name it
    has to fold).
  - *`.pxd` tables* (4, parametrized): name-set equality and bit-equality
    for `pdg` and `legacy`.
  - *Module-local `DEF`s* (10, parametrized): the same two checks for
    each of the five `.pyx`.
  - *Completeness* (2): no `.pyx` in the tree declares a `DEF` this
    module ignores; each derived source `include`s the table it is
    scored against.
  - *Frozen-literal provenance* (3): `_photon/_pion.pyx`'s five
    kinematic literals reconstruct from `legacy` and from no other
    table, its three aliases come from `pdg`, and `R_FACTOR` is the
    Michel normalization over the PDG ratio with the `r^4` log term.
  - *Rule 4* (2): the shared-name partition into divergent (12) and
    identical (7), and the legacy `WIDTH_*` section still empty against
    13 in `pdg`.
- `cargo test --manifest-path rust/Cargo.toml --no-default-features` →
  **`7 passed; 0 failed`** (5 new in `constants::tests`, 2 pre-existing
  in `kernels::tests`).
- `cargo clippy --manifest-path rust/Cargo.toml --all-targets --
  -D warnings` → clean. `cargo fmt --manifest-path rust/Cargo.toml
  --check` → clean.
- Bare `pytest` (via the preflight gate) → **`1088 passed, 13 skipped,
  5 warnings in 557.28s`** (+25 on Phase 02's 1063, all of them the new
  module). The skip count is unchanged at 13, which is the tell that the
  parity suite ran in **bit-equality mode** — all 41 consumed entry
  points, 179,695 pinned
  values, `rtol = 0`.
- `scripts/agents/preflight.sh --paths "test/test_core_constants.py"
  --md "<the three changed .md>"` → **RESULT: PASS** across all eleven
  rows (version bump SKIP: not a closing PR). The first run of it caught
  one real thing, an MD013 over-length line in `../README.md` that this
  note's own re-derivation had introduced; fixed and re-run green.

**Test validity — thirteen mutations, each caught by the test whose name
claims it.** Nine against the Python module (restored between runs):

| Mutation | Caught by |
| --- | --- |
| `pdg::MASS_TAU` last digit | `TestTables::test_values_are_bit_equal[pdg]` |
| `legacy::MASS_OMEGA` last digit | `…[legacy]` |
| `pdg::BR_PHI_TO_MU_MU_A` deleted | `TestTables::test_names_match_exactly[pdg]` + bit-equality |
| `legacy::MASS_E := pdg::MASS_E` (the consolidation) | `test_the_two_tables_diverge_on_exactly_the_recorded_names` + `…[legacy]` |
| `photon_pion::ENG_MU_PIRF` recomputed from `pdg` | `test_photon_pion_literals_come_from_the_legacy_table` + `TestDerived…[photon_pion]` |
| `R_FACTOR` := `0.9972020119096803`, the comment's `r^2` log term | `test_r_factor_is_the_michel_normalization_over_the_pdg_ratio` + both muon modules |
| `WIDTH_K` resurrected in `legacy` | `test_the_legacy_widths_table_is_still_empty` + two others |
| `GAMMA_MU`: `/ MMU` → `/ MPI` | `TestDerived::test_values_are_bit_equal[positron_pion]` |
| a new `DEF R3` added to `_positron/_muon.pyx` | `TestDerived::test_names_match_exactly[positron_muon]` |

One further mutation — re-associating `ENG_MU_PI_RF` as
`0.5 / MPI * (…)` — passed, correctly: at these masses all three
associations give the identical payload
(`0x1.b71ced218b450p+6`, checked directly), so the mutation changed no
value. The value-changing slip in the same expression is the `GAMMA_MU`
row above.

Four against the Rust unit tests, same method:

| Mutation | Caught by |
| --- | --- |
| the consolidation | `the_two_tables_disagree_where_the_cython_says_they_do` |
| `ENG_MU_PIRF` from `pdg` | `photon_pion_mixes_the_two_tables` |
| `R_FACTOR` := the typo'd formula's value | `r_factor_reproduces_from_the_pdg_mass_ratio` |
| `ENG_E_PI_RF` uses `MMU` for `ME` | `const_folding_matches_a_runtime_evaluation` |

Nothing deferred.

### Review round 1 (PR #58)

- **Blocking, accepted:** the mass count was wrong in five places — the
  `constants.rs` header, this note twice, the working memory, and the PR
  body all said "twelve". `constants.pxd` declares **fourteen** `MASS_*`,
  and all fourteen are bit-equal to `hazma/parameters.py`'s. The original
  check *ran*, but over a hand-typed 12-pair list that silently omitted
  `MASS_KL` and `MASS_KS`; the wrong answer then agreed with the correct
  count of twelve `include`-ing spectra extensions two paragraphs above,
  so every internal cross-check looked consistent. Re-derived by
  enumerating `^DEF MASS_` from the header and matching on bit pattern
  instead of on typed names. New class in `docs/agents/lessons.md`:
  `[hand-written-population-in-a-derived-check]`.
- **Non-blocking, accepted:** `cargo doc` gained a fourth warning,
  `redundant explicit link target` at `constants.rs:589` —
  `[`photon_pion::ENG_MU_PIRF`](photon_pion::ENG_MU_PIRF)` where the
  label already resolves. Simplified to `[`photon_pion::ENG_MU_PIRF`]`;
  `cargo doc` is back to the three pre-existing `links to private item`
  warnings in `lib.rs`. **This one was mine to have caught**: the
  verification above grepped `^warning: unresolved`, which is a subset of
  the warnings `cargo doc` emits, so "0 unresolved intra-doc links" was
  true and yet read as "clean". The grep is now `^warning` with a
  by-category count.

## Numerical Impact

**No public value changes** (verified: `git diff origin/master -- hazma`
is empty — 0 lines, on a tree cleaned and rebuilt with
`uv pip install -e .` before anything was run;
`.venv/bin/python -c "import hazma._core; print(hazma._core.__file__)"`
resolves inside this worktree, and the tree carries 21 `.so` — the 20
Cython extensions plus `hazma/_core.abi3.so`).

No grid evaluation applies: the diff adds a Rust module that no Python
imports and no Rust kernel calls, one `pub mod` line, one test module,
and project bookkeeping. No library module, signature, constant or build
*input* under `hazma/` is reachable from it. Measured rather than only
argued: the bare suite ran the parity corpus in bit-equality mode
(`rtol = 0`, 41 entry points, 179,695 pinned values) and passed, at
`1088 passed, 13 skipped`.

The numbers this task *does* move are inside the Rust crate, where they
had no prior value — and the whole task is the argument that they are
the Cython's numbers bit-for-bit rather than approximately.

## Open Questions

- **Which PDG edition each `constants.pxd` value came from is not
  recorded anywhere in the tree.** The `± uncertainty` annotations are
  the only provenance the Cython left; they match `hazma/parameters.py`,
  which says "PDG March 2022" against the CKM block only. Some entries
  are demonstrably older than the current edition (m[τ] = 1776.86 ±
  0.12 MeV; α⁻¹ = 137.035999084, a pre-CODATA-2022 adjustment). The
  module header cites the PDG review index for the tables rather than
  claiming an edition per value, which is honest but weaker than it
  could be. Not blocking, and **not** something to resolve by re-sourcing
  values — rule 4 forbids that. Filing it only if the consolidation
  follow-up below is taken up.
- The two consolidation follow-ups this module makes concrete but does
  not touch:
  [`positron-spectrum-nan-at-legacy-electron-mass`](../../../../docs/followups/todo/positron-spectrum-nan-at-legacy-electron-mass.md)
  (ripens before Phases 05/06) and the general table merge, which
  `../../PLAN.md` scopes out. `constants.rs`'s two namespaces are what
  either would edit.

## Plan Impact

**Impact Level:** None.

Canonical-contract diff performed: the phase file's three Task 3.1 exit
criteria were read one at a time against the shipped diff and each is
satisfied as written — two namespaces preserving the divergences, a test
that extracts from the `.pxd` sources and asserts bit-equality, and
derived module-local `DEF`s as literal consts and `const` expressions
with the same float semantics. Task 3.4's "Depends on: Task 3.1" is now
satisfiable. No gate sentence, exit criterion or active ADR is made
factually wrong by this task, so nothing in `PLAN.md`, the phase file,
`rules.md` or any ADR is patched. `rules.md` rule 4 is *implemented*
here, not revised.

## Stale-state sweep

```text
$ git -C <wt> diff origin/master --stat
 projects/cython-to-rust/task-notes/README.md            | 141 +++-
 projects/cython-to-rust/task-notes/phase-03/README.md   | 104 ++-
 .../task-notes/phase-03/task-3.1-constants.md           | 445 ++++++++++
 rust/src/constants.rs                                   | 750 +++++++++++++++++
 rust/src/lib.rs                                         |   7 +
 test/test_core_constants.py                             | 569 ++++++++++++
 6 files changed, 2002 insertions(+), 14 deletions(-)
 (this note's own row is measured before the block quoting it was
  finalized; the residual is this edit and nothing else)

$ git -C <wt> status --short
M  projects/cython-to-rust/task-notes/README.md
M  projects/cython-to-rust/task-notes/phase-03/README.md
A  projects/cython-to-rust/task-notes/phase-03/task-3.1-constants.md
A  rust/src/constants.rs
M  rust/src/lib.rs
A  test/test_core_constants.py

$ git -C <wt> rev-parse --abbrev-ref HEAD --show-toplevel
claude/cython-to-rust/task-3.1-constants-module
<wt>   (not master, and the worktree - preflight.md's pre-commit assertion)

$ git -C <wt> diff origin/master -- hazma | wc -l
0

$ rg -n 'TODO|FIXME|breakpoint\(|import pdb|[^_a-z]print\(' \
     rust/src/constants.rs rust/src/lib.rs test/test_core_constants.py
(no occurrences)

$ rg -c 'constants\.rs|test_core_constants' <the three changed .md>
task-notes/README.md:5
task-notes/phase-03/README.md:8
task-notes/phase-03/task-3.1-constants.md:19

$ rg -n '_build\.py' rust/src/constants.rs test/test_core_constants.py \
     projects/cython-to-rust/task-notes/phase-03/
(no occurrences; the hits under task-notes/phase-00/ are Task 0.4's own
 dated record of its sweep, untouched here)

$ rg -n 'constants\.pxd|legacy_parameters' docs/
docs/followups/README.md:44            (WIDTH_K/WIDTH_PI, done)
docs/followups/done/legacy-parameters-width-exponent-bug.md   (17 lines)
docs/followups/todo/positron-spectrum-nan-at-legacy-electron-mass.md:14,15,57,73,74
(all still accurate: they describe the divergence this task preserves
 rather than resolves, and none cites a line number this diff moves.
 docs/source/ has no occurrence — no public Python object is renamed.)

$ python3 -c "..."   # every numeric claim in this note, re-derived
199 .pxd constants (151 pdg + 48 legacy); 25 module-local DEFs across 5 .pyx
224 pub const in rust/src/constants.rs  -> 199 + 25, matches
per module: pdg 151, legacy 48, photon_pion 8, photon_rho 3,
            positron_muon 3, positron_pion 6, neutrino_muon 5
19 shared names: 12 divergent, 7 identical
13 WIDTH_* in pdg, 0 in legacy
12 .pyx include ../../_utils/constants.pxd; 4 include legacy_parameters.pxd
14 MASS_* in constants.pxd, all bit-equal to hazma/parameters.py
            (K0/KL/KS share 497.611 -> 14 names, 12 distinct values);
            ALPHA_EM is not

$ cargo doc --no-deps --no-default-features 2>&1 | grep -E '^warning' \
    | sort | uniq -c
   1 warning: `hazma-core` (lib doc) generated 3 warnings
   1 warning: public documentation for `_core` links to private item `dispatch`
   2 warning: public documentation for `_core` links to private item `kernels`
(all three pre-existing in lib.rs and untouched by this diff. Counting by
 category, not grepping for one phrase -- the first pass here grepped
 '^warning: unresolved' and so reported "0 unresolved links" while this
 diff had in fact added a `redundant explicit link target`. Review round 1
 caught it.)
```

**Numerical-impact statement:** No public value changes (verified:
`git diff origin/master -- hazma` is empty — 0 lines; and the bare suite
below ran the parity corpus in bit-equality mode, `rtol = 0` across all
41 consumed entry points and 179,695 pinned values).

Doc-consistency sweep §1/§2/§7/§11/§12 run over the diff. Every count in
this note and in `constants.rs`'s header was re-derived from the live
tree rather than carried from an earlier draft — which caught three
drafting errors before they shipped: the `.pxd` split (151/48, not the
174/25 first written), the include count (twelve spectra extensions, not
eleven), and the number of frozen literals (seven, not six). The sibling
occurrences of each were updated together (§11), and both new files were
swept as fresh creations (§12).

## Handoff to Next Task

**Read first:** this note's Findings, then `rust/src/constants.rs`'s
module header — it is the durable version of the same facts and travels
with the code.

**Now safe to assume:**

- `hazma_core::constants::{pdg, legacy}` are the two Cython tables, and
  `constants::derived::<source_pyx>` holds every module-local `DEF`, all
  bit-equal to the Cython and held there by two independent gates
  (`test/test_core_constants.py`, 25 tests, 0.03s, platform-independent;
  and five `cargo test` units). A Phase 04 kernel names the table its
  `.pyx` `include`s — `pdg` for everything under `hazma/spectra/**`,
  `legacy` for the four mediator spectrum extensions.
- Task 3.4 (interp + boost) is unblocked.

**Still risky:**

- **`derived::photon_pion` mixes both tables and that is correct.** Its
  aliases are PDG, its five frozen literals are legacy. A Phase 04 port
  that "cleans this up" moves published photon spectra; the two tests
  named in that module's doc comment are what stop it.
- `derived::positron_pion::ENG_MU_PI_RF` and
  `derived::photon_pion::ENG_MU_PIRF` are the same physical quantity,
  different numbers, and one underscore apart. Read the module, not the
  name.
- Nothing in the crate consumes these yet, so the first Phase 04 swap is
  the first time a wrong *choice* of table (as opposed to a wrong value)
  can show up. The parity corpus is what catches that.
