# Task 6.3: Positron spectrum pair

**Date:** 2026-08-27
**Project:** cython-to-rust
**Status:** Complete
**Plan References:** `../../phases/phase-06-mediator-spectra.md` (Task 6.3);
`../../rules.md` rules 1–3 (parity), 4 (constants), 6–9 (Rust conventions),
12 (benchmarks)
**Related ADRs:** ADR-0001, ADR-0002
**Depends On:** Task 6.1 (table/cache/mode foundation), Task 6.2 (module
layout, test shape, PyO3 error plumbing)

## Objective

Move `dnde_decay_s`/`dnde_decay_s_pt` and `dnde_decay_v`/`dnde_decay_v_pt`
— the two mediator *positron* spectrum modules — onto `hazma._core`,
delete both Cython twins, and decide the threshold-`nan` follow-up that
ripens exactly here.

## Exit Criteria

From the phase file's Task 6.3 block, plus the two project rules that
bind every swap:

- `dnde_decay_s`/`dnde_decay_s_pt` (scalar) and the vector clone on Rust.
- Corpus green within each function's budget; any budget change
  justified in `test/parity/tolerances.py` and recorded here (rule 2).
- Wrappers swapped; both Cython twins deleted in the same change
  (rule 1 — no dual-implementation drift window).
- Every numerical shift beyond 1e-12 relative declared here and appended
  to `../numerical-impact.md` (rule 3).
- Every `.pyx` numeric edge guard survives explicitly (rule 9).

## Inputs Reviewed

- `../../PLAN.md` — Scope, Numerical impact, Phases table.
- `../../phases/phase-06-mediator-spectra.md` — Task 6.3 exit criteria.
- `../README.md` (project working memory) and `README.md` (phase 06),
  whose `## Handoff to Next Task` is this task's brief.
- `../../rules.md` — all five sections.
- `task-6.2-decay-spectra.md` — `## Findings`, the FMA mutation campaign,
  `## Handoff`.
- `hazma/{scalar,vector}_mediator/*_positron_spec.pyx` — the two sources.
- `hazma/spectra/_positron/{_muon,_pion}.pyx` — the cimported table
  kernels.
- `rust/src/kernels/{mediator_tables,vector_decay_photon}.rs`,
  `rust/src/vector_mediator.rs` — the Task 6.1/6.2 foundation and template.
- `test/parity/{cases,tolerances}.py`, `test/parity/oracles/entry_points.py`.
- `docs/followups/done/positron-spectrum-nan-at-legacy-electron-mass.md`,
  `docs/followups/todo/oracle-restore-revisions-for-the-mediator-decay-pyx.md`.
- `docs/agents/{environment,lessons,doc-consistency}.md`.

## Findings

- **The two `.pyx` are the same text.** Normalise one against the other
  by rewriting `s`↔`v`, `ms`↔`mv`, `eng_s`↔`eng_v`, `eng_p_srf`↔`eng_p_vrf`
  and "scalar"↔"vector", and `diff` reports nothing but those
  substitutions and the order of two `import` lines. Not a clone-pair in
  the decay pair's sense — the *same implementation*, twice. So the port
  is one kernel module, and the four entry points are bit-for-bit equal
  across models by construction.
- **The threshold `nan` is a clang FMA artifact, not source semantics.**
  `sqrt(eng_p * eng_p - me * me)` contracts to
  `fma(eng_p, eng_p, -(me * me))`, which computes the square exactly and
  subtracts the *rounded* `me * me`. For the legacy `m_e` that rounding
  is upward by `1.45e-17`, so at `eng_p == m_e` the radicand is negative
  and `sqrt` answers `NaN` — the whole of
  [`positron-spectrum-nan-at-legacy-electron-mass`](../../../../docs/followups/done/positron-spectrum-nan-at-legacy-electron-mass.md).
  Everything downstream is innocent: both rest-frame tables are
  identically zero at the mass the follow-up used, and the jacobian's own
  radicand is `β² m_e²`, comfortably positive.
- **Finding it needed instrumentation, not reasoning.** Four rounds of
  transliterating the integrand into Python all returned `0.0` where the
  extension returned `NaN`, because none of them reproduced the
  contraction. What settled it was adding a temporary `def` to the `.pyx`
  that returns the intermediates, rebuilding, and reading `p = nan` off
  the real extension. **When a Python replica of a `cdef` disagrees with
  the extension, suspect the compiler before the algebra.**
- **The line term is low by the electron's rest-frame velocity.** The
  `e⁺e⁻` box is `E r β` wide and `pw_ee / (E β)` tall, so it integrates
  to `pw_ee · r` rather than `pw_ee` — a missing `1/r`, worth `3.3e-5` at
  `m = 125` MeV and divergent as `m → 2 m_e`. Reproduced under rule 1 and
  filed as
  [`mediator-positron-line-misses-the-electron-velocity`](../../../../docs/followups/todo/mediator-positron-line-misses-the-electron-velocity.md).
- **Defect A4 reaches all four cases, and it is a pure normalization.**
  Answered from the committed oracle rather than measured again
  (`test/parity/oracles/data/manifest.json`): repairing the inverted
  positron-muon `R_FACTOR` moves 5,237 of 16,740 values in each case, all
  upward, by up to 3.7421e-04 relative — which agrees with
  `R_FACTOR**2 - 1` to eight digits and is identical across all four.
  Contrast A3's reach into the
  photon pair, which changed the *shape* of the vector low-energy tail.
- **`dnde_decay_v` was already taken.** Task 6.2 registered it and
  `dnde_decay_v_pt` in `hazma._core.vector_mediator` for the *photon*
  spectrum; the vector positron `.pyx` exported the identical two names
  from a different extension. Two extensions can; one PyO3 submodule
  cannot. See the naming decision below.
- **Task 6.2 flipped two oracle rows it should not have.**
  `test/parity/oracles/entry_points.py`'s two
  `mediator_spectra.vector.positron.*` rows went `live` → `restored` with
  the note "deleted by Task 6.2", alongside the three photon rows that
  task really was deleting — a search-and-replace on `dnde_decay_v`
  catching the positron pair. Latent, because nothing in `pytest` reads
  the dict. Corrected here, along with the scalar pair this task really
  does delete.
- **The four capi survivors now have no external cimporter.** After this
  swap `hazma/spectra/{_photon,_positron}/_pion.pyx` cimport only their
  own `_muon` twins and `_utils/boost`; nothing else in the tree reads
  any of them. Task 6.4's sweep is therefore already empty.

## Decisions and Implementation Notes

- **One kernel module, `crate::kernels::mediator_decay_positron`**,
  because the two `.pyx` are the same text — a seventh documented naming
  exception in `kernels.rs`, and the only one that is one module for two
  sources. `test_the_two_models_agree_bit_for_bit` is what keeps a later
  edit from quietly giving one model its own arithmetic.
- **The `_core` names are `dnde_positron_decay_{s,v}` and their `_pt`
  twins**, not the Cython's. Forced on the vector half by Task 6.2's
  photon registration; the scalar half follows for symmetry, since
  `dnde_decay_s` beside `dnde_positron_decay_v` for the same function
  would be worse than renaming both. The wrappers re-export under the
  Cython names, so no public import path or call changes, and
  `test/parity/cases.py`'s new `CORE_RENAMES` declares the mapping for
  the served-roster test — the corpus itself still calls the *wrapper*,
  so an alias wired to the wrong `_core` function fails the corpus rather
  than hiding behind the map.
- **The threshold `NaN` becomes `0.0`** — the follow-up's option 2, "keep
  bit-parity and fix the singularity". `momentum` keeps clang's fused
  spelling, so every other energy's arithmetic is unchanged, and clamps a
  negative radicand to zero. Written `if radicand < 0.0` rather than
  `.max(0.0)` so a `NaN` energy still propagates. Consolidating the two
  `MASS_E` tables — the follow-up's option 1 — stays out: `rules.md`
  rule 4 reserves that for a separate declared change after the port.
- **The corpus is not asked to pin the repaired point.** The follow-up
  asked for it, but rule 2 allows corpus data only from pre-port Cython
  and the pre-port value there is `NaN`; pinning it and then changing it
  in the same PR would be circular. `TestTheThresholdSingularity` in
  `test/test_core_mediator_positron.py` pins the new behaviour instead,
  including a 20,001-point sweep of the interval the follow-up swept.
- **`test_core_mediator_tables.py`'s positron-mode oracle is now
  transcription.** It called the shipped `.pyx`; that `.pyx` is gone, so
  it calls the port. Kept and relabelled with its provenance, the
  standing `cython_dispatch_messages()` has had since Task 6.2.
- **Review fix (PR #81): all four remaining `todo/` references to the
  moved follow-up were repointed.** The first pass left them on the
  theory that a pasted transcript and a creation record are frozen
  evidence, generalising from three surviving references to
  `todo/legacy-parameters-width-exponent-bug.md`. `docs/workflow.md:291`
  requires **every** reference to move, and those three are an unswept
  PR rather than a convention — the `[status-encoding-path-reference]`
  ledger entry already said so. The two transcripts now carry the current
  path with a bracketed note of what the command saw when it ran. The two
  older dangling slugs are filed as
  [`moved-followups-leave-dangling-inbound-paths`](../../../../docs/followups/todo/moved-followups-leave-dangling-inbound-paths.md),
  which also proposes the repo-wide preflight gate that would have caught
  this.
- **The restore-revision follow-up stays open.** It cannot be discharged
  here for the same reason Task 6.2 could not discharge it: the revision
  a re-capture needs is the parent of the commit carrying the deletion,
  and that SHA does not exist while the task is authoring the file. The
  follow-up already says "do both in one pass after 6.3 merges"; it is
  extended to name the four positron rows and left open.

## The FMA mutation campaign

Every arithmetic `±` in the new kernel was written both ways, rebuilt
with a forced `rm -f hazma/_core.abi3.so`, and re-measured against the
still-built Cython twins over the corpus's own grids (16,740 values per
model). Baselines: **13,400 / 13,681** before site 5 was adopted,
**13,403 / 13,684** after.

| # | Site | Shipped as | Mutation | scalar | vector | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `sqrt(E² − m_e²)` | fused | unfused | 13,269 | 13,540 | **Killed** (−272) |
| 2 | `E − p β cl` | fused | unfused | 13,369 | 13,650 | **Killed** (−63) |
| 3 | `−1 + cl²` | fused | unfused | 13,402 | 13,686 | Unresolved (+1) |
| 4 | `1 + β²(−1+cl²)` | fused | unfused | 13,402 | 13,685 | Unresolved (0) |
| 5 | `head − coef·m_e·m_e` | **unfused** | fused | 13,400 | 13,681 | **Killed** (−6) |
| 6 | `− 2βcl·E·p` | fused | unfused | 13,390 | 13,675 | **Killed** (−19) |
| 7 | `1 + rβ` (`eplus`) | fused | unfused | 13,400 | 13,681 | Survives *by grid* |
| 8 | `1 − rβ` (`eminus`) | fused | unfused | 13,400 | 13,681 | Survives *by grid* |
| 9 | `1 + (βcl)²` | **unfused** | fused | 13,377 | 13,666 | **Killed** (−38) |
| 10 | `1 − (m/E)²` (2 sites) | **unfused** | fused | 13,292 | 13,572 | **Killed** (−217) |
| 11 | `dnde_cp + dnde_mu` | **unfused** | fused, both orders | 13,392 / 13,386 | 13,654 / 13,668 | **Killed** (−35 / −31) |

Rows 3, 4 and 5 were measured against the post-site-5 baseline; the rest
against the pre-site-5 one, and every one of those moved by far more than
site 5's 3-per-model, so no verdict turns on which baseline it used.

Three results are worth carrying forward.

- **Site 5 contradicted the contraction rule, and measurement won.**
  `head − coef·m_e·m_e` ends in a syntactic multiply, so Task 6.2's rule
  predicts a fusion; fusing it *loses* three bit-equal values per model
  and unfusing it is also the `.pyx`'s own spelling. Sites 9 and 10
  confirm the rule's other half — a `pow` call and a
  `1 − ratio*ratio` written from a `pow` are not contracted, and fusing
  them costs 38 and 217. **The rule predicts; the campaign decides.**
- **Sites 3 and 4 are unresolved, not survivors.** Both sit inside the
  jacobian's radicand, where a `sqrt` and then an adaptive quadrature
  wash out a one-ulp perturbation: the totals move by ±1 in 33,480 and
  the sign flips between models. Kept in the spelling the contraction
  rule predicts, since nothing measured argues otherwise.
- **Sites 7 and 8 survive by grid, in Task 6.2's sense.** The fused and
  unfused line-window edges genuinely differ at **14 of 30**
  `(energy, mass)` pairs — so this is not a power-of-two exactness
  argument — but the edges are only ever *compared* against an energy,
  and **0 of 8,370** corpus grid points change window membership. A
  denser grid, or one landing on an edge, would kill them.

## Files Changed

**Rust.** `rust/src/kernels/mediator_decay_positron.rs` (new — one
kernel for both models), `rust/src/kernels.rs` (module + the naming
exception), `rust/src/scalar_mediator.rs` and
`rust/src/vector_mediator.rs` (two PyO3 entry points each).

**Python package.** `hazma/{scalar,vector}_mediator/_*_positron_spectra.py`
(wrappers repointed), `hazma/{scalar,vector}_mediator/*_positron_spec.pyx`
(deleted), `hazma/vector_mediator/vector_mediator_positron_spec.pyi`
(deleted), `setup.py` (neither mediator package builds an extension now),
`hazma/spectra/{_photon,_positron}/{_muon,_pion}.pyx` (the survivor
comments, which claimed cimporters this task removed).

**Tests.** `test/test_core_mediator_positron.py` (new),
`test/parity/{cases,tolerances,test_parity}.py`,
`test/parity/oracles/entry_points.py`,
`test/test_core_{mediator_tables,positron_pion,scalar_xs}.py`,
`test/test_theory_aggregation.py`.

**Docs.** `docs/followups/done/positron-spectrum-nan-at-legacy-electron-mass.md`
(moved from `todo/`, with its resolution),
`docs/followups/todo/mediator-positron-line-misses-the-electron-velocity.md`
(new), `docs/followups/todo/oracle-restore-revisions-for-the-mediator-decay-pyx.md`
(widened), `docs/followups/README.md`, and the thirteen inbound links to
the moved follow-up.

## Verification

- `scripts/agents/preflight.sh --paths <12 .py> --md <16 .md>` —
  **ten of eleven gates PASS**; `ruff check` is the one FAIL, at
  **38 findings against the trunk's 38** over the same twelve files
  (measured side by side against a stashed trunk, per-file). This change
  adds none. That gate is red on unmodified trunk and is tracked as
  [`preflight-isort-ruff-red-on-trunk`](../../../../docs/followups/todo/preflight-isort-ruff-red-on-trunk.md);
  CI's own ruff step (`--isolated --select E9,F63,F7,F82`) passes.
  Scope the `--paths` argument to `.py` files that still exist — feeding
  it deleted paths and `.pyx`/`.md` turns black and isort red for reasons
  that are not the diff's.
- `cargo fmt --manifest-path rust/Cargo.toml --check` — clean.
- `cargo clippy --manifest-path rust/Cargo.toml --all-targets -- -D warnings`
  — clean.
- `cargo test --manifest-path rust/Cargo.toml --no-default-features` —
  **258 passed** (249 at Task 6.2's close; nine new, covering the FMA
  premise and clamp, the mode dispatch, the lazy `pws` reads and the
  quad options).
- `pytest test/parity -q` — **658 passed, 1 skipped**, with all four
  positron budgets tightened.
- `pytest -q` (the full suite) — **2,389 passed, 15 skipped, 12 subtests
  passed** (2,262 at Task 6.2's close; +126 from the new test module and
  +1 from `test_core_positron_pion.py`'s new cimporter check).
- `pytest test/test_core_mediator_positron.py -q` — **126 passed**:
  reference agreement (56), error paths (24), physics (18), dispatch
  wiring (12), the threshold singularity (12), the twins gone (4).
- `pytest test/test_theory_aggregation.py -q` — **69 passed**, the
  model-layer identities the positron spectra feed.
- `find hazma -name "*.pyx" -o -name "*.pxd"` — **13** (was 15); the two
  deleted are the last in either mediator package.
- **Test validity, measured rather than asserted.** Removing the clamp
  from `momentum` and rebuilding turns
  `pytest test/test_core_mediator_positron.py` to **8 failed, 118
  passed** — the five `test_the_legacy_electron_mass_is_finite`
  parametrisations that reach the integral (`total`, `mu mu`, `pi pi`
  across both models; `e e` short-circuits and stays green, which is
  itself the `.pyx`'s behaviour) plus both
  `test_no_energy_near_the_threshold_is_a_nan`. So the class gates the
  change rather than describing it.
  Two more carry their own guards: `the_momentum_keeps_the_compilers_fused_spelling`
  asserts the fused and unfused spellings actually differ at its chosen
  energy before comparing, and `test_the_two_models_agree_bit_for_bit`
  fails the moment either model is given its own kernel.

## Numerical impact

All four entry points move; the full record is in
[`../numerical-impact.md`](../numerical-impact.md) under Task 6.3.
Summary: worst **2.3319e-12** relative (scalar) and **1.5037e-12**
(vector) against the pinned corpus, both above rule 3's threshold and
both declared; four budgets tightened from `NESTED_RTOL` to
`PORTED_NESTED_RTOL`, none widened; and one value deliberately moved,
`NaN → 0.0` at exactly the legacy `m_e`.

## Plan Impact

**Impact Level:** Phase file patched (a stale exit-criterion sentence),
plus one canonical-contract correction outside this task's own text.

- `phases/phase-04-spectra-kernels.md`'s capi-survivor exception said the
  four extensions' capsules "keep the still-Cython mediator modules
  importable". There are no still-Cython mediator modules after this
  task, so the sentence is put in the past tense and the paragraph now
  records that its own release condition — "once the last cimporter is
  gone" — is met.
- `phases/phase-06-mediator-spectra.md` Task 6.4's exit criteria are
  unchanged and still correct; the `rg` sweep it prescribes is simply
  already empty, which the phase README records so 6.4 does not
  re-derive it.
- No ADR. The threshold repair is an existing follow-up's own option,
  taken and recorded; the `_core` rename is private to the extension and
  changes no public import path.

## Stale-state sweep

Run against this branch, after every prose edit was frozen.

### Identifier sweep

| Identifier | Command | Result |
| --- | --- | --- |
| `scalar_mediator_positron_spec`, `vector_mediator_positron_spec` | `grep -rn ... --include='*.py' --include='*.pyx' --include='*.pxd' --include='*.toml' --include='*.yml' --include='*.rst' .` | 34 hits, **all intended**. No importer: the two wrapper hits are the comments recording the deletion; `PORTED_ENTRY_POINTS` (4) and `oracles/entry_points.py` (4) record the `.pyx` *origin* by contract; `tolerances.py` (4) cites the line the budget describes; the rest are test assertions that the files are gone. The two `hazma/experimental/` hits are unrelated modules whose names merely rhyme. |
| `dnde_positron_decay_*` (new) | `grep -rn dnde_positron_decay projects/ docs/ hazma/ test/ rust/` | 43 — kernel, both PyO3 modules, both wrappers, `CORE_RENAMES`, the roster test, the new test module, the task note and both READMEs. |
| `mediator_decay_positron` (new) | same shape | 41 — the module, its registrations, and every doc that names it. |
| `CORE_RENAMES` (new) | `grep -rln CORE_RENAMES` | 5 files: `cases.py` (defined), `test_parity.py` (the only reader), `test_core_scalar_xs.py` and both project READMEs (documented). |
| survivor cimports | `grep -rn "_{photon,positron}._{muon,pion} cimport" hazma/` | **2** — `_photon/_pion.pyx:9` and `_positron/_pion.pyx:9`, each on its own `_muon` twin. **No external cimporter remains** (KEPT: this is Task 6.4's precondition). |

### Line-number citation sweep

```text
$ scripts/agents/check_doc_citations.py <the 16 touched .md>
docs scanned: 16
in-repo citations checked: 33
  resolved by exact: 24
  resolved by suffix: 9
external citations skipped: 14
out-of-range or ambiguous: NONE
```

Eleven of the fourteen "external" entries are `.pyx` this project has
deleted, which is
[`citation-checker-skips-deleted-inrepo-files`](../../../../docs/followups/todo/citation-checker-skips-deleted-inrepo-files.md),
not a defect here.

### Forward-looking phrase sweep

```text
$ grep -rnE "(Task 6\.3 will|Task 6\.3 owes|Task 6\.3 should|Task 6\.3 does|Task 6\.3 deletes)" projects/ docs/ test/ hazma/
projects/cython-to-rust/task-notes/README.md:575           EDITED
projects/cython-to-rust/task-notes/phase-06/README.md:255  EDITED
test/parity/cases.py:1153                                  EDITED
projects/cython-to-rust/task-notes/phase-06/task-6.2-decay-spectra.md:344,352,456  KEPT
```

The three live claims were wrong once this task landed and were
rewritten: the roster follow-up was widened rather than discharged, the
A4 question was answered, and all seven mediator cases now resolve
through their wrapper. `task-6.2`'s three are a **closed** task note's
record of what was true when it was written, and are left as history:
they are statements about what that task expected, not paths a reader
follows.

### Moved-follow-up path sweep

`positron-spectrum-nan-at-legacy-electron-mass.md` moved from `todo/` to
`done/` in this task, so every inbound reference had to move with it —
`docs/workflow.md:291` requires `rg -l '<slug>\.md'` and an update to
**every** reference, with no exception for a pasted transcript or a
creation record.

The first command below searches the *slug* rather than the old
directory, so it catches a reference in either direction and does not
itself plant the stale path it is looking for. That is not a nicety: the
first draft of this block wrote out the `todo/` path in prose and in the
search pattern, and tripped the second command.

```text
$ rg -n 'positron-spectrum-nan-at-legacy-electron-mass' projects/ docs/ hazma/ test/ rust/ \
    | rg -v 'done/positron-spectrum-nan'
(no occurrences outside this note)

$ for p in $(rg -oN --no-filename 'docs/followups/(todo|done)/[a-z0-9-]+\.md' \
      projects/ docs/ hazma/ test/ rust/ README.md CHANGELOG.md | sort -u); do
    [ -f "$p" ] || echo "DANGLING: $p"
  done
DANGLING: docs/followups/todo/cross-section-prefactor-threshold-cancellation.md
DANGLING: docs/followups/todo/legacy-parameters-width-exponent-bug.md
```

Seventeen references were repointed: thirteen when the file moved, and
four more after review caught them — two creation records
(`phase-01/task-1.4-legacy-npy.md:262`, `phase-01/README.md:419`) and two
inside pasted command output (`task-1.4-legacy-npy.md:456`,
`phase-03/task-3.1-constants.md:408`). The first pass skipped those four
on the theory that a transcript is frozen evidence; `docs/workflow.md`
says otherwise, and the two transcripts now carry the current path with a
bracketed note recording what the command saw when it ran. The
`task-3.1` line also dropped its line numbers, which this task's rewrite
of the follow-up had invalidated.

The two remaining `DANGLING` slugs are older and belong to Phase 00 task
notes, not to this diff; they are the same class and are filed as
[`moved-followups-leave-dangling-inbound-paths`](../../../../docs/followups/todo/moved-followups-leave-dangling-inbound-paths.md).

### Count sweep

| Claim location | Command | Actual | Status |
| --- | --- | --- | --- |
| note, phase README: `258 passed` | `cargo test --manifest-path rust/Cargo.toml --no-default-features` | `258 passed` | OK |
| note, phase README: `2,389 passed, 15 skipped` | `pytest -q` | `2389 passed, 15 skipped, 12 subtests passed` | OK |
| note, phase README: `658 passed, 1 skipped` | `pytest test/parity -q` | `658 passed, 1 skipped` | OK |
| note: `126 passed` and its per-class split | `pytest test/test_core_mediator_positron.py --collect-only -q` | 126: reference 56, errors 24, physics 18, dispatch 12, threshold 12, twins 4 | OK |
| note, phase README: `69 passed` | `pytest test/test_theory_aggregation.py -q` | `69 passed` | OK |
| note, phase README: `13` Cython sources | `find hazma -name "*.pyx" -o -name "*.pxd" \| wc -l` | `13` | OK |
| note, phase README: neither mediator package builds one | `find hazma/{scalar,vector}_mediator -name "*.pyx" -o -name "*.pxd" -o -name "*.pyi" \| wc -l` | `0` | OK |
| `tolerances.py`: no case holds either opening figure | `grep -c "rtol=NESTED_RTOL"` / `"rtol=QUAD_RTOL"` | `0` / `0` | OK |
| note: four budgets tightened | `grep -c "rtol=PORTED_NESTED_RTOL"` | `9` (5 before this task + 4) | OK |
| note: 2.3319e-12 / 1.5037e-12, 13,403 / 13,684 of 16,740 | replay of every corpus block against its stored `.npz` | as claimed | OK |
| note: defect A4 moves 5,237 of 16,740 by 3.7421e-04 | `test/parity/oracles/data/manifest.json`, `defects.A4.diff_against_corpus` | as claimed, identical in all four cases | OK |
| note: the shift is `R_FACTOR**2 - 1` | `1.0001870858234163**2 - 1` | `3.7420664794e-04` vs recorded `3.7420665021e-04` | **EDITED** — the claim said "eleven digits" in three places; it is eight, and all three now say so |
| note: benchmark 32.3x / 42.8x / 43.3x | release builds of both sides, run from `/tmp` | as claimed | OK |
| ruff findings added by this change | per-file `ruff check` over the 12 touched `.py`, branch vs a stashed trunk | `38` vs `38` | OK |

### Numerical-impact statement

Every corpus block of all four entry points was replayed against its
stored `.npz` — 16,740 values per entry point, over three mediator
masses x five parent energies x four modes. **Values moved:** 3,337 of
16,740 on the scalar pair (worst 2.3319e-12 relative) and 3,056 on the
vector pair (worst 1.5037e-12), both above rule 3's 1e-12 threshold and
both recorded in [`../numerical-impact.md`](../numerical-impact.md).
**One value moved deliberately:** `NaN` to `0.0` at exactly the legacy
`m_e`, which the corpus does not pin. No other public function is
reachable from this diff: `pytest test/test_theory_aggregation.py`
(69 passed) covers the model-layer identities these spectra feed.

### Exit Criteria -> test mapping

| Exit criterion | Satisfied by |
| --- | --- |
| Both pairs on Rust | `rust/src/kernels/mediator_decay_positron.rs`; `test_core_mediator_positron.py::TestAgainstAnIndependentReference` (56) |
| Corpus green within budget | `pytest test/parity` — 658 passed, 1 skipped |
| Budget changes justified in `tolerances.py` and here | the four `why=` strings, each carrying its own measurement; the count sweep above |
| Wrappers swapped, twins deleted in the same change | `TestTheCythonTwinsAreGone` (4), and the identifier sweep's "no importer" row |
| Shifts beyond 1e-12 declared | `../numerical-impact.md`, Task 6.3 section |
| Every `.pyx` edge guard survives explicitly | `TestErrorPaths` (24) and `TestTheThresholdSingularity` (12); the `eng_m < mass`, `eng_p < m_e`, line-window and lazy-`pws` guards each have a test |

### Task-note self-consistency

`**Status:** Complete` matches the phase README's Tasks-table cell and
the Exit Criteria mapping above. Every file named in §Files Changed
appears in `git diff --cached --name-only` (38 paths). Note length 442
lines, inside ADR-0002's ~500 budget.

## Handoff to Next Task

**Task 6.4 (retire the capi survivors and `_utils` headers) is next**,
and its ground is already clear. Read `../../PLAN.md`, `../README.md`,
this phase's `README.md`, then the phase file.

**Now safe to assume:**

- **6.4's `rg` sweep is already empty.** After this task
  `hazma/spectra/{_photon,_positron}/_pion.pyx` cimport only their own
  `_muon` twins and `hazma/_utils/boost`; nothing else in the tree reads
  any of the four. The stale comments that claimed otherwise — in all
  four `.pyx`, in `test/parity/oracles/entry_points.py` and in
  `test/test_core_positron_pion.py` — were corrected here, so what 6.4
  reads is current.
- **Both mediator packages are Cython-free.** `setup.py` builds no
  extension for either, and
  `test_core_mediator_positron.py::TestTheCythonTwinsAreGone` asserts it.
  Thirteen `.pyx`/`.pxd` remain, all of them 6.4's.
- **Every mode oracle is now transcription.** No `.pyx` spells a mode
  string or a dispatch message; `test_core_mediator_tables.py` and
  `cython_dispatch_messages()` carry the provenance instead.
- **`test/parity/oracles` needs no re-capture.** Two of the four defect
  patches (A3, A4) reach the mediator spectra and their arrays are
  committed; nothing changes unless a patch does. 6.4 should say so
  explicitly when it deletes the last `.pyx`.

**Still owed:**

- **The restore-revision follow-up is still open**, now covering all
  seven `mediator_spectra.*` cases —
  [`oracle-restore-revisions-for-the-mediator-decay-pyx`](../../../../docs/followups/todo/oracle-restore-revisions-for-the-mediator-decay-pyx.md).
  Neither 6.2 nor 6.3 could discharge it, for the same reason: the SHA a
  re-capture needs is the parent of the commit carrying the deletion.
  **6.4 can**, because by then both are merged — and 6.4 is also where it
  becomes moot, so decide rather than defer.
- **A new follow-up from this task**, not blocking:
  [the line's missing electron velocity](../../../../docs/followups/todo/mediator-positron-line-misses-the-electron-velocity.md).
  It moves published numbers above the budgets these four cases now
  hold, so it needs a corpus re-capture or a declared exception — which
  makes it a *post*-6.4 item, not a Phase 06 one.

**Still risky:**

- **Deleting a `.pyx` does not make its module unimportable.** The built
  `.so` and generated `.c` sit beside the source, are gitignored, and
  survive `git rm`. This task used that deliberately — both twins stayed
  callable for the drift measurement after `git rm` — and then removed
  them by hand. 6.4 deletes four at once: `rm` the orphans, or the next
  `pip install -e .` measures extensions nothing builds.
- **When a Python replica of a `cdef` disagrees with the extension,
  suspect the compiler.** This task lost four rounds to that before
  instrumenting the `.pyx` itself. `cargo`-side and Python-side replicas
  are both unfused; the shipped C is not.
