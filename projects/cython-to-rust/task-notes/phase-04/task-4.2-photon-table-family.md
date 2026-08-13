# Task 4.2: the photon table family (`_kaon`, `_eta`, `_omega`, `_eta_prime`, `_phi`)

**Date:** 2026-08-12
**Project:** cython-to-rust
**Status:** Complete
**Plan References:** `../../phases/phase-04-spectra-kernels.md` (Goal — the
eight-step swap recipe and the capi-survivor exception; Part 2; Task 4.2),
`../../rules.md` rules 1–4 and 9, `../../PLAN.md` "Numerical impact"
**Related ADRs:** ADR-0001 (framework), ADR-0002 (license-clean numerics)
**Depends On:** Task 4.1 (the per-kernel swap template), Phase 03 (interp,
boost, dispatch, constants)

## Objective

Port the five tabulated photon `.pyx` modules — seven public entry points
between them — to one shared Rust implementation parameterised by
(embedded table, parent mass, line terms); swap the wrapper; delete the
Cython outright.

## Exit Criteria

Copied from the phase file's Task 4.2 block:

- One shared Rust implementation parameterized by (embedded table, mass,
  delta terms); the 7 CSVs under `spectra/_photon/data/` embedded via
  `include_str!` parsed once at init (CSVs stay in-repo as source of
  truth).
- 7 entry points swapped, corpus-green; Cython twins + their ~170 lines
  of commented-out dead code gone.
- Import-time file I/O for these modules eliminated (note the
  package-data globs to retire in Phase 07).

## Inputs Reviewed

- `../../PLAN.md`; `../README.md`; `README.md` (this phase's working
  memory, including the two "currently risky" bullets that name this
  task); `../../phases/phase-04-spectra-kernels.md`; `../../rules.md`.
- `../phase-04/task-4.1-positron-muon.md` — the swap template and its
  test module's shape.
- `../phase-03/task-3.4-interp-boost.md` — the FMA method, the boost
  integral's four reproduced defects, and the `[platform-scoped-oracle]`
  lesson.
- `rust/src/{boost,interp,dispatch,constants}.rs`; `rust/src/kernels.rs`
  and `rust/src/kernels/positron_muon.rs`.
- The five `.pyx` and their `.pxd`/`.pyi` at `origin/master` (665aed5).
- [`docs/followups/todo/parity-corpus-pins-ill-conditioned-points.md`](../../../../docs/followups/todo/parity-corpus-pins-ill-conditioned-points.md)
  — the phase README told this task to resolve or explicitly waive it
  before starting. **Waived, with the reason under "Decisions" below.**
- `docs/agents/lessons.md`, `docs/agents/environment.md`.

## Findings

- **Five near-identical files, and the differences are all data.** The
  kaon module already had one `kaon_interp_spec` helper serving three
  entry points; the other four repeated it with the names changed. What
  actually varies is the table, the parent mass, and a list of
  `(line energy, weight)` pairs — so the port is one `dnde` over seven
  `Spectrum` statics. Writing it that way is also what surfaced the two
  defects below: they are invisible one file at a time and obvious once
  the five weight expressions sit in one column.
- **Eight FMA sites across the five shipped objects, one per line term,
  and no others.** `objdump -d` finds exactly one `fmadd` in `_eta.so`
  and `_eta_prime.so`, two in each of `_kaon.so`, `_omega.so` and
  `_phi.so`, all of the form `fmadd d0, d8, d0, d9` — the boosted line
  times its folded weight plus the running total. Everything else in
  these modules is a call into `_utils/boost` or `np.interp`, which are
  *separate extensions* reached through `__pyx_capi__` function pointers
  and therefore not inlined, so their arithmetic is already covered by
  Task 3.4's port. The rest-frame tail `dnde[0] * emin / photon_energy`
  is a multiply then a divide and cannot contract. Written from that map,
  the port was bit-equal on the first build — no bisection round, the
  same outcome as Task 4.1.
- **`numpy.sum(axis=0)` over the CSV's mode columns is pairwise, and one
  table is wide enough to notice.** The Cython built each rest-frame
  spectrum as `np.sum(np.loadtxt(csv).T[1:], axis=0)`. For six of the
  seven tables the reduction is over 2–7 columns, where NumPy's pairwise
  routine degenerates to a sequential fold and a naive sum agrees. The φ
  table has **ten** mode columns, past the eight where the eight-accumulator
  path starts, and there a sequential sum is **not** bit-equal. Reusing
  `boost::pairwise_sum` (Task 3.4 wrote it for `np.trapezoid`) fixes it;
  a mutation to the sequential form fails six tests, all of them φ.
- **A `NaN` photon energy reached a Rust panic, and the Cython raised
  `IndexError`.** With `lb = ub = NaN` every comparison in
  `boost_integrate_linear_interp` is false, so the Cython falls through
  to `np.flatnonzero(lb <= x)[0]` on an empty match — measured on the
  shipped build: `dnde_photon_eta(float('nan'), 1000.0)` raises
  `IndexError: index 0 is out of bounds for axis 0 with size 0`. The Rust
  reached `.expect("lb <= xmax, so some node is at or above it")` and
  produced a `PanicException` (reproduced through the `hazma._core.boost`
  probe, which has had this since Task 3.4). Neither type can be
  reproduced from inside `dispatch::map_unary`, which maps element by
  element and has no per-element error channel — see "Decisions".
- **The port's second and third live 2.1.0 numerical defects.** Both are
  in the line terms, both are reproduced rather than repaired per rule 1,
  and both are filed and blocked behind Phase 06 Task 6.4:
  - `_eta_prime.pyx:107` weights its `η′ → γγ` line with `BR` where its
    four two-photon siblings use `2·BR`, so the mode contributes
    **0.02307 photons per decay instead of 0.04614** — 0.63% of the η′
    photon yield, missing, all of it at `M_η′/2 = 478.89` MeV. Measured
    by integrating the line term alone (`quad`: η 0.78819993 against
    `2·BR = 0.7882`; η′ 0.02306998 against `BR = 0.02307`). The ω and φ
    weights are *correctly* un-doubled — their modes are `X → Yγ`, one
    photon each — which is what makes the η′ the odd one out rather than
    the family a mixed convention.
    [`eta-prime-two-photon-line-missing-factor-two.md`](../../../../docs/followups/todo/eta-prime-two-photon-line-missing-factor-two.md).
  - `_phi.pyx:111,113` place both photon lines at `(M² + m²)/(2M)`,
    which is the **daughter meson's** energy, not the photon's — 656.94
    MeV where 362.52 is right for `φ → ηγ` (×1.81) and **959.65 where
    59.82 is right** for `φ → η′γ` (×16.0, i.e. 94% of the φ's whole rest
    mass in one photon). The ω's two lines use `(M² − m²)/(2M)` and are
    right, which is the control. The local is even named `eng_eta`.
    [`phi-photon-lines-use-the-daughter-meson-energy.md`](../../../../docs/followups/todo/phi-photon-lines-use-the-daughter-meson-energy.md).
- **Deleting an extension strands whatever read its module globals.**
  `test/test_core_interp.py` and `test/test_core_boost.py` (Task 3.4)
  both built their seven-table fixtures from `_eta.eta_data_energies` and
  friends. Both went red at *collection*, so the full suite reported two
  errors and nothing else — a failure mode that looks like a broken
  build. Repaired by loading the CSVs the way the deleted modules did
  (`np.loadtxt(...).T`, `np.sum(rows[1:], axis=0)`), which also makes
  those oracles independent of the Rust that now consumes them. Same
  class as Task 0.2's "a stranded dependent belongs to the task that
  strands it", one level down: not an import of a deleted *module* but a
  read of a deleted module's *state*.
- **A monkeypatch that shadows a real submodule stops measuring a
  delta.** `test/parity/test_parity.py`'s two served-kernel meta-tests
  patched `hazma._core.photon` and `hazma._core.positron` with fakes and
  asserted against `baseline + 1` and `[]`. Filling `photon` with seven
  real kernels made the fake *replace* seven and add one. Repointed at
  `hazma._core.not_a_real_domain`, a name no domain will take, so the
  delta stays a delta as Phases 04–06 fill the rest.
- **The Rust and Python halves of a kernel port do not accept the same
  physics notation** (Task 4.2). `rust/src/kernels/photon_tables.rs`
  writes `η′ → γγ` and `(M² − m²)/(2M)` freely; the same strings in a
  Python docstring produce 22 `RUF002` "ambiguous unicode" findings,
  because ruff reads `γ` as a Latin `y`, `′` as a backtick, `−` as a
  hyphen and `×` as an `x`. Every other `test/test_core_*.py` is clean, so
  this is a rule the suite already follows silently and a new module has
  to learn: **spell final states the way hazma's own CSV headers do**
  (`a` for a photon — `a_a`, `pi0_a`, `eta_a`), and use ASCII `-`, `x` and
  `'`. `η`, `φ`, `ω`, `β`, `δ`, `→`, `·` and superscripts are *not*
  flagged, so the notation stays readable. Three `PLR2004` magic-value
  comparisons and one missing return annotation came with it — all four
  worth fixing rather than silencing.

## Decisions and Implementation Notes

- **The ill-conditioned-corpus follow-up is explicitly waived for this
  task**, not resolved. The phase README flagged that Task 4.2 is the
  first to meet one of the six blocks (`spectra.photon.eta`
  `[boosted_strong]`). The waiver rests on a measurement rather than on
  optimism: the port is **bit-equal to the Cython at all 336,000 sampled
  points**, so on the capturing platform there is nothing for a
  conditioning budget to absorb, and off it the parity suite does not run
  at all (`.github/workflows/ci.yml` passes `--ignore=test/parity`). The
  follow-up's own text says the block "will produce a false failure the
  moment a Rust implementation lands" — that prediction is now refuted
  for this family, which is worth recording there when it is next
  touched. It stays open for the five cross-section blocks and for the
  day the CI scoping is lifted.
- **A `NaN` photon energy now returns `NaN` instead of raising
  `IndexError`.** Declared, in `rust/src/boost.rs`'s "Faithfulness
  notes", in the numerical-impact log, and in
  `test/test_core_photon_tables.py`. The alternatives were both worse:
  reproducing the `IndexError` from inside `map_unary` is impossible
  element-wise, and leaving the `.expect` turns a catchable Python
  exception into a `PanicException`. `NaN` is what the same kernels'
  *rest-frame* branch already returns (`np.interp` propagates), so the
  change makes one kernel self-consistent rather than inventing a
  convention. The parity corpus samples no `NaN` abscissa, so no pinned
  value moves.
- **The parent-energy branch is resolved once per call, not per point.**
  `photon_tables::branch(parent_energy, mass)` returns
  `BelowThreshold | RestFrame | InFlight { beta }`, and `dnde` takes it.
  Two reasons: it is the same arithmetic on the same inputs (the Cython
  recomputed `beta` per point and got the identical double), and it gives
  the `0 < beta < 1` guard somewhere to fail *before* any element is
  evaluated — which is what lets `crate::photon` raise a `ValueError` for
  a `NaN` parent energy while `map_unary` stays infallible per element.
- **`boost::pairwise_sum` became `pub(crate)`** rather than being copied.
  Two reproductions of a NumPy implementation detail would drift; one
  with two callers and a docstring naming both does not.
- **The tables are `LazyLock<Spectrum>` statics, parsed on first use.**
  The Cython parsed all seven at *import*, so `import hazma.spectra` paid
  seven file reads and seven `np.loadtxt` calls whether or not a spectrum
  was ever evaluated. The CSVs stay in the repository: they are the
  source of truth, `test/parity/generate.py` hashes them into the corpus
  kernel digest, and `test/test_core_photon_tables.py` re-parses them
  with NumPy as its oracle. **The `hazma.spectra._photon.data`
  package-data glob and the matching `MANIFEST.in` line are now unread at
  runtime** — Phase 07 Task 7.1 should retire them along with the
  `hazma.spectra._photon` `*.pyx` patterns, and not before, since the
  sdist must still carry the CSVs for `include_str!` to find at build
  time.
- **`hazma/spectra/_photon/path.py` went with the five modules.** It
  existed only to give them `DATA_DIR`; nothing else imported it
  (checked). Callerless after the swap, so deleted in the same PR — the
  same call Task 0.2 made for the `electron` helper.
- **This module's test has one comparison mode where Task 4.1 has two**,
  and the departure is deliberate. Task 4.1's twin survives as a capi
  provider and could be called bit-for-bit; these five are deleted, so
  there is no twin to call. The replacement oracle is a Python reference
  built from the CSVs plus the `hazma._core.{boost,interp}` probes, with
  the one fused multiply-add reproduced by a `Fraction`-based `fma`. That
  comparison is exact arithmetic on both sides and needs no platform
  scoping. It does not re-test the foundation — `test/test_core_boost.py`
  and `test/test_core_interp.py` do, against the Cython twin and NumPy —
  it tests the wiring and the tables, which is what this task wrote.
- **The `TABULATED` budget class is kept rather than tightened to
  `EXACT`.** The port reproduces the capturing platform bit-for-bit, so
  `EXACT` would pass today. It would be the wrong contract: unlike
  `spectra.positron.muon`, bit-equality here rests on reproducing
  *NumPy's summation order*, an implementation detail a future NumPy may
  change. The `why` strings on the charged-kaon and η entries now record
  both facts.
- **The wrapper's eleven pre-existing ruff findings were fixed rather
  than measured around.** `ruff check` on
  `hazma/spectra/_photon/__init__.py` reported the same 11 `D205` / `D412`
  / `D400` docstring-style findings on `origin/master` as on this branch —
  zero delta, and the situation
  [`preflight-isort-ruff-red-on-trunk.md`](../../../../docs/followups/todo/preflight-isort-ruff-red-on-trunk.md)
  describes. But preflight FAILs on the absolute count, not on a delta, so
  "no delta" is not a green gate. They are mechanical (a numpydoc summary
  is one line; a section header takes no blank line; a summary ends with a
  period) and they are in the file this task already edits, so fixing them
  was cheaper than arguing. **CI's ruff step is unaffected either way** —
  `--isolated --select E9,F63,F7,F82 --exclude hazma/experimental` reports
  the same two `F821`s in `hazma/experimental/` on both trees, which that
  exclusion drops.
- **Three docstrings named an argument the signature does not have.**
  `dnde_photon_{charged,short,long}_kaon` take `photon_energy`; their
  `Parameters` blocks said `photon_energies`, and the short- and long-kaon
  blocks both called their argument the "Charged kaon energy" — copies
  from the charged-kaon block above them. Corrected, along with adding the
  units sentence every one of the seven now carries (`AGENTS.md`:
  "units stated for every physical quantity").

## Files Changed

- New: `rust/src/kernels/photon_tables.rs`,
  `test/test_core_photon_tables.py`,
  `docs/followups/todo/eta-prime-two-photon-line-missing-factor-two.md`,
  `docs/followups/todo/phi-photon-lines-use-the-daughter-meson-energy.md`.
- Changed: `rust/src/{kernels,photon,boost,interp}.rs`,
  `hazma/spectra/_photon/__init__.py`, `hazma/_core.pyi`, `setup.py`,
  `test/parity/{cases,tolerances,test_parity}.py`,
  `test/test_core_{boost,interp}.py`, `docs/followups/README.md`,
  `docs/followups/todo/{boost-integral-drops-last-interior-cell,positron-muon-spectrum-normalization-inverted}.md`.
- Deleted: `hazma/spectra/_photon/{_eta,_eta_prime,_kaon,_omega,_phi}.{pyx,pxd,pyi}`
  (15 files) and `hazma/spectra/_photon/path.py`.

## Verification

Environment: CPython 3.12.12, NumPy 2.5.1, SciPy 1.18.0, Cython 3.2.9,
macOS/arm64 — the parity corpus's capturing environment, built with
`uv pip install -e . --no-build-isolation` into the worktree (confirmed:
`hazma.__file__` and `hazma._core.__file__` both resolve inside it).

- `cargo test --manifest-path rust/Cargo.toml --no-default-features` —
  `test result: ok. 96 passed; 0 failed; 0 ignored`. 15 of those are
  `kernels::photon_tables::tests`: the folded constants against the
  disassembled immediates, the parsed tables against NumPy's bit
  patterns, the row counts, the three branch boundaries, the `NaN`
  parent-energy guard, the below-threshold zeros, the rest-frame tails,
  each line's boosted plateau, and the two reproduced defects.
- `cargo fmt --check` and `cargo clippy --all-targets -- -D warnings` —
  clean. One `#[allow(clippy::float_equality_without_abs)]` on `branch`,
  for the same reason Task 4.1 needed it on `dnde_positron_muon`: the
  comparison is a genuine one-sided threshold, not a disguised equality.
- `pytest -q test/test_core_photon_tables.py` — `184 passed, 1 skipped`.
  The skip is the charged kaon in the per-line photon-count test, which
  has no monochromatic line. Coverage by class: **12 dispatch-wiring
  assertions × 7 entry points** (scalar / NumPy scalar / 0-d array / 1-D
  array / scalar-array agreement / sequence / empty grid / rank error /
  the verbatim assert message / dtype error / type error / keyword
  arguments), plus one that the public wrapper delegates to the
  extension; **7 reference comparisons × 5 parent energies** plus the
  below-threshold and at-every-table-node cases; and **9 physics
  statements** (threshold, boosted endpoint, `1/E` tail, `NaN` parent,
  `NaN` photon, per-line photon count, the two reproduced defects, and
  the Task 3.4 boost divergence seen through a public entry point).
- `pytest -q test/parity` — `629 passed, 1 skipped`, with all seven
  tabulated cases green at `TABULATED` (1e-12) and the corpus's own
  mode reported as budget mode (`hazma._core serves 8 kernel(s)`, and a
  kernel-digest change from the five deleted `.pyx`).
- `python test/parity/generate.py --check` —
  `corpus OK: 41 cases / 1580 arrays match the manifest`. The stored
  corpus is untouched.
- `pytest -q test/test_theory_aggregation.py` — the model-layer identity
  gate Task 1.4 built for exactly this risk (a swap that repoints a
  kernel but drops a branching-fraction weight passes the corpus).
- Bare `pytest -q` — the full suite:
  **`1628 passed, 15 skipped, 5 warnings in 587.90s`**. Collection goes
  1458 → 1643 against `origin/master` (measured with
  `pytest --collect-only -q` on both trees), **+185 and all of them
  `test/test_core_photon_tables.py`** — no other module gains or loses a
  test. The 15th skip is the charged kaon in the per-line photon-count
  test; the 14 inherited skips are unchanged, including
  `test_running_on_the_capturing_tree`, which has skipped since Task 4.1
  put the corpus in budget mode.
- **Mutation checks (the tests are not vacuous).** Two production
  mutations, each rebuilt and re-run:
  - `boost::pairwise_sum(&components)` → a sequential fold: **6 failures**,
    every one of them φ (`test_the_embedded_table_is_the_csv_numpy_reads[phi]`
    and the five `test_the_kernel_reproduces_the_reference[*-phi]`), and
    none on the other six tables — exactly the predicted signature.
  - `delta.mul_add(weight, result)` → `result += weight * delta`:
    **15 failures** across all six spectra that carry a line, at every
    parent energy above rest.
- `scripts/agents/preflight.sh --paths <the nine changed Python files>
  --md <the eight changed markdown files>` — **RESULT: PASS**, every gate
  green: black, isort, ruff, the three cargo gates, `pytest` (the
  1628/15 above, re-run at 596.33s), the import smoke, markdownlint, and
  the forbidden-token scan. Two earlier runs failed and are worth
  recording rather than hiding: the first on black plus 29 ruff findings
  in the new test module, the second on the wrapper's 11 pre-existing
  ruff findings (see "Decisions").
- **Deferred:** nothing. The two defects found are filed rather than
  fixed, which rule 1 requires.

## Numerical impact

**No public value changes** (measured, not argued).

- **Kernel level, against the Cython being replaced, before it was
  deleted.** All seven entry points × six parent energies (`E = M`,
  `M(1+1e-12)`, `1.05 M`, `2 M`, `10 M`, `1000 M`) × 8,000 photon
  energies each — half log-spaced and half log-uniform random over
  `[1e-5 M, 100 E]`, so the below-table tail, the interpolated interior,
  the boosted window and the hard zero above it are all sampled.
  **336,000 points, 0 bitwise mismatches, max relative deviation
  0.000e+00.** This is the strongest statement available and it is the
  one this family gets, because the twins do not survive the PR.
- **Public-API level, across the whole diff.** `origin/master` (665aed5)
  built in a scratch worktree with the same pinned environment, and the
  same script run on both trees: 12 `dnde_photon_*` × 4 parent energies,
  2 `dnde_positron_*` and 2 `dnde_neutrino_*` × 3 each, plus both models'
  `spectra()`, `positron_spectra()`, `annihilation_cross_sections()` and
  `thermal_cross_section()` — **97 arrays / 18,694 values, bit-for-bit
  identical** (0 differing bit patterns). Nothing above the compiled
  boundary is touched by this diff and the measurement says so.
- **One declared behavior change, at `NaN` only.**
  `dnde_photon_{charged,long,short}_kaon`, `dnde_photon_eta`,
  `dnde_photon_eta_prime`, `dnde_photon_omega` and `dnde_photon_phi` with
  a `NaN` **photon** energy and a parent in flight returned `IndexError`
  and now return `NaN`; with a `NaN` **parent** energy they raised
  `AssertionError` and now raise `ValueError` (rule 9, the same
  tightening the whole port declares once). No finite input moves. This
  belongs in the Phase 07 CHANGELOG's behavior-change list, not its
  numerical one.

## Open Questions

- **Does the φ also omit a `φ → π⁰γ` line entirely?** `constants.pxd:283`
  defines `BR_PHI_TO_PI0_A = 1.32e-3`, nothing reads it (checked against
  `origin/master` as well as the current tree), and the ω adds exactly
  the analogous line for its own `π⁰γ` mode. The φ's `pi0_a` CSV column
  integrates to 0.002612 against `2 · BR(π⁰ → γγ) · BR(φ → π⁰γ) =
  0.002609` for the π⁰'s own decay photons — suggestive but not
  conclusive, since the tables are truncated at low energy. Recorded as
  the first open question on
  [`phi-photon-lines-use-the-daughter-meson-energy.md`](../../../../docs/followups/todo/phi-photon-lines-use-the-daughter-meson-energy.md);
  whoever repairs the line energies should settle it in the same PR.
- **Retiring the photon package-data glob** is Phase 07 Task 7.1's, and
  is now *half* true: the CSVs are unread at runtime but must still ship
  in the sdist for `include_str!`. Noted in the decisions above so that
  task does not remove the wrong one.

## Plan Impact

**Impact Level:** None.

The canonical contract is unchanged. The phase file's Task 4.2 block, its
Goal (the eight-step recipe and the capi-survivor exception, which names
`_photon/_muon` and `_photon/_pion` but not these five), its Part 2
heading and the phase Exit Criteria all read correctly against what
shipped — checked sentence by sentence. `rules.md` rules 1–4 and 9 were
followed rather than amended: the two defects are reproduced and filed,
the constants are bit-parity, and the one `assert` in the live path
became a `ValueError`. `PLAN.md`'s one-line Phase 04 summary ("16 entry
points swapped; twins deleted (4 capi survivors defer to 06)") is still
accurate.

## Stale-state sweep

Run against this branch
(`claude/cython-to-rust/task-4.2-photon-table-family-kaon-eta-omega-eta-phi`)
after Step 8's bookkeeping.

### Identifier sweep

Every name the deleted modules exported, and the module paths themselves,
across code, tests and docs:

```console
$ rg -n 'hazma\.spectra\._photon\.(_eta|_eta_prime|_kaon|_omega|_phi|path)' \
    --glob '!.venv' --glob '!rust/target' .
./test/parity/data/manifest.json      7 hits ("entry_point": ...)
./test/parity/cases.py:1380,1384,1388,1391,1393,1396,1397
./projects/cython-to-rust/task-notes/phase-01/task-1.1-corpus-generator.md:265,266
```

Three populations, all correct as they stand:

- the seven `cases.py` hits are the `PORTED_ENTRY_POINTS` rows this task
  added, which record a pinned value's `.pyx` origin on purpose —
  `assert_full_coverage` reads them and fails if such a module still
  exports its `def`;
- the seven `manifest.json` hits are the corpus's stored provenance for
  values captured from those modules. **The manifest must not be
  edited** (rule 2), and the runner keys everything by case name, never
  by `module:function` — which is why repointing a case disturbs no
  stored data (Task 4.1's finding, re-confirmed here);
- the two Task 1.1 hits are quoted error-message samples in a historical
  task note.

No live import of a deleted module remains.

```console
$ rg -n 'eta_data_|kaon_data_|omega_data_|phi_data_|eta_prime_data_' \
    hazma test --glob '!.venv'
test/test_core_interp.py:148:#: (``_eta.eta_data_energies`` and friends); that task moved the parse
test/test_core_boost.py:466:#: (``_eta.eta_data_energies`` and friends); that task moved the parse
```

Both are the comment this task added, recording where those fixtures used
to come from. No *read* of a deleted module global survives; `DATA_DIR`
now resolves only to `test/parity/`'s own npz directory, to
`cases.PHOTON_DATA_DIR` (which reads the CSVs straight from the tree),
and to the three test modules' own CSV loaders.

### Line-number citation sweep

```console
$ rg -n '_photon/(_eta|_eta_prime|_kaon|_omega|_phi)\.pyx:[0-9]' \
    --glob '!.venv' --glob '!rust/target' .
./test/parity/cases.py:56,310                             (grid-design docstring)
./test/parity/tolerances.py:229,240,246,252               (budget `why` strings)
./docs/followups/todo/eta-prime-two-photon-line-missing-factor-two.md:34-37
./docs/followups/todo/phi-photon-lines-use-the-daughter-meson-energy.md:28-39
./rust/src/interp.rs:22                                   (call-site table)
./rust/src/kernels/photon_tables.rs:519                   (the η′ defect)
./projects/cython-to-rust/task-notes/phase-00/task-0.1-relocate-constants.md:371-376
```

Every one is a citation to the pre-port source, which is the convention
both parity modules state in their own docstrings ("Phases 04-06 delete
the files they point into; the line numbers are historical evidence, not
live references"), and which the Task 0.1 note predates. The two Rust
sites and the two follow-ups were **patched in this task** to say so
explicitly rather than leaving the present tense: `rust/src/interp.rs`
and `rust/src/boost.rs` now name `crate::kernels::photon_tables` as the
live consumer, and each follow-up carries a "quoted from the pre-port
sources" note naming the SHA that recovers them.

### Forward-looking phrase sweep

- `rust/src/photon.rs` — rewritten; the "Empty scaffold. Phase 04 fills
  it with the `dnde_photon_*` kernels currently in
  `hazma/spectra/_photon/*.pyx`" docstring is gone, replaced by a live
  description plus a pointer to Tasks 4.3-4.5.
- `rust/src/kernels.rs:14` — `photon_tables` listed beside
  `positron_muon`, with the one-module-per-`.pyx` convention's exception
  stated rather than left as a silent violation.
- `rust/src/positron.rs:15` — "Phase 04 Task 4.6 adds
  `dnde_positron_charged_pion` beside the muon" is still true and is left
  alone.
- `hazma/_core.pyi:8` — now names `photon` and Task 4.2 beside Task 4.1.
- `test/parity/tolerances.py` — the charged-kaon and η `why` strings
  record the swap and why the class is kept.

### Count sweep

```console
$ ls hazma/spectra/_photon/*.pyx | wc -l
3
$ find hazma -name '*.pyx' | wc -l
15
$ git ls-tree -r --name-only origin/master hazma | grep -c '\.pyx$'
20
```

`setup.py`'s photon list is now `["_muon", "_pion", "_rho"]`, matching
the three `.pyx` on disk; the tree-wide count falls 20 → 15. The phase
file's Exit Criteria ("Remaining Cython under `hazma/spectra/` +
`hazma/_utils/` is exactly the four capi survivors and their headers")
is a Task 4.6 statement, not a Task 4.2 one — three photon modules and
the neutrino trio are still to go.

```console
$ git diff origin/master --numstat -- 'hazma/spectra/_photon/' \
    | awk '{s+=$2} END {print s}'
1037
$ for f in _eta _eta_prime _kaon _omega _phi; do
      git show origin/master:hazma/spectra/_photon/$f.pyx \
        | grep -cE '^\s*#\s*(cdef|@cython|res|return|if |gamma|beta|pre|emin|emax|eng|#)'
  done
28 29 83 32 32          # 204 lines of commented-out dead code
$ cargo test --manifest-path rust/Cargo.toml --no-default-features 2>&1 \
    | grep -c photon_tables
15
```

1,037 deleted lines under `hazma/spectra/_photon/` — 977 of `.pyx`, of
which **204 are the commented-out `quad`-based bodies** the phase file
estimated at "~170", plus 27 of `.pxd`, 7 of `.pyi` and 5 of `path.py`.
The replacement is 873 lines of `rust/src/kernels/photon_tables.rs`, of
which 400 are the `#[cfg(test)]` module (it begins at line 473) — so the
non-test implementation-plus-docs is 473 lines against 977, for five
files' worth of behaviour.

### Numerical-impact statement

`No public value changes` — verified by two independent measurements:
336,000 points against the Cython twins before deletion (0 mismatches),
and 97 arrays / 18,694 values of the public surface against a built
`origin/master` (0 differing bit patterns). Both commands and their
outputs are under "Numerical impact" above. One declared behavior change
at `NaN` inputs only, stated there and in
`rust/src/boost.rs`'s faithfulness notes.

### Exit Criteria → test mapping

| Criterion | Where it is checked |
| --- | --- |
| one shared implementation parameterised by table/mass/lines | `rust/src/kernels/photon_tables.rs` — one `dnde`, seven `Spectrum` statics; `tests::each_line_contributes_its_own_boosted_plateau` exercises the parameterisation per spectrum |
| 7 CSVs embedded via `include_str!`, parsed once at init | the seven `LazyLock<Spectrum>` statics; `tests::the_parsed_tables_are_bit_equal_to_numpys` and `TestAgainstAnIndependentReference::test_the_embedded_table_is_the_csv_numpy_reads` |
| CSVs stay in-repo as source of truth | untouched under `hazma/spectra/_photon/data/`; still hashed by `generate.kernel_digest` and read by `cases.table_edges` |
| 7 entry points swapped | `TestDispatchWiring::test_the_public_wrapper_delegates_to_this_kernel` × 7 |
| corpus-green | `pytest -q test/parity` → 629 passed |
| twins + ~170 lines of commented-out dead code gone | 15 deleted files; `git diff --stat` shows the five `.pyx` at −1,163 lines total, of which the commented-out `quad`-based bodies are ~170 |
| import-time file I/O eliminated | no `np.loadtxt` reachable from `import hazma.spectra`; `path.py` deleted |
| package-data globs noted for Phase 07 | "Decisions" and "Open Questions" above |

### Task-note self-consistency

`**Status:** Complete` here, `Complete (2026-08-12)` in this phase's
`README.md` Tasks table, phase frontmatter still `status: In Progress`
(Tasks 4.3–4.6 remain), `../README.md`'s Phases row updated to name 4.2
as done. `PLAN.md` untouched — no canonical change.

## Handoff to Next Task

**Read first:** `../../phases/phase-04-spectra-kernels.md` (the Goal's
eight-step recipe), then `README.md` in this directory, then this note.
Task 4.3 (`_photon/_muon`, spence) is next and is a **capi survivor** —
delete its `def`, not its file.

**Currently safe to assume:**

- `hazma._core.photon` serves seven kernels and `hazma._core.positron`
  one; `rust/src/photon.rs` is no longer a scaffold and shows the
  registration shape for a kernel with a fixed second argument.
- `boost::pairwise_sum` is `pub(crate)` and reproduces
  `numpy.sum(axis=0)` for any column count; `boost_integrate_linear_interp`
  is now total (a `NaN` window returns `NaN` rather than panicking).
- The two Task 3.4 test modules load their photon tables from the CSVs,
  so deleting further `.pyx` will not strand them again.
- The corpus is in budget mode and **cannot be regenerated**; seven more
  cases now compare a Rust implementation against pre-port pins, all
  bit-equal on this platform.

**Currently risky / unknown:**

- Four blocked defects now share one eventual corpus regeneration
  (positron normalization, the boost integral, the η′ line weight, the φ
  line energies). Do not "fix" any of them in passing — each fails the
  gate that governs the remaining swaps.
- Task 4.5's nested-ρ drift is still the project's numerical stress test.
- The φ's possibly-missing `π⁰γ` line (see Open Questions) is unresolved
  and is *not* reproduced-or-repaired either way — the port carries
  exactly what the Cython carried.
