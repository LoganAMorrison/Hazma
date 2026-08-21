# Phase 04 Learnings: Spectra kernels

Synthesized at phase close (2026-08-20) from the six task notes
([4.1](../task-notes/phase-04/task-4.1-positron-muon.md),
[4.2](../task-notes/phase-04/task-4.2-photon-table-family.md),
[4.3](../task-notes/phase-04/task-4.3-photon-muon.md),
[4.4](../task-notes/phase-04/task-4.4-photon-pion.md),
[4.5](../task-notes/phase-04/task-4.5-photon-rho.md),
[4.6](../task-notes/phase-04/task-4.6-positron-pion-neutrino.md)) and
[`../task-notes/phase-04/README.md`](../task-notes/phase-04/README.md).
Read this instead of the notes; the notes are history.

## 1. Implementation Reality Check

The phase delivered what it promised: **16 public entry points** moved
from Cython to `hazma._core` across six tasks, every swap gated on the
parity corpus, every twin either deleted outright or reduced to its
`cdef` capsules. `hazma/spectra/` now holds **no Cython Python entry
point of any kind** — four `.pyx` survive there for their capsules alone,
read only by the four mediator spectrum modules Phase 06 ports.
`cases.rust_core_kernels()` → 16. Six of six tasks landed, no ADR was
needed, and no exit criterion turned out to be wrong.

Two things the plan did not anticipate, and both are worth carrying:

**The phase found seven live numerical defects in hazma 2.1.0 — one per
task, except 4.2 which found two.** None was found by porting; every one
was found by *writing a statement the original never made* — an analytic
normalization check, a sibling-to-sibling diff, a rest-frame limit, a
forward-cone argument, a continuum subtraction. The roster, in task
order: the positron-muon inverted normalization (4.1), the η′ two-photon
line's missing factor 2 and the φ lines' daughter-meson energies (4.2),
the muon photon spectrum's rest-frame endpoint (4.3), the charged pion's
lost forward cone (4.4), the ρ's rest-frame branch returning its
integrand (4.5), and the charged pion's doubled `π → e ν` neutrino line
(4.6). All are reproduced under rule 1 and all are filed under
`docs/followups/todo/`. Six of the seven are in the
`parity-pinned-defect-repair` project's queue; 4.6's arrived after that
project's roster was fixed and is called out in §5.

**Every task's numerical prediction was wrong, in a different direction
each time.** 4.3 predicted a `5/β` amplification and found one — then
fixed it at the source rather than widening a budget. 4.4 predicted the
same for the pion and found the boost enters as a Jacobian instead. 4.5
was billed as the project's numerical stress test and came in at 1.5e-13,
five decades inside its own class, because a nested integral *averages*
rather than amplifies. 4.6 expected the neutrino muon to need a budget
and got bit-equality at all 3,795 pinned values. **Re-derive; do not
inherit.** Nothing in the six tasks' history supports predicting a
kernel's drift from its shape.

## 2. Critical Context for Future Work

- **The per-kernel swap recipe in the phase file's Goal is eight steps,
  and steps 1 and 5 are the ones that get skipped.** Step 1 —
  disassemble the shipped `.so` and read the FMA sites before writing
  any Rust — is what made 4.1, 4.3, 4.4 and 4.6 bit-equal on the first
  build. Step 5 — repoint the corpus case from the `.pyx` to the wrapper
  and add a `PORTED_ENTRY_POINTS` row — is what keeps the gate from
  going green and vacuous. Phases 05 and 06 copy both.
- **The capi-survivor exception is now fully spent.** Four `.pyx` remain
  under `hazma/spectra/`: `_photon/{_muon,_pion}` and
  `_positron/{_muon,_pion}`, each with its `def` deleted and its `cdef`s
  intact, because the four mediator spectrum modules `cimport` them.
  Phase 06 Task 6.4 is the only place they go.
- **`kernels::{photon_muon, photon_pion, photon_rho, photon_tables,
  positron_muon, positron_pion, neutrino_flavors, neutrino_muon,
  neutrino_pion}` are all `pub` and PyO3-free.** Phase 06's mediator
  spectra boost exactly these, natively, the way the `.pyx` cimport the
  Cython today — `photon_pion`'s four `pub` fns and
  `positron_pion::dnde_positron_charged_pion` most of all.
- **The corpus left bit-equality mode permanently at Task 4.1** and
  cannot be regenerated (`rules.md` rule 2, enforced by
  `assert_no_rust_core`). 19 of 41 cases are still `EXACT` class at
  `rtol = 0` on the capturing platform. **Five budgets were tightened
  across the phase and none was widened**: `PORTED_QUAD_RTOL` (1e-12) for
  `spectra.photon.charged_pion` (4.4), `spectra.positron.charged_pion`
  and `spectra.neutrino.charged_pion` (both 4.6); `PORTED_NESTED_RTOL`
  (1e-9) for both ρ cases (4.5). `QUAD_RTOL`'s two remaining holders are
  the thermal cross sections, which are Phase 05's.
- **Four test-module shapes, and the twin's fate forces the choice.**
  A twin that survives *and* admits bit-equality →
  `test_core_positron_muon.py` (two modes, platform-scoped). A twin that
  survives but is quadrature-backed → `test_core_photon_pion.py`'s
  charged half and `test_core_positron_pion.py` (one measured budget, no
  platform branch: two adaptive integrators are not bit-equal anywhere).
  A twin carrying two oracle classes at two standards →
  `test_core_photon_pion.py` whole. A twin that does **not** survive →
  `test_core_photon_tables.py`, `test_core_photon_rho.py`,
  `test_core_neutrino.py` (an independent Python reference, plus the
  against-the-Cython numbers measured *before* the deletion).
- **`test_core_dispatch.py`'s Cython oracle has moved four times and has
  one host left.** It needs a live `.pyx` `def` with the unary
  `hasattr(__len__)` / `assert` / array-return shape; `hazma/spectra/`
  has none after 4.6, so it is `scalar_mediator_decay_spectrum` now.
  Phase 06 deletes that, and has to decide whether to retire
  `TestDeclaredDivergencesFromCython` or re-express its widenings against
  `cython_xs`. The `assert`-message roster shrinks with the tree — it is
  down from four wordings to two — and the wordings the port still emits
  are pinned in each kernel's own test module.
- **`crate::quad` short-circuits an empty interval** as of Task 4.6, the
  way `scipy/integrate/_quadpack_py.py:436` does. Any Phase 05/06 kernel
  whose limits can coincide inherits the fix.
- **`test/test_theory_aggregation.py` (69 tests) is the model-layer gate
  the corpus cannot be**, and the only numerical gate in the repo not
  scoped to the capturing platform. Every task in this phase ran it
  either side of the swap; Phases 05–06 should keep doing so.

## 3. Quirk Log & Edge Cases

- **Clang contracts, and which expressions it contracts is a
  per-expression fact.** Every task read the disassembly rather than
  pattern-matching, and every task found at least one expression that
  looks fusable and is not — `x² − 4r²`, `1 − β²` inside `boost_beta`,
  `e² − m_e²`, `xm² + 3r⁴`, and any sum whose operand went through a
  division. `_photon/_rho.pyx` contracts **nothing at all**, and the
  reason is untyped `cdef` locals: Cython boxes them into
  `PyFloatObject`s and evaluates through `PyNumber_*`, leaving clang no
  expression to fuse. `_neutrino/_pion.pyx` also contracts nothing, for
  the *opposite* reason — every local is typed, and the file simply
  contains no multiply-add. **Two files, same count, different cause.**
- **Compile-time constants can be folded *with* contraction.**
  `_positron/_pion.pyx`'s `emax_pi_rf` is a module-level `cdef double`
  that clang folds to a single immediate, and the immediate is one ulp
  above the unfused expression because `1.0 + β·√…` fused. Reproducing it
  needs a literal plus a `mul_add` re-derivation, not a `const`
  expression. Expect the same wherever a `.pyx` computes a constant at
  module init.
- **A rest-frame branch is not always the limit of the boosted one.**
  Three kernels disagree about what to do within one `DBL_EPSILON` of
  rest, and all three are shipped behavior: the photon/positron muon
  kernels return a genuine rest-frame spectrum, the ρ returns its bare
  *integrand* (MeV⁻² where the other branch is MeV⁻¹), and
  `_positron/_pion` returns **exactly zero**. Two of the three are filed
  defects; the third is not, because a delta function has no rest-frame
  representation in this API at all.
- **An absolute `DBL_EPSILON` threshold on a MeV quantity is not a
  tolerance band.** `fabs(epi - mpi) < DBL_EPSILON` at `m_π = 139.57`
  admits exactly one double — `m_π` itself — because one ulp there is
  2.8e-14, 128x `DBL_EPSILON`. Several kernels write the guard
  two-sided; the second side is inoperative in all of them.
- **A `NaN` energy does not propagate through a kernel that clips with
  `fmax`/`fmin`**, in either language: both limits collapse onto the
  rest-frame support and a finite number comes back. The corpus samples
  no `NaN`, so only a hand-written test sees a port that differs.
- **Deleting an extension strands whatever read its module *globals***,
  not only whatever imported it (Task 4.2 — two test modules failed at
  *collection*). And deleting a `.pyx` does not make its module
  unimportable: the built `.so` and generated `.c` sit beside it,
  gitignored, and neither `git checkout` nor `git stash` removes them.
  Assert on the source files and the `setup.py` entry, never on an
  `ImportError`.
- **The two muon files disagree about the Michel normalization and only
  one is wrong.** `_positron/_muon.pyx` divides by `R_FACTOR` where it
  should multiply (0.0374% low); `_neutrino/_muon.pyx` multiplies, and
  both its rows integrate to exactly one neutrino. A reader who meets the
  positron defect first will be tempted to "fix" the neutrino kernel;
  both sides are now asserted so that attempt fails.

## 4. Test Infrastructure State

- **Bare `pytest -q` → `1934 passed, 15 skipped`** at phase close, from
  `1378 / 13` at Phase 03's. The series across the phase:
  1628 (4.2) → 1682 (4.3) → 1755 (4.4) → 1802 (4.5) → 1934 (4.6).
  Re-derive rather than quoting.
- **`cargo test --no-default-features` → `169 passed`**, from 69 at Phase
  03's close: 80 (4.1) → 96 (4.2) → 109 (4.3) → 120 (4.4) → 133 (4.5) →
  169 (4.6).
- **`pytest test/parity` → `658 passed, 1 skipped`**; 41 cases, all green,
  and the corpus is platform-portable as of 2026-08-18 so CI runs it on
  every matrix entry.
- **Six new per-kernel test modules**, 4,000+ lines between them:
  `test_core_{positron_muon, photon_tables, photon_muon, photon_pion,
  photon_rho, positron_pion, neutrino}.py`. Each carries its own oracle
  argument in its module docstring; that docstring is the deliverable as
  much as the assertions are.
- **Run a mutation campaign on every kernel, and interrogate the
  survivors.** The phase ran five (4.3), eleven (4.4), six (4.5) and
  eleven (4.6). The discipline that emerged: a survivor is either
  *unobservable by construction* — in which case say so in the test, with
  the argument — or it is a **seam that needs lifting out**. Task 4.5
  lifted `photon_rho::boost_window`; Task 4.6 lifted
  `neutrino_pion::boost_window` and killed a survivor that was a **real
  error the gates could not see**: writing `γ` as `E/m` instead of
  `1/√(1−β²)` sits 29x outside the corpus's own budget at `E_π = 10⁵`
  MeV, and the corpus stops at `10 m_π`. **Ask "can this be lifted out?"
  before writing a limitation into the source.**
- **CI, not local runs, is where a mis-scoped bit-equality test fails.**
  The capturing platform cannot see a bug in its own skip logic. Read a
  Linux-only failure in a bit-equality assertion as "the scope is wrong"
  before "the port is wrong", and scope the class to
  `test/parity/data/manifest.json`'s machine rather than to a
  does-this-compiler-contract probe.
- **Do not chase a divergent-regime measurement one platform at a time.**
  PR #68 raised a budget from 1e-10 to 1e-8 on a macOS measurement and
  the *next* Linux point failed at 3.06e-08. Wynn's epsilon-algorithm is
  chaotic on a non-converging sequence, so there is no honest tolerance
  there; assert same-order-of-magnitude instead and keep the precision
  claim in the converged half.

## 5. Follow-on seeds

- **Six pinned-defect repairs from this phase** are queued in
  [`../../parity-pinned-defect-repair/PLAN.md`](../../parity-pinned-defect-repair/PLAN.md):
  [positron-muon normalization](../../../docs/followups/todo/positron-muon-spectrum-normalization-inverted.md),
  [η′ two-photon line](../../../docs/followups/todo/eta-prime-two-photon-line-missing-factor-two.md),
  [φ photon line energies](../../../docs/followups/todo/phi-photon-lines-use-the-daughter-meson-energy.md),
  [muon rest-frame endpoint](../../../docs/followups/todo/photon-muon-rest-frame-endpoint-uses-the-wrong-power-of-r.md),
  [charged-pion forward cone](../../../docs/followups/todo/charged-pion-photon-spectrum-misses-the-forward-cone.md),
  [ρ rest-frame integrand](../../../docs/followups/todo/rho-rest-frame-branch-returns-the-integrand.md).
  Task 4.5 measured that the forward-cone defect **compounds** through the
  ρ rather than merely propagating (0.945 of endpoint predicted, 0.537
  measured at `γ_ρ = 10`), so that repair is larger than originally
  scoped.
- **The doubled `π → e ν` neutrino line** —
  [`neutrino-pion-electron-line-counted-twice.md`](../../../docs/followups/todo/neutrino-pion-electron-line-counted-twice.md),
  filed in Task 4.6 and not yet in that project's roster. Unlike its six
  siblings it needs **no** Cython oracle: the excess is a closed-form
  plateau over a computable window, so its twin's deletion costs nothing.
- **Worth telling the maintainer separately from this project's
  schedule.** Several of the seven affect published numbers today, and
  two of them affect the *shape* of a spectrum rather than a total —
  which is the kind a limit calculation notices.
- **`test_core_dispatch.py`'s divergence class needs a decision in Phase
  06**, when its last Cython oracle goes. See §2.
- **The `derived::` namespace shrinks with the `.pyx` it is scored
  against.** Two submodules retired in this phase — `photon_rho` (4.5,
  bare aliases, vanished) and `neutrino_muon` (4.6, real arithmetic,
  moved into the kernel module). Phase 06 retires `photon_pion`,
  `positron_muon` and `positron_pion` the same way, and
  `test/test_core_constants.py` dies with the last `.pxd`.
