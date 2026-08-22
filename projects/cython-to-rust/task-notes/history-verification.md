# Archived working memory: Verification (Phases 00–04)

**Project:** cython-to-rust
**Moved:** 2026-08-21, from [`README.md`](README.md)
**Source lines:** 1954–2179 of that file at commit `c57ce4f`

This file is a verbatim archive. Nothing below the rule was edited,
summarised or reordered when it moved, and it sits in the same
directory as the README so every relative link in the moved text
still resolves. Reproduce the move with

```sh
git show c57ce4f:projects/cython-to-rust/task-notes/README.md | sed -n '1954,2179p'
```

The phase learnings under [`../learnings/`](../learnings/)
condense this material and are what a new task reads first — see
[ADR-0002](../../../docs/adrs/ADR-0002-read-phase-learnings-not-closed-task-notes.md).
Come here when a learnings entry, a task note or a citation sends
you to the original entry. Later phase-close sweeps append the
closed phase's entries below, verbatim, under a
`### Swept YYYY-MM-DD (Phase XX)` heading.

---

## Verification

- Scaffolding PR: `scripts/agents/preflight.sh` (repo gate; no code
  changes).
- **Phase 04 closing state (2026-08-20, Task 4.6):** bare `pytest -q` →
  **`1935 passed, 15 skipped, 7 warnings in 151.28s`** on the capturing
  environment, from `1831 / 15` on `origin/master`. **+104 = 48 new tests
  in `test/test_core_positron_pion.py` plus 58 in
  `test/test_core_neutrino.py` less 2 parameterized rows retired from
  `test/test_core_constants.py`** with `derived::neutrino_muon` (23 → 21);
  `test/test_core_dispatch.py` unchanged at 118, its spectra oracle module
  swapped rather than its assertions.
  `pytest -q test/parity` → `658 passed, 1 skipped`, all three swapped
  cases green — `spectra.neutrino.muon` **bit-equal at all 3,795 pinned
  values** at `EXACT_RTOL`, and both quadrature cases **tightened** to
  `PORTED_QUAD_RTOL` (1e-12) against measured worsts of 5.494e-15
  (`positron.charged_pion`, 1,304 / 1,460 bit-equal) and 9.739e-16
  (`neutrino.charged_pion`, 3,793 / 4,185).
  `cargo test --no-default-features` → `169 passed`, from 133 — the +36
  is exactly the four new kernel modules; fmt and clippy clean.
  `pytest -q test/test_theory_aggregation.py` → `69 passed` either side
  of the swap. An **eleven-mutation** validity campaign is in the task
  note with two survivors, both resolved: one lifted into
  `neutrino_pion::boost_window` and killed — it was a **real error the
  gates could not see**, a γ spelling 29x outside the corpus's budget at
  energies the corpus does not sample — and one shown unobservable *by
  construction* and recorded as such, closing the campaign 11 / 11.
  **PR #74 round 1 was red on all five Linux jobs and green on macOS**, in
  one assertion of the new positron-pion module at `E_π = 1e6` MeV:
  `emin = γ(E − βk)` is a cancellation conditioned at `2γ²ε` (2.3e-8 at
  γ = 7165), and x86-64's baseline has no FMA so the shipped Cython is
  unfused there while the port's `mul_add` is fused. Resolved by bounding
  the module's grids to `E_π = 1e4` and asserting the mechanism.
- **Phase 04 Task 4.5 state (2026-08-18) — the nested ρ, and the photon
  domain closed:** bare `pytest -q` →
  **`1802 passed, 15 skipped, 6 warnings in 47.15s`** on the capturing
  environment, from `1755 / 15` measured by stashing this branch's diff
  and re-running on the pre-task tree. **+47 = 49 new tests in
  `test/test_core_photon_rho.py` less 2 parameterized rows retired from
  `test/test_core_constants.py`** with `derived::photon_rho` (25 → 23);
  `test/test_core_dispatch.py` is unchanged at 118, because its spectra
  oracle module was swapped rather than its assertions.
  `pytest -q test/parity` → `629 passed, 1 skipped`, both ρ cases green
  in all five blocks at the **tightened** `PORTED_NESTED_RTOL` (1e-9)
  against a measured worst of 1.5e-13 (charged, 1,070 / 1,395 bit-equal)
  and 3.2e-15 (neutral, 1,052 / 1,395).
  `cargo test --no-default-features` → `133 passed`, from 121 — the +12
  is exactly `kernels::photon_rho`; fmt and clippy clean.
  `pytest -q test/test_theory_aggregation.py` → `69 passed` either side
  of the swap. A **six-mutation** validity campaign is in the task note
  with one survivor, and unlike Task 4.4's the survivor was **fixed**
  rather than documented: the fused boost-window arithmetic was lifted
  into a `boost_window(e, erho) -> (emin, emax, pre)` `fn` and pinned
  bit-for-bit, closing the campaign 6 / 6.
- **Phase 04 Task 4.4 state (2026-08-17) — the pion pair:** bare
  `pytest -q` → **`1755 passed, 15 skipped, 8 warnings in 605.61s`** on
  the capturing environment. Collection goes 1697 → 1770 against
  `origin/master`, **+73 and every one of them
  `test/test_core_photon_pion.py`**. `pytest -q test/parity` →
  `629 passed, 1 skipped`, with `spectra.photon.charged_pion` at 2.618e-15
  worst relative against the 1e-12 budget this task tightened it to and
  `spectra.photon.neutral_pion` bit-equal at all 1,305 values.
  `cargo test --no-default-features` → `120 passed` (11 new, all
  `kernels::photon_pion`); fmt and clippy clean.
  `pytest -q test/test_theory_aggregation.py` → `69 passed` either side of
  the swap. An **eleven-mutation** validity campaign is in the task note,
  with two survivors — an FMA site inside the quadrature integrand and a
  one-ulp constant — both unobservable through the entry point for the
  same reason and both recorded in the source. The campaign's first run
  had to be discarded and rebuilt: see Findings.
- **Phase 04 Task 4.2 state (2026-08-12) — the tabulated photon family:**
  bare `pytest -q` → **`1628 passed, 15 skipped in 587.90s`** on the
  capturing environment. Collection goes 1458 → 1643 against
  `origin/master` (`pytest --collect-only -q` on both trees), **+185 and
  every one of them `test/test_core_photon_tables.py`** — no other module
  gains or loses a test, which is the check that a swap this large moved
  no existing coverage. `pytest -q test/parity` →
  `629 passed, 1 skipped`; `python test/parity/generate.py --check` →
  `corpus OK: 41 cases / 1580 arrays`;
  `cargo test --no-default-features` → `96 passed` (16 new: 15 for
  `photon_tables` and one `NaN`-window test in `boost`); clippy and fmt
  clean; `scripts/agents/preflight.sh` **RESULT: PASS**.
  Two earlier preflight runs failed on lint alone — 29 ruff findings in
  the new test module, then the wrapper's 11 pre-existing ones — with no
  gate other than black and ruff ever red. The skip count goes 14 → 15,
  and the new one is the charged kaon in the per-line photon-count test
  — it has no monochromatic line.

- **Phase 04 Task 4.1 state (2026-08-11) — first kernel swap:** bare
  `pytest -q` → **`1424 passed, 14 skipped in 555.96s`** on the
  capturing environment (from 1378/13 at Task 3.5: +47 passes for
  `test/test_core_positron_muon.py`, and −1 pass / +1 skip for
  `test_running_on_the_capturing_tree`, which now skips because the
  corpus is in budget mode — **that is the designed signal, and the skip
  count does not go back down**). `pytest test/parity -q` →
  `629 passed, 1 skipped`; `pytest test/test_core_positron_muon.py -q` →
  `47 passed`; `pytest test/test_theory_aggregation.py -q` → `69 passed`;
  `cargo test --no-default-features` → `80 passed` (11 new); clippy, fmt
  and `markdownlint` clean; `scripts/agents/preflight.sh` RESULT: PASS.
  **Eighteen mutations against `rust/src/kernels/positron_muon.rs`**, run
  sequentially from a green baseline with the baseline re-asserted after,
  each gated by `cargo test` *and* a rebuild plus the Python module —
  **13 caught, 16 after three tests were added, and the two that remain
  are provably equivalent mutants** (`x.mul_add(2.0, C)` is bit-identical
  to `x * 2.0 + C` because doubling is exact, which also means one of the
  nine `fmadd` sites in the shipped object code is unobservable; and
  `beta + beta` vs `2.0 * beta`, included as a control). The three caught
  late all moved a **branch boundary** without moving a value — Task
  3.4's shape exactly — and one of them exposed that `dndx`'s
  `beta < DBL_EPSILON` short circuit is **unreachable from
  `dnde_positron_muon`**, because the outer `E − m_μ < DBL_EPSILON` guard
  already routes everything that could reach it.
- **Phase 03 Task 3.5 state (2026-08-11) — phase closed:** bare
  `pytest -q` → `1378 passed, 13 skipped in 564.55s` on the capturing
  environment, parity suite included and in bit-equality mode (skip count
  unchanged at 13, and `tolerances.provenance` → `exact=True` checked
  directly); `pytest test/test_core_dispatch.py -q` →
  `118 passed in 4.19s`;
  `cargo test --manifest-path rust/Cargo.toml --no-default-features` →
  `69 passed` (2 new); clippy, fmt and `markdownlint --dot` clean;
  `scripts/agents/preflight.sh` RESULT: PASS. Fourteen mutations against
  `rust/src/{dispatch,kernels}.rs`, sequential from a green baseline with
  the baseline re-asserted after — **13 caught**. The survivor is the
  interesting one: it swapped two arms of the classification that the
  implementation's own comment called load-bearing, and left all 118
  tests green — so the *comment* was wrong (the real guard against a
  string parsing as a number is the 0-d dtype check), and it was
  corrected rather than the mutation dropped.
- **Phase 03 Task 3.4 state (2026-08-10):** bare `pytest -q` →
  `1314 passed, 13 skipped` on the capturing environment, parity suite
  included and in bit-equality mode (skip count unchanged at 13, and
  `tolerances.provenance` → `exact=True` checked directly);
  `pytest test/test_core_interp.py -q` → `33 passed in 0.46s`;
  `pytest test/test_core_boost.py -q` → `69 passed in 0.91s`;
  `cargo test --no-default-features` → `67 passed` (24 new); clippy,
  fmt and `markdownlint --dot` clean; `scripts/agents/preflight.sh`
  RESULT: PASS. Twenty-one mutations against `rust/src/{interp,boost}.rs`,
  run sequentially behind a lock with a green baseline asserted before
  and after — 17 of the first 20 caught, and all 21 after two tests were
  added. **The three survivors shared one shape**: each moved a *branch
  boundary* by a single double without touching any value the function
  returns, so no grid sample could see it. What catches that is
  bisecting on the bit pattern (`test_the_window_edges_sit_on_the_same
  _double_as_the_cython`) — and the parameter space matters as much as
  the sampling, since with `m = 0` the fused and unfused momenta are
  bit-identical and only massive-product draws can distinguish them.
- **Phase 03 Task 3.3 state (2026-08-10):** bare `pytest -q` →
  `1212 passed, 13 skipped` on the capturing environment, parity suite
  included and in bit-equality mode (skip count unchanged at 13);
  `pytest test/test_core_quad.py -q` → `58 passed in 5.10s`;
  `cargo test --no-default-features` → `43 passed` (27 new); clippy and
  fmt clean; `scripts/agents/preflight.sh` RESULT: PASS. Seventeen
  mutations against `rust/src/quad.rs`, 15 caught on the first pass and
  the two survivors (`qagpe`'s `ndin`, `qagse`'s roundoff threshold)
  covered by tests written afterwards against inputs found by searching
  with each mutation in place. The Gauss–Kronrod literals are bit-equal
  to the netlib Fortran (47 values, checked by a script that parses both
  sides independently of the crate).
- **Phase 03 Task 3.2 state (2026-08-09; PR #59 review round 1,
  2026-08-10):** bare `pytest -q` →
  `1154 passed, 13 skipped` on the capturing environment, parity suite
  included and in bit-equality mode (skip count unchanged at 13);
  `pytest test/test_core_special.py -q` → `65 passed in 0.50s`;
  `cargo test --no-default-features` → `16 passed` (9 new), clippy and
  fmt clean; `scripts/agents/preflight.sh` RESULT: PASS. Eleven
  mutations — nine against `rust/src/special.rs`, two against the
  corpus's served-kernel guard — each caught by the test whose name
  claimed it (tables in the task note). One of them, dropping the
  recurrence's order factor, passed `cargo test` on the first pass and
  is why the
  Wronskian unit test now runs at ν = 2 as well as ν = 1.
- **Phase 03 Task 3.1 state (2026-08-09):** bare `pytest -q` →
  `1088 passed, 13 skipped` on the capturing environment, parity suite
  included and in bit-equality mode;
  `pytest test/test_core_constants.py -q` → `25 passed in 0.03s`;
  `cargo test --no-default-features` → `7 passed` (5 new), clippy and
  fmt clean; `scripts/agents/preflight.sh` RESULT: PASS. Thirteen
  mutations — nine Python, four Rust — each caught by the test whose
  name claimed it (table in the task note).
- Per-phase Verification sections live in `phase-XX/README.md`.
- **Phase 02 closing state (2026-08-09):** bare `pytest -q` →
  `1063 passed, 13 skipped` on the capturing environment (1076
  collected), parity suite included and in **bit-equality mode**;
  `scripts/agents/preflight.sh` RESULT: PASS across all eleven rows, the
  three cargo gates green. `git diff origin/master -- hazma` is empty, so
  the public compiled surface is exactly where Phase 00 left it — the
  whole phase's only change under `hazma/` is the non-executable
  `hazma/_core.pyi` stub. Task 2.3's 54 new tests were validated by a
  six-mutation campaign against `rust/src/{dispatch,lib}.rs`, each
  mutation rebuilt and caught by the test whose name claimed it.
- **Phase 02 Task 2.2 state (2026-08-08):** `scripts/agents/preflight.sh`
  RESULT: PASS across all eleven rows — the three cargo gates green,
  `pytest` at `1009 passed, 13 skipped` (byte-identical to Task 2.1's, so
  no test outcome moved) with the parity suite still in bit-equality
  mode, markdownlint green over 16 changed docs. `git diff origin/master
  -- hazma rust` and `-- setup.py pyproject.toml MANIFEST.in` are both
  empty, so the compiled artifacts are the trunk's. PR #56's eight
  checks are green (including the new `rust` job, 30s), and the
  dispatched `release.yml` run 31297673951 is `success` on both
  platforms with `publish` skipped.
- **Phase 02 Task 2.1 state (2026-08-08):** bare `pytest -q` →
  `1009 passed, 13 skipped` in 569.45s on the capturing environment
  (1022 collected), parity suite in **bit-equality mode**;
  `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings` and
  `cargo test --no-default-features` (2 unit tests) all green from
  `rust/`; `python test/parity/generate.py --check` → `corpus OK: 41
  cases / 1580 arrays`; wheel and sdist both build, the sdist installs
  from source into a fresh venv on a *different* interpreter and runs
  both toolchains from outside the repo. `scripts/agents/preflight.sh`
  RESULT: PASS.
- **Phase 01 closing state (2026-08-08):** bare `pytest -q` →
  `1006 passed, 13 skipped` on the capturing environment (1019 collected:
  67 `hazma` + 952 `test`), parity suite included and in exact mode;
  `python test/parity/generate.py --check` → `corpus OK: 41 cases / 1580
  arrays`. (Off macOS, CI ran `pytest --ignore=test/parity` from PR #52
  until the parity-corpus stability follow-up landed on 2026-08-18; every
  matrix entry runs the corpus now.) No skip
  anywhere in the repo is waiting on this project. The public compiled
  surface is still exactly where Phase 00 left it.
- **Phase 00 closing state (2026-08-06):** 20 `.pyx` ↔ 20 declared
  `Extension`s ↔ 20 `.so`, verified as a set equality; zero C++;
  `pytest -q test` → `244 passed, 20 skipped`, bare `pytest -q` →
  `57 passed, 10 skipped`; sdist and wheel both build, and the sdist
  installs and runs in a fresh venv from outside the repo. The public
  compiled surface is unchanged from where the phase started.
