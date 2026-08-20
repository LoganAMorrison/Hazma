# Task 2: Capture the corrected-value oracles from the four live twins

**Date:** 2026-08-19
**Project:** parity-pinned-defect-repair
**Status:** Complete
**Plan References:** `../PLAN.md` Task 2; `../rules.md` rules 1, 8, 10, 11;
`../references/corpus-repinning.md` "The oracle capture protocol";
`../references/defect-blast-radius.md` (the whole file)
**Related ADRs:** none. `../PLAN.md`'s second anticipated ADR — whether
these arrays stay committed after `cython-to-rust` closes — is not
settled here; see Open Questions.
**Depends On:** none

## Objective

Take the one measurement that stops being possible when the port deletes
the Cython, and commit it: the corrected value of each Group A defect, on
the corpus's own grids, from a patched copy of the twin that predates the
Rust port.

## Exit Criteria

- `test/parity/oracles/` committed and self-checking, with a `--check`
  mode that needs no build, mirroring `test/parity/generate.py --check`.
- A test that the oracle manifest's platform matches the corpus
  manifest's.
- A recorded per-defect diff between each oracle and the corresponding
  stored corpus array.
- `git diff -- hazma` empty on the final tree, and
  `git diff --stat -- test/parity/data` empty.

## Inputs Reviewed

- `../PLAN.md` (whole file), `../rules.md` (all 11),
  `../references/corpus-repinning.md`,
  `../references/defect-blast-radius.md`.
- `projects/cython-to-rust/rules.md` rules 1–3.
- The four Group A follow-ups under `docs/followups/todo/` — each one's
  "What" section is where this task's patch came from.
- `test/test_core_boost.py` — the `__pyx_capi__` + `ctypes.PYFUNCTYPE`
  precedent, and the only place in the repo that had driven a `cdef`
  this way before.
- `test/parity/generate.py` (the `--check` contract, the manifest shape),
  `test/parity/cases.py` (`Block`, `build_cases`),
  `test/parity/tolerances.py` (`_libm_identity`).
- `docs/agents/lessons.md`: `[mutation-harness-poisons-its-own-baseline]`,
  `[platform-scoped-oracle-asserted-globally]`,
  `[measured-tree-vs-imported-module]`, `[derived-count-not-rederived]`.

## Findings

### The spike passed for all four twins, and for a fifth

Before any capture machinery, the four twins' *unpatched* `cdef`s were
driven through `__pyx_capi__` and compared against the stored corpus
arrays. All four export a `double (double, double)` capsule, which is
directly `ctypes`-callable; the `*_array` capsules take a
`__Pyx_memviewslice` and are not, but every one of them is a loop over
the `*_point` `cdef`, so looping the point form in Python reproduces the
array path exactly. Measured, not argued: bit-for-bit on every block of

```text
spectra.photon.muon           spectra.positron.muon
spectra.photon.charged_pion   spectra.positron.charged_pion
spectra.photon.neutral_pion
```

`hazma/spectra/_positron/_pion.pyx` is the fifth — not a twin of any
defect, but A4's composition chain and the earliest deletion (Task 4.6).

### Two of the four composition chains were already stranded

This is the finding that shaped the task. `../PLAN.md`'s deadline table
lists three waves, all in the future. It is missing two that have already
passed:

| Wave | Task | Deleted | Chain lost |
| --- | --- | --- | --- |
| 0a | 4.2 (2026-08-12, `0954e5a`) | `_photon/{_eta,_eta_prime,_kaon,_omega,_phi}.pyx` | **all seven of A1's cases** |
| 0b | 4.5 (`b5f7f90`) | `_photon/_rho.pyx` | A2's and A3's two rho cases |

`boost_integrate_linear_interp` survives, but nothing in the tree calls
it any more — `rust/src/kernels/photon_tables.rs` is its only consumer,
which `../references/defect-blast-radius.md` states. So A1's twin being
"live" is true of the primitive and false of every corpus case it
reaches. Same for the rho: the pion kernel A3 patches is live, the outer
quadrature that turns it into `spectra.photon.charged_rho` is not.

**Both were recovered rather than written off.** The deleted `.pyx` are
in git history, and `../PLAN.md` Task 3 already uses `git show` to
recover B1/B2/B3 sources for review; this task compiles them instead of
reading them. Thirteen files, from two revisions, listed in
`defects.RESTORED_SOURCES` with the revision each comes from, hashed
against `git show <rev>:<path>` before every capture and again by
`test/parity/test_oracles.py`. The alternative — capturing A1 at the
`boost_integrate_linear_interp` boundary only — would have handed Task 4
a primitive-level oracle and no way to check the seven published arrays
its gate names.

### The baseline pass is what makes the rest believable

Nine of the twenty cases run through resurrected modules and four through
a shim that rebuilds a `def` the port deleted. Neither is assumed
faithful. `capture.py --baseline` drives every case through the
*unpatched* build and requires bit-for-bit equality with the committed
corpus:

```text
$ python test/parity/oracles/capture.py --baseline
hazma resolved at hazma
  ... 20 cases, each OK ...
baseline OK: 20 cases / 640 arrays / 152815 values reproduce the corpus bit-for-bit
```

Not one array differed, including the seven driven through modules that
had been deleted from the tree and the four driven through a `ctypes`
shim standing in for a `def` the port removed.

Bit-equality rather than a tolerance, because the capture runs on the
corpus's own capturing platform with the same Cython: anything short of
identical would mean the harness differs, not the libm.

### A2's repair exposes a negative sliver at the endpoint

Found by the physics check, not by the oracle comparison — which is what
`../rules.md` rule 4 exists for. Extending the rest-frame guard to
`y >= 1 - r` regains 0.254264 MeV below the endpoint, and the O(α)
formula the branch evaluates (hep-ph/9909265) has its own zero at
**52.808176 MeV**, just under the kinematic endpoint at 52.827952. So
the corrected guard admits a window where the spectrum is negative:

| quantity | value |
| --- | --- |
| window regained by `y >= 1 - r` | 0.254264 MeV |
| of which the formula is negative | 0.019774 MeV — the top 7.8% |
| most negative value | −6.433683e-09 MeV⁻¹ |
| that, against the spectrum's peak | 3.66e-07 |
| photons the repair adds over the window | +5.454359e-08 |
| photons the negative sliver removes | −8.974639e-11 (0.16% of the above) |

Measured on the patched build over 200,001 samples between the shipped
and corrected cuts, with the peak taken over the branch's whole support.

The magnitude says this is the formula's own accuracy limit near its
endpoint rather than a coding error — 1e-7 of peak is where an O(α)
expression stops being trustworthy. It is still a published spectrum
going below zero, and Task 7's gate as written ("the endpoint invariant
… now holds over the extra 0.25 MeV") does not anticipate it. The oracle
records what the follow-up's stated repair actually produces, negative
sliver included; deciding whether to clamp is Task 7's, and it now has
the numbers to decide with. See Open Questions.

### The physics checks, per defect

`../rules.md` rule 4: an oracle comparison cannot tell you two
implementations are both wrong. Each patched build was therefore also
asked a question the corpus cannot answer, while it still existed —
these are not re-runnable, since the build is reverted.

**A1**, against the follow-up's own two hand-computable cases. With
`x == y` the integrand `y/x` is 1, so the integral is the window width
clamped to the table and the return is that over `2 γ β`:

| case | shipped | patched | closed form |
| --- | --- | --- | --- |
| `x=y=[1..4]`, β=0.6, E=2.2 | 1.266667 | **1.933333** | 1.933333 |
| `x=y=[1..6]`, β=0.01, E=3.5 | 53.497500 | **3.500000** | 3.500000 |

The shipped column reproduces the follow-up's stated figures
(`1.9/(2γβ)` and `53.497`) exactly, which is what says the two cases were
transcribed right; the patched column lands on the closed form to better
than 1e-12 relative.

**A2**: the negative-sliver finding above — that check is what produced
it.

**A3**, against the specific figure the follow-up pins, and a yield that
should barely move with boost:

| quantity | value |
| --- | --- |
| `dnde_photon_charged_pion(900, 1396)` | **3.585860e-07** MeV⁻¹ (shipped `0.0`; follow-up predicted `3.586e-07`) |
| yield at `E_π` = 1000 / 1396 / 5000 MeV | 0.080880 / 0.080743 / 0.080599 photons per decay |
| support, as a fraction of `E_π` | ~0.983 at all three, against roughly the top quarter missing before |
| negatives or NaNs anywhere on the grid | none |

Hitting the follow-up's four-significant-figure prediction from an
independently written patch is the check that the repair *form* chosen
here is the one it meant, not merely a change that moves the number.

**A4**, against the invariant a normalization defect cannot survive: a
muon decays to exactly one positron.

| | integral of the spectrum over its support |
| --- | --- |
| patched, at rest | **1.000000000000** — deviation `+2.220e-16`, one ulp |
| patched, `E_μ` = 211.316749 MeV | 1.0000000000 |
| patched, `E_μ` = 1056.583745 MeV | 0.9999999993 |
| shipped | 0.999625933330 |

`R_FACTOR² = 1.000374206648`, so the repair moves the yield by
**0.0374%** — the figure the follow-up states, recovered here from the
normalization integral rather than from the constant. This is the
strongest of the four checks: the corrected form is not merely different,
it is exactly right, and the shipped form is exactly `1/R_FACTOR²` away
from right.

### The measured radii disagree with the prediction three times out of four

`../references/defect-blast-radius.md` says of itself that it is a
prediction and that a repair task re-derives its own row. Task 2 is the
first task in a position to do that for Group A, so it did.

**A1's sign is not one-signed, and the split is by block.** `../PLAN.md`
Task 4's gate says "the sign is one-signed and upward at every declared
position". Measured over all seven cases:

| block | moved | up | down | median &#124;shipped / oracle&#124; |
| --- | --- | --- | --- | --- |
| `rest` | 0 | 0 | 0 | — |
| `rest_plus_eps` | 1156 | 0 | **1156** | 9768.55 |
| `near_rest` | 1130 | 1130 | 0 | 0.967695 |
| `boosted_mild` | 1028 | 1028 | 0 | 0.997818 |
| `boosted_strong` | 840 | 839 | 1 | 0.999929 |
| **total** | **4154** | **2997** | **1157** | |

Both directions are the same off-by-one read from opposite sides, which
is what the follow-up's "Why" section describes and what the plan's gate
compressed away. In `rest_plus_eps` the two partial-cell terms overlap
and the shipped value is ~9,800× too *high*; in the boosted blocks a
whole cell is dropped and the shipped value is slightly too *low*,
converging toward 1 as the boost strengthens. "Systematically low" is
true away from threshold and false at it.

It also answers the question that file explicitly leaves open — "whether
`rest` moves depends on whether the integral runs at β = 0 — **measure
it**". It does not: **0 of 1750 `rest` positions move**, because all
seven callers short-circuit to the rest-frame spectrum before the
integral.

**A2 reaches one corpus case, not seven.** The rest-frame branch is
guarded by `emu - MASS_MU < DBL_EPSILON`, and every composed caller
boosts the muon before calling it — the charged pion at
`ENG_MU_PIRF = 109.778` MeV, both mediators at `m/2 ≥ 125` MeV. So the
branch is unreachable from any composition chain, and
`spectra.photon.charged_pion`, both rho cases and all three
`mediator_spectra.*.photon` cases move **0 values each**. Only
`spectra.photon.muon`'s `rest` block moves, and only four positions in
it:

| `E_γ` (MeV) | shipped | oracle |
| --- | --- | --- |
| 52.827897242 | 0.0 | −2.995874e-09 |
| 52.827950017 | 0.0 | −2.916321e-09 |
| 52.827950070 | 0.0 | −2.916240e-09 |
| 52.827950123 | 0.0 | −2.916160e-09 |

**Every one of them is negative** — the corpus's `rest` grid puts four
points in the top 0.02 MeV of the regained window and none at all in the
0.234 MeV of positive spectrum below it. So A2's declared delta, under
the repair exactly as the follow-up specifies it, would consist entirely
of small negative values replacing zeros. That makes the clamp question
in Open Questions the whole of Task 7's numerical impact rather than a
footnote to it.

**A3 and A4 match their predictions.** A3 moves all six predicted cases,
including both rho cases — so the inner fix does propagate through the
outer quadrature, even though Task 4.5 showed it is not *sufficient*
there. A4 moves all six, uniformly: every non-zero position is multiplied
by `R_FACTOR²`, giving the same max relative shift of `0.000374207` on
all six cases and `0` positions moving down. That is the signature of an
overall factor, and it is as clean a confirmation as this measurement
produces.

### The capture environment matches the corpus manifest

| Key | Corpus manifest | This capture |
| --- | --- | --- |
| `python` | 3.12.12 | 3.12.12 |
| `numpy` | 2.5.1 | 2.5.1 |
| `scipy` | 1.18.0 | 1.18.0 |
| `cython` | 3.2.9 | 3.2.9 |
| `machine` | arm64 | arm64 |
| `platform` | macOS-26.5.2-arm64-arm-64bit | macOS-26.6.1-arm64-arm-64bit |

The one difference is an OS point release, and the repo already has a
ruling on it: `tolerances._libm_identity` compares
`platform.platform().partition("-")[0]` and `machine`, deliberately
coarser than the version string, because the capturing machine had
already moved from 26.5.2 to 26.6.1 before this task and calling that a
platform change would silently drop the corpus's `EXACT` class off
bit-equality. `test_oracles.py` reuses that function rather than
introducing a second answer. `numpy` was pinned back to 2.5.1 for the
capture — the environment resolved 2.5.2 by default, and `np.trapezoid`
is on the boost integral's path.

## Decisions and Implementation Notes

- **One patch per capture, never two.** Each defect's build carries
  exactly one patched `.pyx`, so a captured array states one repair's
  size rather than a combination. `capture.py` re-derives
  `git diff -- <source>` and refuses unless it is byte-identical to the
  committed patch, which is the `[mutation-harness-poisons-its-own-baseline]`
  guard in code rather than in a checklist. The tree was also snapshotted
  outside the worktree and `cmp`-ed before every mutation and after every
  revert.
- **The patches are committed, and they are the specification.**
  `test/parity/oracles/patches/*.patch` is what "corrected" means for
  each defect, in a form Tasks 4, 7, 8 and 10 can read and match. The
  arrays alone would leave the repair tasks re-deriving four fixes from
  four follow-ups and hoping they landed on the same one.
- **A4's operator order is a choice, and it is recorded.** The follow-up
  flags it: `/ (2 * beta * R_FACTOR)` becomes `* R_FACTOR / (2 * beta)`,
  not `/ (2 * beta) * R_FACTOR`, and the two differ in the last ulp.
  Task 10 must match the patch or re-derive its own oracle.
- **A3's repair form is the follow-up's "cheapest faithful fix" and no
  more.** `cos_min = (1 - ENG_GAM_MAX_PIRG / (E_γ γ_π)) / β_π`, clamped
  to `[-1, 1]`, with the legacy `ENG_GAM_MAX_PIRG` literal left alone —
  the follow-up's point 1 says deriving a PDG-consistent edge instead
  moves the answer a second time and to do one or the other deliberately.
  This capture does the first. See Open Questions for what that leaves
  Task 8.
- **Grids are not stored.** Only `values` and `scalar_values`. The
  capture drives `test/parity/cases.py`'s own `Block` objects, so the
  grids are byte-identical to the corpus's by construction and a second
  copy would only be something else to keep in sync.
- **The oracle manifest carries the corpus manifest's hash.** Rule 1
  forbids rewriting the corpus, so it should never move; if it does,
  every recorded diff is against something that no longer exists, and
  `test_the_corpus_has_not_moved_under_the_capture` says so.

## Numerical impact

**No public value moves in this task.** The four patches exist only
inside the capture and are reverted; `git diff -- hazma` is empty on the
final tree and `test/parity/data` is untouched. What follows is the
*measurement*, which is the deliverable — the first statement of each
repair's size taken from Cython rather than from the Rust that will be
repaired. `../task-notes/README.md`'s "Numerical impact so far" carries
the same figures per `../rules.md` rule 10.

The blast-radius reference calls itself a prediction. Three of its four
Group A rows are wrong, and two of `../PLAN.md`'s gates are wrong with
them — measured by running each patched build over every case the
prediction names and seeing what moved. Details under Findings; the
per-case table is below, derived with:

```sh
python3 -c "import json,pathlib; m=json.loads(pathlib.Path('test/parity/oracles/data/manifest.json').read_text()); \
  print(sum(c['values_moved'] for d in m['defects'].values() for c in d['diff_against_corpus'].values()))"
```

### A1 — the boost integral mis-covers its window at both ends

Patch: `test/parity/oracles/patches/A1-boost-integral-window.patch` on `hazma/_utils/boost.pyx`.

| Corpus case | values moved / total | max abs shift | max rel shift | up / down |
| --- | --- | --- | --- | --- |
| `spectra.photon.charged_kaon` | 631 / 1435 (44.0%) | 483975 | 0.999993 | 449 / 182 |
| `spectra.photon.eta` | 560 / 1435 (39.0%) | 6209.45 | 0.999997 | 412 / 148 |
| `spectra.photon.eta_prime` | 552 / 1435 (38.5%) | 2502.46 | 0.999952 | 404 / 148 |
| `spectra.photon.long_kaon` | 631 / 1435 (44.0%) | 659277 | 0.999993 | 449 / 182 |
| `spectra.photon.omega` | 633 / 1435 (44.1%) | 481699 | 0.999993 | 449 / 184 |
| `spectra.photon.phi` | 551 / 1435 (38.4%) | 3435.51 | 0.999856 | 404 / 147 |
| `spectra.photon.short_kaon` | 596 / 1435 (41.5%) | 587910 | 0.999899 | 430 / 166 |

### A2 — the muon photon rest-frame branch stops short of the endpoint

Patch: `test/parity/oracles/patches/A2-muon-rest-frame-endpoint.patch` on `hazma/spectra/_photon/_muon.pyx`.

| Corpus case | values moved / total | max abs shift | max rel shift | up / down |
| --- | --- | --- | --- | --- |
| `mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum` | 0 / 8610 (0.0%) | 0 | 0 | 0 / 0 |
| `mediator_spectra.vector.photon.dnde_decay_v` | 0 / 29295 (0.0%) | 0 | 0 | 0 / 0 |
| `mediator_spectra.vector.photon.dnde_decay_v_pt` | 0 / 29295 (0.0%) | 0 | 0 | 0 / 0 |
| `spectra.photon.charged_pion` | 0 / 1500 (0.0%) | 0 | 0 | 0 / 0 |
| `spectra.photon.charged_rho` | 0 / 1435 (0.0%) | 0 | 0 | 0 / 0 |
| `spectra.photon.muon` | 4 / 1370 (0.3%) | 2.99587e-09 | 0 | 0 / 4 |
| `spectra.photon.neutral_rho` | 0 / 1435 (0.0%) | 0 | 0 | 0 / 0 |

### A3 — the charged-pion photon spectrum returns zero in the forward cone

Patch: `test/parity/oracles/patches/A3-charged-pion-forward-cone.patch` on `hazma/spectra/_photon/_pion.pyx`.

| Corpus case | values moved / total | max abs shift | max rel shift | up / down |
| --- | --- | --- | --- | --- |
| `mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum` | 1032 / 8610 (12.0%) | 4.8511e-09 | 1.62664e-06 | 609 / 423 |
| `mediator_spectra.vector.photon.dnde_decay_v` | 2013 / 29295 (6.9%) | 7.33366e-10 | 7.77494 | 1209 / 804 |
| `mediator_spectra.vector.photon.dnde_decay_v_pt` | 2013 / 29295 (6.9%) | 7.33366e-10 | 7.77494 | 1209 / 804 |
| `spectra.photon.charged_pion` | 245 / 1500 (16.3%) | 7.85331e-07 | 0.0169925 | 128 / 117 |
| `spectra.photon.charged_rho` | 528 / 1435 (36.8%) | 3.04384e-10 | 8.73349e-08 | 285 / 243 |
| `spectra.photon.neutral_rho` | 528 / 1435 (36.8%) | 9.11613e-08 | 0.563695 | 276 / 252 |

### A4 — the muon positron spectrum divides by its normalization

Patch: `test/parity/oracles/patches/A4-positron-muon-normalization.patch` on `hazma/spectra/_positron/_muon.pyx`.

| Corpus case | values moved / total | max abs shift | max rel shift | up / down |
| --- | --- | --- | --- | --- |
| `mediator_spectra.scalar.positron.dnde_decay_s` | 5237 / 16740 (31.3%) | 7.96537e-06 | 0.000374207 | 5237 / 0 |
| `mediator_spectra.scalar.positron.dnde_decay_s_pt` | 5237 / 16740 (31.3%) | 7.96537e-06 | 0.000374207 | 5237 / 0 |
| `mediator_spectra.vector.positron.dnde_decay_v` | 5237 / 16740 (31.3%) | 3.34892e-06 | 0.000374207 | 5237 / 0 |
| `mediator_spectra.vector.positron.dnde_decay_v_pt` | 5237 / 16740 (31.3%) | 3.34892e-06 | 0.000374207 | 5237 / 0 |
| `spectra.positron.charged_pion` | 525 / 1500 (35.0%) | 1.15841e-05 | 0.000374207 | 525 / 0 |
| `spectra.positron.muon` | 502 / 1370 (36.6%) | 1.4163e-05 | 0.000374207 | 502 / 0 |

## Files Changed

- `test/parity/oracles/capture.py` — the harness: `--baseline`,
  `--defect`, `--assemble`, `--check`.
- `test/parity/oracles/defects.py` — the four defects, each one's patch,
  source and case list; the restored-source roster.
- `test/parity/oracles/entry_points.py` — where each case's Cython value
  comes from now, and the `__pyx_capi__` shim.
- `test/parity/oracles/patches/*.patch` — one unified diff per defect.
- `test/parity/oracles/data/{A1,A2,A3,A4}.npz`, `data/manifest.json` —
  the capture.
- `test/parity/oracles/README.md` — the directory's own account, and the
  recapture loop.
- `test/parity/test_oracles.py` — the gate.
- `projects/parity-pinned-defect-repair/task-notes/README.md` — Task 2
  status, and "Numerical impact so far".
- `projects/parity-pinned-defect-repair/references/defect-blast-radius.md`
  — the two already-closed deletion waves, the measured A1 and A2 rows,
  and the re-derived coverage arithmetic.
- `projects/parity-pinned-defect-repair/PLAN.md` — the Task 4, 7 and 8
  gates, corrected against the measurement.
- `projects/parity-pinned-defect-repair/rules.md` — rule 7's example,
  which named a pair of repairs that no longer overlap.
- `test/parity/README.md` — a row for `oracles/` and a paragraph saying
  why seven pinned values being wrong still does not make the corpus
  regenerable.

## Verification

Every count below was re-derived after the last edit to the thing it
measures, with the command beside it (`../rules.md` rule 11).

**The capture itself.** Baseline first, then one patched build per
defect, then a final unpatched rebuild:

```sh
python test/parity/oracles/capture.py --baseline    # 20 cases / 640 arrays / 152815 values, bit-for-bit
git apply test/parity/oracles/patches/A1-boost-integral-window.patch
uv pip install -e .
python test/parity/oracles/capture.py --defect A1   # and A2, A3, A4 the same way
python test/parity/oracles/capture.py --assemble    # 4 defects / 1559.7 KiB
```

**The committed capture, re-hashed without a build:**

```sh
$ python test/parity/oracles/capture.py --check
oracles OK: 4 defects / 940 arrays match the manifest (corpus manifest f476fb420caf)
$ python test/parity/generate.py --check
corpus OK: 41 cases / 1580 arrays match the manifest (generated at 010747c6125d, kernel digest f5e6e269be47)
```

940 arrays / 224,385 values / 1,559.7 KiB, against the 10 MiB budget
`capture.py` inherits from `generate.py`.

**The gate:**

```sh
$ pytest test/parity/test_oracles.py -n 0
18 passed
```

**The leak guard is not vacuous.** Run as a mutation, not argued:

```sh
$ git apply test/parity/oracles/patches/A1-boost-integral-window.patch
$ pytest test/parity/test_oracles.py::test_the_capture_left_no_library_behavior_behind -n 0
1 failed
$ git checkout -- hazma/_utils/boost.pyx
$ pytest test/parity/test_oracles.py::test_the_capture_left_no_library_behavior_behind -n 0
1 passed
```

Its first form searched the tracked source for the patch's added lines
and reported a leak on a clean tree: A1's added block re-uses lines the
surrounding partial-cell arithmetic already contains verbatim
(`x2 = x[ilow]`, `b = y1 - m * x1`). Replaced with
`git apply --reverse --check`, which succeeds exactly when a patch is
applied and so asks the question directly.

**Nothing leaked into the library.** Checked three ways, on the final
tree, after the last rebuild:

```sh
git diff --stat -- hazma             # no output
git diff --stat -- test/parity/data  # no output
cmp <snapshot>/hazma/_utils/boost.pyx hazma/_utils/boost.pyx
cmp <snapshot>/setup.py setup.py     # and the other three .pyx, all silent
```

The snapshot lives outside the worktree and was `cmp`-ed before every
mutation and after every revert, eight times in all —
`[mutation-harness-poisons-its-own-baseline]`. The thirteen restored
sources and the `setup.py` entry that built them were removed afterwards;
`git status --porcelain` shows no untracked file under `hazma/`.

**The tree that was measured is the tree that was hashed.** Recorded in
the manifest as `hazma_package: "hazma"` for all four defects, and
asserted at capture time by `cases.hazma_package_path()`:

```sh
$ python -c "import hazma, hazma._core; print(hazma.__file__); print(hazma._core.__file__)"
.../hazma-oracles-capture-027594/hazma/__init__.py
.../hazma-oracles-capture-027594/hazma/_core.abi3.so
```

**One CI-only failure, found and fixed on PR #73's first run.**
`test_the_restored_sources_are_still_recoverable` shells out to
`git show <rev>:<path>`, and `actions/checkout` clones at
`fetch-depth: 1`, so on CI the recorded revisions are simply absent and
git exits 128. Reproduced locally with
`git clone --depth 1 file://$(git rev-parse --show-toplevel)`, which is
what turned it from a guess into a diagnosis. The test now probes
`git cat-file -e <rev>^{commit}` first and skips with that reason when
the history is not there; a revision that *does* resolve and then hashes
wrong is still a failure. 18 passed on a full clone, 17 passed and 1
skipped on a shallow one.

**Docs.** `markdownlint --dot` clean over all seven touched markdown
files; `scripts/agents/check_doc_citations.py` run separately over the
same set (it is not in `preflight.sh` —
`[gate-green-is-not-citations-green]`), 0 in-repo citations, none out of
range; and a script confirming all relative markdown links in them
resolve.

**The gate:** `scripts/agents/preflight.sh` with `--paths` scoped to the
four new Python modules and `--md` over the changed markdown.

*Deferred:* nothing. The one thing this task could not do is re-run its
own captures from a clean tree — that needs a patched Cython build, and
`--check` is the standing verification in its place, by the same
reasoning `generate.py --check` uses for the corpus.

## Open Questions

- **A3's repaired value is not final, and the rho half of it is known
  incomplete.** Task 4.5 measured that repairing the charged-pion kernel
  is necessary but not sufficient for the rho: the *outer* boost integral
  hits the same QUADPACK failure a second time, and the follow-up's own
  onset/endpoint table shows it. The A3 oracle therefore states the
  inner-fix-only value for `spectra.photon.charged_rho` and
  `spectra.photon.neutral_rho`; Task 8 has to restrict the outer interval
  too, and cannot expect those two arrays to match this oracle once it
  does. The four non-rho cases in A3's radius are unaffected by that
  caveat.
- **Does A2's repair need a floor at zero?** The finding above: `y >= 1 - r`
  is the right kinematic endpoint and it admits 0.019774 MeV where the
  rest-frame formula evaluates negative, at 3.66e-07 of peak. Three
  options, none of them this task's to pick — cut at the formula's own
  zero instead of at the kinematic endpoint; clamp the branch at zero;
  or accept it as the expression's accuracy limit and say so in the
  docstring. Whichever Task 7 chooses, the A2 oracle here is the
  unclamped one, so a clamped repair will legitimately differ from it in
  that window and the declaration has to say so rather than widen a
  tolerance.
- **Whether these arrays stay committed after `cython-to-rust` closes**
  is `../PLAN.md`'s second anticipated ADR and is still open. Nothing in
  this task forecloses either answer; the argument for keeping them is
  that they are the last evidence a repaired value was ever checked
  against a non-Rust implementation.

## Plan Impact

**Impact Level:** Update `../PLAN.md`, `../rules.md` and
`../references/defect-blast-radius.md`. No ADR.

This task was supposed to produce arrays. It also produced measurements
that falsify parts of the plan, and under "Change control" those get
patched rather than noted. Five edits, all inside this project:

1. **`../references/defect-blast-radius.md`, the deletion schedule.**
   Two waves had already passed when this task started and the table
   listed neither. It now carries them, marked closed, with what each
   cost and how it was recovered.
2. **Same file, the A1 row.** It asked whether `rest` moves and said
   "measure it". Measured: it does not, 0 of 1750 positions.
3. **Same file, the A2 row.** Predicted 7 cases; measured 1. The row now
   says so, and says why the graph misled — a composition edge is not a
   path to a defect that sits behind a guard.
4. **Same file's coverage arithmetic, and `../rules.md` rule 7.** With
   A2 at one case, `A3 ⊆ A2` inverts into disjointness and rule 7's
   example ("A2 and A3 on both rho cases") becomes false; it is now
   "A3 and B3". The union still comes to 20 and the sum still comes to
   41, which is the check that file asks for.
5. **`../PLAN.md`, the Task 4, 7 and 8 gates.** Task 4 asserted an
   upward one-signed delta, which is false in `rest_plus_eps`; Task 7
   named 7 cases, which is 1, and did not anticipate that its whole
   delta is negative; Task 8 called A3 a strict subset of A2 and named
   the narrower of two candidate integration bounds. All three now state
   what was measured and point here for the evidence.

None of this touches `cython-to-rust` — no phase, task, or ordering of
the port moved, and nothing here needs its change control. Within this
project the corrections are to gates that no task has executed yet, so
they invalidate no completed work.

An ADR was considered and is not warranted: none of the five changes a
*decision*, they correct statements of fact against a measurement. The
one genuine decision this task surfaced — whether A2's repair needs a
floor at zero — is Task 7's to make, and is filed under Open Questions
rather than pre-empted here.

## Handoff to Next Task

- **Read first:** `test/parity/oracles/README.md`, then the patch for
  whichever defect you are repairing. The patch is the statement of the
  repair; the arrays are the check on it.
- **Safe to assume:** the deadline is bought out. Every Group A oracle is
  committed, and `cython-to-rust` Tasks 4.6, 6.2, 6.3 and 6.4 can now run
  in any order without stranding a repair — `../rules.md` rule 8 is
  discharged.
- **Still risky:** the A3 rho caveat above, and the fact that the case
  lists in `defects.py` come from a *prediction*. Every repair task still
  re-derives its own radius by running the repaired kernel over the whole
  corpus; a case the oracle does not cover that turns out to move is a
  finding about the composition graph, not a tolerance to widen.
