# The sequencing premise, verified

**Audience:** anyone reading `PLAN.md`'s "The premise this project
corrects", and Task 11 (which sweeps the superseded copies).
**Nature:** Grounded facts. Every claim carries the command that
produced it, run on this tree at `3e01590` on 2026-08-19.

## What the seven follow-ups said

Each carried some form of *"blocked until after `cython-to-rust`
Phase 06 Task 6.4"*, on the reasoning that the parity corpus pins the
shipped-but-wrong values by construction, so a repair fails the gate
that governs the remaining kernel swaps, and the fix therefore needs a
declared corpus regeneration that `projects/cython-to-rust/rules.md`
rule 2 forbids until the port is done.

The premise is right. The conclusion inverts the causality: Task 6.4 is
not the task that *unblocks* re-pinning, it is the task that *destroys
the ability to re-pin*.

## Claim by claim

### 1. Four of the seven defects still have a live Cython twin

```sh
find hazma -name "*.pyx" -o -name "*.pxd" | sort
```

Present, among others: `hazma/spectra/_photon/_pion.pyx`,
`hazma/spectra/_photon/_muon.pyx`, `hazma/spectra/_positron/_muon.pyx`,
`hazma/_utils/boost.pyx` — the twins of the charged-pion forward cone,
the muon photon endpoint, the positron-muon normalization and the boost
integral window respectively.

### 2. Three do not

```sh
git log --diff-filter=D --name-only --oneline --all \
    -- 'hazma/spectra/_photon/*.pyx' 'hazma/spectra/_positron/*.pyx'
```

`hazma/spectra/_photon/_eta_prime.pyx` and
`hazma/spectra/_photon/_phi.pyx` were deleted in `0954e5a`
("feat(spectra): port the tabulated photon spectra to rust", Task 4.2);
`hazma/spectra/_photon/_rho.pyx` in `b5f7f90` ("feat(spectra): port the
rho photon spectra to rust", Task 4.5). Each went in the same PR as its
swap, per `projects/cython-to-rust/rules.md` rule 1's no-drift-window
requirement.

The sources are still recoverable for review —
`git show 665aed5:hazma/spectra/_photon/_eta_prime.pyx` and
`git show b5f7f90^:hazma/spectra/_photon/_rho.pyx` both resolve — but a
historical blob is not a *runnable* oracle without reconstructing the
build it was compiled by, which is a different and much larger job than
calling a twin that is still in the tree.

### 3. Task 6.4 deletes exactly the four live twins

`projects/cython-to-rust/phases/phase-06-mediator-spectra.md`, Task 6.4
exit criteria, verbatim in substance: delete the four capi-survivor
extensions (`_photon/_muon`, `_photon/_pion`, `_positron/_muon`,
`_positron/_pion` `.pyx` + `.pxd`), `hazma/_utils/boost.{pyx,pxd}`,
`constants.pxd`, `kinematics.pxd`, `legacy_parameters.pxd`, and the
`spectra/_neutrino/_neutrino` struct module — after which
`find hazma -name "*.pyx" -o -name "*.pxd"` returns nothing.

### 4. A whole-corpus regeneration is *already* impossible

`test/parity/generate.py`'s `generate()` calls
`corpus.assert_no_rust_core()` before it evaluates anything, and
`test/parity/cases.py`'s implementation raises as soon as
`rust_core_kernels()` is non-empty — that is, as soon as `hazma._core`
*serves* a kernel, which it has since Phase 04 Task 4.1 (2026-08-11)
swapped `dnde_positron_muon` — the project's own walking skeleton, and
the first repointed wrapper of the port.

This is the load-bearing correction. Five of the seven follow-ups
proposed "one declared regeneration after Phase 06 Task 6.4, not four"
as the cheap path. That path has not existed since 2026-08-11, and after
Task 6.4 it could only be reopened by relaxing the very guard rule 2
asks for. The mechanism this project builds instead — declare the delta,
keep the array — is not a workaround for a scheduling problem; it is the
only shape a repair can take.

### 5. The twins are `cdef`-only, so the oracle route is `__pyx_capi__`

```sh
grep -n "^def \|^cdef .*(" hazma/spectra/_photon/_pion.pyx \
    hazma/spectra/_photon/_muon.pyx hazma/spectra/_positron/_muon.pyx \
    hazma/_utils/boost.pyx
```

Zero top-level `def`s across all four; every entry is `cdef`. They
survive to Task 6.4 solely because the mediator spectrum `.pyx` still
`cimport` their `__pyx_capi__` symbols. `test/test_core_boost.py`
already drives `hazma._utils.boost` through those capsules as an oracle
for the Rust port, which is the existing precedent Task 2 reuses.

### 6. The deadline is three dates, not one

A corrected corpus value has to be reachable through the *whole* Cython
composition chain, and the chain dies before the twin does. From
`projects/cython-to-rust/phases/phase-04-spectra-kernels.md` Task 4.6
and `phase-06-mediator-spectra.md` Tasks 6.2–6.4 — see
[`defect-blast-radius.md`](defect-blast-radius.md) for the per-defect
mapping. Task 4.6 is the only Phase 04 task still open
(`projects/cython-to-rust/task-notes/README.md` Phases table), so the
first window is the one closing soonest.

## What this does *not* claim

- It does not claim the Cython twins are *correct*. They are the
  implementation the defects were found in. What they provide is an
  **independent** implementation, which is a different property, and
  it is why `rules.md` rule 4 requires a physics invariant alongside
  every oracle comparison.
- It does not claim the three twin-less defects are harder. They are
  easier: each has a closed-form delta (`PLAN.md` Task 3), which is why
  their corrected follow-up wording is "corpus re-pinning only" rather
  than a deadline.
- It does not claim `cython-to-rust` erred. The port preserved these
  defects deliberately, under its own rule 1, and filed every one of
  them. The error is in the follow-ups' sequencing conclusion, drawn
  once and then copied across seven files.
