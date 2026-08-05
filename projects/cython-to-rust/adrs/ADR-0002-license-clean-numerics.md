# ADR 0002: License-clean numerics — cephes in-tree, GSL-derived code out

**Date:** 2026-08-03
**Status:** Accepted (signed off by Logan 2026-08-04 — Hazma stays MIT;
no GPL-3 crates)
**Scope:** Project-scoped (applies only within `projects/cython-to-rust/`).

## Context

The Rust core needs three special functions (`spence`/Li₂, `K₁`, `Kₙ`)
and a finite-interval adaptive integrator matching
`scipy.integrate.quad` (QUADPACK `qags`/`qagp`). Logan's existing
rust-cyphus crates cover much of this and were assessed on 2026-08-03
(full results in `../references/numerics-replacements.md`):
`cyphus-integration` passes 43/44 tests on rustc 1.96 after a one-line
fix and includes exactly the needed `qags`/`qagp`; `cyphus-specfun`
passes 99/102 and has Bessel K but no dilogarithm.

Both crates are **GPL-3**: they are ports of the GNU Scientific Library
and carry GSL's copyright headers. Authorship does not help — a port of
GPL code is a derivative work, so they cannot be relicensed. Hazma is
MIT (LICENSE, © Logan Morrison and Adam Coogan). Linking or vendoring
GPL-3 code into the extension would make the distributed wheels GPL-3.

Meanwhile scipy's own `spence`/`k1`/`kn` are **cephes** wrappers, and
QUADPACK itself (Piessens et al., netlib Fortran — what scipy vendors)
is **public domain**; GSL's GPL applies to GSL's reimplementation, not
to the upstream QUADPACK sources or the published algorithms.

## Decision

1. **No GSL-derived code enters this repository or its dependency graph**
   — not vendored, not a crates.io/git dependency, not a dev-dependency.
2. Special functions come from the **cephes lineage**: the `spec_math`
   crate (MIT OR Apache-2.0, pure-Rust cephes; `bessel_k1`, `bessel_kn`,
   `Polylog::li2` → `cephes64::spence`). If a gap or parity miss
   appears, the fallback is a direct in-tree Rust translation of the
   specific cephes routine. This is also the numerically correct choice:
   scipy's functions are cephes, so cephes-lineage code matches scipy
   more closely than GSL-lineage code would.
3. The integrator is a Rust port of **netlib QUADPACK** (`qk15`, `qk21`,
   `qelg`, `qags`, `qagp`; finite intervals only), translated from the
   public-domain Fortran — explicitly *not* from GSL or from
   cyphus-integration.
4. The cyphus crates are used only as **out-of-repo, dev-time oracles**:
   running them locally to cross-check values or subdivision behavior is
   fine (no different from testing against GSL itself); their code is
   not read side-by-side while writing the Hazma port, and nothing from
   them is committed. Primary parity oracle remains scipy, via the
   Phase 01 corpus and direct comparisons in `test/`.

## Consequences

- **Positive:** Hazma stays MIT with an unambiguous provenance chain
  (public-domain QUADPACK + permissive cephes); parity with scipy is
  algorithm-for-algorithm, minimizing corpus drift; a second independent
  implementation (cyphus) is available as a cross-check without license
  exposure.
- **Negative:** the QUADPACK port (~1,500–2,500 lines) is written fresh
  instead of reusing cyphus-integration, costing roughly 2–3 days that
  reuse might have saved; Fortran-to-Rust translation of `qelg`
  ε-extrapolation is fiddly.
- **Mitigation:** scope is the finite-interval subset only (`qagi` has
  zero live callers); the corpus and per-integrand scipy comparisons
  catch translation faults; cyphus-integration's passing test suite
  demonstrates the job is bounded and was done once by the same author.
- **Foreclosed alternative (settled by the sign-off):** accepting GPL-3
  for the wheels and depending on modernized cyphus crates directly.
  Rejected because it changes Hazma's license terms for every
  downstream user and requires co-author agreement, to save at most a
  few days of foundation work. Hazma stays MIT; revisiting this needs a
  new ADR superseding this one, not a judgment call inside a task.
