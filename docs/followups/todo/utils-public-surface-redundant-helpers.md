# `hazma.utils` carries two helpers that should not become public API

- **Added:** 2026-08-05
- **Source:** conversation — a docs audit asking why
  `docs/source/utils.rst` omits `hazma.utils.minkowski_dot`
- **Scope:** cross-cutting
- **Status:** open
- **Triggers / blockers:** ripens when cython-to-rust ADR-0003 deletes
  `hazma/gamma_ray.py` and `docs/source/gamma_ray.rst` with it — that
  removes `minkowski_dot`'s only public-docs reference. Both cleanups
  are name removals, so they want the same major bump the
  cython-to-rust project already carries.

## Why

`docs/source/utils.rst` documents five of `hazma/utils.py`'s functions
(`kallen_lambda`, `two_body_momentum`, `cross_section_prefactor`,
`ldot`, `lnorm_sqr`). Two public names are missing from it, and in both
cases the right fix is to remove the function rather than to document
it. This note exists so
the omission is not re-filed as a doc bug and "fixed" by adding
`autofunction` entries — which would promote both names to the
supported surface that `docs/versioning.md` defines, making later
removal a breaking change.

**`minkowski_dot` duplicates `ldot`.** Both are pure Python, and both
evaluate `p0*q0 - p1*q1 - p2*q2 - p3*q3` in that term order, so on a
single four-vector there is no room for a difference —
`test/test_utils.py::test_minkowski_dot_matches_ldot` pins them as
bit-for-bit equal over 100 random pairs. The only behavioral
difference is the input contract: `minkowski_dot` is index-based and
so accepts plain lists and tuples, while `ldot` asserts
`lv.shape[axis] == 4` and additionally handles an `axis` for stacked
arrays. Every current call site passes a shape-`(4,)` `ndarray`, which
both accept.

It exists only because cython-to-rust Task 0.3 deleted the Cython
`hazma.field_theory_helper_functions.common_functions.minkowski_dot`
and gave the name a pure-Python home so its callers kept working. Its
one in-library consumer is
`hazma/experimental/axial_vector_mediator/avm_msqrd.py`, and
`docs/versioning.md` excludes `hazma/experimental/` from the public
surface outright. Its one public-docs reference is a worked
`gamma_ray_fsr` example — in the page ADR-0003 removes.

**`kinematically_accessable` is dead.** It has zero callers repo-wide,
no docstring, no type annotations, and a misspelled name
("accessable"). An `autofunction` entry would render an empty stub.

## What

Deliberately **not** done: adding `autofunction` entries for either
name to `docs/source/utils.rst`. Instead, when the trigger above
fires:

- Fold `minkowski_dot` into `ldot`. Either relax `ldot` to accept any
  length-4 sequence (a length check in place of the `.shape`
  assertion, which is what currently rejects lists) or leave `ldot`
  alone if no caller needs the looser contract. Then repoint
  `hazma/experimental/axial_vector_mediator/avm_msqrd.py`, delete
  `minkowski_dot`, and drop or retarget the five `minkowski_dot` tests
  in `test/test_utils.py` — keeping the sign-convention and
  on-shell-invariant coverage against `ldot`, since those pin physics
  and not the wrapper.
- Delete `kinematically_accessable`, or give it a docstring,
  annotations, the corrected spelling, and a caller if some code
  should be using it.

`minkowski_dot` landed in `e94fb21`, after the `2.1.0` tag, so it is
unreleased today and removing it currently costs nothing. That is only
true until the next release ships it — which is the argument for
resolving this inside the cython-to-rust major rather than letting it
drift.

## Entry points

- `hazma/utils.py:71` — `kinematically_accessable`.
- `hazma/utils.py:191` — `ldot`, the survivor.
- `hazma/utils.py:215` — `minkowski_dot`.
- `docs/source/utils.rst` — the five-entry API Reference list.
- `docs/source/gamma_ray.rst:85` — the sole public-docs reference,
  inside the `gamma_ray_fsr` example.
- `hazma/experimental/axial_vector_mediator/avm_msqrd.py:10` — the sole
  in-library consumer (non-public per `docs/versioning.md`).
- `test/test_utils.py:258-311` — the `minkowski_dot` block.
- Prior decision: `projects/cython-to-rust/adrs/ADR-0003-remove-gamma-ray-module.md`.
- Prior task: `projects/cython-to-rust/task-notes/phase-00/task-0.3-delete-superseded.md`.

## Risks / open questions

- Relaxing `ldot`'s assertion is a widening of a public contract, not a
  narrowing, so it is `minor` on its own — but it changes the error a
  caller sees for a malformed input from `AssertionError` to whatever
  the length check raises. Prefer a `ValueError` from
  `hazma/hazma_errors.py` if one fits.
- The old Cython `minkowski_dot` was *not* bit-identical to today's
  Python one (the C compiler contracts `a*b - c*d` into an FMA;
  measured ≤2.7e-14 relative, declared in
  `projects/cython-to-rust/task-notes/phase-00/README.md`). That drift
  is already absorbed and is not re-opened by this cleanup: collapsing
  onto `ldot` is exactly zero further change, per the bit-for-bit test.
- Sequencing: doing this before ADR-0003 lands means editing a
  `gamma_ray.rst` example that is about to be deleted anyway. After is
  simpler and strictly less work.
