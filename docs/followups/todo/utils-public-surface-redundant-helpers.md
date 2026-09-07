# `hazma.utils.kinematically_accessable` should not be public API

- **Added:** 2026-08-05
- **Source:** conversation — a docs audit asking why
  `docs/source/utils.rst` omits `hazma.utils.minkowski_dot`
- **Scope:** cross-cutting
- **Status:** open — half resolved. `minkowski_dot` was removed on
  2026-09-06, before the 2.2.0 tag that would have made it public
  (see "What"). `kinematically_accessable` is what remains.
- **Triggers / blockers:** blocked on the next `major`.
  `kinematically_accessable` shipped in 2.1.0, so removing it breaks the
  public surface: it is importable and not in `hazma/deprecated/`, which
  puts it outside the reachability carve-out in `docs/versioning.md`.
  Giving it a docstring, annotations and the corrected spelling instead
  is `minor` and needs no window — but that is only worth doing if some
  code should be calling it, and none does.

## Why

`docs/source/utils.rst` documents five of `hazma/utils.py`'s functions
(`kallen_lambda`, `two_body_momentum`, `cross_section_prefactor`,
`ldot`, `lnorm_sqr`). `kinematically_accessable` is missing from it, and
the right fix is to remove the function rather than to document it. This
note exists so the omission is not re-filed as a doc bug and "fixed" by
adding an `autofunction` entry — which would advertise as supported a
name that is already awkward to remove.

**`kinematically_accessable` is dead.** It has zero callers repo-wide,
no docstring, no type annotations, and a misspelled name
("accessable"). An `autofunction` entry would render an empty stub.

## What

Deliberately **not** done: adding an `autofunction` entry for
`kinematically_accessable` to `docs/source/utils.rst`. Instead, delete
it, or give it a docstring, annotations, the corrected spelling, and a
caller if some code should be using it. Deleting it needs a `major`.

### Resolved: `minkowski_dot`, removed 2026-09-06

It duplicated `ldot` — both pure Python, both evaluating
`p0*q0 - p1*q1 - p2*q2 - p3*q3` in that term order, pinned bit-for-bit
equal on a single four-vector. The only difference was the input
contract: `minkowski_dot` was index-based and so took lists and tuples,
while `ldot` asserts `lv.shape[axis] == 4` and handles an `axis` for
stacked arrays. Every call site passed a shape-`(4,)` `ndarray`, so
`ldot` absorbed them unchanged and its contract was left alone.

It landed in `e94fb21`, after the `2.1.0` tag, so it appeared in no
release and removing it was invisible to users — but only until 2.2.0
shipped it, after which it would have been an ordinary public name
outside the reachability carve-out in `docs/versioning.md` and removable
only in a `major`. It was dropped inside that window, in the 2.2.0
closing PR.

Its consumers were repointed at `ldot`:
`hazma/experimental/axial_vector_mediator/avm_msqrd.py` (excluded from
the public surface) and `notebooks/dev/gamma_ray_fsr/partial_integration.py`,
which stays dead on its own `hazma.rambo` and `hazma.gamma_ray` imports.
Both squared matrix elements in `avm_msqrd.py` were checked bit-for-bit
against the deleted implementation over 200 random momentum draws. The
three `test/test_utils.py` tests that pinned the metric signature and the
on-shell invariant were retargeted onto `ldot`, which had no direct
coverage of its own; the two that only exercised the wrapper — bit-for-bit
agreement with `ldot`, and accepting plain lists — were dropped.

## Entry points

- `hazma/utils.py:71` — `kinematically_accessable`.
- `docs/source/utils.rst` — the five-entry API Reference list.
- Prior decision: `projects/cython-to-rust/adrs/ADR-0003-remove-gamma-ray-module.md`.
- Prior task: `projects/cython-to-rust/task-notes/phase-00/task-0.3-delete-superseded.md`.

## Risks / open questions

- ~~Sequencing: doing this before ADR-0003 lands means editing a
  `gamma_ray.rst` example that is about to be deleted anyway.~~ Moot —
  the page is gone (Task 0.5, 2026-08-05), so this cleanup no longer
  has to touch it either way.
