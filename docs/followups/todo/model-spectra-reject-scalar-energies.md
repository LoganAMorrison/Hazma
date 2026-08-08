# The model spectrum dicts reject the scalar energies they document

- **Added:** 2026-08-08
- **Source:** `projects/cython-to-rust/task-notes/phase-01/task-1.4-legacy-npy.md`
- **Scope:** cross-cutting
- **Status:** open
- **Triggers / blockers:** ripens with Phase 04–06 of `cython-to-rust`. The
  `positron_spectra` half is a *compiled* signature constraint, so the Rust
  port decides its fate either way — better to make the call deliberately than
  to inherit it.

## Why

Both methods advertise `e_gams : float or float numpy.array` (and
`e_ps` likewise) in their NumPy-style docstrings, and `AGENTS.md` states the
repo-wide contract as "**Arrays in, arrays out.** Spectrum functions accept
scalars or NumPy arrays and broadcast". Neither method does. Measured on the
current tree (2026-08-08, `hazma` built from `master` at `7a81ce4`):

```text
ScalarMediator   spectra            scalar TypeError: object of type 'float' has no len()
ScalarMediator   positron_spectra   scalar TypeError: Argument 'eng_ps' has incorrect type (expected numpy.ndarray, got float)
KineticMixing    spectra            scalar TypeError: object of type 'float' has no len()
KineticMixing    positron_spectra   scalar TypeError: Argument 'eng_ps' has incorrect type (expected numpy.ndarray, got float)
```

`total_spectrum` and `total_positron_spectrum` accept a scalar, because they
wrap the argument in a one-element array first — so the failure is confined to
the per-channel dict methods, which is exactly why nobody has hit it.

The two failures have different causes and want different fixes:

- **`spectra`** dies in pure Python. `Theory.spectra` branches on
  `type(e_gams) == float` for *closed* channels only, so an open channel
  reaches a kernel wrapper that assumes a sequence — e.g.
  `hazma/scalar_mediator/_scalar_mediator_spectra.py:20`,
  `np.array([0.0 for _ in range(len(e_gams))])`. The `type(...) == float`
  test is itself wrong for a NumPy scalar.
- **`positron_spectra`** dies at the Cython boundary: the compiled positron
  kernels declare `np.ndarray` parameters, so a Python float is refused before
  any Hazma code runs.

## What

Decide the contract, then make the code and the docstrings agree — the
current state is the one option that is not defensible.

If the contract stays "scalars broadcast":

- Normalize once at the public boundary (`np.asarray(..., dtype=float)`,
  remember `ndim == 0`, reshape the result back) rather than in each of the
  ~10 channel wrappers.
- Replace the `type(e_gams) == float` branch in `Theory.spectra` with a
  shape-based test; it misses `np.float64` today.
- The Rust port can take scalars directly, so the `positron_spectra` half
  resolves itself if the boundary is done in Phase 04–06.

If the contract becomes "arrays only", say so in every affected docstring and
raise a `TypeError` with a message naming the fix, rather than letting a
`len()` failure surface from three frames down.

Either way add the scalar-input case to `test/test_theory_aggregation.py`,
which covers this aggregation layer and deliberately omits the scalar case
today (with a pointer to this file).

## Entry points

- `hazma/theory/__init__.py:152` — `Theory.spectra`, and its
  `type(e_gams) == float` branch
- `hazma/theory/__init__.py:323` — `Theory.positron_spectra`
- `hazma/scalar_mediator/_scalar_mediator_spectra.py:20` — the `len(e_gams)`
  that raises
- `hazma/scalar_mediator/_scalar_mediator_positron_spectra.py`,
  `hazma/vector_mediator/_vector_mediator_positron_spectra.py` — the compiled
  `np.ndarray` boundary
- `test/test_theory_aggregation.py` — the suite that would carry the
  regression test
- Related project: `projects/cython-to-rust/` (Phases 04–06 rewrite these kernels)

## Risks / open questions

- Normalizing at the boundary changes the *return* type for scalar input
  (currently an exception, so nothing can depend on it) but must not change
  it for array input. That is a `docs/versioning.md` question the fixing PR
  answers, not this file.
- `AGENTS.md` states the broadcast contract for "new public functions". These
  are old ones, so the "arrays only" resolution is available without
  contradicting it — but it should then be written down.
