# Versioning

Hazma follows [Semantic Versioning](https://semver.org/) over its
**public Python API**. This document defines that surface, gives the
litmus test for choosing a bump, and lists the files a closing PR must
touch.

## The version lives in one place

```python
# hazma/__init__.py
VERSION: Final[str] = "2.0.2"
__version__ = VERSION
```

`pyproject.toml` reads it dynamically (`version = { attr =
"hazma.VERSION" }`), so there is exactly one number to edit. Do not add a
second copy.

> `_build.py` carries an unrelated, stale `VERSION` constant that is not
> the package version and is not read by the build metadata. Leave it
> alone; it is not part of a version bump.

## The public surface

A change is user-facing if it changes any of these:

1. **Import paths.** `hazma.spectra.dnde_photon`, `hazma.theory.Theory`,
   the model packages — anything a user writes in an `import` statement.
2. **Signatures.** Function and method names, positional order, keyword
   names, defaults, and whether an argument is required.
3. **Return shapes and units.** Scalar vs array, tuple arity, dict keys,
   dtype, and the physical units of every returned quantity.
4. **Numerical output.** The values the library computes. A spectrum that
   moves is a user-facing change even when no signature changed.
5. **Exception types.** What is raised, and when (`hazma/hazma_errors.py`).
6. **`hazma/deprecated/`.** It stays importable; removing or changing
   anything there is a break.
7. **Supported Python versions** and required runtime dependencies.

Explicitly **not** the public surface: `hazma/experimental/`, anything
under a leading-underscore package that is not re-exported
(`hazma/_decay/`, `hazma/_utils/`, …), `notebooks/`, `test/`, internal
helper names, docstring wording, and performance characteristics.

## Choosing the bump

The litmus test, applied in order — the first line that matches wins:

**`major`** — existing correct user code stops working or silently
changes meaning:

- A public function, class, module, or keyword argument is **removed or
  renamed**.
- A return shape or unit changes (`dN/dE` in `MeV⁻¹` becomes `GeV⁻¹`; a
  scalar becomes an array).
- A required argument is added, or an argument's default changes in a way
  that changes results.
- The minimum Python version rises.
- Anything in `hazma/deprecated/` is removed.

**`minor`** — additive, or a deliberate correction to a published number:

- A new public function, class, model, channel, or keyword argument with
  a backward-compatible default.
- **A numerical result changes because a physics bug was fixed.** This is
  the carve-out worth reading twice: the API is unchanged, so it is not
  `major`, but a user's plot moves, so it is not `patch`. Name the
  affected functions and the size of the change in `CHANGELOG.md`.
- A previously-raising input now returns a value (or vice versa) as a
  deliberate correctness fix.
- A dependency's minimum version rises.

**`patch`** — no user-visible behavior change:

- Bug fix whose output was previously an exception, a NaN, or an obvious
  crash — not a plausible-looking wrong number.
- Performance work with numerically identical output.
- Docs, tests, typing, packaging, CI, internal refactors.

Default to `patch`. Raising `patch → minor → major` mid-project is fine
and expected when scope shifts. Lowering requires a one-line note in
`task-notes/README.md` explaining why the change is no longer
user-facing.

### The numerical-change rule

If you cannot tell whether a change moves a number, **measure it**. Run
the affected function before and after on a representative grid and diff
the arrays. "The tests still pass" is not evidence: a `rtol=1e-3`
assertion absorbs a 0.1% shift, and 0.1% on a published spectrum is a
`minor`, not a `patch`.

## What a closing PR must contain

The PR that flips a `projects/<slug>/PLAN.md` frontmatter `status:` to
`Complete` carries the bump in the same diff:

1. **`hazma/__init__.py`** — `VERSION` set to the new value.
2. **`CHANGELOG.md`** — a new `## [X.Y.Z] — YYYY-MM-DD` section that
   names the project slug and lists user-facing changes under
   `Added` / `Changed` / `Fixed` / `Removed`. Numerical changes go under
   `Changed` with the magnitude stated.

Verify locally before committing:

```bash
scripts/agents/preflight.sh --closing
```

It checks that `VERSION` actually moved relative to the trunk and that
`CHANGELOG.md` has a matching section. A partial closure (status flipped,
version untouched) fails the gate.

## Cheat sheet

| Change                                              | Bump    |
|-----------------------------------------------------|---------|
| New `dnde_photon` final state                        | `minor` |
| Renamed a keyword argument                           | `major` |
| Fixed a wrong interpolation → spectrum shifts 2%     | `minor` |
| Fixed a crash on an empty energy array               | `patch` |
| Vectorized a loop, identical output                  | `patch` |
| Changed returned units from MeV to GeV               | `major` |
| Added a new model package                            | `minor` |
| Tightened a docstring                                | `patch` |
| Dropped Python 3.10 support                          | `major` |
