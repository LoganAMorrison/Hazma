# `pip install -e .` builds `hazma._core` unoptimized

- **Added:** 2026-08-20
- **Source:** cython-to-rust Task 5.1 (benchmarking the vector cross sections)
- **Scope:** cross-cutting
- **Status:** done (2026-08-27, cython-to-rust Task 7.1)
- **Resolution:** the backend swap resolved it without a choice having to
  be made. The debug default belonged to `setuptools_rust.build_rust`,
  which reads `debug = self.inplace or self.debug` and so built debug for
  any `--inplace` (editable) install. maturin's PEP 517 hooks — the ones
  `pip install -e .` calls — build release unconditionally; only the
  separate `maturin develop` **CLI** defaults to debug, and nothing in
  this repo invokes it. Measured after the cutover on macOS/arm64:
  `uv pip install -e .` leaves `rust/target/release/` and no
  `rust/target/debug/`, and `vector_mediator.thermal_cross_section` at
  `x = 0.5` runs at **35.8 us** per call from that editable tree — the
  release order of magnitude in the table below, not the 1866 us debug
  one. `rules.md` rule 12's benchmark-against-Cython instruction is
  therefore sound as written from an ordinary editable tree again.

## Why

`setuptools_rust.build_rust` decides the cargo profile with
`debug = self.inplace or self.debug` when a `RustExtension` leaves
`debug` unset, and hazma's does (`setup.py`). An **editable** install
passes `--inplace`, so the documented dev loop —
`pip install -e .`, which `AGENTS.md` names as *the* rebuild command and
which `test/parity` requires — produces a `cargo build` **debug**
extension. A non-editable `pip install .` does not, so released wheels
are release builds; it is only the developer's tree that is slow.

The gap is not marginal. Measured on macOS/arm64 (M-series, Python
3.13.7), the same vector cross sections against the pre-swap Cython:

| entry point | Cython | Rust (debug) | Rust (release) |
| --- | --- | --- | --- |
| `sigma_xx_to_vv`, 1k-point array | 7.0 us | 133.8 us | 6.6 us |
| `sigma_xx_to_v_to_pi0g`, 1k scalars | 94.6 us | 2524.6 us | 97.2 us |
| `thermal_cross_section`, `x = 0.5` | 126.4 us | 1866.2 us | 65.2 us |

**Twenty times slower**, and the sign of the comparison flips: in debug
the port looks like a 20x regression against the Cython it replaces, and
in release it is a 1.1x-3.2x improvement. `projects/cython-to-rust/rules.md`
rule 12 asks every kernel task to benchmark against the pre-swap Cython
"on the same machine", so every such measurement taken from an editable
tree without knowing this is not merely imprecise — it points the wrong
way.

It also costs test wall-clock: `test/parity` is minutes of adaptive
quadrature and every kernel in it now runs 20x slower than it needs to,
locally and in CI (CI reinstalls editable before its test step —
`.github/workflows/ci.yml`).

## What

Set `debug=False` on the `RustExtension` in `setup.py`, so both install
forms build release. One line.

What makes it a decision rather than a fix is the rebuild cost:
`rust/Cargo.toml`'s `[profile.release]` sets `lto = true` and
`codegen-units = 1`, so a one-file edit costs **64 s** to reinstall
against roughly 10 s in debug. That is paid on every kernel edit, and
Phases 05-06 are nothing but kernel edits.

Three ways out, and the third is probably right:

1. `debug=False` and accept the 64 s loop.
2. `debug=False` plus a `[profile.dev]` that is optimized but not
   LTO'd — `opt-level = 2`, `lto = false`, default `codegen-units` —
   which recovers most of the speed at a fraction of the link time.
   Needs `debug` left inferred and the profile chosen instead.
3. Leave the default and **document it**: add the fact to
   `docs/agents/environment.md` beside "editing a `.rs` requires a
   rebuild", and have `rules.md` rule 12 say which profile a benchmark
   must be taken in. Cheapest, and it makes the trap visible where
   agents already read.

Whichever is chosen, `docs/agents/environment.md` should say it: today
that file explains that `cargo build` is not the rebuild without saying
that the rebuild is unoptimized.

## Entry points

- `setup.py` — the `RustExtension(...)` block.
- `rust/Cargo.toml` — `[profile.release]`, and where a `[profile.dev]`
  would go.
- `docs/agents/environment.md` — the "editing a `.rs`" paragraph.
- `projects/cython-to-rust/rules.md` — rule 12 (Process 3), the
  benchmark rule this undermines.
- Related project: `projects/cython-to-rust/` (Phase 07 Task 7.1)

## Risks / open questions

- ~~**Not a risk: the two profiles are numerically identical.**~~
  **Corrected 2026-08-27 by Task 7.1, which switched the profile and
  measured the result.** The original reading — cython-to-rust Task 5.3
  re-ran its whole relic-density sweep against a debug and a release
  `hazma._core`, 12 relic densities and 78 `thermal_cross_section`
  values across six mediator model points, every one at `rtol = 0` —
  holds for those functions and does not generalize. `[profile.release]`'s
  `lto = true` and `codegen-units = 1` do move doubles elsewhere: the
  mediator table grids come out **one ulp** from `numpy.logspace` at 4 of
  500 abscissae at m = 550 MeV and 1 of 500 at m = 900 MeV under release,
  and bit-equal under debug.

  Nothing users receive moved, on two independent grounds. Every
  published wheel has always been a release build, so the release values
  are the shipped ones and the debug values were only ever the
  developer's — and the two agree anyway on the public surface: a
  16-function sweep over 7,206 values (the ten `dnde_*` spectra at
  m = 900 MeV over 400 log-spaced energies, plus both mediator models'
  total photon and positron spectra and three `thermal_cross_section`
  points each) is **bit-equal** between a debug and a release
  `hazma._core` built from identical sources. The one-ulp grid difference
  does not reach a published number; it is visible only to a test that
  reads `mediator_tables` directly. What the switch changed is which of
  the two builds the test suite measures, and
  `test/test_core_mediator_tables.py`'s grid comparison had encoded the
  debug values as an exact claim scoped by *platform*. Task 7.1 moved that
  comparison to the one-ulp budget the same module already derived; see
  `assert_matches_numpy_grid` for the reasoning.

  The durable lesson is the one this bullet got backwards: **a
  profile-parity result is scoped to the functions it measured.** A debug
  tree is the right place to take a parity reading only for kernels whose
  arithmetic has been checked across the profiles; timing taken there is
  wrong unconditionally.

- Phase 07 replaces setuptools-rust with maturin, whose `develop`
  command has the same debug default and a `--release` flag. Anything
  written into `setup.py` now is discarded then; anything written into
  `docs/agents/environment.md` is not.
- CI time is a real budget. Option 2 is the only one that does not
  trade one of the two costs for the other.
