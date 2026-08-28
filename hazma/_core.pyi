from typing import overload

import numpy as np

# `hazma._core` is the Rust extension (cython-to-rust ADR-0001). Its
# per-domain submodules — photon, positron, neutrino, scalar_mediator,
# vector_mediator — are filled one kernel at a time by Phases 04-06.
# Phase 04 closed on 2026-08-20 having filled the first three: `photon`
# carries all twelve public decay spectra, `positron` both of its
# compiled ones, and `neutrino` both of its. Phase 05 then filled both
# mediator submodules with their cross sections: `vector_mediator` with
# the six consumed `def`s of `_c_vector_mediator_cross_sections.pyx`
# (Task 5.1) and `scalar_mediator` with the twelve of its scalar twin
# (Task 5.2). Both submodules grow again in Phase 06.
#
# The live roster is discovered rather than listed — `test/parity/
# cases.py`'s `rust_core_kernels()` walks the extension — so this comment
# names phases rather than kernels, which is what keeps it from going
# stale one swap at a time.
#
# They are deliberately unstubbed, and a stub file is not the cheap fix
# it looks like: `_core` is a single extension, so a submodule stub needs
# a `hazma/_core/` stub *package* (`__init__.pyi` plus one file per
# submodule) that would then shadow this file. The typed surface users
# see is the wrapper, not the extension — every ported entry point is
# re-exported through an `@overload`-annotated function in
# `hazma/spectra/**/__init__.py`, which is what a caller imports and what
# `docs/versioning.md` defines the public API against. Phase 07 revisits
# the packaging, and the stub layout belongs with it.
#
# Five further submodules — `special` (cython-to-rust Task 3.2), `quad`
# (Task 3.3), `interp` and `boost` (Task 3.4), and `dispatch` (Task 3.5) —
# are deliberately not
# stubbed. `special` exposes `spence`, `bessel_k1` and `bessel_kn` only so
# `test/test_core_special.py` can sweep them against scipy; `interp` and
# `boost` expose the interpolation and boost foundation only so
# `test/test_core_interp.py` can sweep against `np.interp` and
# `test/test_core_boost.py` against a Python transcription of
# `rust/src/boost.rs` (it swept the Cython twin through
# `hazma._utils.boost.__pyx_capi__` until cython-to-rust Task 6.4 deleted
# that extension). Those three follow the same dispatch contract
# `roundtrip` documents below, on the argument their Cython call sites
# swept. `quad` exposes the QUADPACK port only so
# `test/test_core_quad.py` can put one Python integrand through both it
# and `scipy.integrate.quad`; it takes a callable rather than an array and
# so has no dispatch contract to describe. `dispatch` exposes three probes
# over the argument-and-error layer itself, each taking the quantity
# wording as an argument, only so `test/test_core_dispatch.py` can render
# every message and compare bytes against a frozen roster — the `.pyx`
# sources it used to extract them from are gone. The kernels
# that will use any of them call the Rust side directly, and nothing under
# `hazma/` imports them — a stub would advertise a surface this package
# does not mean to offer.

# `roundtrip` is the scaffold's plumbing probe, not physics. It follows
# the dispatch contract every ported entry point uses: a Python float, a
# NumPy scalar, or a 0-d numeric array returns a float; a 1-D float64
# array — or a sequence that converts to one — returns a fresh 1-D float64
# array. A higher-rank or non-float64 array raises ValueError; anything
# that is neither a real number nor a sequence raises TypeError.
# The 0-d-array case is a float at runtime, which the second overload
# below cannot express — read it as "ndarray in, ndarray out for ndim 1".
@overload
def roundtrip(x: float) -> float: ...
@overload
def roundtrip(x: np.ndarray) -> np.ndarray: ...
