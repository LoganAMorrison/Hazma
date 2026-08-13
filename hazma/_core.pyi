from typing import overload

import numpy as np

# `hazma._core` is the Rust extension (cython-to-rust ADR-0001). Its
# per-domain submodules — photon, positron, neutrino, scalar_mediator,
# vector_mediator — are filled one kernel at a time by Phases 04-06;
# `positron` carries the first, `dnde_positron_muon` (Task 4.1), and
# `photon` the seven tabulated meson spectra (Task 4.2).
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
# `test/test_core_boost.py` against the Cython twin through
# `hazma._utils.boost.__pyx_capi__`. Those three follow the same dispatch
# contract `roundtrip` documents below, on the argument their Cython call
# sites sweep. `quad` exposes the QUADPACK port only so
# `test/test_core_quad.py` can put one Python integrand through both it
# and `scipy.integrate.quad`; it takes a callable rather than an array and
# so has no dispatch contract to describe. `dispatch` exposes three probes
# over the argument-and-error layer itself, each taking the quantity
# wording as an argument, only so `test/test_core_dispatch.py` can render
# every message the `.pyx` sources contain and compare bytes. The kernels
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
