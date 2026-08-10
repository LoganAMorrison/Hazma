from typing import overload

import numpy as np

# `hazma._core` is the Rust extension (cython-to-rust ADR-0001). Its
# per-domain submodules — photon, positron, neutrino, scalar_mediator,
# vector_mediator — exist but are empty until Phases 03-06 fill them;
# each gets its own stub alongside its first kernel.
#
# Two further submodules, `special` (cython-to-rust Task 3.2) and `quad`
# (Task 3.3), are deliberately not stubbed. `special` exposes `spence`,
# `bessel_k1` and `bessel_kn` only so `test/test_core_special.py` can
# sweep them against scipy — its three functions follow the same dispatch
# contract `roundtrip` documents below. `quad` exposes the QUADPACK port
# only so `test/test_core_quad.py` can put one Python integrand through
# both it and `scipy.integrate.quad`; it takes a callable rather than an
# array and so has no dispatch contract to describe. The kernels that
# will use either call the Rust side directly, and nothing under `hazma/`
# imports them — a stub would advertise a surface this package does not
# mean to offer.

# `roundtrip` is the scaffold's plumbing probe, not physics. It follows
# the dispatch contract every ported entry point uses: a Python float, a
# NumPy scalar, or a 0-d array returns a float; a 1-D float64 array
# returns a fresh 1-D float64 array. Anything else raises ValueError.
# The 0-d-array case is a float at runtime, which the second overload
# below cannot express — read it as "ndarray in, ndarray out for ndim 1".
@overload
def roundtrip(x: float) -> float: ...
@overload
def roundtrip(x: np.ndarray) -> np.ndarray: ...
