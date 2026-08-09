from typing import overload

import numpy as np

# `hazma._core` is the Rust extension (cython-to-rust ADR-0001). Its
# per-domain submodules — photon, positron, neutrino, scalar_mediator,
# vector_mediator — exist but are empty until Phases 03-06 fill them;
# each gets its own stub alongside its first kernel.

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
