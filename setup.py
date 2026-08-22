"""Build script declaring hazma's compiled extension modules.

Project metadata lives in pyproject.toml; this file only declares
ext_modules and rust_extensions. They must be declared eagerly (not
injected from a build_py subclass) so setuptools knows the distribution
is impure before it picks wheel tags — otherwise wheels come out
mistagged as py3-none-any.

Two toolchains coexist here for the duration of the cython-to-rust
migration (ADR-0001): the surviving Cython extensions, and the Rust
crate in ``rust/`` that becomes ``hazma._core``. Phases 03-06 move
kernels from the first to the second; Phase 07 deletes the Cython half
and swaps the backend to maturin, at which point this file goes away.
"""

# pylint: disable=invalid-name

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools_rust import Binding, RustExtension


def make_extension(module: list[str], sources: list[str]) -> list[Extension]:
    """Build a Cython extension module.

    Every surviving extension compiles as C. The C++ ones went with
    ``_gamma_ray/`` and ``_phase_space/`` in cython-to-rust Task 0.2, so
    this helper no longer carries a C++ branch.
    """
    package = ".".join(["hazma", *module])
    path = "/".join(["hazma", *module])

    extensions = []
    for src in sources:
        m = package + "." + src
        p = [path + "/" + src + ".pyx"]
        exts = Extension(m, p, include_dirs=[numpy.get_include()])
        for ext in cythonize(exts):
            extensions.append(ext)
    return extensions


extensions = []

# Cython utilities
extensions += make_extension(["_utils"], ["boost"])

# Decay Spectra
# Every spectra entry point is served by hazma._core as of cython-to-rust
# Task 4.6, so no extension under hazma/spectra/ has a top-level `def`
# left. (The mediator extensions further down still do — they are
# Phases 05-06.) The five
# tabulated photon extensions (_kaon, _eta, _omega, _eta_prime, _phi)
# went in Task 4.2, _rho in Task 4.5, and all three _neutrino extensions
# (_muon, _pion, _neutrino) in Task 4.6, so those nine .pyx are gone.
#
# The four below stay built for their `__pyx_capi__` capsules alone:
# _photon/_pion cimports _photon/_muon, _positron/_pion cimports
# _positron/_muon, and both mediator decay-spectrum modules and both
# mediator positron-spectrum modules cimport from these four. Phase 06
# Task 6.4 is where the files go.
extensions += make_extension(
    ["spectra", "_photon"],
    ["_muon", "_pion"],
)
extensions += make_extension(
    ["spectra", "_positron"],
    ["_muon", "_pion"],
)

# Scalar mediator
# `_c_scalar_mediator_cross_sections` went in cython-to-rust Task 5.2 —
# all twelve of its consumed entry points are served by
# `hazma._core.scalar_mediator`, and the thirteenth (`sigma_xx_to_all`)
# was dropped rather than ported because nothing imported it. The two
# spectrum modules below are Phase 06's.
extensions += make_extension(
    ["scalar_mediator"],
    [
        "scalar_mediator_decay_spectrum",
        "scalar_mediator_positron_spec",
    ],
)

# Vector mediator
# `_c_vector_mediator_cross_sections` went in cython-to-rust Task 5.1,
# the same way and for the same reasons as its scalar twin above. The
# two spectrum modules below are Phase 06's.
extensions += make_extension(
    ["vector_mediator"],
    [
        "vector_mediator_decay_spectrum",
        "vector_mediator_positron_spec",
    ],
)

# The Rust half. One cdylib, at its final import path from day one, built
# against CPython's limited API so a single shared object serves every
# supported interpreter. That is an *extension-level* property: the wheels
# stay CPython-tagged while any Cython extension remains, and the abi3
# claim is verified by the installed file being named `_core.abi3.so`.
rust_extensions = [
    RustExtension(
        "hazma._core",
        path="rust/Cargo.toml",
        binding=Binding.PyO3,
        py_limited_api=True,
    )
]

setup(ext_modules=extensions, rust_extensions=rust_extensions)
