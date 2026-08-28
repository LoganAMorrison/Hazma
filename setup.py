"""Build script declaring hazma's compiled extension module.

Project metadata lives in pyproject.toml; this file only declares
rust_extensions. They must be declared eagerly (not injected from a
build_py subclass) so setuptools knows the distribution is impure before
it picks wheel tags — otherwise wheels come out mistagged as
py3-none-any.

One toolchain remains. The Cython half this file also carried for the
duration of the cython-to-rust migration (ADR-0001) is gone: Phases
03-06 moved every kernel to the Rust crate in ``rust/``, and Phase 06
deleted the last ``.pyx`` and ``.pxd``. Phase 07 swaps the backend to
maturin, at which point this file goes away too.
"""

# pylint: disable=invalid-name

from setuptools import setup
from setuptools_rust import Binding, RustExtension

# One cdylib, at its final import path from day one, built against
# CPython's limited API so a single shared object serves every supported
# interpreter. Now that no Cython extension is left to hold the wheels to
# a single CPython, the abi3 tag is the distribution's to claim; the
# claim is verified by the installed file being named `_core.abi3.so`.
rust_extensions = [
    RustExtension(
        "hazma._core",
        path="rust/Cargo.toml",
        binding=Binding.PyO3,
        py_limited_api=True,
    )
]

setup(rust_extensions=rust_extensions)
