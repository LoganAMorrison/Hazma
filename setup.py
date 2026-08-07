"""Build script declaring hazma's Cython extension modules.

Project metadata lives in pyproject.toml; this file only declares
ext_modules. They must be declared eagerly (not injected from a build_py
subclass) so setuptools knows the distribution is impure before it picks
wheel tags — otherwise wheels come out mistagged as py3-none-any.
"""

# pylint: disable=invalid-name

from typing import List

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup


def make_extension(module: List[str], sources: List[str], cpp=False):
    """Build a Cython extension module."""
    package = ".".join(["hazma", *module])
    path = "/".join(["hazma", *module])

    extensions = []
    for src in sources:
        m = package + "." + src
        p = [path + "/" + src + ".pyx"]
        include_dirs = [numpy.get_include()]
        if cpp:
            exts = Extension(
                m,
                p,
                extra_compile_args=["-std=c++11"],
                language="c++",
                include_dirs=include_dirs,
            )
        else:
            exts = Extension(m, p, include_dirs=include_dirs)
        for ext in cythonize(exts):
            extensions.append(ext)
    return extensions


extensions = []

# Cython utilities
extensions += make_extension(["_utils"], ["boost"])

# Decay Spectra
extensions += make_extension(
    ["spectra", "_photon"],
    ["_muon", "_pion", "_rho", "_kaon", "_eta", "_omega", "_eta_prime", "_phi"],
)
extensions += make_extension(
    ["spectra", "_positron"],
    ["_muon", "_pion"],
)
extensions += make_extension(
    ["spectra", "_neutrino"],
    ["_muon", "_pion", "_neutrino"],
)

# Scalar mediator
extensions += make_extension(
    ["scalar_mediator"],
    [
        "scalar_mediator_decay_spectrum",
        "scalar_mediator_positron_spec",
        "_c_scalar_mediator_cross_sections",
    ],
)

# Vector mediator
extensions += make_extension(
    ["vector_mediator"],
    [
        "vector_mediator_decay_spectrum",
        "vector_mediator_positron_spec",
        "_c_vector_mediator_cross_sections",
    ],
)

setup(ext_modules=extensions)
