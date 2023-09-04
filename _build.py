"""Install script for Hazma."""

# pylint: disable=invalid-name

from typing import List

from Cython.Build import cythonize
from setuptools import Extension
from setuptools.command.build_py import build_py as _build_py

VERSION = "2.0.0-rc1"


class build_py(_build_py):
    def run(self):
        self.run_command("build_ext")
        return super().run()

    def initialize_options(self) -> None:
        super().initialize_options()
        if self.distribution.ext_modules == None:
            self.distribution.ext_modules = []

        if self.distribution.include_dirs == None:
            self.distribution.include_dirs = []

        import numpy

        self.distribution.include_dirs.append(numpy.get_include())

        extensions = []

        # Cython utilities
        extensions += make_extension(["_utils"], ["boost"])

        # Gamma-Ray Helper
        extensions += make_extension(
            ["_gamma_ray"], ["gamma_ray_generator", "gamma_ray_fsr"], cpp=True
        )

        # Phase space
        extensions += make_extension(
            ["_phase_space"], ["generator", "histogram", "modifiers"], cpp=True
        )

        # Field Theory helper
        extensions += make_extension(
            ["field_theory_helper_functions"],
            ["common_functions", "three_body_phase_space"],
            cpp=True,
        )

        # Positron
        extensions += make_extension(
            ["_positron"], ["positron_muon", "positron_charged_pion", "positron_decay"]
        )
        # Neutrino
        extensions += make_extension(["_neutrino"], ["charged_pion", "muon"])

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

        # RH-neutrino
        # extensions += [
        #     Extension(
        #         "hazma.rh_neutrino._rh_neutrino_fsr_four_body",
        #         sources=["hazma/rh_neutrino/_rh_neutrino_fsr_four_body.pyx"],
        #         extra_compile_args=["-g", "-std=c++11"],
        #         language="c++",
        #     )
        # ]

        self.distribution.ext_modules.extend(extensions)


def make_extension(module: List[str], sources: List[str], cpp=False):
    """Build a Cython extension module."""
    package = ".".join(["hazma", *module])
    path = "/".join(["hazma", *module])

    extensions = []
    for src in sources:
        m = package + "." + src
        p = [path + "/" + src + ".pyx"]
        if cpp:
            exts = Extension(m, p, extra_compile_args=["-std=c++11"], language="c++")
        else:
            exts = Extension(m, p)
        for ext in cythonize(exts):
            extensions.append(ext)
    return extensions
