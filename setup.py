from setuptools import setup, Extension, find_packages

import numpy as np

VERSION = "1.1"


def long_description():
    """Remove first four lines from the README for PyPI.
    """
    with open("README.md", encoding="utf-8") as f:
        ld = f.read()
    return "\n".join([str(l) for l in ld.split("\n")[4:]])


decay_dir = "hazma/decay_helper_functions/"
gr_dir = "hazma/gamma_ray_helper_functions/"
ps_dir = "hazma/phase_space_helper_functions/"
ft_dir = "hazma/field_theory_helper_functions/"
pos_dir = "hazma/positron_helper_functions/"
sm_dir = "hazma/scalar_mediator/"
vm_dir = "hazma/vector_mediator/"

decay_pack = "hazma.decay_helper_functions"
ft_pack = "hazma.field_theory_helper_functions"
gr_pack = "hazma.gamma_ray_helper_functions"
theory_pack = "hazma.theory"
ps_pack = "hazma.phase_space_helper_functions"
pos_pack = "hazma.positron_helper_functions"
sm_pack = "hazma.scalar_mediator"
vm_pack = "hazma.vector_mediator"

psm_pack = "hazma.pseudo_scalar_mediator"
avm_pack = "hazma.axial_vector_mediator"
unit_pack = "hazma.unitarization"

packs = [
    "hazma",
    avm_pack,
    decay_pack,
    ft_pack,
    gr_pack,
    theory_pack,
    ps_pack,
    pos_pack,
    psm_pack,
    sm_pack,
    unit_pack,
    vm_pack,
]

extensions = []

# Decay helper functions extensions
extensions += [
    Extension(
        decay_pack + ".decay_charged_pion",
        sources=[decay_dir + "decay_charged_pion.pyx"],
    )
]
extensions += [
    Extension(
        decay_pack + ".decay_neutral_pion",
        sources=[decay_dir + "decay_neutral_pion.pyx"],
    )
]
extensions += [
    Extension(decay_pack + ".decay_muon",
              sources=[decay_dir + "decay_muon.pyx"])
]
extensions += [
    Extension(
        decay_pack + ".decay_charged_kaon",
        sources=[decay_dir + "decay_charged_kaon.pyx"],
    )
]
extensions += [
    Extension(
        decay_pack + ".decay_long_kaon", sources=[decay_dir + "decay_long_kaon.pyx"]
    )
]
extensions += [
    Extension(
        decay_pack + ".decay_short_kaon", sources=[decay_dir + "decay_short_kaon.pyx"]
    )
]

# Gamma-Ray Helper functions extensions
extensions += [
    Extension(
        gr_pack + ".gamma_ray_generator", sources=[gr_dir + "gamma_ray_generator.pyx"]
    )
]

# Phase space helper functions extensions
extensions += [
    Extension(
        ps_pack + ".generator",
        sources=[ps_dir + "generator.pyx"],
        extra_compile_args=["-g", "-std=c++11"],
        language="c++",
    )
]
extensions += [
    Extension(
        ps_pack + ".histogram",
        sources=[ps_dir + "histogram.pyx"],
        extra_compile_args=["-g", "-std=c++11"],
        language="c++",
    )
]
extensions += [
    Extension(
        ps_pack + ".modifiers",
        sources=[ps_dir + "modifiers.pyx"],
        extra_compile_args=["-g", "-std=c++11"],
        language="c++",
    )
]

# Field Theory helper functions extensions
extensions += [
    Extension(ft_pack + ".common_functions",
              sources=[ft_dir + "common_functions.pyx"])
]
extensions += [
    Extension(
        ft_pack + ".three_body_phase_space",
        sources=[ft_dir + "three_body_phase_space.pyx"],
    )
]

# Positron Helper functions
extensions += [
    Extension(
        pos_pack + ".positron_charged_pion",
        sources=[pos_dir + "positron_charged_pion.pyx"],
    )
]
extensions += [
    Extension(pos_pack + ".positron_muon",
              sources=[pos_dir + "positron_muon.pyx"])
]
extensions += [
    Extension(pos_pack + ".positron_decay",
              sources=[pos_dir + "positron_decay.pyx"])
]

# Scalar mediator
extensions += [
    Extension(
        sm_pack + ".scalar_mediator_decay_spectrum",
        sources=[sm_dir + "scalar_mediator_decay_spectrum.pyx"],
    )
]
extensions += [
    Extension(
        sm_pack + ".scalar_mediator_positron_spec",
        sources=[sm_dir + "scalar_mediator_positron_spec.pyx"],
    )
]
extensions += [
    Extension(
        sm_pack + "._c_scalar_mediator_cross_sections",
        sources=[sm_dir + "_c_scalar_mediator_cross_sections.pyx"],
    )
]
# Vector mediator
extensions += [
    Extension(
        vm_pack + ".vector_mediator_decay_spectrum",
        sources=[vm_dir + "vector_mediator_decay_spectrum.pyx"],
    )
]
extensions += [
    Extension(
        vm_pack + ".vector_mediator_positron_spec",
        sources=[vm_dir + "vector_mediator_positron_spec.pyx"],
    )
]
extensions += [
    Extension(
        vm_pack + "._c_vector_mediator_cross_sections",
        sources=[vm_dir + "_c_vector_mediator_cross_sections.pyx"],
    )
]


setup(
    name="hazma",
    version=VERSION,
    author="Logan Morrison and Adam Coogan",
    author_email="loanmorr@ucsc.edu",
    maintainer="Logan Morrison",
    maintainer_email="loanmorr@ucsc.edu",
    url="http://hazma.readthedocs.io",
    description="Python package for computing indirect detection constraints on sub-GeV dark matter.",
    long_description=long_description(),
    long_description_content_type="text/markdown",
    keywords="dark-matter mev-dark-matter gamma-ray-spectra",
    packages=find_packages(),
    ext_modules=extensions,
    include_dirs=[
        np.get_include(),
        "hazma/decay_helper_functions",
        "hazma/positron_helper_functions",
    ],
    package_data={
        "hazma/decay_helper_functions": ["*.pxd"],
        "hazma/positron_helper_functions": ["*.pxd"],
    },
    setup_requires=["pytest-runner"],
    install_requires=[
        "pip>=9.0.1",
        "matplotlib>=2.2.3",
        "scipy>=1.0.0",
        "numpy>=1.13.3",
        "cython>=0.27.3",
        "numpydoc>=0.7.0",
        "scikit-image>=0.14",
        "setuptools>=27.3",
        "flake8>=3.5.0",
    ],
    tests_require=["pytest>=3.2.5"],
    zip_safe=False,
    include_package_data=True,
    license="gpl-3.0",
    platforms="MacOS and Linux",
    download_url="https://github.com/LoganAMorrison/Hazma",
    classifiers=["Programming Language :: Python"],
)
