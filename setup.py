import numpy as np
from setuptools import Extension, find_packages, setup  # type: ignore

VERSION = "2.0.0-alpha"


def long_description():
    """Remove first four lines from the README for PyPI."""
    with open("README.md", encoding="utf-8") as f:
        ld = f.read()
    return "\n".join([str(line) for line in ld.split("\n")[4:]])


def make_extension(module, name, cpp=False):
    package = ".".join(["hazma", module, name])
    sources = ["/".join(["hazma", module, name]) + ".pyx"]
    if cpp:
        return Extension(
            package, sources, extra_compile_args=["-g", "-std=c++11"], language="c++"
        )
    else:
        return Extension(package, sources)


def make_extensions(module, names, cpp=False):
    extensions = []
    for name in names:
        extensions += [make_extension(module, name, cpp)]
    return extensions


EXTENSIONS = []


# Decay helper
EXTENSIONS += make_extensions(
    "decay_helper_functions",
    [
        "decay_charged_pion",
        "decay_neutral_pion",
        "decay_muon",
        "decay_charged_kaon",
        "decay_long_kaon",
        "decay_short_kaon",
    ],
    cpp=True,
)

# Gamma-Ray Helper
EXTENSIONS += make_extensions(
    "gamma_ray_helper_functions", ["gamma_ray_generator", "gamma_ray_fsr"], cpp=True
)

# Phase space
EXTENSIONS += make_extensions(
    "phase_space_helper_functions", ["generator", "histogram", "modifiers"], cpp=True
)

# Field Theory helper
EXTENSIONS += make_extensions(
    "field_theory_helper_functions",
    ["common_functions", "three_body_phase_space"],
    cpp=True,
)

# Positron Helper
EXTENSIONS += make_extensions(
    "positron_helper_functions",
    ["positron_muon", "positron_charged_pion", "positron_decay"],
)

# Scalar mediator
EXTENSIONS += make_extensions(
    "scalar_mediator",
    [
        "scalar_mediator_decay_spectrum",
        "scalar_mediator_positron_spec",
        "_c_scalar_mediator_cross_sections",
    ],
)

# Vector mediator
EXTENSIONS += make_extensions(
    "vector_mediator",
    [
        "vector_mediator_decay_spectrum",
        "vector_mediator_positron_spec",
        "_c_vector_mediator_cross_sections",
    ],
)

# RH-neutrino
EXTENSIONS += [
    Extension(
        "hazma.rh_neutrino._rh_neutrino_fsr_four_body",
        sources=["hazma/rh_neutrino/_rh_neutrino_fsr_four_body.pyx"],
        extra_compile_args=["-g", "-std=c++11"],
        language="c++",
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
    description=(
        "Python package for computing indirect detection constraints"
        + " on sub-GeV dark matter."
    ),
    long_description=long_description(),
    long_description_content_type="text/markdown",
    keywords="dark-matter mev-dark-matter gamma-ray-spectra",
    packages=find_packages(),
    ext_modules=EXTENSIONS,
    include_dirs=[
        np.get_include(),
        "hazma/decay_helper_functions",
        "hazma/positron_helper_functions",
        "hazma/gamma_ray_helper_functions",
    ],
    package_data={
        "hazma/cython_utils": ["*.pxd"],
        "hazma/decay_helper_functions": ["*.pxd"],
        "hazma/positron_helper_functions": ["*.pxd"],
        "hazma/gamma_ray_helper_functions": ["*.pxd"],
    },
    setup_requires=["pytest-runner"],
    install_requires=[
        "pip",
        "matplotlib",
        "scipy",
        "numpy",
        "cython",
        "numpydoc",
        "scikit-image",
        "setuptools",
        "flake8",
        "importlib_resources ; python_version<'3.7'",
    ],
    python_requires=">=3",
    tests_require=["pytest>=3.2.5"],
    zip_safe=False,
    include_package_data=True,
    license="gpl-3.0",
    platforms="MacOS and Linux",
    download_url="https://github.com/LoganAMorrison/Hazma",
    classifiers=["Programming Language :: Python"],
)
