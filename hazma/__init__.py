from importlib.metadata import version
from typing import Final

#: The installed distribution's version.
#:
#: The number itself lives in `pyproject.toml`'s `[project] version`, which
#: is what a closing PR edits (`docs/versioning.md`). Reading it back out of
#: the installed metadata keeps this attribute -- part of the public API,
#: alongside `__version__` -- without a second copy to drift.
#:
#: This raises `importlib.metadata.PackageNotFoundError` when hazma is
#: importable but not installed, which is a state the library cannot work
#: in anyway: every spectrum goes through the compiled `hazma._core`, and
#: only an install puts that extension in the package. A sentinel fallback
#: would trade a precise error for a version string that lies.
VERSION: Final[str] = version("hazma")
__version__ = VERSION

# import warnings

# from hazma import (
#     background_model,
#     constraint_parameters,
#     cmb,
#     flux_measurement,
#     # gamma_ray,
#     gamma_ray_parameters,
#     hazma_errors,
#     parameters,
#     # rambo,
#     relic_density,
#     spectra,
#     target_params,
#     utils,
# )

# # Models
# from hazma import (
#     pbh,
#     rh_neutrino,
#     scalar_mediator,
#     single_channel,
#     theory,
#     vector_mediator,
# )

# __all__ = [
#     "background_model",
#     "cmb",
#     "constraint_parameters",
#     "flux_measurement",
#     "gamma_ray_parameters",
#     # "gamma_ray",
#     "hazma_errors",
#     "parameters",
#     "relic_density",
#     "spectra",
#     "target_params",
#     "utils",
#     # Models
#     "pbh",
#     "rh_neutrino",
#     "scalar_mediator",
#     "single_channel",
#     "theory",
#     "vector_mediator",
# ]
