from typing import Union

import numpy as np

from hazma.utils import kallen_lambda


def width_v_to_p_p(
    mv: float,
    mass: float,
    form_factor: Union[complex, np.ndarray],
    symmetry: float = 1.0,
) -> float:
    """
    Compute the partial width for the decay of the vector mediator into two
    mesons.

    Parameters
    ----------
    mass: float
        Mass of the final state meson.
    form_factor: complex
        Vector form factor for the V-meson-meson vertex.
    symmetry: float
        Symmetry factor. If the final state mesons are identical, then this
        should be 1/2. Default is 1.0

    Returns
    -------
    gamma: float
        Partial width for the vector to decay into two mesons.
    """
    if mv < 2 * mass:
        return 0.0
    mu = mass / mv
    return (
        symmetry
        / 48.0
        / np.pi
        * mv
        * (1.0 - 4.0 * mu**2) ** 1.5
        * np.abs(form_factor) ** 2
    )


def width_v_to_p_a(mv: float, mass: float, ff: complex) -> float:
    """
    Compute the partial width for the decay of the vector mediator
    into a meson and photon.

    Parameters
    ----------
    mass: float
        Mass of the final state meson.
    ff: complex
        Vector form factor for the V-meson-meson vertex.

    Returns
    -------
    gamma: float
        Partial width for the vector to decay into a meson and photon.
    """
    if mv < mass:
        return 0.0
    # Note: form-factor has units of 1/GeV
    mu = mass / mv
    q = 0.5 * np.sqrt(kallen_lambda(1.0, mu**2, 0))
    return mv * q**3 * np.abs(ff) ** 2 / (12.0 * np.pi)


def width_v_to_v_p(mv: float, ff: complex, mvector: float, mscalar: float) -> float:
    """
    Compute the partial width for the decay of the vector mediator
    into a vector meson and a scalar meson.

    Parameters
    ----------
    ff: complex
        Form factor.
    mvector: float
        Mass of the vector meson.
    mscalar: float
        Mass of the scalar meson.
    """
    if mv < mvector + mscalar:
        return 0.0
    q = 0.5 * np.sqrt(kallen_lambda(1.0, (mvector / mv) ** 2, (mscalar / mv) ** 2))
    return mv * q**3 * np.abs(ff) ** 2 / (12.0 * np.pi)
