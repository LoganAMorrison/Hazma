"""
Module for computing the form factor V-eta-gamma.
"""
from typing import Union

import numpy as np

from hazma.vector_mediator.form_factors.utils import MPI_GEV

_res_masses = np.array([0.77526, 0.78284, 1.01952, 1.465, 1.70])
_res_widths = np.array([0.1491, 0.00868, 0.00421, 0.40, 0.30])
_amps = np.array([0.0861, 0.00824, 0.0158, 0.0147, 0.0])
_phases = np.array([0.0, 11.3, 170.0, 61.0, 0.0])


def form_factor_eta_gamma(
        s: Union[float, np.ndarray],
        gvuu: float,
        gvdd: float,
        gvss: float
) -> Union[complex, np.ndarray]:
    """
    Compute the form factor for V-eta-gamma at given squared center of mass
    energ(ies).

    Parameters
    ----------
    s: Union[float, np.ndarray]
        Array of squared center-of-mass energies or a single value.
    gvuu : float
        Coupling of vector mediator to the up quark.
    gvdd : float
        Coupling of vector mediator to the down quark.
    gvss : float
        Coupling of vector mediator to the strange quark.

    Returns
    -------
    ff: Union[float, np.ndarray]
        The form factors.
    """
    ci0 = 3.0 * (gvuu + gvdd)
    ci1 = gvuu - gvdd
    cs = -3.0 * gvss

    # TODO This was different for Pi-Gamma. Typo?
    c_rho_om_phi = np.array([ci1, ci0, cs, ci1, cs])

    if hasattr(s, '__len__'):
        ss = np.array(s)
    else:
        ss = np.array([s])

    q = np.sqrt(ss)
    di = (
        _res_masses ** 2
        - ss[:, np.newaxis]
        - 1j * q[:, np.newaxis]
        * _res_widths
    )
    di[:, 0] = (
        _res_masses[0] ** 2
        - ss[:, np.newaxis]
        - 1j * q[:, np.newaxis] * (
            _res_widths[0]
            * _res_masses[0] ** 2
            / ss[:, np.newaxis] * (
                (ss[:, np.newaxis] - 4.0 * MPI_GEV ** 2)
                / (_res_masses[0] ** 2 - 4.0 * MPI_GEV ** 2)
            )**1.5
        ))

    ff = np.sum(
        c_rho_om_phi
        * _amps
        * _res_masses ** 2
        * np.exp(1j * np.radians(_phases))
        / di,
        axis=1
    )

    if len(ss) == 1:
        return ff[0]
    return ff
