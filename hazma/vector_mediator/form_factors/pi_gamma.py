from typing import Union

import numpy as np

from hazma.vector_mediator.form_factors.utils import MPI_GEV

# Constant parameters for the V-pi-gamma form factor.
_res_masses = np.array([0.77526, 0.78284, 1.01952, 1.45, 1.70])
_res_widths = np.array([0.1491, 0.00868, 0.00421, 0.40, 0.30])
_amps = np.array([0.0426, 0.0434, 0.00303, 0.00523, 0.0])
_phases = np.array([-12.7, 0.0, 158.0, 180.0, 0.0])


def form_factor_pi_gamma(
    s: Union[float, np.ndarray], gvuu: float, gvdd: float, gvss: float
) -> Union[complex, np.ndarray]:
    """
    Compute the form factor for V-gamma-pi at given squared center of mass
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

    c_rho_om_phi = np.array([ci1, ci0, cs, ci0, ci0])

    if hasattr(s, "__len__"):
        ss = np.array(s)
    else:
        ss = np.array([s])

    q = np.sqrt(ss)
    di = _res_masses ** 2 - ss[:, np.newaxis] - 1j * q[:, np.newaxis] * _res_widths
    di[:, 0] = (
        _res_masses[0] ** 2
        - s
        - 1j
        * q
        * (
            _res_widths[0]
            * _res_masses[0] ** 2
            / s
            * ((s - 4.0 * MPI_GEV ** 2) / (_res_masses[0] ** 2 - 4.0 * MPI_GEV ** 2))
            ** 1.5
        )
    )

    ff = np.sum(
        c_rho_om_phi * _amps * _res_masses ** 2 * np.exp(1j * np.radians(_phases)) / di,
        axis=1,
    )
    if type(s) == float:
        return ff[0]
    return ff
