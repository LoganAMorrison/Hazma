"""
Module for computing various vector-meson form factors.
"""
from typing import Union

import numpy as np
import numpy.typing as npt

from hazma.vector_mediator.form_factors.eta_gamma import (
    form_factor_eta_gamma as __ff_eta_gamma,
)
from hazma.vector_mediator.form_factors.kk import form_factor_kk as __ff_kk
from hazma.vector_mediator.form_factors.pi_gamma import (
    form_factor_pi_gamma as __ff_pi_gamma,
)
from hazma.vector_mediator.form_factors.pipi import form_factor_pipi as __ff_pipi


def _form_factor_pipi(
    self, s: Union[float, npt.NDArray[np.float64]], imode: int = 1
) -> Union[complex, npt.NDArray[np.complex128]]:
    """
    Compute the pi-pi-V form factor.

    Parameters
    ----------
    s: Union[float,npt.NDArray[np.float64]
        Square of the center-of-mass energy in MeV.
    imode: Optional[int]
        Iso-spin channel. Default is 1.

    Returns
    -------
    ff: Union[complex,npt.NDArray[np.complex128]]
        Form factor from pi-pi-V.
    """
    return __ff_pipi(
        s * 1e-6,  # Convert to GeV
        self._ff_pipi_params,
        self._gvuu,
        self._gvdd,
    )


def _form_factor_kk(
    self,
    s: Union[float, npt.NDArray[np.float64]],
    imode: int,
) -> Union[complex, npt.NDArray[np.complex128]]:
    """
    Compute the K-K-V form factor.

    Parameters
    ----------
    s: Union[float,npt.NDArray[np.float64]
        Square of the center-of-mass energy in MeV.
    imode: int
        Iso-spin channel. Using imode=0 for K0 K0bar final state and imode=1
        for K^+ K^-.

    Returns
    -------
    ff: Union[complex,npt.NDArray[np.complex128]]
        Form factor from K-K-V.
    """
    return __ff_kk(
        s * 1e-6,  # Convert to GeV
        self._ff_kk_params,
        self._gvuu,
        self._gvdd,
        self._gvss,
        imode,
    )


def _form_factor_pi_gamma(
    self,
    s: Union[float, npt.NDArray[np.float64]],
) -> Union[complex, npt.NDArray[np.complex128]]:
    """
    Compute the pi-gamma-V form factor.

    Parameters
    ----------
    s: Union[float,npt.NDArray[np.float64]
        Square of the center-of-mass energy in MeV.

    Returns
    -------
    ff: Union[complex,npt.NDArray[np.complex128]]
        Form factor from pi-gamma-V.
    """
    return __ff_pi_gamma(
        s * 1e-6,  # Convert to GeV
        self._gvuu,
        self._gvdd,
        self._gvss,
    )


def _form_factor_eta_gamma(
    self,
    s: Union[float, npt.NDArray[np.float64]],
) -> Union[complex, npt.NDArray[np.complex128]]:
    """
    Compute the eta-gamma-V form factor.

    Parameters
    ----------
    s: Union[float,npt.NDArray[np.float64]
        Square of the center-of-mass energy in MeV.

    Returns
    -------
    ff: Union[complex,npt.NDArray[np.complex128]]
        Form factor from eta-gamma-V.
    """
    return __ff_eta_gamma(
        s * 1e-6,  # Convert to GeV
        self._gvuu,
        self._gvdd,
        self._gvss,
    )
