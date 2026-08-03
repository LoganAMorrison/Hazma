"""
Module for computing various vector-meson form factors.
"""

from typing import Union

import numpy as np
import numpy.typing as npt

from hazma.vector_mediator.form_factors.eta_gamma import FormFactorEtaGamma
from hazma.vector_mediator.form_factors.eta_omega import FormFactorEtaOmega
from hazma.vector_mediator.form_factors.eta_phi import FormFactorEtaPhi
from hazma.vector_mediator.form_factors.omega_pi import FormFactorOmegaPi0
from hazma.vector_mediator.form_factors.phi_pi import FormFactorPhiPi0

# from hazma.vector_mediator.form_factors.kk import form_factor_kk as __ff_kk
# from hazma.vector_mediator.form_factors.pipi import form_factor_pipi as __ff_pipi
# from hazma.vector_mediator.form_factors.pipi import FormFactorPiPi
from hazma.vector_mediator.form_factors.pi_gamma import FormFactorPiGamma
from hazma.vector_mediator.form_factors.pi_k_k import (
    FormFactorPi0K0K0,
    FormFactorPi0KpKm,
    FormFactorPiKK0,
)
from hazma.vector_mediator.form_factors.utils import (
    META_GEV,
    MOMEGA_GEV,
    MPHI_GEV,
    MPI0_GEV,
    ComplexArray,
    RealArray,
)

# def form_factor_pipi(
#     self, s: Union[float, npt.NDArray[np.float64]], imode: int = 1
# ) -> Union[complex, npt.NDArray[np.complex128]]:
#     """
#     Compute the pi-pi-V form factor.

#     Parameters
#     ----------
#     s: Union[float,npt.NDArray[np.float64]
#         Square of the center-of-mass energy in MeV.
#     imode: Optional[int]
#         Iso-spin channel. Default is 1.

#     Returns
#     -------
#     ff: Union[complex,npt.NDArray[np.complex128]]
#         Form factor from pi-pi-V.
#     """
#     return __ff_pipi(
#         s * 1e-6,  # Convert to GeV
#         self._ff_pipi_params,
#         self._gvuu,
#         self._gvdd,
#         imode=imode,
#     )


# def form_factor_kk(
#     self,
#     s: Union[float, npt.NDArray[np.float64]],
#     imode: int = 1,
# ) -> Union[complex, npt.NDArray[np.complex128]]:
#     """
#     Compute the K-K-V form factor.

#     Parameters
#     ----------
#     s: Union[float,npt.NDArray[np.float64]
#         Square of the center-of-mass energy in MeV.
#     imode: int
#         Iso-spin channel. Using imode=0 for K0 K0bar final state and imode=1
#         for K^+ K^-.

#     Returns
#     -------
#     ff: Union[complex,npt.NDArray[np.complex128]]
#         Form factor from K-K-V.
#     """
#     return __ff_kk(
#         s * 1e-6,  # Convert to GeV
#         self._ff_kk_params,
#         self._gvuu,
#         self._gvdd,
#         self._gvss,
#         imode=imode,
#     )


# def form_factor_pi_gamma(
#     self,
#     q: Union[float, RealArray],
# ) -> Union[complex, ComplexArray]:
#     """
#     Compute the pi-gamma-V form factor.

#     Parameters
#     ----------
#     s: Union[float,npt.NDArray[np.float64]
#         Square of the center-of-mass energy in MeV.

#     Returns
#     -------
#     ff: Union[complex,npt.NDArray[np.complex128]]
#         Form factor from pi-gamma-V.
#     """
#     ff = FormFactorPiGamma()
#     if hasattr(q, "__len__"):
#         if self.mv < MPI0_GEV:
#             return np.zeros_like(q)
#         qq = 1e-3 * np.array(q)
#         return ff.form_factor(
#             qq**2,
#             self._gvuu,
#             self._gvdd,
#             self._gvss,
#         )
#     else:
#         if self.mv < MPI0_GEV:
#             return 0.0
#         qq = 1e-3 * np.array([q])
#         return ff.form_factor(
#             qq**2,
#             self._gvuu,
#             self._gvdd,
#             self._gvss,
#         )[0]


# def form_factor_eta_gamma(
#     self,
#     q: Union[float, RealArray],
# ) -> Union[complex, ComplexArray]:
#     """
#     Compute the eta-gamma-V form factor.

#     Parameters
#     ----------
#     s: Union[float,npt.NDArray[np.float64]
#         Square of the center-of-mass energy in MeV.

#     Returns
#     -------
#     ff: Union[complex,npt.NDArray[np.complex128]]
#         Form factor from eta-gamma-V.
#     """
#     ff = FormFactorEtaGamma()
#     if hasattr(q, "__len__"):
#         if self.mv < META_GEV:
#             return np.zeros_like(q)
#         qq = 1e-3 * np.array(q)
#         return ff.form_factor(
#             qq**2,
#             self._gvuu,
#             self._gvdd,
#             self._gvss,
#         )
#     else:
#         if self.mv < META_GEV:
#             return 0.0
#         qq = 1e-3 * np.array([q])
#         return ff.form_factor(
#             qq**2,
#             self._gvuu,
#             self._gvdd,
#             self._gvss,
#         )[0]


# def form_factor_omega_pi(
#     self,
#     q: Union[float, npt.NDArray[np.float64]],
# ) -> Union[complex, npt.NDArray[np.complex128]]:
#     """
#     Compute the V-omega-pi form factor.

#     Parameters
#     ----------
#     q: Union[float,npt.NDArray[np.float64]
#         Center-of-mass energy in MeV.

#     Returns
#     -------
#     ff: Union[complex,npt.NDArray[np.complex128]]
#         Form factor from V-omega-pi.
#     """
#     ff = FormFactorOmegaPi0()
#     if hasattr(q, "__len__"):
#         if self.mv < MOMEGA_GEV + MPI0_GEV:
#             return np.zeros_like(q)
#         qq = 1e-3 * np.array(q)
#         return ff.form_factor(qq**2, self._gvuu, self._gvdd)
#     else:
#         if self.mv < MOMEGA_GEV + MPI0_GEV:
#             return 0.0
#         qq = 1e-3 * np.array([q])
#         return ff.form_factor(qq**2, self._gvuu, self._gvdd)[0]


# def form_factor_phi_pi(
#     self,
#     q: Union[float, npt.NDArray[np.float64]],
# ) -> Union[complex, npt.NDArray[np.complex128]]:
#     """
#     Compute the V-phi-pi form factor.

#     Parameters
#     ----------
#     q: Union[float,npt.NDArray[np.float64]
#         Center-of-mass energy in MeV.

#     Returns
#     -------
#     ff: Union[complex,npt.NDArray[np.complex128]]
#         Form factor from V-phi-pi.
#     """
#     ff = FormFactorPhiPi0()
#     if self.mv < MPHI_GEV + MPI0_GEV:
#         return np.zeros_like(q)
#     qq = 1e-3 * q
#     return ff.form_factor(qq**2, self._gvuu, self._gvdd)


# def form_factor_eta_phi(
#     self,
#     q: Union[float, npt.NDArray[np.float64]],
# ) -> Union[complex, npt.NDArray[np.complex128]]:
#     """
#     Compute the V-eta-phi form factor.

#     Parameters
#     ----------
#     q: Union[float,npt.NDArray[np.float64]
#         Center-of-mass energy in MeV.

#     Returns
#     -------
#     ff: Union[complex,npt.NDArray[np.complex128]]
#         Form factor from V-eta-phi.
#     """
#     ff = FormFactorEtaPhi()
#     if hasattr(q, "__len__"):
#         if self.mv < META_GEV + MPHI_GEV:
#             return np.zeros_like(q)
#         qq = 1e-3 * np.array(q)
#         return ff.form_factor(qq**2, self._gvss)
#     else:
#         if self.mv < META_GEV + MPHI_GEV:
#             return 0.0
#         qq = 1e-3 * np.array([q])
#         return ff.form_factor(qq**2, self._gvss)[0]


# def form_factor_eta_omega(
#     self,
#     q: Union[float, npt.NDArray[np.float64]],
# ) -> Union[complex, npt.NDArray[np.complex128]]:
#     """
#     Compute the V-eta-omega form factor.

#     Parameters
#     ----------
#     q: Union[float,npt.NDArray[np.float64]
#         Center-of-mass energy in MeV.

#     Returns
#     -------
#     ff: Union[complex,npt.NDArray[np.complex128]]
#         Form factor from V-eta-omega.
#     """
#     ff = FormFactorEtaOmega()
#     if hasattr(q, "__len__"):
#         if self.mv < META_GEV + MOMEGA_GEV:
#             return np.zeros_like(q)
#         qq = 1e-3 * np.array(q)
#         return ff.form_factor(qq**2, self._gvuu, self._gvdd)
#     else:
#         if self.mv < META_GEV + MOMEGA_GEV:
#             return 0.0
#         qq = 1e-3 * np.array([q])
#         return ff.form_factor(qq**2, self._gvuu, self._gvdd)[0]
