"""
Module for integrating two-body phase space.
"""

from typing import Sequence, Tuple, Callable, Any, Optional

import numpy as np
from scipy import integrate

from hazma.utils import kallen_lambda

from ._base import AbstractPhaseSpaceIntegrator


def _msqrd_flat(z):
    if np.isscalar(z):
        return 1.0

    return np.zeros_like(z)


class TwoBody(AbstractPhaseSpaceIntegrator):
    r"""Class for working with 2-body phase space."""

    def __init__(
        self,
        cme: float,
        masses: Sequence[float],
        msqrd: Optional[Callable[..., Any]] = None,
        msqrd_signature: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        cme: float
            Center-of-mass energy.
        masses: sequence float
            The two final state particle masses.
        msqrd: callable, optional
            Function to compute squared matrix element. The signature of the
            function depends on the value of `msqrd_signature`. If no matrix
            element is passed, it is taken to be flat (i.e. |M|^2 = 1).
        msqrd_signature: str, optional
            Signature of squared matrix element. If 'momenta', then the function
            is assumed to take in a NumPy array containing the momenta of the
            final state particles. If 'z', then it is assumed to take in
            the angle between the 3-momenta of the two final state particles.
            Default is 'z'.
        """
        assert (
            len(masses) == 2
        ), f"Expected 'masses' to have length 2, found {len(masses)}."

        self.__cme = cme
        self.__masses = (masses[0], masses[1])

        if msqrd is not None:
            assert callable(msqrd), "The squared matrix element must be callable."

        self.__msqrd_signature_z = True
        if msqrd_signature is not None:
            assert msqrd_signature in ["momenta", "z"], (
                f"Invalid 'msqrd_signature' {msqrd_signature}." "Use 'momenta' or 'z'."
            )
            if msqrd_signature == "momenta":
                self.__msqrd_signature_z = False

        if msqrd is None:
            self.__msqrd = _msqrd_flat
        else:
            self.__msqrd = msqrd

    @property
    def cme(self) -> float:
        r"""Center-of-mass energy of the proccess."""
        return self.__cme

    @cme.setter
    def cme(self, val) -> None:
        self.__cme = val

    @property
    def masses(self) -> Tuple[float, float]:
        r"""Masses of the final state particles."""
        return self.__masses

    @masses.setter
    def masses(self, masses: Sequence[float]) -> None:
        assert (
            len(masses) == 2
        ), f"Expected 'masses' to have length 3, found {len(masses)}."

        self.__masses = (masses[0], masses[1])

    @property
    def msqrd(self):
        r"""Squared matrix element of the proccess."""
        return self.__msqrd

    @msqrd.setter
    def msqrd(self, fn) -> None:
        r"""Squared matrix element of the proccess."""
        self.__msqrd = fn

    def __integrate_momenta(self):
        r"""Integrate over phase space assuming msqrd take the final-state
        particle momenta as its argument.
        """
        cme = self.cme
        m1, m2 = self.masses

        p = np.sqrt(kallen_lambda(cme**2, m1**2, m2**2)) / (2 * cme)
        e1 = np.hypot(m1, p)
        e2 = np.hypot(m2, p)
        ps = np.zeros((4, 2), dtype=np.float64)

        def integrand(z):
            sin = np.sqrt(1 - z**2)
            ps[:, 0] = np.array([e1, sin * p, 0.0, z * p])
            ps[:, 1] = np.array([e2, -sin * p, 0.0, -z * p])
            return self.__msqrd(ps)

        pre = 1.0 / (8.0 * np.pi) * p / cme

        integral, error = integrate.quad(integrand, -1.0, 1.0)

        return integral * pre, error * pre

    def __integrate_angle(self):
        r"""Integrate over phase space assuming msqrd take the angle as
        argument.
        """
        cme = self.cme
        m1, m2 = self.masses

        p = np.sqrt(kallen_lambda(cme**2, m1**2, m2**2)) / (2 * cme)
        pre = 1.0 / (8.0 * np.pi) * p / cme

        integral, error = integrate.quad(self.__msqrd, -1.0, 1.0)
        return integral * pre, error * pre

    def integrate(self):  # pylint: disable=arguments-differ
        r"""Integrate over phase space.

        Returns
        -------
        integral: float
            Value of the phase space integration.
        error_estimate: float
            Estimation of the error.
        """
        if self.__msqrd_signature_z:
            return self.__integrate_angle()
        return self.__integrate_momenta()
