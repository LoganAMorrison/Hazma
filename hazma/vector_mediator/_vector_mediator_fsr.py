from hazma.parameters import alpha_em, qe
from hazma.parameters import charged_pion_mass as mpi
from hazma.parameters import electron_mass as me
from hazma.parameters import muon_mass as mmu

from cmath import sqrt, log, pi
import numpy as np


class VectorMediatorFSR:
    def __dnde_xx_to_v_to_ffg(self, egam, Q, f):
        """Return the fsr spectra for fermions from decay of vector mediator.

        Computes the final state radiaton spectrum value dNdE from a vector
        mediator given a gamma ray energy of `egam`, center of mass energy `Q`
        and final state fermion `f`.

        Paramaters
        ----------
        egam : float
            Gamma ray energy.
        Q: float
            Center of mass energy of mass of off-shell vector mediator.
        f : float
            Name of the final state fermion: "e" or "mu".

        Returns
        -------
        spec_val : float
            Spectrum value dNdE from vector mediator.
        """
        if f == "e":
            mf = me
        elif f == "mu":
            mf = mmu

        mu_l = mf / Q
        x = 2 * egam / Q

        ret_val = 0.0

        if (
            4.0 * mf**2 <= (Q**2 - 2.0 * Q * egam) <= Q**2
            and Q > 2.0 * mf
            and Q > 2.0 * self.mx
        ):
            val = (
                (
                    2
                    * alpha_em
                    * (
                        -(
                            sqrt(1 + (4 * mu_l**2) / (-1 + x))
                            * (2 - 2 * x + x**2 - 4 * (-1 + x) * mu_l**2)
                        )
                        + (2 + (-2 + x) * x - 4 * x * mu_l**2 - 8 * mu_l**4)
                        * log(
                            -(
                                (1 + sqrt(1 + (4 * mu_l**2) / (-1 + x)))
                                / (-1 + sqrt(1 + (4 * mu_l**2) / (-1 + x)))
                            )
                        )
                    )
                )
                / (pi * Q * x * sqrt(1 - 4 * mu_l**2) * (1 + 2 * mu_l**2))
            ).real

            assert val >= 0
            return val
        else:
            return 0.0

    def dnde_xx_to_v_to_ffg(self, egam, Q, f):
        """Return the fsr spectra for fermions from decay of vector mediator.

        Computes the final state radiaton spectrum value dNdE from a vector
        mediator given a gamma ray energy of `egam`, center of mass energy `Q`
        and final state fermion `f`.

        Paramaters
        ----------
        egam : float
            Gamma ray energy.
        Q: float
            Center of mass energy of mass of off-shell vector mediator.
        f : float
            Mass of the final state fermion.

        Returns
        -------
        spec_val : float
            Spectrum value dNdE from vector mediator.
        """
        if hasattr(egam, "__len__"):
            return np.array([self.__dnde_xx_to_v_to_ffg(e, Q, f) for e in egam])
        else:
            return self.__dnde_xx_to_v_to_ffg(egam, Q, f)

    def __dnde_xx_to_v_to_pipig(self, egam, Q):
        """Unvectorized dnde_xx_to_v_to_pipig"""
        mx = self.mx

        mu_pi = mpi / Q
        x = 2.0 * egam / Q
        x_min = 0.0
        x_max = 1 - 4.0 * mu_pi**2

        if x < x_min or x > x_max or Q < 2 * mpi or Q < 2.0 * mx:
            return 0.0
        else:
            val = (
                (
                    4
                    * alpha_em
                    * (
                        sqrt(1 + (4 * mu_pi**2) / (-1 + x))
                        * (-1 + x + x**2 - 4 * (-1 + x) * mu_pi**2)
                        + (-1 + x + 2 * mu_pi**2)
                        * (-1 + 4 * mu_pi**2)
                        * log(
                            -(
                                (1 + sqrt(1 + (4 * mu_pi**2) / (-1 + x)))
                                / (-1 + sqrt(1 + (4 * mu_pi**2) / (-1 + x)))
                            )
                        )
                    )
                )
                / (pi * Q * x * (1 - 4 * mu_pi**2) ** 1.5)
            ).real

            assert val >= 0
            return val

    def dnde_xx_to_v_to_pipig(self, eng_gams, Q):
        """
        Returns the gamma ray energy spectrum for two fermions annihilating
        into two charged pions and a photon.

        Parameters
        ----------
        eng_gam : numpy.ndarray or double
            Gamma ray energy.
        Q : double
            Center of mass energy, or sqrt((ppip + ppim + pg)^2).

        Returns
        -------
        Returns gamma ray energy spectrum for
        :math:`\chi\bar{\chi}\to\pi^{+}\pi^{-}\gamma` evaluated at the gamma
        ray energy(ies).

        """

        if hasattr(eng_gams, "__len__"):
            return np.array(
                [self.__dnde_xx_to_v_to_pipig(eng_gam, Q) for eng_gam in eng_gams]
            )
        else:
            return self.__dnde_xx_to_v_to_pipig(eng_gams, Q)
