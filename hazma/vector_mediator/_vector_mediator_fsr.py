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

        e, m = egam / Q, mf / Q

        s = Q ** 2 - 2.0 * Q * egam

        ret_val = 0.0

        if 4.0 * mf ** 2 <= s <= Q ** 2 and Q > 2.0 * mf and Q > 2.0 * self.mx:
            ret_val = -(
                alpha_em
                * (
                    4.0
                    * sqrt(1.0 - 2.0 * e - 4.0 * m ** 2)
                    * (1.0 - 2.0 * m ** 2 + 2.0 * e * (-1 + e + 2.0 * m ** 2))
                    + sqrt(1.0 - 2.0 * e)
                    * (1.0 + 2.0 * (-1 + e) * e - 4.0 * e * m ** 2 - 4.0 * m ** 4)
                    * (
                        log(1.0 - 2.0 * e)
                        - 4.0
                        * log(sqrt(1.0 - 2.0 * e) + sqrt(1.0 - 2.0 * e - 4.0 * m ** 2))
                        + 2.0
                        * log(
                            (sqrt(1.0 - 2.0 * e) - sqrt(1.0 - 2.0 * e - 4.0 * m ** 2))
                            * (1.0 - sqrt(1.0 + (4.0 * m ** 2) / (-1 + 2.0 * e)))
                        )
                    )
                )
            ) / (
                2.0
                * e
                * (1.0 + 2.0 * m ** 2)
                * sqrt((-1 + 2.0 * e) * (-1 + 4.0 * m ** 2))
                * pi
                * Q
            )

            assert ret_val.imag == 0.0
            ret_val = ret_val.real

        assert ret_val >= 0

        return ret_val.real

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

        mupi = mpi / Q
        x = 2.0 * egam / Q
        xmin = 0.0
        xmax = 1 - 4.0 * mupi ** 2

        if x < xmin or x > xmax or Q < 2 * mpi or Q < 2.0 * mx:
            return 0.0

        coeff = qe ** 2 / (4.0 * (1 - 4 * mupi ** 2) ** 1.5 * pi ** 2)

        dynamic = (
            (
                2
                * sqrt(1 - 4 * mupi ** 2 - x)
                * (-1 - 4 * mupi ** 2 * (-1 + x) + x + x ** 2)
            )
            / sqrt(1 - x)
            + (-1 + 4 * mupi ** 2)
            * (-1 + 2 * mupi ** 2 + x)
            * log(
                (1 + sqrt(1 - x) * sqrt(1 - 4 * mupi ** 2 - x) - x) ** 2
                / (-1 + sqrt(1 - x) * sqrt(1 - 4 * mupi ** 2 - x) + x) ** 2
            )
        ) / x

        ret_val = 2.0 * dynamic * coeff / Q

        assert ret_val.imag == 0.0
        assert ret_val.real >= 0
        return ret_val.real

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
