"""Module for computing fsr spectrum from a pseudo-scalar mediator.

@author - Logan Morrison and Adam Coogan.
@data - December 2017

"""
import numpy as np

from cmath import sqrt, log, pi

from hazma.parameters import alpha_em


class PseudoScalarMediatorFSR:
    def __dnde_xx_to_p_to_ffg(self, egam, Q, mf):
        """
        Returns the fsr spectra for fermions from decay of pseudo-scalar
        mediator.

        Computes the final state radiaton spectrum value dNdE from a
        pseudo-scalar mediator given a gamma ray energy of `eng_gam`,
        center of mass
        energy `cme` and final state fermion mass `mass_f`.

        Paramaters
        ----------
        eng_gam : float
            Gamma ray energy.
        cme: float
            Center of mass energy of mass of off-shell pseudo-scalar mediator.
        mass_f : float
            Mass of the final state fermion.

        Returns
        -------
        spec_val : float
            Spectrum value dNdE from pseudo-scalar mediator.
        """
        e, m = egam / Q, mf / Q

        s = Q ** 2 - 2.0 * Q * egam

        ret_val = 0.0

        if 4.0 * mf ** 2 <= s <= Q ** 2:
            ret_val = (
                2.0
                * alpha_em
                * (
                    -sqrt((-1.0 + 2.0 * e) * (-1.0 + 2.0 * e + 4.0 * m ** 2))
                    + (2.0 + 4.0 * (-1.0 + e) * e) * log(m)
                    + m ** 2
                    * (
                        2.0
                        * log(sqrt(1.0 - 2.0 * e) - sqrt(1.0 - 2.0 * e - 4.0 * m ** 2))
                        - log(
                            2.0
                            * (
                                1.0
                                - 2.0 * e
                                - 2.0 * m ** 2
                                + sqrt(
                                    (-1.0 + 2.0 * e) * (-1.0 + 2.0 * e + 4.0 * m ** 2)
                                )
                            )
                        )
                    )
                    + (1.0 + 2.0 * (-1.0 + e) * e)
                    * log(
                        -2.0
                        / (
                            -1.0
                            + 2.0 * e
                            + 2.0 * m ** 2
                            + sqrt((-1.0 + 2.0 * e) * (-1.0 + 2.0 * e + 4.0 * m ** 2))
                        )
                    )
                )
            ) / (e * sqrt(1.0 - 4.0 * m ** 2) * pi * Q)

            assert ret_val.imag == 0.0

            ret_val = ret_val.real

            assert ret_val >= 0.0

        return ret_val

    def dnde_xx_to_p_to_ffg(self, egam, Q, mf):
        """Returns the fsr spectra for fermions from decay of pseudo-scalar
        mediator.

        Computes the final state radiaton spectrum value dNdE from a
        pseudo-scalar mediator given a gamma ray energy of `eng_gam`,
        center of mass energy `cme` and final state fermion mass `mass_f`.

        Paramaters
        ----------
        eng_gam : float
            Gamma ray energy.
        cme: float
            Center of mass energy of mass of off-shell pseudo-scalar mediator.
        mass_f : float
            Mass of the final state fermion.

        Returns
        -------
        spec_val : float
            Spectrum value dNdE from pseudo-scalar mediator.
        """
        if hasattr(egam, "__len__"):
            return np.array([self.__dnde_xx_to_p_to_ffg(e, Q, mf) for e in egam])
        else:
            return self.__dnde_xx_to_p_to_ffg(egam, Q, mf)
