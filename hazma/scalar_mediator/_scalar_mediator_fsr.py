import numpy as np
from cmath import sqrt, pi, atanh, log

from hazma.parameters import (qe, alpha_em, charged_pion_mass as mpi)


class ScalarMediatorFSR:
    def __dnde_xx_to_s_to_ffg(self, egam, Q, mf):
        """ Unvectorized dnde_xx_to_s_to_ffg """
        e, rf, s = egam / Q, mf / Q, Q**2 - 2. * Q * egam

        mx = self.mx

        if 2. * mf < Q and 4. * mf**2 < s < Q**2 and 2. * mx < Q:
            ret_val = (alpha_em *
                       (2 * (-1 + 4 * rf**2) *
                        sqrt((-1 + 2 * e) * (-1 + 2 * e + 4 * rf**2)) +
                        4 * (1 + 2 * (-1 + e) * e - 6 * rf**2 + 8 * e * rf**2 +
                             8 * rf**4) *
                        atanh(sqrt(1 + (4 * rf**2) / (-1 + 2 * e))))) / \
                (e * (1 - 4 * rf**2)**1.5 * pi * Q)

            assert ret_val.imag == 0
            ret_val = ret_val.real
            assert ret_val >= 0

            return ret_val
        else:
            return 0.0

    def dnde_xx_to_s_to_ffg(self, egam, Q, mf):
        """Return the fsr spectra for fermions from decay of scalar mediator.

        Computes the final state radiaton spectrum value dNdE from a scalar
        mediator given a gamma ray energy of `eng_gam`, center of mass
        energy `cme`
        and final state fermion mass `mass_f`.

        Paramaters
        ----------
        egam : float
            Gamma ray energy.
        Q: float
            Center of mass energy of mass of off-shell scalar mediator.
        mf : float
            Mass of the final state fermion.
        params: namedtuple
            Namedtuple of the model parameters.

        Returns
        -------
        spec_val : float
            Spectrum value dNdE from scalar mediator.

        """
        if hasattr(egam, '__len__'):
            return np.array([self.__dnde_xx_to_s_to_ffg(e, Q, mf)
                             for e in egam])
        else:
            return self.__dnde_xx_to_s_to_ffg(egam, Q, mf)

    # ######################
    """ Charged Pion FSR """
    # ######################

    def __dnde_xx_to_s_to_pipig(self, eng_gam, Q):
        """Unvectorized dnde_xx_to_s_to_pipig"""

        mupi = mpi / Q
        x = 2 * eng_gam / Q

        if x < 0. or 1. - 4. * mupi**2 < x:
            return 0.

        dynamic = (2 * sqrt(-1 + x) * sqrt(-1 + 4 * mupi**2 + x) +
                   (-1 + 2 * mupi**2 + x) *
                   log((1 - x + sqrt(-1 + x) *
                        sqrt(-1 + 4 * mupi**2 + x))**2 /
                       (-1 + x + sqrt(-1 + x) *
                        sqrt(-1 + 4 * mupi**2 + x))**2)) / x

        coeff = qe**2 / (4. * sqrt(1 - 4 * mupi**2) * pi**2)

        ret_val = 2. * dynamic * coeff / Q

        assert ret_val.imag == 0.0
        assert ret_val.real >= 0.0

        return ret_val.real

    def dnde_xx_to_s_to_pipig(self, eng_gams, Q):
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

        if hasattr(eng_gams, '__len__'):
            return np.array([self.__dnde_xx_to_s_to_pipig(eng_gam, Q)
                             for eng_gam in eng_gams])
        else:
            return self.__dnde_xx_to_s_to_pipig(eng_gams, Q)
