import numpy as np
from cmath import sqrt, pi, atanh, log

from hazma.parameters import qe, alpha_em, charged_pion_mass as mpi


class ScalarMediatorFSR:
    def __dnde_xx_to_s_to_ffg(self, photon_energy, Q, mf):
        """ Unvectorized dnde_xx_to_s_to_ffg """
        e, rf, s = photon_energy / Q, mf / Q, Q ** 2 - 2.0 * Q * photon_energy

        mx = self.mx

        if 2.0 * mf < Q and 4.0 * mf ** 2 < s < Q ** 2 and 2.0 * mx < Q:
            ret_val = (
                alpha_em
                * (
                    2
                    * (-1 + 4 * rf ** 2)
                    * sqrt((-1 + 2 * e) * (-1 + 2 * e + 4 * rf ** 2))
                    + 4
                    * (
                        1
                        + 2 * (-1 + e) * e
                        - 6 * rf ** 2
                        + 8 * e * rf ** 2
                        + 8 * rf ** 4
                    )
                    * atanh(sqrt(1 + (4 * rf ** 2) / (-1 + 2 * e)))
                )
            ) / (e * (1 - 4 * rf ** 2) ** 1.5 * pi * Q)

            assert ret_val.imag == 0
            ret_val = ret_val.real
            assert ret_val >= 0

            return ret_val
        else:
            return 0.0

    def dnde_xx_to_s_to_ffg(self, photon_energies, Q, mf):
        """Return the fsr spectra for dark matter annihilating into a pair of
        fermions.

        Computes the final state radiation spectrum value dNdE from a scalar
        mediator given a gamma ray energy of `eng_gam`, center of mass
        energy `cme`
        and final state fermion mass `mass_f`.

        Parameters
        ----------
        photon_energies : float or np.array
            Energy(ies) of the final state photon.
        Q: float
            Center of mass energy of mass of off-shell scalar mediator.
        mf : float
            Mass of the final state fermion.

        Returns
        -------
        spec_val : float
            Spectrum value dNdE from scalar mediator.

        """
        if hasattr(photon_energies, "__len__"):
            return np.array(
                [self.__dnde_xx_to_s_to_ffg(e, Q, mf) for e in photon_energies]
            )
        else:
            return self.__dnde_xx_to_s_to_ffg(photon_energies, Q, mf)

    def __dnde_xx_to_s_to_pipig(self, photon_energy, Q):
        """Unvectorized dnde_xx_to_s_to_pipig"""

        mupi = mpi / Q
        x = 2 * photon_energy / Q

        if x < 0.0 or 1.0 - 4.0 * mupi ** 2 < x or 2.0 * self.mx > Q:
            return 0.0

        dynamic = (
            2 * sqrt(-1 + x) * sqrt(-1 + 4 * mupi ** 2 + x)
            + (-1 + 2 * mupi ** 2 + x)
            * log(
                (1 - x + sqrt(-1 + x) * sqrt(-1 + 4 * mupi ** 2 + x)) ** 2
                / (-1 + x + sqrt(-1 + x) * sqrt(-1 + 4 * mupi ** 2 + x)) ** 2
            )
        ) / x

        coeff = qe ** 2 / (4.0 * sqrt(1 - 4 * mupi ** 2) * pi ** 2)

        ret_val = 2.0 * dynamic * coeff / Q

        assert ret_val.imag == 0.0
        assert ret_val.real >= 0.0

        return ret_val.real

    def dnde_xx_to_s_to_pipig(self, photon_energies, Q):
        """
        Returns the gamma ray energy spectrum for two fermions annihilating
        into two charged pions and a photon.

        Parameters
        ----------
        photon_energies : float or np.array
            Energy(ies) of the final state photon.
        Q : double
            Center of mass energy

        Returns
        -------
        dnde: float or np.array
            Gamma-ray spectrum for dark matter annihilating into charged pions.

        """

        if hasattr(photon_energies, "__len__"):
            return np.array(
                [
                    self.__dnde_xx_to_s_to_pipig(eng_gam, Q)
                    for eng_gam in photon_energies
                ]
            )
        else:
            return self.__dnde_xx_to_s_to_pipig(photon_energies, Q)
