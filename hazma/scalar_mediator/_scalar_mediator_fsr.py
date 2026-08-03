import numpy as np
from cmath import sqrt, pi, log

from hazma.parameters import alpha_em, charged_pion_mass as mpi


def __dnde_xx_to_s_to_ffg(self, photon_energy, Q, mf):
    """Unvectorized dnde_xx_to_s_to_ffg"""
    s = Q**2 - 2.0 * Q * photon_energy

    mx = self.mx

    if 2.0 * mf < Q and 4.0 * mf**2 < s < Q**2 and 2.0 * mx < Q:
        x = 2 * photon_energy / Q
        mu_l = mf / Q

        val = (
            (
                2
                * alpha_em
                * (
                    -2
                    * (-1 + x)
                    * (-1 + 4 * mu_l**2)
                    * sqrt(1 + (4 * mu_l**2) / (-1 + x))
                    + ((x + 4 * mu_l**2) ** 2 - 2 * (-1 + x + 6 * mu_l**2))
                    * log(
                        (1 + sqrt(1 + (4 * mu_l**2) / (-1 + x)))
                        / (1 - sqrt(1 + (4 * mu_l**2) / (-1 + x)))
                    )
                )
            )
            / (pi * Q * x * (1 - 4 * mu_l**2) ** 1.5)
        ).real

        assert val >= 0

        return val
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
            [__dnde_xx_to_s_to_ffg(self, e, Q, mf) for e in photon_energies]
        )
    else:
        return __dnde_xx_to_s_to_ffg(self, photon_energies, Q, mf)


def __dnde_xx_to_s_to_pipig(self, photon_energy, Q):
    """Unvectorized dnde_xx_to_s_to_pipig"""

    mu_pi = mpi / Q
    x = 2 * photon_energy / Q

    if x < 0.0 or 1.0 - 4.0 * mu_pi**2 < x or 2.0 * self.mx > Q:
        return 0.0
    else:
        val = (
            (
                4
                * alpha_em
                * (
                    (-1 + x) * sqrt(1 + (4 * mu_pi**2) / (-1 + x))
                    - (-1 + x + 2 * mu_pi**2)
                    * log(
                        -(
                            (1 + sqrt(1 + (4 * mu_pi**2) / (-1 + x)))
                            / (-1 + sqrt(1 + (4 * mu_pi**2) / (-1 + x)))
                        )
                    )
                )
            )
            / (pi * Q * x * sqrt(1 - 4 * mu_pi**2))
        ).real

        assert val >= 0

        return val


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
            [__dnde_xx_to_s_to_pipig(self, eng_gam, Q) for eng_gam in photon_energies]
        )
    else:
        return __dnde_xx_to_s_to_pipig(self, photon_energies, Q)
