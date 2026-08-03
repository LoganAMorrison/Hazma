"""
This file contains the decay spectra from a right-handed neutrino at rest.
"""

import numpy as np

from hazma.spectra import (
    dnde_photon_neutral_pion as decay_pi0,
    dnde_photon_charged_pion as decay_pi,
    dnde_photon_charged_kaon as decay_k,
    dnde_photon_muon as decay_mu,
)

from hazma.parameters import (
    GF,
    Vud,
    lepton_masses,
    sin_theta_weak as sw,
    cos_theta_weak as cw,
    neutral_pion_mass as mpi0,
    charged_pion_mass as mpi,
    charged_kaon_mass as mk,
)
from hazma.gamma_ray import gamma_ray_decay


def dnde_nu_pi0(self, photon_energies, spectrum_type="all"):
    """
    Compute the gamma-ray spectrum from the decay of a right-handed
    neutrino into a neutral pion and neutrino.

    Parameters
    ----------
    photon_energies: float or np.array
        Photon energies where the spectrum should be computed.
    spectrum_type: str, optional
        String specifying which spectrum component should be computed.
        Options are: "all", "decay" or "fsr"
    """
    if self.mx < mpi0:
        if hasattr(photon_energies, "__len__"):
            return np.zeros_like(photon_energies)
        else:
            return 0.0

    if spectrum_type in ["all", "decay"]:
        epi = (self.mx**2 + mpi0**2) / (2.0 * self.mx)
        return decay_pi0(photon_energies, epi)
    elif spectrum_type == "fsr":
        if hasattr(photon_energies, "__len__"):
            return np.array([0.0 for _ in photon_energies])
        else:
            return 0.0
    else:
        raise ValueError(
            "Type {} is invalid. Use 'all', 'fsr' or 'decay'".format(spectrum_type)
        )


def dnde_pi_l(self, photon_energies, spectrum_type="all"):
    """
    Compute the gamma-ray spectrum from the decay of a right-handed
    neutrino into a charged pion and lepton.

    Parameters
    ----------
    photon_energies: float or np.array
        Photon energies where the spectrum should be computed.
    spectrum_type: str, optional
        String specifying which spectrum component should be computed.
        Options are: "all", "decay" or "fsr"
    """
    if self.mx < mpi + self.ml:
        if hasattr(photon_energies, "__len__"):
            return np.zeros_like(photon_energies)
        else:
            return 0.0

    if spectrum_type == "all":
        return self.dnde_pi_l(photon_energies, "fsr") + self.dnde_pi_l(
            photon_energies, "decay"
        )
    elif spectrum_type == "fsr":
        return self.dnde_pi_l_fsr(photon_energies)
    elif spectrum_type == "decay":
        epi = (self.mx**2 + mpi**2 - self.ml**2) / (2.0 * self.mx)
        el = (self.mx**2 - mpi**2 + self.ml**2) / (2.0 * self.mx)
        if self.lepton == "e":
            return decay_pi(photon_energies, epi)
        elif self.lepton == "mu":
            return decay_pi(photon_energies, epi) + decay_mu(photon_energies, el)
    else:
        raise ValueError(
            "Type {} is invalid. Use 'all', 'fsr' or 'decay'".format(spectrum_type)
        )


def dnde_k_l(self, photon_energies, spectrum_type="all"):
    """
    Compute the gamma-ray spectrum from the decay of a right-handed
    neutrino into a charged kaon and lepton.

    Parameters
    ----------
    photon_energies: float or np.array
        Photon energies where the spectrum should be computed.
    spectrum_type: str, optional
        String specifying which spectrum component should be computed.
        Options are: "all", "decay" or "fsr"
    """
    if self.mx < mk + self.ml:
        if hasattr(photon_energies, "__len__"):
            return np.zeros_like(photon_energies)
        else:
            return 0.0

    if spectrum_type == "all":
        return self.dnde_k_l(photon_energies, "fsr") + self.dnde_k_l(
            photon_energies, "decay"
        )
    elif spectrum_type == "fsr":
        return self.dnde_k_l_fsr(photon_energies)
    elif spectrum_type == "decay":
        ek = (self.mx**2 + mk**2 - self.ml**2) / (2.0 * self.mx)
        el = (self.mx**2 - mk**2 + self.ml**2) / (2.0 * self.mx)
        if self.lepton == "e":
            return decay_k(photon_energies, ek)
        elif self.lepton == "mu":
            return decay_k(photon_energies, ek) + decay_mu(photon_energies, el)
    else:
        raise ValueError(
            "Type {} is invalid. Use 'all', 'fsr' or 'decay'".format(spectrum_type)
        )


def __lnorm_sqr(p):
    return p[0] ** 2 - p[1] ** 2 - p[2] ** 2 - p[3] ** 2


def __msqrd_nu_l_l(momenta, mx, tmix, ml):
    s = __lnorm_sqr(momenta[0] + momenta[2])
    t = __lnorm_sqr(momenta[1] + momenta[2])
    return -(
        (
            GF**2
            * (
                2
                * ml**4
                * (1 + 4 * cw**4 - 4 * sw**2 + 8 * sw**4 + cw**2 * (-4 + 8 * sw**2))
                - 2
                * ml**2
                * (
                    -((1 - 2 * cw**2) ** 2 * mx**2)
                    + (1 - 2 * cw**2) ** 2 * s
                    + 2
                    * (1 + 4 * cw**4 - 4 * sw**2 + 8 * sw**4 + cw**2 * (-4 + 8 * sw**2))
                    * t
                )
                + (1 + 4 * cw**4 - 4 * sw**2 + 8 * sw**4 + cw**2 * (-4 + 8 * sw**2))
                * (s**2 + 2 * s * t + 2 * t**2 - mx**2 * (s + 2 * t))
            )
            * np.sin(2 * tmix) ** 2
        )
        / cw**4
    )


def __msqrd_nu_lp_lp(momenta, mx, tmix, ml):
    s = __lnorm_sqr(momenta[0] + momenta[2])
    t = __lnorm_sqr(momenta[1] + momenta[2])
    return -(
        (
            GF**2
            * (
                2 * ml**4 * (1 - 4 * sw**2 + 8 * sw**4) * np.sin(2 * tmix) ** 2
                + (1 - 4 * sw**2 + 8 * sw**4)
                * (s**2 + 2 * s * t + 2 * t**2 - mx**2 * (s + 2 * t))
                * np.sin(2 * tmix) ** 2
                - 2
                * ml**2
                * (
                    -(mx**2 * np.sin(2 * tmix) ** 2)
                    + (s + 2 * (1 - 4 * sw**2 + 8 * sw**4) * t) * np.sin(2 * tmix) ** 2
                )
            )
        )
        / cw**4
    )


def __msqrd_nup_l_lp(momenta, mx, tmix, mli, mlk):
    t = __lnorm_sqr(momenta[1] + momenta[2])
    return -16 * GF**2 * (-(mli**2) + t) * (-(mlk**2) - mx**2 + t) * np.sin(tmix) ** 2


def __dnde_nu_l_l_decay(self, photon_energies, j, n, m):
    i = self._gen
    leps = ["electron", "muon"]

    if self.include_3body and (n == 2 or m == 2):
        fs = ["neutrino", leps[n - 1], leps[m - 1]]
        if i == j == n == m:

            def msqrd_nu_l_l(momenta):
                return __msqrd_nu_l_l(momenta, self.mx, self.theta, self.ml)

            return gamma_ray_decay(fs, self.mx, photon_energies, msqrd_nu_l_l)
        if (i == j) and (n == m):

            def msqrd_nu_lp_lp(momenta):
                return __msqrd_nu_lp_lp(
                    momenta, self.mx, self.theta, lepton_masses[n - 1]
                )

            return gamma_ray_decay(fs, self.mx, photon_energies, msqrd_nu_lp_lp)
        if ((i == n) and (j == m)) or ((i == m) and (j == n)):

            def msqrd_nup_l_lp(momenta):
                return __msqrd_nup_l_lp(
                    momenta,
                    self.mx,
                    self.theta,
                    lepton_masses[n - 1],
                    lepton_masses[m - 1],
                )

            return gamma_ray_decay(fs, self.mx, photon_energies, msqrd_nup_l_lp)
    else:
        return np.zeros_like(photon_energies)


def dnde_nu_l_l(self, photon_energies, j, n, m, spectrum_type="all"):
    """
    Compute the gamma-ray spectrum from the decay of a right-handed
    neutrino into an active neutrino and two charged leptons.

    Parameters
    ----------
    photon_energies: float or np.array
        Photon energies where the spectrum should be computed.
    spectrum_type: str, optional
        String specifying which spectrum component should be computed.
        Options are: "all", "decay" or "fsr"
    """
    accessable = self.mx > lepton_masses[n - 1] + lepton_masses[m - 1]

    if not accessable:
        if hasattr(photon_energies, "__len__"):
            return np.zeros_like(photon_energies)
        return 0.0

    if spectrum_type == "all":
        return self.dnde_nu_l_l(photon_energies, j, n, m, "fsr") + self.dnde_nu_l_l(
            photon_energies, j, n, m, "decay"
        )
    elif spectrum_type == "fsr":
        if self.include_3body:
            return self.dnde_nu_l_l_fsr(photon_energies, j, n, m)
        else:
            return np.zeros_like(photon_energies)
    elif spectrum_type == "decay":
        return __dnde_nu_l_l_decay(self, photon_energies, j, n, m)

    else:
        raise ValueError(
            "Type {} is invalid. Use 'all', 'fsr' or 'decay'".format(spectrum_type)
        )


def __msqrd_l_pi_pi0(momenta, mx, tmix, ml):
    s = __lnorm_sqr(momenta[0] + momenta[2])
    t = __lnorm_sqr(momenta[1] + momenta[2])
    return (
        2
        * GF**2
        * Vud**2
        * (
            ml**4
            + mx**4
            + 4 * mpi**2 * (mpi0**2 - t)
            + 4 * t * (-(mpi0**2) + s + t)
            - mx**2 * (s + 4 * t)
            - ml**2 * (-2 * mx**2 + s + 4 * t)
        )
        * np.sin(tmix) ** 2
    )


def dnde_l_pi_pi0(self, photon_energies, spectrum_type="all"):
    """
    Compute the gamma-ray spectrum from the decay of a right-handed
    neutrino into a charged pion, neutral pion and charged lepton.

    Parameters
    ----------
    photon_energies: float or np.array
        Photon energies where the spectrum should be computed.
    spectrum_type: str, optional
        String specifying which spectrum component should be computed.
        Options are: "all", "decay" or "fsr"
    """
    if self.mx < self.ml + mpi + mpi0:
        if hasattr(photon_energies, "__len__"):
            return np.zeros_like(photon_energies)
        else:
            return 0.0

    if spectrum_type == "all":
        return self.dnde_l_pi_pi0(photon_energies, "fsr") + self.dnde_l_pi_pi0(
            photon_energies, "decay"
        )
    elif spectrum_type == "fsr":
        if self.include_3body:
            return self.dnde_l_pi_pi0_fsr(photon_energies)
        else:
            return np.zeros_like(photon_energies)
    elif spectrum_type == "decay":
        if self.include_3body:
            if self.lepton == "e":
                lepton = "electron"
            else:
                lepton = "muon"

            def msqrd(momenta):
                return __msqrd_l_pi_pi0(momenta, self.mx, self.theta, self.ml)

            return gamma_ray_decay(
                [lepton, "charged_pion", "neutral_pion"],
                self.mx,
                photon_energies,
                msqrd,
            )
        else:
            return np.zeros_like(photon_energies)

    else:
        raise ValueError(
            "Type {} is invalid. Use 'all', 'fsr' or 'decay'".format(spectrum_type)
        )


def __msqrd_nu_pi_pi(momenta, mx, tmix):
    s = __lnorm_sqr(momenta[0] + momenta[2])
    t = __lnorm_sqr(momenta[1] + momenta[2])
    return (
        GF**2
        * (1 - 2 * sw**2) ** 2
        * (4 * mpi**4 + mx**4 - 8 * mpi**2 * t + 4 * t * (s + t) - mx**2 * (s + 4 * t))
        * np.sin(2 * tmix) ** 2
    ) / (2.0 * cw**4)


def dnde_nu_pi_pi(self, photon_energies, spectrum_type="all"):
    """
    Compute the gamma-ray spectrum from the decay of a right-handed
    neutrino into an active neutrino and two charged pions.

    Parameters
    ----------
    photon_energies: float or np.array
        Photon energies where the spectrum should be computed.
    spectrum_type: str, optional
        String specifying which spectrum component should be computed.
        Options are: "all", "decay" or "fsr"
    """
    if self.mx < 2.0 * mpi:
        if hasattr(photon_energies, "__len__"):
            return np.zeros_like(photon_energies)
        else:
            return 0.0

    if spectrum_type == "all":
        return self.dnde_nu_pi_pi(photon_energies, "fsr") + self.dnde_nu_pi_pi(
            photon_energies, "decay"
        )
    elif spectrum_type == "fsr":
        if self.include_3body:
            return self.dnde_nu_pi_pi_fsr(photon_energies)
        else:
            return np.zeros_like(photon_energies)
    elif spectrum_type == "decay":
        if self.include_3body:

            def msqrd(momenta):
                return __msqrd_nu_pi_pi(momenta, self.mx, self.theta)

            return gamma_ray_decay(
                ["neutrino", "charged_pion", "charged_pion"],
                self.mx,
                photon_energies,
                msqrd,
            )
        else:
            return np.zeros_like(photon_energies)

    else:
        raise ValueError(
            "Type {} is invalid. Use 'all', 'fsr' or 'decay'".format(spectrum_type)
        )


def dnde_nu_g_g(self, photon_energies, spectrum_type="all"):
    """
    Compute the gamma-ray spectrum from the decay of a right-handed
    neutrino into an active neutrino and two photons through an off-shell
    neutral pion. This is only included if the RH neutrino mass is less than
    the neutral pion mass.

    Parameters
    ----------
    photon_energies: float or np.array
        Photon energies where the spectrum should be computed.
    spectrum_type: str, optional
        String specifying which spectrum component should be computed.
        Options are: "all", "decay" or "fsr"
    """
    mx = self.mx
    if mx > mpi0:
        return np.zeros_like(photon_energies)

    spec = (
        32
        * photon_energies**3
        * (
            6 * photon_energies * mx * (-8 * photon_energies + 5 * mx)
            + 5 * (-3 * photon_energies + 2 * mx) * mpi0**2
        )
    ) / (mx**5 * (mx**2 + mpi0**2))

    if spectrum_type == "all":
        return spec
    elif spectrum_type == "fsr":
        return spec
    elif spectrum_type == "decay":
        return np.zeros_like(photon_energies)

    else:
        raise ValueError(
            "Type {} is invalid. Use 'all', 'fsr' or 'decay'".format(spectrum_type)
        )
