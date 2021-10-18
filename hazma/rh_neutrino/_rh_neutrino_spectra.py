"""
This file contains the decay spectra from a right-handed neutrino at rest.
"""
import numpy as np

from hazma.decay import (
    neutral_pion as decay_pi0,
    charged_pion as decay_pi,
    charged_kaon as decay_k,
    muon as decay_mu,
)
from hazma.parameters import (
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

    if spectrum_type == "all":
        return self.dnde_nu_pi0(photon_energies, "fsr") + self.dnde_nu_pi0(
            photon_energies, "decay"
        )
    elif spectrum_type == "fsr":
        if hasattr(photon_energies, "__len__"):
            return np.array([0.0 for _ in photon_energies])
        else:
            return 0.0
    elif spectrum_type == "decay":
        epi = (self.mx ** 2 + mpi0 ** 2) / (2.0 * self.mx)
        return decay_pi0(photon_energies, epi)
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
        epi = (self.mx ** 2 + mpi ** 2 - self.ml ** 2) / (2.0 * self.mx)
        el = (self.mx ** 2 - mpi ** 2 + self.ml ** 2) / (2.0 * self.mx)
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
        ek = (self.mx ** 2 + mk ** 2 - self.ml ** 2) / (2.0 * self.mx)
        el = (self.mx ** 2 - mk ** 2 + self.ml ** 2) / (2.0 * self.mx)
        if self.lepton == "e":
            return decay_k(photon_energies, ek)
        elif self.lepton == "mu":
            return decay_k(photon_energies, ek) + decay_mu(photon_energies, el)
    else:
        raise ValueError(
            "Type {} is invalid. Use 'all', 'fsr' or 'decay'".format(spectrum_type)
        )


def dnde_nu_l_l(self, photon_energies, spectrum_type="all"):
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
    if self.mx < 2.0 * self.ml:
        if hasattr(photon_energies, "__len__"):
            return np.zeros_like(photon_energies)
        else:
            return 0.0

    if spectrum_type == "all":
        return self.dnde_nu_l_l(photon_energies, "fsr") + self.dnde_nu_l_l(
            photon_energies, "decay"
        )
    elif spectrum_type == "fsr":
        if self.include_3body:
            return self.dnde_nu_l_l_fsr(photon_energies)
        else:
            return np.zeros_like(photon_energies)
    elif spectrum_type == "decay":
        if self.include_3body and self.lepton == "mu":
            return gamma_ray_decay(
                ["neutrino", "muon", "muon"], self.mx, photon_energies
            )
        else:
            return np.zeros_like(photon_energies)

    else:
        raise ValueError(
            "Type {} is invalid. Use 'all', 'fsr' or 'decay'".format(spectrum_type)
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
            return gamma_ray_decay(
                [lepton, "charged_pion", "neutral_pion"],
                self.mx,
                photon_energies,
            )
        else:
            return np.zeros_like(photon_energies)

    else:
        raise ValueError(
            "Type {} is invalid. Use 'all', 'fsr' or 'decay'".format(spectrum_type)
        )


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
            return gamma_ray_decay(
                ["neutrino", "charged_pion", "charged_pion"],
                self.mx,
                photon_energies,
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
        * photon_energies ** 3
        * (
            6 * photon_energies * mx * (-8 * photon_energies + 5 * mx)
            + 5 * (-3 * photon_energies + 2 * mx) * mpi0 ** 2
        )
    ) / (mx ** 5 * (mx ** 2 + mpi0 ** 2))

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
