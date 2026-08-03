"""
This file contains implemenations of the FSR spectra from RH neutrino decays.
"""

import numpy as np
from hazma.parameters import (
    qe,
    charged_pion_mass as mpi,
    charged_kaon_mass as mk,
    electron_mass as me,
    muon_mass as mmu,
)
from hazma.utils import dnde_altarelli_parisi_fermion, dnde_altarelli_parisi_scalar


def _dnde_pi_l_fsr(self, photon_eng):
    """
    Compute the FSR spectra from a right-handed neutrino decay into a
    charged pion and lepton.

    Parameters
    ----------
    self: object
        Instance of the `RHNeutrino` class. (see
        `hazma.rh_neutrino.__init__.py`)
    photon_eng: float
       The energy of the final state photon in MeV.

    Returns
    -------
    dnde: float
        Photon spectrum.
    """
    mvr = self.mx
    ml = self.ml
    s = mvr**2 - 2.0 * mvr * photon_eng
    if mvr < mpi + ml or (mpi + ml) ** 2 >= s or mvr**2 <= s:
        return 0.0

    return (
        qe**2
        * (
            (
                -2 * ml**6
                + 2 * ml**4 * (2 * mvr**2 + s)
                + ml**2
                * (
                    2 * mpi**4
                    - mvr**4
                    + mpi**2 * (6 * mvr**2 - 2 * s)
                    - 6 * mvr**2 * s
                    + s**2
                )
                + mvr**2 * (2 * mpi**4 + mvr**4 + s**2 - 2 * mpi**2 * (mvr**2 + s))
            )
            * np.log(
                (
                    ml**2
                    - mpi**2
                    + s
                    + np.sqrt(ml**4 + (mpi**2 - s) ** 2 - 2 * ml**2 * (mpi**2 + s))
                )
                / (
                    ml**2
                    - mpi**2
                    + s
                    - np.sqrt(ml**4 + (mpi**2 - s) ** 2 - 2 * ml**2 * (mpi**2 + s))
                )
            )
            + 2
            * (ml**4 - mpi**2 * mvr**2 + mvr**4 - ml**2 * (mpi**2 + 2 * mvr**2))
            * (
                -2 * np.sqrt(ml**4 + (mpi**2 - s) ** 2 - 2 * ml**2 * (mpi**2 + s))
                + (ml**2 + mpi**2 - s)
                * np.log(
                    (
                        -(ml**2)
                        + mpi**2
                        + s
                        - np.sqrt(ml**4 + (mpi**2 - s) ** 2 - 2 * ml**2 * (mpi**2 + s))
                    )
                    / (
                        -(ml**2)
                        + mpi**2
                        + s
                        + np.sqrt(ml**4 + (mpi**2 - s) ** 2 - 2 * ml**2 * (mpi**2 + s))
                    )
                )
            )
        )
    ) / (
        8.0
        * np.sqrt(-(ml**2) + (ml**2 - mpi**2 + mvr**2) ** 2 / (4.0 * mvr**2))
        * (ml**4 - mpi**2 * mvr**2 + mvr**4 - ml**2 * (mpi**2 + 2 * mvr**2))
        * np.pi**2
        * (mvr**2 - s)
    )


def dnde_pi_l_fsr(self, photon_energies):
    """
    Compute the FSR spectra from a right-handed neutrino decay into a
    charged pion and lepton.

    Parameters
    ----------
    self: object
        Instance of the `RHNeutrino` class. (see
        `hazma.rh_neutrino.__init__.py`)
    photon_energies: float or np.array
       The energy of the final state photon in MeV.

    Returns
    -------
    dnde: float or np.array
        Photon spectrum.
    """
    if hasattr(photon_energies, "__len__"):
        return np.array([_dnde_pi_l_fsr(self, e) for e in photon_energies])
    return _dnde_pi_l_fsr(self, photon_energies)


def _dnde_k_l_fsr(self, photon_eng):
    """
    Compute the FSR spectra from a right-handed neutrino decay into a
    charged kaon and lepton.

    Parameters
    ----------
    self: object
        Instance of the `RHNeutrino` class. (see
        `hazma.rh_neutrino.__init__.py`)
    photon_eng: float
       The energy of the final state photon in MeV.

    Returns
    -------
    dnde: float
        Photon spectrum.
    """
    mvr = self.mx
    ml = self.ml
    s = mvr**2 - 2.0 * mvr * photon_eng
    if mvr < mk + ml or (mk + ml) ** 2 >= s or mvr**2 <= s:
        return 0.0

    return -(
        qe**2
        * (
            4
            * (-((ml**2 - mvr**2) ** 2) + mk**2 * (ml**2 + mvr**2))
            * np.sqrt(mk**4 + (ml**2 - s) ** 2 - 2 * mk**2 * (ml**2 + s))
            - 2
            * (-((ml**2 - mvr**2) ** 2) + mk**2 * (ml**2 + mvr**2))
            * (mk**2 + ml**2 - s)
            * np.log(
                (
                    mk**2
                    - ml**2
                    + s
                    - np.sqrt(mk**4 + (ml**2 - s) ** 2 - 2 * mk**2 * (ml**2 + s))
                )
                / (
                    mk**2
                    - ml**2
                    + s
                    + np.sqrt(mk**4 + (ml**2 - s) ** 2 - 2 * mk**2 * (ml**2 + s))
                )
            )
            + (
                -2 * ml**6
                + 2 * mk**4 * (ml**2 + mvr**2)
                + 2 * ml**4 * (2 * mvr**2 + s)
                + mvr**2 * (mvr**4 + s**2)
                + ml**2 * (-(mvr**4) - 6 * mvr**2 * s + s**2)
                - 2 * mk**2 * (ml**2 * (-3 * mvr**2 + s) + mvr**2 * (mvr**2 + s))
            )
            * np.log(
                (
                    -(mk**2)
                    + ml**2
                    + s
                    + np.sqrt(mk**4 + (ml**2 - s) ** 2 - 2 * mk**2 * (ml**2 + s))
                )
                / (
                    -(mk**2)
                    + ml**2
                    + s
                    - np.sqrt(mk**4 + (ml**2 - s) ** 2 - 2 * mk**2 * (ml**2 + s))
                )
            )
        )
    ) / (
        8.0
        * (-((ml**2 - mvr**2) ** 2) + mk**2 * (ml**2 + mvr**2))
        * np.sqrt(-(ml**2) + (-(mk**2) + ml**2 + mvr**2) ** 2 / (4.0 * mvr**2))
        * np.pi**2
        * (mvr**2 - s)
    )


def dnde_k_l_fsr(self, photon_energies):
    """
    Compute the FSR spectra from a right-handed neutrino decay into a
    charged kaon and lepton.

    Parameters
    ----------
    self: object
        Instance of the `RHNeutrino` class. (see
        `hazma.rh_neutrino.__init__.py`)
    photon_energies: float or np.array
       The energy of the final state photon in MeV.

    Returns
    -------
    dnde: float or np.array
        Photon spectrum.
    """
    if hasattr(photon_energies, "__len__"):
        return np.array([_dnde_k_l_fsr(self, e) for e in photon_energies])
    return _dnde_k_l_fsr(self, photon_energies)


def dnde_nu_l_l_fsr(self, photon_energies, j, n, m):
    """
    Compute the FSR contribution to the gamma-ray spectrum fom the decay of a
    right-handed neutrino into an active neutrino and two charged leptons.

    Parameters
    ----------
    photon_energies: float or array-like
        Photon energies where the spectrum should be computed.

    Returns
    -------
    spectrum: float or array-like
        Gamma-ray spectrum.
    """
    br = self.width_nu_l_l(j, n, m) / self.decay_widths()["total"]
    ml1 = me if n == 1 else mmu
    ml2 = me if m == 1 else mmu

    if br > 0.0:
        if n == m:
            return br * dnde_altarelli_parisi_fermion(photon_energies, self.mx, ml1)
        else:
            return (
                br
                * 0.5
                * (
                    dnde_altarelli_parisi_fermion(photon_energies, self.mx, ml1)
                    + dnde_altarelli_parisi_fermion(photon_energies, self.mx, ml2)
                )
            )
    else:
        if hasattr(photon_energies, "__len__"):
            return np.zeros_like(photon_energies)
        return 0.0


def dnde_l_pi_pi0_fsr(self, photon_energies):
    """
    Compute the FSR contribution to the gamma-ray spectrum fom the decay of a
    right-handed neutrino a charged lepton, charged pion and neutral pion.

    Parameters
    ----------
    photon_energies: float or array-like
        Photon energies where the spectrum should be computed.

    Returns
    -------
    spectrum: float or array-like
        Gamma-ray spectrum.
    """
    br = self.width_l_pi_pi0() / self.decay_widths()["total"]
    ml = self.ml

    if br > 0.0:
        return (
            br
            * 0.5
            * (
                dnde_altarelli_parisi_fermion(photon_energies, self.mx, ml)
                + dnde_altarelli_parisi_scalar(photon_energies, self.mx, mpi)
            )
        )
    else:
        if hasattr(photon_energies, "__len__"):
            return np.zeros_like(photon_energies)
        return 0.0


def dnde_nu_pi_pi_fsr(self, photon_energies):
    """
    Compute the FSR contribution to the gamma-ray spectrum fom the decay of a
    right-handed neutrino into an active neutrino and two charged pions.

    Parameters
    ----------
    photon_energies: float or array-like
        Photon energies where the spectrum should be computed.

    Returns
    -------
    spectrum: float or array-like
        Gamma-ray spectrum.
    """
    br = self.width_nu_pi_pi() / self.decay_widths()["total"]

    if br > 0.0:
        return br * (dnde_altarelli_parisi_scalar(photon_energies, self.mx, mpi))
    else:
        if hasattr(photon_energies, "__len__"):
            return np.zeros_like(photon_energies)
        return 0.0
