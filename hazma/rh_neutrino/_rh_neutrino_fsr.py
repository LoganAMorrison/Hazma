"""
This file contains implemenations of the FSR spectra from RH neutrino decays.
"""

import numpy as np
from hazma.parameters import (
    fpi,
    GF,
    qe,
    Vud,
    charged_pion_mass as mpi,
    charged_kaon_mass as mk,
    neutral_pion_mass as mpi0,
)
from hazma.gamma_ray import gamma_ray_fsr
from scipy.interpolate import UnivariateSpline


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
    s = mvr ** 2 - 2.0 * mvr * photon_eng
    if mvr < mpi + ml or (mpi + ml) ** 2 >= s or mvr ** 2 <= s:
        return 0.0

    return (
        qe ** 2
        * (
            (
                -2 * ml ** 6
                + 2 * ml ** 4 * (2 * mvr ** 2 + s)
                + ml ** 2
                * (
                    2 * mpi ** 4
                    - mvr ** 4
                    + mpi ** 2 * (6 * mvr ** 2 - 2 * s)
                    - 6 * mvr ** 2 * s
                    + s ** 2
                )
                + mvr ** 2
                * (
                    2 * mpi ** 4
                    + mvr ** 4
                    + s ** 2
                    - 2 * mpi ** 2 * (mvr ** 2 + s)
                )
            )
            * np.log(
                (
                    ml ** 2
                    - mpi ** 2
                    + s
                    + np.sqrt(
                        ml ** 4
                        + (mpi ** 2 - s) ** 2
                        - 2 * ml ** 2 * (mpi ** 2 + s)
                    )
                )
                / (
                    ml ** 2
                    - mpi ** 2
                    + s
                    - np.sqrt(
                        ml ** 4
                        + (mpi ** 2 - s) ** 2
                        - 2 * ml ** 2 * (mpi ** 2 + s)
                    )
                )
            )
            + 2
            * (
                ml ** 4
                - mpi ** 2 * mvr ** 2
                + mvr ** 4
                - ml ** 2 * (mpi ** 2 + 2 * mvr ** 2)
            )
            * (
                -2
                * np.sqrt(
                    ml ** 4
                    + (mpi ** 2 - s) ** 2
                    - 2 * ml ** 2 * (mpi ** 2 + s)
                )
                + (ml ** 2 + mpi ** 2 - s)
                * np.log(
                    (
                        -(ml ** 2)
                        + mpi ** 2
                        + s
                        - np.sqrt(
                            ml ** 4
                            + (mpi ** 2 - s) ** 2
                            - 2 * ml ** 2 * (mpi ** 2 + s)
                        )
                    )
                    / (
                        -(ml ** 2)
                        + mpi ** 2
                        + s
                        + np.sqrt(
                            ml ** 4
                            + (mpi ** 2 - s) ** 2
                            - 2 * ml ** 2 * (mpi ** 2 + s)
                        )
                    )
                )
            )
        )
    ) / (
        8.0
        * np.sqrt(
            -(ml ** 2)
            + (ml ** 2 - mpi ** 2 + mvr ** 2) ** 2 / (4.0 * mvr ** 2)
        )
        * (
            ml ** 4
            - mpi ** 2 * mvr ** 2
            + mvr ** 4
            - ml ** 2 * (mpi ** 2 + 2 * mvr ** 2)
        )
        * np.pi ** 2
        * (mvr ** 2 - s)
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
        return np.array([self._dnde_pi_l_fsr(e) for e in photon_energies])
    else:
        return self._dnde_pi_l_fsr(photon_energies)


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
    s = mvr ** 2 - 2.0 * mvr * photon_eng
    if mvr < mk + ml or (mk + ml) ** 2 >= s or mvr ** 2 <= s:
        return 0.0

    return -(
        qe ** 2
        * (
            4
            * (-((ml ** 2 - mvr ** 2) ** 2) + mk ** 2 * (ml ** 2 + mvr ** 2))
            * np.sqrt(
                mk ** 4 + (ml ** 2 - s) ** 2 - 2 * mk ** 2 * (ml ** 2 + s)
            )
            - 2
            * (-((ml ** 2 - mvr ** 2) ** 2) + mk ** 2 * (ml ** 2 + mvr ** 2))
            * (mk ** 2 + ml ** 2 - s)
            * np.log(
                (
                    mk ** 2
                    - ml ** 2
                    + s
                    - np.sqrt(
                        mk ** 4
                        + (ml ** 2 - s) ** 2
                        - 2 * mk ** 2 * (ml ** 2 + s)
                    )
                )
                / (
                    mk ** 2
                    - ml ** 2
                    + s
                    + np.sqrt(
                        mk ** 4
                        + (ml ** 2 - s) ** 2
                        - 2 * mk ** 2 * (ml ** 2 + s)
                    )
                )
            )
            + (
                -2 * ml ** 6
                + 2 * mk ** 4 * (ml ** 2 + mvr ** 2)
                + 2 * ml ** 4 * (2 * mvr ** 2 + s)
                + mvr ** 2 * (mvr ** 4 + s ** 2)
                + ml ** 2 * (-(mvr ** 4) - 6 * mvr ** 2 * s + s ** 2)
                - 2
                * mk ** 2
                * (ml ** 2 * (-3 * mvr ** 2 + s) + mvr ** 2 * (mvr ** 2 + s))
            )
            * np.log(
                (
                    -(mk ** 2)
                    + ml ** 2
                    + s
                    + np.sqrt(
                        mk ** 4
                        + (ml ** 2 - s) ** 2
                        - 2 * mk ** 2 * (ml ** 2 + s)
                    )
                )
                / (
                    -(mk ** 2)
                    + ml ** 2
                    + s
                    - np.sqrt(
                        mk ** 4
                        + (ml ** 2 - s) ** 2
                        - 2 * mk ** 2 * (ml ** 2 + s)
                    )
                )
            )
        )
    ) / (
        8.0
        * (-((ml ** 2 - mvr ** 2) ** 2) + mk ** 2 * (ml ** 2 + mvr ** 2))
        * np.sqrt(
            -(ml ** 2)
            + (-(mk ** 2) + ml ** 2 + mvr ** 2) ** 2 / (4.0 * mvr ** 2)
        )
        * np.pi ** 2
        * (mvr ** 2 - s)
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
        return np.array([self._dnde_k_l_fsr(e) for e in photon_energies])
    else:
        return self._dnde_k_l_fsr(photon_energies)


def dnde_nu_l_l_fsr(self, photon_energies):
    """
    Compute the FSR spectra from a right-handed neutrino decaying into
    an active neutrino and two charged leptons.

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
    spectrum = gamma_ray_fsr(
        photon_energies,
        self.mx,
        np.array([self.mx]),
        np.array([0.0, self.ml, self.ml]),
        self.width_nu_l_l(),
        self.msqrd_nu_l_l_g,
        nevents=1000,
    )
    return np.array([spec[0] for spec in spectrum])


def dnde_l_pi_pi0_fsr(self, photon_energies):
    """
    Compute the FSR spectra from a right-handed neutrino decaying into
    a neutral pion, charged pion and charged lepton.

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
    spectrum = gamma_ray_fsr(
        photon_energies,
        self.mx,
        np.array([self.mx]),
        np.array([self.ml, mpi, mpi0]),
        self.width_l_pi_pi0(),
        self.msqrd_l_pi_pi0_g,
        nevents=1000,
    )
    return np.array([spec[0] for spec in spectrum])


def dnde_nu_pi_pi_fsr(self, photon_energies):
    """
    Compute the FSR spectra from a right-handed neutrino decaying into
    an active neutrino and two charged pions.

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
    spectrum = gamma_ray_fsr(
        photon_energies,
        self.mx,
        np.array([self.mx]),
        np.array([0.0, mpi, mpi]),
        self.width_nu_pi_pi(),
        self.msqrd_nu_pi_pi_g,
        nevents=1000,
    )
    return np.array([spec[0] for spec in spectrum])
