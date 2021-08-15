"""
This file contains implemenations of the FSR spectra from RH neutrino decays.
"""

import numpy as np
from hazma.parameters import (
    qe,
    charged_pion_mass as mpi,
    charged_kaon_mass as mk,
)


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
                * (2 * mpi ** 4 + mvr ** 4 + s ** 2 - 2 * mpi ** 2 * (mvr ** 2 + s))
            )
            * np.log(
                (
                    ml ** 2
                    - mpi ** 2
                    + s
                    + np.sqrt(
                        ml ** 4 + (mpi ** 2 - s) ** 2 - 2 * ml ** 2 * (mpi ** 2 + s)
                    )
                )
                / (
                    ml ** 2
                    - mpi ** 2
                    + s
                    - np.sqrt(
                        ml ** 4 + (mpi ** 2 - s) ** 2 - 2 * ml ** 2 * (mpi ** 2 + s)
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
                * np.sqrt(ml ** 4 + (mpi ** 2 - s) ** 2 - 2 * ml ** 2 * (mpi ** 2 + s))
                + (ml ** 2 + mpi ** 2 - s)
                * np.log(
                    (
                        -(ml ** 2)
                        + mpi ** 2
                        + s
                        - np.sqrt(
                            ml ** 4 + (mpi ** 2 - s) ** 2 - 2 * ml ** 2 * (mpi ** 2 + s)
                        )
                    )
                    / (
                        -(ml ** 2)
                        + mpi ** 2
                        + s
                        + np.sqrt(
                            ml ** 4 + (mpi ** 2 - s) ** 2 - 2 * ml ** 2 * (mpi ** 2 + s)
                        )
                    )
                )
            )
        )
    ) / (
        8.0
        * np.sqrt(-(ml ** 2) + (ml ** 2 - mpi ** 2 + mvr ** 2) ** 2 / (4.0 * mvr ** 2))
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
            * np.sqrt(mk ** 4 + (ml ** 2 - s) ** 2 - 2 * mk ** 2 * (ml ** 2 + s))
            - 2
            * (-((ml ** 2 - mvr ** 2) ** 2) + mk ** 2 * (ml ** 2 + mvr ** 2))
            * (mk ** 2 + ml ** 2 - s)
            * np.log(
                (
                    mk ** 2
                    - ml ** 2
                    + s
                    - np.sqrt(
                        mk ** 4 + (ml ** 2 - s) ** 2 - 2 * mk ** 2 * (ml ** 2 + s)
                    )
                )
                / (
                    mk ** 2
                    - ml ** 2
                    + s
                    + np.sqrt(
                        mk ** 4 + (ml ** 2 - s) ** 2 - 2 * mk ** 2 * (ml ** 2 + s)
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
                        mk ** 4 + (ml ** 2 - s) ** 2 - 2 * mk ** 2 * (ml ** 2 + s)
                    )
                )
                / (
                    -(mk ** 2)
                    + ml ** 2
                    + s
                    - np.sqrt(
                        mk ** 4 + (ml ** 2 - s) ** 2 - 2 * mk ** 2 * (ml ** 2 + s)
                    )
                )
            )
        )
    ) / (
        8.0
        * (-((ml ** 2 - mvr ** 2) ** 2) + mk ** 2 * (ml ** 2 + mvr ** 2))
        * np.sqrt(
            -(ml ** 2) + (-(mk ** 2) + ml ** 2 + mvr ** 2) ** 2 / (4.0 * mvr ** 2)
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
    return self._dnde_k_l_fsr(photon_energies)
