"""
This file contains the mixin class which implements the partial widths of the
right-handed neutrino.
"""
from hazma.parameters import (
    GF,
    fpi,
    Vud,
    neutral_pion_mass as mpi0,
    charged_pion_mass as mpi,
    sin_theta_weak as sw,
    cos_theta_weak as cw,
)
from scipy.integrate import quad
import numpy as np


# =======================
# ---- 2-Body Widths ----
# =======================


class RHNeutrinoWidths:
    def width_pi0_nu(self):
        """
        Computes the width of the right-handed neutrino into a neutral pion and
        active neutrino.

        Parameters
        ----------
        mvr:
            Right-handed neutrino mass in MeV.
        stheta:
            Mixing angle between right-handed and active neutrinos.
        """
        mvr = self.mx
        stheta = self.stheta
        if mvr < mpi0:
            return 0.0
        return -(
            fpi ** 2
            * GF ** 2
            * stheta ** 2
            * (
                mvr ** 2 * (1 - 2 * stheta ** 2) ** 2
                - mpi0 ** 2 * (-1 + stheta ** 2) ** 2
            )
            * np.sqrt(
                -((mvr ** 2 * stheta ** 4) / (-1 + stheta ** 2) ** 2)
                + (
                    mpi0 ** 2
                    - mvr ** 2 * (1 + stheta ** 4 / (-1 + stheta ** 2) ** 2)
                )
                ** 2
                / (4.0 * mvr ** 2)
            )
        ) / (4.0 * cw ** 2 * np.pi * (-1 + stheta ** 2) ** 3)

    def width_pi_l(self):
        """
        Computes the width of the right-handed neutrino into a charged pion and
        a lepton. This only includes the contribution from one charge
        configuration (i.e. N -> pi^+ + e^-).

        Parameters
        ----------
        mvr: float
            Right-handed neutrino mass in MeV.
        stheta: float
            Mixing angle between right-handed and active neutrinos.
        ml: float
            Mass of the final-state lepton in MeV.
        """
        mvr = self.mx
        stheta = self.stheta
        ml = self.ml

        if mvr < mpi + ml:
            return 0.0
        return (
            fpi ** 2
            * GF ** 2
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
            * stheta ** 2
            * Vud ** 2
        ) / (4.0 * mvr ** 2 * np.pi)


# =======================
# ---- 3-Body Widths ----
# =======================


def width_n_to_nu_pi0_pi0(mvr, stheta):
    """
    Compute the width for a right-handed neutrino to decay into an active
    neutrino and two neutral pion. (NOTE: The amplitude for this process is
    order G_F^4. For consistancy of the EFT, we must ignore this process.)

    Parameters
    ----------
    mvr: float
        Right-handed neutrino mass in MeV.
    stheta: float
        Mixing angle between right-handed and active neutrinos.

    Returns
    -------
    width: float
        Partial decay with for N -> nu + pi0 + pi0.

    """
    # Higher order in G_F
    return 0.0


def width_n_to_nu_pi_pi(mvr, stheta):
    """
    Compute the width of the right-handed neutrino into an active-neutrino
    and two neutral pions.

    Parameters
    ----------
    mvr: float
        Right-handed neutrino mass in MeV.
    stheta: float
        Mixing angle between right-handed and active neutrinos.

    Returns
    -------
    width: float
        Partial decay with for N -> nu + pi^+ + pi^-.
    """
    ctheta = np.sqrt(1.0 - stheta)
    return (
        ctheta ** 2
        * GF ** 2
        * stheta ** 2
        * (1 + 2 * sw ** 2) ** 2
        * (
            mvr
            * np.sqrt(-4 * mpi ** 2 + mvr ** 2)
            * (
                12 * mpi ** 6
                - 10 * mpi ** 4 * mvr ** 2
                + 24 * mpi ** 2 * mvr ** 4
                + mvr ** 6
            )
            + (
                6 * mpi ** 8
                - 6 * mpi ** 6 * mvr ** 2
                - 3 * mpi ** 2 * mvr ** 6
            )
            * np.log(
                (
                    2 * mpi ** 4
                    + mvr ** 3 * (mvr + np.sqrt(-4 * mpi ** 2 + mvr ** 2))
                    - 2
                    * mpi ** 2
                    * mvr
                    * (2 * mvr + np.sqrt(-4 * mpi ** 2 + mvr ** 2))
                )
                / (
                    2 * mpi ** 4
                    + mvr ** 3 * (mvr - np.sqrt(-4 * mpi ** 2 + mvr ** 2))
                    + 2
                    * mpi ** 2
                    * mvr
                    * (-2 * mvr + np.sqrt(-4 * mpi ** 2 + mvr ** 2))
                )
            )
        )
    ) / (768.0 * cw ** 2 * mvr ** 3 * np.pi ** 3)


def width_n_to_l_pi_pi0(mvr, stheta, ml):
    """
    Compute the width for right-handed neutrino to a lepton, charged pion and
    neutral pion. This only includes a single charge configuration
    (i.e. N -> e^- + pi^+ + pi0).

    Parameters
    ----------
    mvr: float
        Right-handed neutrino mass in MeV.
    stheta: float
        Mixing angle between right-handed and active neutrinos.
    ml: float
        Mass of the lepton associated with the active neutrino which the
        right-handed neutrino mixes with.
    """
    return 0.0


def width_n_to_nu_nu_nu():
    """
    Compute the width for right-handed neutrino to three active neutrinos.
    """
    return 0.0


def width_n_to_nu_l_l():
    """
    Compute the width for right-handed neutrino to an active neutrino and two
    leptons.
    """
    return 0.0
