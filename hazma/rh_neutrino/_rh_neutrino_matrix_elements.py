"""
This file contains matrix elements for various 3- and 4-body decays of the
right-handed neutrino for use with RAMBO.
"""
import numpy as np

from hazma.parameters import (
    qe,
    GF,
    Vud,
    cos_theta_weak as cw,
    sin_theta_weak as sw,
    charged_pion_mass as mpi,
    neutral_pion_mass as mpi0,
)
from hazma.field_theory_helper_functions.common_functions import (
    minkowski_dot as MDot,
)


def msqrd_pi_pi0_l(self, momenta):
    """
    Compute the squared matrix-element for a RH neutrino decaying into a
    charged pion, a neutral pion and a charged-lepton at leading order in the
    Fermi constant.

    Parameters
    ----------
    momenta: List
        List of NumPy arrays storing the four-momenta of the charged pion,
        neutral pion and charged lepton.

    Returns
    -------
    msqrd: float
        The matrix element for N->pi + pi^0 + l for the given model and
        four-momenta.
    """
    pl = momenta[0]
    ppi = momenta[1]
    ppi0 = momenta[2]
    P = pl + ppi + ppi0
    s = MDot(P - pl, P - pl)
    t = MDot(P - ppi, P - ppi)
    mvr = self.mx
    ml = self.ml
    stheta = self.stheta

    return (
        2
        * GF ** 2
        * stheta ** 2
        * (
            ml ** 4
            + mvr ** 4
            - mvr ** 2 * s
            + ml ** 2 * (2 * mvr ** 2 - s - 4 * t)
            + 4 * mpi ** 2 * (mpi0 ** 2 - t)
            - 4 * mpi0 ** 2 * t
            - 4 * mvr ** 2 * t
            + 4 * s * t
            + 4 * t ** 2
        )
        * Vud ** 2
    )


def msqrd_pi_pi0_l_g(self, momenta):
    """
    Compute the squared matrix-element for a RH neutrino decaying into a
    charged pion, a neutral pion, a charged-lepton and a photon at leading
    order in the Fermi constant.

    Parameters
    ----------
    momenta: List
        List of NumPy arrays storing the four-momenta of the charged pion,
        neutral pion, charged lepton and the photon.

    Returns
    -------
    msqrd: float
        The matrix element for N->pi + pi^0 + l + photon for the given model
        and four-momenta.
    """
    pl = momenta[0]
    ppi = momenta[1]
    ppi0 = momenta[2]
    k = momenta[3]

    Ml = self.ml
    smix = self.stheta

    return (
        -4
        * GF ** 2
        * qe ** 2
        * smix ** 2
        * Vud ** 2
        * (
            2 * mpi ** 2 * MDot(k, pl) ** 4
            + MDot(k, pl) ** 3
            * (
                -2 * MDot(k, ppi) ** 2
                + MDot(k, ppi)
                * (4 * mpi ** 2 - 4 * MDot(pl, ppi) - 6 * MDot(ppi, ppi0))
                + mpi ** 2
                * (
                    mpi ** 2
                    - 3 * mpi0 ** 2
                    + 2 * MDot(k, ppi0)
                    + 4 * MDot(pl, ppi)
                    - 4 * MDot(pl, ppi0)
                    + 2 * MDot(ppi, ppi0)
                )
            )
            + Ml ** 2
            * MDot(k, ppi) ** 2
            * (
                2 * MDot(k, ppi) ** 2
                + 2 * MDot(k, ppi0) ** 2
                + mpi ** 2 * MDot(pl, ppi)
                - 3 * mpi0 ** 2 * MDot(pl, ppi)
                + 2 * MDot(pl, ppi) ** 2
                - 3 * mpi ** 2 * MDot(pl, ppi0)
                + mpi0 ** 2 * MDot(pl, ppi0)
                - 4 * MDot(pl, ppi) * MDot(pl, ppi0)
                + 2 * MDot(pl, ppi0) ** 2
                - Ml ** 2 * (mpi ** 2 + mpi0 ** 2 - 2 * MDot(ppi, ppi0))
                + 2 * (MDot(pl, ppi) + MDot(pl, ppi0)) * MDot(ppi, ppi0)
                + MDot(k, ppi)
                * (
                    mpi ** 2
                    - 3 * mpi0 ** 2
                    - 4 * MDot(k, ppi0)
                    + 4 * MDot(pl, ppi)
                    - 4 * MDot(pl, ppi0)
                    + 2 * MDot(ppi, ppi0)
                )
                + MDot(k, ppi0)
                * (
                    -3 * mpi ** 2
                    + mpi0 ** 2
                    - 4 * MDot(pl, ppi)
                    + 4 * MDot(pl, ppi0)
                    + 2 * MDot(ppi, ppi0)
                )
            )
            - MDot(k, pl)
            * MDot(k, ppi)
            * (
                2 * MDot(k, ppi) ** 3
                + MDot(k, ppi) ** 2
                * (
                    -2 * Ml ** 2
                    + mpi ** 2
                    - 3 * mpi0 ** 2
                    - 4 * MDot(k, ppi0)
                    + 6 * MDot(pl, ppi)
                    - 4 * MDot(pl, ppi0)
                    + 2 * MDot(ppi, ppi0)
                )
                + MDot(k, ppi)
                * (
                    -(Ml ** 2 * (mpi ** 2 - 3 * mpi0 ** 2))
                    + 2 * MDot(k, ppi0) ** 2
                    + 8 * MDot(pl, ppi) ** 2
                    + MDot(pl, ppi0)
                    * (
                        2 * Ml ** 2
                        - 3 * mpi ** 2
                        + mpi0 ** 2
                        + 2 * MDot(pl, ppi0)
                    )
                    + 2 * (-(Ml ** 2) + MDot(pl, ppi0)) * MDot(ppi, ppi0)
                    + 2
                    * MDot(pl, ppi)
                    * (
                        -2 * Ml ** 2
                        + mpi ** 2
                        - 3 * mpi0 ** 2
                        - 7 * MDot(pl, ppi0)
                        + 2 * MDot(ppi, ppi0)
                    )
                    + MDot(k, ppi0)
                    * (
                        2 * Ml ** 2
                        - 3 * mpi ** 2
                        + mpi0 ** 2
                        - 8 * MDot(pl, ppi)
                        + 4 * MDot(pl, ppi0)
                        + 2 * MDot(ppi, ppi0)
                    )
                )
                + MDot(pl, ppi)
                * (
                    2 * MDot(k, ppi0) ** 2
                    + 4 * MDot(pl, ppi) ** 2
                    - 2
                    * Ml ** 2
                    * (mpi ** 2 + mpi0 ** 2 - 2 * MDot(ppi, ppi0))
                    + 2
                    * MDot(pl, ppi)
                    * (
                        mpi ** 2
                        - 3 * mpi0 ** 2
                        - 4 * MDot(pl, ppi0)
                        + 2 * MDot(ppi, ppi0)
                    )
                    + 2
                    * MDot(pl, ppi0)
                    * (
                        -3 * mpi ** 2
                        + mpi0 ** 2
                        + 2 * MDot(pl, ppi0)
                        + 2 * MDot(ppi, ppi0)
                    )
                    + MDot(k, ppi0)
                    * (
                        2 * Ml ** 2
                        - 3 * mpi ** 2
                        + mpi0 ** 2
                        - 2 * MDot(pl, ppi)
                        + 6 * MDot(pl, ppi0)
                        + 2 * MDot(ppi, ppi0)
                    )
                )
            )
            + MDot(k, pl) ** 2
            * (
                -4 * MDot(k, ppi) ** 3
                + MDot(k, ppi) ** 2
                * (
                    mpi ** 2
                    + 3 * mpi0 ** 2
                    + 4 * MDot(k, ppi0)
                    - 10 * MDot(pl, ppi)
                    + 4 * MDot(pl, ppi0)
                    - 8 * MDot(ppi, ppi0)
                )
                + MDot(k, ppi)
                * (
                    (mpi ** 2 - 2 * MDot(pl, ppi))
                    * (
                        mpi ** 2
                        - 3 * mpi0 ** 2
                        + 4 * MDot(pl, ppi)
                        - 4 * MDot(pl, ppi0)
                    )
                    + (
                        -2 * Ml ** 2
                        - mpi ** 2
                        + mpi0 ** 2
                        - 10 * MDot(pl, ppi)
                        + 2 * MDot(pl, ppi0)
                    )
                    * MDot(ppi, ppi0)
                    + 2 * MDot(ppi, ppi0) ** 2
                    + 2
                    * MDot(k, ppi0)
                    * (mpi ** 2 + MDot(pl, ppi) + MDot(ppi, ppi0))
                )
                + mpi ** 2
                * (
                    -(Ml ** 2 * (mpi ** 2 + mpi0 ** 2))
                    + (mpi ** 2 - 3 * mpi0 ** 2) * MDot(pl, ppi)
                    + (-3 * mpi ** 2 + mpi0 ** 2) * MDot(pl, ppi0)
                    + 2
                    * (
                        (MDot(pl, ppi) - MDot(pl, ppi0)) ** 2
                        + MDot(k, ppi0)
                        * (Ml ** 2 + MDot(pl, ppi) + MDot(pl, ppi0))
                        + (Ml ** 2 + MDot(pl, ppi) + MDot(pl, ppi0))
                        * MDot(ppi, ppi0)
                    )
                )
            )
        )
    ) / (MDot(k, pl) ** 2 * MDot(k, ppi) ** 2)


def msqrd_nu_l_l(self, momenta):
    """
    Compute the squared matrix-element for a RH neutrino decaying into a
    active neutrino, and two charged-leptons at leading order in the Fermi
    constant.

    Parameters
    ----------
    momenta: List
        List of NumPy arrays storing the four-momenta of the active neutino,
        and charged leptons.

    Returns
    -------
    msqrd: float
        The matrix element for N-> nu+ l + lbar for the given model and
        four-momenta.
    """
    pnu = momenta[0]
    pl = momenta[1]
    plb = momenta[2]
    P = pnu + pl + plb
    s = MDot(P - pl, P - pl)
    t = MDot(P - plb, P - plb)

    mvr = self.mx
    ml = self.ml
    stheta = self.stheta
    ctheta = np.sqrt(1.0 - stheta ** 2)

    return (
        -4
        * ctheta ** 2
        * GF ** 2
        * stheta ** 2
        * (
            ml ** 4
            * (
                2
                + 8 * cw ** 8
                - 24 * sw ** 2
                + 48 * sw ** 4
                + 8 * cw ** 4 * (-1 + 6 * sw ** 2)
            )
            - (
                1
                + 4 * cw ** 8
                - 4 * sw ** 2
                + 8 * sw ** 4
                + cw ** 4 * (-4 + 8 * sw ** 2)
            )
            * (-(s ** 2) - t ** 2 + mvr ** 2 * (s + t))
            + 2
            * ml ** 2
            * (
                mvr ** 2
                * (
                    1
                    + 4 * cw ** 8
                    - 4 * sw ** 2
                    + 8 * sw ** 4
                    + cw ** 4 * (-4 + 8 * sw ** 2)
                )
                - (-1 + 2 * cw ** 4 + 4 * sw ** 2) ** 2 * (s + t)
            )
        )
    ) / cw ** 8


def msqrd_nu_l_l_g(self, momenta):
    """
    Compute the squared matrix-element for a RH neutrino decaying into a
    active neutrino, two charged-leptons and a photon at leading order in the
    Fermi constant.

    Parameters
    ----------
    momenta: List
        List of NumPy arrays storing the four-momenta of the active neutino,
        charged leptons and the photon.

    Returns
    -------
    msqrd: float
        The matrix element for N-> nu + l + lbar + photon for the given model
        and four-momenta.
    """
    pnu = momenta[0]
    pl = momenta[1]
    plb = momenta[2]
    k = momenta[3]

    mvr = self.mx
    ml = self.ml
    stheta = self.stheta
    ctheta = np.sqrt(1.0 - stheta ** 2)

    return 0.0


def msqrd_nu_pi_pi_g(self, momenta):

    pnu = momenta[0]
    ppim = momenta[1]
    ppip = momenta[2]
    k = momenta[3]

    stheta = self.stheta
    return (
        -8
        * GF ** 2
        * qe ** 2
        * (-1 + stheta)
        * stheta ** 2
        * (1 + stheta)
        * (1 - 2 * sw ** 2) ** 2
        * (
            mpi ** 2 * MDot(k, ppim) ** 3 * (MDot(pnu, ppim) + MDot(pnu, ppip))
            + MDot(k, pnu) ** 2
            * (
                mpi ** 2 * MDot(k, ppim) ** 2
                + mpi ** 2 * MDot(k, ppip) ** 2
                + 2 * MDot(k, ppim) * MDot(k, ppip) * MDot(ppim, ppip)
            )
            + mpi ** 2
            * MDot(k, ppip) ** 2
            * (
                MDot(pnu, ppim) ** 2
                + MDot(pnu, ppim)
                * (
                    -(mpi ** 2)
                    + MDot(k, ppip)
                    - 2 * MDot(pnu, ppip)
                    + MDot(ppim, ppip)
                )
                + MDot(pnu, ppip)
                * (
                    -(mpi ** 2)
                    + MDot(k, ppip)
                    + MDot(pnu, ppip)
                    + MDot(ppim, ppip)
                )
            )
            + MDot(k, ppim) ** 2
            * (
                mpi ** 2 * MDot(pnu, ppim) ** 2
                + MDot(pnu, ppim)
                * (
                    -4 * MDot(k, ppip) ** 2
                    + MDot(k, ppip)
                    * (mpi ** 2 + 4 * MDot(pnu, ppip) - 2 * MDot(ppim, ppip))
                    + mpi ** 2
                    * (-(mpi ** 2) - 2 * MDot(pnu, ppip) + MDot(ppim, ppip))
                )
                + MDot(pnu, ppip)
                * (
                    -4 * MDot(k, ppip) ** 2
                    + MDot(k, ppip)
                    * (mpi ** 2 - 4 * MDot(pnu, ppip) - 2 * MDot(ppim, ppip))
                    + mpi ** 2
                    * (-(mpi ** 2) + MDot(pnu, ppip) + MDot(ppim, ppip))
                )
            )
            + MDot(k, ppim)
            * MDot(k, ppip)
            * (
                MDot(k, ppip)
                * (
                    -4 * MDot(pnu, ppim) ** 2
                    + MDot(pnu, ppip) * (mpi ** 2 - 2 * MDot(ppim, ppip))
                    + MDot(pnu, ppim)
                    * (mpi ** 2 + 4 * MDot(pnu, ppip) - 2 * MDot(ppim, ppip))
                )
                - 2
                * MDot(ppim, ppip)
                * (
                    MDot(pnu, ppim) ** 2
                    + MDot(pnu, ppim)
                    * (-(mpi ** 2) - 2 * MDot(pnu, ppip) + MDot(ppim, ppip))
                    + MDot(pnu, ppip)
                    * (-(mpi ** 2) + MDot(pnu, ppip) + MDot(ppim, ppip))
                )
            )
            + MDot(k, pnu)
            * (
                mpi ** 2 * MDot(k, ppim) ** 3
                + mpi ** 2
                * MDot(k, ppip) ** 2
                * (
                    -(mpi ** 2)
                    + MDot(k, ppip)
                    + 2 * MDot(pnu, ppim)
                    - 2 * MDot(pnu, ppip)
                    + MDot(ppim, ppip)
                )
                + MDot(k, ppim)
                * MDot(k, ppip)
                * (
                    MDot(k, ppip)
                    * (mpi ** 2 - 4 * MDot(pnu, ppim) - 2 * MDot(ppim, ppip))
                    + 2 * (mpi ** 2 - MDot(ppim, ppip)) * MDot(ppim, ppip)
                )
                - MDot(k, ppim) ** 2
                * (
                    4 * MDot(k, ppip) ** 2
                    + mpi ** 2
                    * (
                        mpi ** 2
                        + 2 * MDot(pnu, ppim)
                        - 2 * MDot(pnu, ppip)
                        - MDot(ppim, ppip)
                    )
                    + MDot(k, ppip)
                    * (
                        -(mpi ** 2)
                        + 4 * MDot(pnu, ppip)
                        + 2 * MDot(ppim, ppip)
                    )
                )
            )
        )
    ) / ((-1 + sw ** 2) * MDot(k, ppim) ** 2 * MDot(k, ppip) ** 2)

