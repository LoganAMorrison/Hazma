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
    vh as vev,
)
from hazma.field_theory_helper_functions.common_functions import minkowski_dot as LDot

MW = 80.379e3
MH = 125.10e3


def msqrd_nu_pi_pi(self, momenta):
    """
    Compute the squared matrix-element for a RH neutrino decaying into a
    neutrino and two charged pions at leading order in the Fermi constant.
    Momenta are ordered as follows: {nu,pi+,pi-}.

    Parameters
    ----------
    momenta: List
        List of NumPy arrays storing the four-momenta of the final state
        particles.

    Returns
    -------
    msqrd: float
        The matrix element for N -> nu + pi + pi for the given model and
        four-momenta.
    """
    smix = self.stheta
    mx = self.mx

    pnu = momenta[0]
    ppip = momenta[1]
    ppim = momenta[2]
    P = pnu + ppim + ppip

    s = LDot(P - pnu, P - pnu)
    t = LDot(P - ppip, P - ppip)

    return (
        qe ** 4
        * (-1 + smix ** 2)
        * (smix - 2 * smix * sw ** 2) ** 2
        * (
            mx ** 4
            - mx ** 2 * (s + 4 * t)
            + 4 * (mpi ** 4 - 2 * mpi ** 2 * t + t * (s + t))
        )
    ) / (16.0 * MW ** 4 * sw ** 4 * (-1 + sw ** 2))


def msqrd_nu_pi_pi_g(self, momenta):
    """
    Compute the squared matrix-element for a RH neutrino decaying into a
    neutrino, two charged pions and a photon at leading order in the Fermi
    constant. Momenta are ordered as follows: {nu,pi+,pi-,photon}.

    Parameters
    ----------
    momenta: List
        List of NumPy arrays storing the four-momenta of the final state
        particles.

    Returns
    -------
    msqrd: float
        The matrix element for N -> nu + pi + pi + gamma for the given model
        and four-momenta.
    """
    k1 = momenta[0]
    k2 = momenta[1]
    k3 = momenta[2]
    k4 = momenta[3]

    smix = self.stheta

    return (
        -16
        * GF ** 2
        * qe ** 2
        * (-1 + smix ** 2)
        * (smix - 2 * smix * sw ** 2) ** 2
        * (
            LDot(k1, k2) ** 2
            * (
                mpi ** 2 * LDot(k2, k4) ** 2
                - 2 * LDot(k2, k3) * LDot(k2, k4) * LDot(k3, k4)
                + mpi ** 2 * LDot(k3, k4) ** 2
            )
            + LDot(k1, k3) ** 2
            * (
                mpi ** 2 * LDot(k2, k4) ** 2
                - 2 * LDot(k2, k3) * LDot(k2, k4) * LDot(k3, k4)
                + mpi ** 2 * LDot(k3, k4) ** 2
            )
            + LDot(k1, k2)
            * (
                mpi ** 2 * LDot(k2, k4) ** 3
                + mpi ** 2
                * LDot(k3, k4) ** 2
                * (
                    -(mpi ** 2)
                    - 2 * LDot(k1, k3)
                    + 2 * LDot(k1, k4)
                    + LDot(k2, k3)
                    + LDot(k3, k4)
                )
                - LDot(k2, k4) ** 2
                * (
                    mpi ** 4
                    + 2 * mpi ** 2 * LDot(k1, k3)
                    + 2 * mpi ** 2 * LDot(k1, k4)
                    - mpi ** 2 * LDot(k2, k3)
                    + 3 * mpi ** 2 * LDot(k3, k4)
                    + 2 * LDot(k2, k3) * LDot(k3, k4)
                )
                + LDot(k2, k4)
                * LDot(k3, k4)
                * (
                    -2 * LDot(k2, k3) ** 2
                    + mpi ** 2 * LDot(k3, k4)
                    + 2 * LDot(k2, k3) * (mpi ** 2 + 2 * LDot(k1, k3) + LDot(k3, k4))
                )
            )
            + LDot(k1, k4)
            * (
                mpi ** 2 * LDot(k2, k4) ** 3
                + mpi ** 2
                * LDot(k3, k4) ** 2
                * (-(mpi ** 2) + LDot(k1, k4) + LDot(k2, k3) + LDot(k3, k4))
                + LDot(k2, k4)
                * LDot(k3, k4)
                * (
                    -2 * LDot(k2, k3) ** 2
                    + mpi ** 2 * LDot(k3, k4)
                    + 2 * LDot(k2, k3) * (mpi ** 2 + LDot(k1, k4) + LDot(k3, k4))
                )
                + LDot(k2, k4) ** 2
                * (
                    -(mpi ** 4)
                    + mpi ** 2 * LDot(k1, k4)
                    + mpi ** 2 * LDot(k3, k4)
                    + LDot(k2, k3) * (mpi ** 2 + 2 * LDot(k3, k4))
                )
            )
            + LDot(k1, k3)
            * (
                mpi ** 2 * LDot(k2, k4) ** 3
                + mpi ** 2
                * LDot(k3, k4) ** 2
                * (-(mpi ** 2) - 2 * LDot(k1, k4) + LDot(k2, k3) + LDot(k3, k4))
                - LDot(k2, k4)
                * LDot(k3, k4)
                * (
                    2 * LDot(k2, k3) ** 2
                    - 2 * LDot(k2, k3) * (mpi ** 2 - LDot(k3, k4))
                    + 3 * mpi ** 2 * LDot(k3, k4)
                )
                + LDot(k2, k4) ** 2
                * (
                    -(mpi ** 4)
                    + 2 * mpi ** 2 * LDot(k1, k4)
                    + mpi ** 2 * LDot(k3, k4)
                    + LDot(k2, k3) * (mpi ** 2 + 2 * LDot(k3, k4))
                )
            )
        )
    ) / ((-1 + sw ** 2) * LDot(k2, k4) ** 2 * LDot(k3, k4) ** 2)


def msqrd_l_pi_pi0(self, momenta):
    """
    Compute the squared matrix-element for a RH neutrino decaying into a
    charged lepton, neutral pion and a charged pion at leading order in the
    Fermi constant. Momenta are ordered as follows: {l+,pi-,pi0}

    Parameters
    ----------
    momenta: List
        List of NumPy arrays storing the four-momenta of the final state
        particles.

    Returns
    -------
    msqrd: float
        The matrix element for N -> l + pi0 + pi for the given model and
        four-momenta.
    """
    pl = momenta[0]
    ppi = momenta[1]
    ppi0 = momenta[2]
    P = pl + ppi + ppi0
    s = LDot(P - pl, P - pl)
    t = LDot(P - ppi, P - ppi)

    ml = self.ml
    mx = self.mx
    smix = self.stheta

    return (
        2
        * GF ** 2
        * smix ** 2
        * (
            ml ** 4
            + mx ** 4
            + ml ** 2 * (2 * mx ** 2 - s - 4 * t)
            + 4 * mpi ** 2 * (mpi0 ** 2 - t)
            + 4 * t * (-(mpi0 ** 2) + s + t)
            - mx ** 2 * (s + 4 * t)
        )
        * Vud ** 2
    )


def msqrd_l_pi_pi0_g(self, momenta):
    """
    Compute the squared matrix-element for a RH neutrino decaying into a
    charged lepton, neutral pion, a charged pion and a photon at leading order
    in the Fermi constant. Momenta are ordered as follows: {l-,pi+,pi0,photon}.

    Parameters
    ----------
    momenta: List
        List of NumPy arrays storing the four-momenta of the final state
        particles.

    Returns
    -------
    msqrd: float
        The matrix element for N -> l + pi + pi0 + gamma for the given model
        and four-momenta.
    """
    k1 = momenta[0]
    k2 = momenta[1]
    k3 = momenta[2]
    k4 = momenta[3]

    smix = self.stheta
    ml = self.ml

    return (
        -8
        * GF ** 2
        * qe ** 2
        * smix ** 2
        * Vud ** 2
        * (
            2 * mpi ** 2 * LDot(k1, k4) ** 4
            + LDot(k1, k4) ** 3
            * (
                mpi ** 4
                - 3 * mpi ** 2 * mpi0 ** 2
                - 4 * mpi ** 2 * LDot(k1, k3)
                + 2 * mpi ** 2 * LDot(k2, k3)
                + 4 * LDot(k1, k2) * (mpi ** 2 - LDot(k2, k4))
                + 4 * mpi ** 2 * LDot(k2, k4)
                - 6 * LDot(k2, k3) * LDot(k2, k4)
                - 2 * LDot(k2, k4) ** 2
                + 2 * mpi ** 2 * LDot(k3, k4)
            )
            + ml ** 2
            * LDot(k2, k4) ** 2
            * (
                -(ml ** 2 * mpi ** 2)
                - ml ** 2 * mpi0 ** 2
                + 2 * LDot(k1, k2) ** 2
                + 2 * LDot(k1, k3) ** 2
                + 2 * ml ** 2 * LDot(k2, k3)
                + mpi ** 2 * LDot(k2, k4)
                - 3 * mpi0 ** 2 * LDot(k2, k4)
                + 2 * LDot(k2, k3) * LDot(k2, k4)
                + 2 * LDot(k2, k4) ** 2
                + LDot(k1, k2)
                * (
                    mpi ** 2
                    - 3 * mpi0 ** 2
                    - 4 * LDot(k1, k3)
                    + 2 * LDot(k2, k3)
                    + 4 * LDot(k2, k4)
                    - 4 * LDot(k3, k4)
                )
                - 3 * mpi ** 2 * LDot(k3, k4)
                + mpi0 ** 2 * LDot(k3, k4)
                + 2 * LDot(k2, k3) * LDot(k3, k4)
                - 4 * LDot(k2, k4) * LDot(k3, k4)
                + 2 * LDot(k3, k4) ** 2
                + LDot(k1, k3)
                * (
                    -3 * mpi ** 2
                    + mpi0 ** 2
                    + 2 * LDot(k2, k3)
                    - 4 * LDot(k2, k4)
                    + 4 * LDot(k3, k4)
                )
            )
            + LDot(k1, k4) ** 2
            * (
                -(ml ** 2 * mpi ** 4)
                - ml ** 2 * mpi ** 2 * mpi0 ** 2
                + 2 * mpi ** 2 * LDot(k1, k3) ** 2
                + 2 * ml ** 2 * mpi ** 2 * LDot(k2, k3)
                + 2 * LDot(k1, k2) ** 2 * (mpi ** 2 - 4 * LDot(k2, k4))
                + mpi ** 4 * LDot(k2, k4)
                - 3 * mpi ** 2 * mpi0 ** 2 * LDot(k2, k4)
                - 2 * ml ** 2 * LDot(k2, k3) * LDot(k2, k4)
                - mpi ** 2 * LDot(k2, k3) * LDot(k2, k4)
                + mpi0 ** 2 * LDot(k2, k3) * LDot(k2, k4)
                + 2 * LDot(k2, k3) ** 2 * LDot(k2, k4)
                + mpi ** 2 * LDot(k2, k4) ** 2
                + 3 * mpi0 ** 2 * LDot(k2, k4) ** 2
                - 8 * LDot(k2, k3) * LDot(k2, k4) ** 2
                - 4 * LDot(k2, k4) ** 3
                + 2 * ml ** 2 * mpi ** 2 * LDot(k3, k4)
                + 2 * mpi ** 2 * LDot(k2, k4) * LDot(k3, k4)
                + 2 * LDot(k2, k3) * LDot(k2, k4) * LDot(k3, k4)
                + 4 * LDot(k2, k4) ** 2 * LDot(k3, k4)
                + LDot(k1, k2)
                * (
                    mpi ** 4
                    - 3 * mpi ** 2 * mpi0 ** 2
                    + 2 * LDot(k2, k3) * (mpi ** 2 - 5 * LDot(k2, k4))
                    - 4 * LDot(k1, k3) * (mpi ** 2 - 2 * LDot(k2, k4))
                    + 2 * mpi ** 2 * LDot(k2, k4)
                    + 6 * mpi0 ** 2 * LDot(k2, k4)
                    - 10 * LDot(k2, k4) ** 2
                    + 2 * mpi ** 2 * LDot(k3, k4)
                    + 2 * LDot(k2, k4) * LDot(k3, k4)
                )
                + LDot(k1, k3)
                * (
                    -4 * mpi ** 2 * LDot(k2, k4)
                    + 4 * LDot(k2, k4) ** 2
                    + 2 * LDot(k2, k3) * (mpi ** 2 + LDot(k2, k4))
                    + mpi ** 2 * (-3 * mpi ** 2 + mpi0 ** 2 + 2 * LDot(k3, k4))
                )
            )
            - LDot(k1, k4)
            * LDot(k2, k4)
            * (
                4 * LDot(k1, k2) ** 3
                + 2
                * LDot(k1, k2) ** 2
                * (
                    mpi ** 2
                    - 3 * mpi0 ** 2
                    - 4 * LDot(k1, k3)
                    + 2 * LDot(k2, k3)
                    + 4 * LDot(k2, k4)
                    - LDot(k3, k4)
                )
                + LDot(k1, k2)
                * (
                    -2 * ml ** 2 * mpi ** 2
                    - 2 * ml ** 2 * mpi0 ** 2
                    + 4 * LDot(k1, k3) ** 2
                    - 4 * ml ** 2 * LDot(k2, k4)
                    + 2 * mpi ** 2 * LDot(k2, k4)
                    - 6 * mpi0 ** 2 * LDot(k2, k4)
                    + 6 * LDot(k2, k4) ** 2
                    + 2 * ml ** 2 * LDot(k3, k4)
                    - 3 * mpi ** 2 * LDot(k3, k4)
                    + mpi0 ** 2 * LDot(k3, k4)
                    - 8 * LDot(k2, k4) * LDot(k3, k4)
                    + 2 * LDot(k3, k4) ** 2
                    + 2 * LDot(k2, k3) * (2 * ml ** 2 + 2 * LDot(k2, k4) + LDot(k3, k4))
                    + 2
                    * LDot(k1, k3)
                    * (
                        -3 * mpi ** 2
                        + mpi0 ** 2
                        + 2 * LDot(k2, k3)
                        - 7 * LDot(k2, k4)
                        + 3 * LDot(k3, k4)
                    )
                )
                + LDot(k2, k4)
                * (
                    -(ml ** 2 * mpi ** 2)
                    + 3 * ml ** 2 * mpi0 ** 2
                    + 2 * LDot(k1, k3) ** 2
                    - 2 * ml ** 2 * LDot(k2, k4)
                    + mpi ** 2 * LDot(k2, k4)
                    - 3 * mpi0 ** 2 * LDot(k2, k4)
                    + 2 * LDot(k2, k4) ** 2
                    + 2 * ml ** 2 * LDot(k3, k4)
                    - 3 * mpi ** 2 * LDot(k3, k4)
                    + mpi0 ** 2 * LDot(k3, k4)
                    - 4 * LDot(k2, k4) * LDot(k3, k4)
                    + 2 * LDot(k3, k4) ** 2
                    + 2 * LDot(k2, k3) * (-(ml ** 2) + LDot(k2, k4) + LDot(k3, k4))
                    + LDot(k1, k3)
                    * (
                        2 * ml ** 2
                        - 3 * mpi ** 2
                        + mpi0 ** 2
                        + 2 * LDot(k2, k3)
                        - 4 * LDot(k2, k4)
                        + 4 * LDot(k3, k4)
                    )
                )
            )
        )
    ) / (LDot(k1, k4) ** 2 * LDot(k2, k4) ** 2)


def msqrd_nu_l_l(self, momenta):
    """
    Compute the squared matrix-element for a RH neutrino decaying into a
    neutrino and two charged leptons at leading order in the Fermi constant.
    Momenta are ordered as follows: {nu,l+,l-}.

    Parameters
    ----------
    momenta: List
        List of NumPy arrays storing the four-momenta of the final state
        particles.

    Returns
    -------
    msqrd: float
        The matrix element for N -> nu + l + l for the given model and
        four-momenta.
    """
    pnu = momenta[0]
    plp = momenta[1]
    plm = momenta[2]
    P = pnu + plp + plm
    s = LDot(P - pnu, P - pnu)
    t = LDot(P - plp, P - plp)

    smix = self.stheta
    ml = self.ml
    mx = self.mx
    return (
        8
        * GF ** 2
        * smix ** 2
        * (-1 + smix ** 2)
        * (
            2 * ml ** 4 * (1 + 4 * sw ** 2 + 8 * sw ** 4)
            + 2 * ml ** 2 * (mx ** 2 - s - 2 * (1 + 4 * sw ** 2 + 8 * sw ** 4) * t)
            + (1 + 4 * sw ** 2 + 8 * sw ** 4)
            * (s ** 2 + 2 * s * t + 2 * t ** 2 - mx ** 2 * (s + 2 * t))
        )
    )


def msqrd_nu_l_l_g(self, momenta):
    """
    Compute the squared matrix-element for a RH neutrino decaying into a
    neutrino, two charged leptons and a photon at leading order in the
    Fermi constant. Momenta are ordered as follows: {nu, l+, l-, photon}.

    Parameters
    ----------
    momenta: List
        List of NumPy arrays storing the four-momenta of the final state
        particles.

    Returns
    -------
    msqrd: float
        The matrix element for N -> nu + l + l + gamma for the given model and
        four-momenta.
    """
    k1 = momenta[0]
    k2 = momenta[1]
    k3 = momenta[2]
    k4 = momenta[3]

    smix = self.stheta
    ml = self.ml

    return (
        16
        * GF ** 2
        * qe ** 2
        * smix ** 2
        * (-1 + smix ** 2)
        * (
            2
            * (1 + 4 * sw ** 2 + 8 * sw ** 4)
            * LDot(k1, k3) ** 2
            * LDot(k2, k4) ** 2
            * LDot(k3, k4)
            + 2
            * (1 + 4 * sw ** 2 + 8 * sw ** 4)
            * LDot(k1, k2) ** 2
            * LDot(k2, k4)
            * LDot(k3, k4) ** 2
            + LDot(k1, k4)
            * (
                (1 + 4 * sw ** 2 + 8 * sw ** 4)
                * LDot(k2, k4) ** 3
                * (ml ** 2 - LDot(k3, k4))
                + ml ** 2
                * LDot(k3, k4) ** 2
                * (
                    (ml + 4 * ml * sw ** 2) ** 2
                    + (1 + 4 * sw ** 2 + 8 * sw ** 4) * LDot(k2, k3)
                    + (1 + 4 * sw ** 2 + 8 * sw ** 4) * LDot(k3, k4)
                )
                + LDot(k2, k4) ** 2
                * (
                    (1 + 4 * sw ** 2 + 8 * sw ** 4)
                    * LDot(k2, k3)
                    * (ml ** 2 - 2 * LDot(k3, k4))
                    + ml ** 2
                    * (
                        (ml + 4 * ml * sw ** 2) ** 2
                        + (-1 + 4 * sw ** 2 + 8 * sw ** 4) * LDot(k3, k4)
                    )
                )
                - LDot(k2, k4)
                * LDot(k3, k4)
                * (
                    -8 * ml ** 2 * sw ** 2 * (1 + 2 * sw ** 2) * LDot(k1, k4)
                    + 2 * (1 + 4 * sw ** 2 + 8 * sw ** 4) * LDot(k2, k3) ** 2
                    + 2
                    * LDot(k2, k3)
                    * (
                        (ml + 4 * ml * sw ** 2) ** 2
                        + (1 + 4 * sw ** 2 + 8 * sw ** 4) * LDot(k3, k4)
                    )
                    + LDot(k3, k4)
                    * (
                        ml ** 2 * (1 - 4 * sw ** 2 - 8 * sw ** 4)
                        + (1 + 4 * sw ** 2 + 8 * sw ** 4) * LDot(k3, k4)
                    )
                )
            )
            + LDot(k1, k3)
            * (
                (1 + 4 * sw ** 2 + 8 * sw ** 4)
                * LDot(k2, k4) ** 3
                * (ml ** 2 - LDot(k3, k4))
                + ml ** 2
                * LDot(k3, k4) ** 2
                * (
                    (ml + 4 * ml * sw ** 2) ** 2
                    + 2 * (1 + 4 * sw ** 2 + 8 * sw ** 4) * LDot(k1, k4)
                    + LDot(k2, k3)
                    + LDot(k3, k4)
                    + 4 * sw ** 2 * (1 + 2 * sw ** 2) * (LDot(k2, k3) + LDot(k3, k4))
                )
                + LDot(k2, k4) ** 2
                * (
                    ml ** 4 * (1 + 4 * sw ** 2) ** 2
                    + (1 + 4 * sw ** 2 + 8 * sw ** 4)
                    * (
                        LDot(k2, k3) * (ml ** 2 - 2 * LDot(k3, k4))
                        + ml ** 2 * LDot(k3, k4)
                    )
                )
                - LDot(k2, k4)
                * LDot(k3, k4)
                * (
                    2 * (1 + 4 * sw ** 2 + 8 * sw ** 4) * LDot(k2, k3) ** 2
                    + (1 + 4 * sw ** 2 + 8 * sw ** 4)
                    * LDot(k3, k4)
                    * (-(ml ** 2) + 2 * LDot(k1, k4) + LDot(k3, k4))
                    + 2
                    * LDot(k2, k3)
                    * (
                        (ml + 4 * ml * sw ** 2) ** 2
                        + (1 + 4 * sw ** 2 + 8 * sw ** 4) * LDot(k1, k4)
                        + (1 + 4 * sw ** 2 + 8 * sw ** 4) * LDot(k3, k4)
                    )
                )
            )
            + LDot(k1, k2)
            * (
                (1 + 4 * sw ** 2 + 8 * sw ** 4)
                * LDot(k2, k4) ** 3
                * (ml ** 2 - LDot(k3, k4))
                + LDot(k2, k4)
                * LDot(k3, k4)
                * (
                    -2
                    * LDot(k2, k3)
                    * (
                        (ml + 4 * ml * sw ** 2) ** 2
                        + 2 * (1 + 4 * sw ** 2 + 8 * sw ** 4) * LDot(k1, k3)
                        + LDot(k1, k4)
                        + LDot(k2, k3)
                        + 4
                        * sw ** 2
                        * (1 + 2 * sw ** 2)
                        * (LDot(k1, k4) + LDot(k2, k3))
                    )
                    + (1 + 4 * sw ** 2 + 8 * sw ** 4)
                    * (ml ** 2 - 2 * LDot(k1, k3) - 2 * LDot(k2, k3))
                    * LDot(k3, k4)
                    - (1 + 4 * sw ** 2 + 8 * sw ** 4) * LDot(k3, k4) ** 2
                )
                + ml ** 2
                * LDot(k3, k4) ** 2
                * (
                    (ml + 4 * ml * sw ** 2) ** 2
                    + 2 * (1 + 4 * sw ** 2 + 8 * sw ** 4) * LDot(k1, k3)
                    + LDot(k2, k3)
                    + LDot(k3, k4)
                    + 4 * sw ** 2 * (1 + 2 * sw ** 2) * (LDot(k2, k3) + LDot(k3, k4))
                )
                + LDot(k2, k4) ** 2
                * (
                    ml ** 4 * (1 + 4 * sw ** 2) ** 2
                    + (1 + 4 * sw ** 2 + 8 * sw ** 4)
                    * (
                        ml ** 2 * (2 * (LDot(k1, k3) + LDot(k1, k4)) + LDot(k2, k3))
                        + (
                            ml ** 2
                            - 2 * LDot(k1, k3)
                            - 2 * LDot(k1, k4)
                            - 2 * LDot(k2, k3)
                        )
                        * LDot(k3, k4)
                    )
                )
            )
        )
    ) / (LDot(k2, k4) ** 2 * LDot(k3, k4) ** 2)

