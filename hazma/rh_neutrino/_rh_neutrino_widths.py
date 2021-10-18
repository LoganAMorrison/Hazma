"""
This file contains the mixin class which implements the partial widths of the
right-handed neutrino.
"""
from scipy.integrate import quad
import numpy as np

from hazma.parameters import (
    qe,
    GF,
    fpi,
    Vud,
    Vus,
    neutral_pion_mass as mpi0,
    charged_pion_mass as mpi,
    charged_kaon_mass as mk,
    sin_theta_weak as sw,
    cos_theta_weak as cw,
    electron_mass as me,
    muon_mass as mmu,
)


# =======================
# ---- 2-Body Widths ----
# =======================


def width_pi0_nu(self):
    """
    Computes the width of the right-handed neutrino into a neutral pion and
    active neutrino.

    Returns
    -------
    width: float
        Partial width for N -> pi^0 + nu.
    """
    mx = self.mx
    smix = self.stheta
    if mx < mpi0:
        return 0.0
    return (
        fpi ** 2 * GF ** 2 * (mx ** 2 - mpi0 ** 2) ** 2 * smix ** 2 * (-1 + smix ** 2)
    ) / (8.0 * mx * np.pi * (-1 + sw ** 2))


def width_pi_l(self):
    """
    Computes the width of the right-handed neutrino into a charged pion and
    a lepton. This only includes the contribution from one charge
    configuration (i.e. N -> pi^+ + e^-).

    Returns
    -------
    width: float
        Partial width for N -> pi + l.
    """
    mx = self.mx
    smix = self.stheta
    ml = self.ml

    if mx < mpi + ml:
        return 0.0
    return (
        fpi ** 2
        * GF ** 2
        * np.sqrt((ml - mx - mpi) * (ml + mx - mpi) * (ml - mx + mpi) * (ml + mx + mpi))
        * ((ml ** 2 - mx ** 2) ** 2 - (ml ** 2 + mx ** 2) * mpi ** 2)
        * smix ** 2
        * Vud ** 2
    ) / (8.0 * mx ** 3 * np.pi)


def width_k_l(self):
    """
    Computes the width of the right-handed neutrino into a charged kaon and
    a lepton. This only includes the contribution from one charge
    configuration (i.e. N -> K^+ + e^-).

    Returns
    -------
    width: float
        Partial width for N -> K + l.
    """
    mx = self.mx
    smix = self.stheta
    ml = self.ml

    if mx < mk + ml:
        return 0.0

    return (
        fpi ** 2
        * GF ** 2
        * np.sqrt((mk - ml - mx) * (mk + ml - mx) * (mk - ml + mx) * (mk + ml + mx))
        * ((ml ** 2 - mx ** 2) ** 2 - mk ** 2 * (ml ** 2 + mx ** 2))
        * smix ** 2
        * Vus ** 2
    ) / (8.0 * mx ** 3 * np.pi)


def width_nu_gamma(self):
    """
    Compute the width for a right-handed neutrino to decay into an active
    neutrino and a photon at one-loop.

    Returns
    -------
    width: float
        Partial decay with for N -> nu + pi0 + pi0.

    """
    mx = self.mx
    smix = self.stheta
    return -(GF ** 2 * mx ** 5 * qe ** 2 * smix ** 2 * (6 - 5 * smix ** 2) ** 2) / (
        4096.0 * np.pi ** 9 * (-1 + smix ** 2)
    )


# =======================
# ---- 3-Body Widths ----
# =======================


def width_nu_pi0_pi0(self):
    """
    Compute the width for a right-handed neutrino to decay into an active
    neutrino and two neutral pion. (NOTE: The amplitude for this process is
    order G_F^4. For consistancy of the EFT, we must ignore this process.)

    Returns
    -------
    width: float
        Partial decay with for N -> nu + pi0 + pi0.

    """
    # Higher order in G_F
    return 0.0


def width_nu_pi_pi(self):
    """
    Compute the width of the right-handed neutrino into an active-neutrino
    and two neutral pions.

    Returns
    -------
    width: float
        Partial decay with for N -> nu + pi^+ + pi^-.
    """
    mx = self.mx
    smix = self.stheta
    if mx < 2.0 * mpi:
        return 0.0
    return (
        GF ** 2
        * (-1 + smix ** 2)
        * (smix - 2 * smix * sw ** 2) ** 2
        * (
            mx ** 2
            * np.sqrt(1 - (4 * mpi ** 2) / mx ** 2)
            * (
                mx ** 6
                + 24 * mx ** 4 * mpi ** 2
                - 10 * mx ** 2 * mpi ** 4
                + 12 * mpi ** 6
            )
            - 24
            * mpi ** 2
            * (mx ** 6 + 2 * mx ** 2 * mpi ** 4 - 2 * mpi ** 6)
            * np.arctanh(np.sqrt(1 - (4 * mpi ** 2) / mx ** 2))
        )
    ) / (768.0 * mx ** 3 * np.pi ** 3 * (-1 + sw ** 2))


def width_l_pi_pi0(self):
    """
    Compute the width for right-handed neutrino to a lepton, charged pion and
    neutral pion. This only includes a single charge configuration
    (i.e. N -> e^- + pi^+ + pi0).

    Returns
    -------
    width: float
        Partial width for N -> l + pi + pi^0.
    """
    # TODO: Get Mathematica to compute width analytically.

    mx = self.mx
    ml = self.ml
    smix = self.stheta

    if mx < ml + mpi + mpi0:
        return 0.0

    def integrand(s):
        """Returns the integrand s integral"""
        return (
            -2
            * GF ** 2
            * np.sqrt(
                (ml ** 4 + (mx ** 2 - s) ** 2 - 2 * ml ** 2 * (mx ** 2 + s))
                * (mpi ** 4 + (mpi0 ** 2 - s) ** 2 - 2 * mpi ** 2 * (mpi0 ** 2 + s))
            )
            * (
                ml ** 4
                * (
                    -4 * (mpi ** 2 - mpi0 ** 2) ** 2
                    + 2 * (mpi ** 2 + mpi0 ** 2) * s
                    - s ** 2
                )
                + ml ** 2
                * (
                    s
                    * (
                        2 * (mpi ** 2 - mpi0 ** 2) ** 2
                        + 2 * (mpi ** 2 + mpi0 ** 2) * s
                        - s ** 2
                    )
                    + 2
                    * mx ** 2
                    * (
                        4 * (mpi ** 2 - mpi0 ** 2) ** 2
                        - 2 * (mpi ** 2 + mpi0 ** 2) * s
                        + s ** 2
                    )
                )
                - (mx ** 2 - s)
                * (
                    mx ** 2
                    * (
                        4 * (mpi ** 2 - mpi0 ** 2) ** 2
                        - 2 * (mpi ** 2 + mpi0 ** 2) * s
                        + s ** 2
                    )
                    + 2 * s * (mpi ** 4 + (mpi0 ** 2 - s) ** 2)
                )
            )
            * smix ** 2
            * Vud ** 2
        ) / (3.0 * s ** 3)

    lb, ub = (mpi + mpi0) ** 2, (ml - mx) ** 2
    pre = 1 / (256.0 * mx ** 3 * np.pi ** 3)

    return pre * quad(integrand, lb, ub)[0]


def width_nu_nu_nu(self, j, n, m):
    """
    Compute the width for right-handed neutrino to three active neutrinos.

    Parameters
    ----------
    j: int
        Generation of first active neutrino.
    n: int
        Generation of second active neutrino.
    m: int
        Generation of third active neutrino.

    Returns
    -------
    width: float
        Partial width for N -> 3nu.
    """
    mx = self.mx
    tmix = self.theta
    i = self._gen

    w = (GF ** 2 * mx ** 5 * np.sin(2 * tmix) ** 2) / (768.0 * cw ** 4 * np.pi ** 3)

    if i == j == n == m:
        return w * np.cos(tmix) ** 4
    if (j == i and n == m) or (j == n and i == m) or (j == m and i == n):
        return w / 2.0
    return 0.0


def __width_nu_l_l(self):
    """nuR_i -> nuL_j + l_n + l_m, with i=j=n=m."""
    mx = self.mx
    ml = self.ml
    r = self.ml / self.mx

    if mx < 2.0 * ml:
        return 0.0

    tmix = self.theta

    return (
        GF ** 2
        * mx ** 5
        * (
            -(
                np.sqrt(1 - 4 * r ** 2)
                * (
                    -1
                    - 4 * sw ** 4
                    + 12 * r ** 6 * (1 + 8 * sw ** 2 + 4 * sw ** 4)
                    + r ** 4 * (2 - 80 * sw ** 2 + 8 * sw ** 4)
                    + 2 * r ** 2 * (7 - 8 * sw ** 2 + 28 * sw ** 4)
                )
            )
            + 24
            * (
                -8 * r ** 6 * sw ** 2
                - r ** 4 * (1 - 2 * sw ** 2) ** 2
                + r ** 8 * (1 + 8 * sw ** 2 + 4 * sw ** 4)
            )
            * np.log((4 * r ** 2) / (1 + np.sqrt(1 - 4 * r ** 2)) ** 2)
        )
        * np.sin(2 * tmix) ** 2
    ) / (1536.0 * np.pi ** 3 * (-1 + sw ** 2) ** 2)


def __width_nu_lp_lp(self):
    """nuR_i -> nuL_i + l_j + l_j, with i!=j"""
    mx = self.mx
    ml = self.ml
    r = self.ml / self.mx

    if mx < 2.0 * ml:
        return 0.0

    tmix = self.theta

    return (
        GF ** 2
        * mx ** 5
        * (
            np.sqrt(1 - 4 * r ** 2)
            * (
                1
                - 4 * sw ** 2
                + 8 * sw ** 4
                - 12 * r ** 6 * (1 - 12 * sw ** 2 + 24 * sw ** 4)
                - 2 * r ** 2 * (7 - 20 * sw ** 2 + 40 * sw ** 4)
                + 2 * r ** 4 * (-1 - 36 * sw ** 2 + 72 * sw ** 4)
            )
            + 24
            * r ** 4
            * (
                -1
                + 8 * r ** 2 * sw ** 2 * (1 - 2 * sw ** 2)
                + r ** 4 * (1 - 12 * sw ** 2 + 24 * sw ** 4)
            )
            * np.log((4 * r ** 2) / (1 + np.sqrt(1 - 4 * r ** 2)) ** 2)
        )
        * np.sin(2 * tmix) ** 2
    ) / (1536.0 * np.pi ** 3 * (-1 + sw ** 2) ** 2)


def __width_nup_l_lp(self, k):
    """nuR_i -> nuL_j + l_j + l_i, with i!=j"""
    mx = self.mx
    ml = self.ml
    ri = self.ml / self.mx
    if k == 1:
        rk = me / self.mx
    elif k == 2:
        rk = mmu / self.mx
    else:
        return 0.0

    if mx < 2.0 * ml:
        return 0.0

    tmix = self.theta

    return (
        GF ** 2
        * mx ** 5
        * (
            np.sqrt(ri ** 4 + (-1 + rk ** 2) ** 2 - 2 * ri ** 2 * (1 + rk ** 2))
            * (
                1
                + ri ** 6
                - 7 * rk ** 2
                - 7 * rk ** 4
                + rk ** 6
                - 7 * ri ** 4 * (1 + rk ** 2)
                + ri ** 2 * (-7 + 12 * rk ** 2 - 7 * rk ** 4)
            )
            + 12
            * ri ** 4
            * (-1 + rk ** 4)
            * np.log(
                (4 * ri ** 2 * rk ** 2)
                / (
                    -1
                    + ri ** 2
                    + rk ** 2
                    - np.sqrt(
                        ri ** 4 + (-1 + rk ** 2) ** 2 - 2 * ri ** 2 * (1 + rk ** 2)
                    )
                )
                ** 2
            )
            + 6
            * (ri ** 4 - rk ** 4)
            * np.log(
                (
                    ri ** 4
                    + rk ** 2 * (-1 + rk ** 2)
                    - ri ** 2 * (1 + 2 * rk ** 2)
                    + (ri ** 2 - rk ** 2)
                    * np.sqrt(
                        ri ** 4 + (-1 + rk ** 2) ** 2 - 2 * ri ** 2 * (1 + rk ** 2)
                    )
                )
                ** 2
                / (
                    -1
                    + ri ** 2
                    + rk ** 2
                    - np.sqrt(
                        ri ** 4 + (-1 + rk ** 2) ** 2 - 2 * ri ** 2 * (1 + rk ** 2)
                    )
                )
                ** 2
            )
        )
        * np.sin(tmix) ** 2
    ) / (192.0 * np.pi ** 3)


def width_nu_l_l(self, j: int, n: int, m: int):
    """
    Compute the width for right-handed neutrino to an active neutrino and
    two charged leptons.

    Parameters
    ----------
    j: int
        Generation of active neutrino.
    n: int
        Generation of charged-lepton.
    m: int
        Generation of anti charged-lepton.

    Returns
    -------
    width: float
        Partial width for N -> nu_j + l_n + l_m.
    """
    i = self._gen

    if j == i and n == i and m == i:
        return __width_nu_l_l(self)
    elif j == i and n == m:
        return __width_nu_lp_lp(self)
    elif j == n and i == m:
        return __width_nup_l_lp(self, j)
    elif j == m and i == n:
        return __width_nup_l_lp(self, j)
    else:
        return 0.0


def width_nu_g_g(self):
    """
    Compute the width for a right-handed neutrino to decay into an active
    neutrino and an off-shell pion which decays into two photons.
    """
    mx = self.mx

    # Return zero if we can go into an on-shell pion
    if mx > mpi0:
        return 0.0

    smix = self.stheta

    return 0.5 * (GF ** 2 * mx ** 11 * qe ** 4 * smix ** 2 * (-1 + smix ** 2)) / (
        245760.0 * mpi0 ** 6 * np.pi ** 7 * (-1 + sw ** 2)
    ) + (GF ** 2 * mx ** 9 * qe ** 4 * smix ** 2 * (-1 + smix ** 2)) / (
        245760.0 * mpi0 ** 4 * np.pi ** 7 * (-1 + sw ** 2)
    )
