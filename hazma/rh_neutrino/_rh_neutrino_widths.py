"""
This file contains the mixin class which implements the partial widths of the
right-handed neutrino.
"""
from scipy.integrate import quad
import numpy as np
from typing import Optional, Tuple

from hazma.parameters import (
    qe,
    GF,
    fpi,
    Vud,
    Vus,
    neutral_pion_mass as mpi0,
    eta_mass as meta,
    charged_pion_mass as mpi,
    charged_kaon_mass as mk,
    sin_theta_weak as sw,
    cos_theta_weak as cw,
    lepton_masses,
    alpha_em,
)


def kallen_lambda(a, b, c):
    """
    Compute the Kallen-Lambda (triangle) function.
    """
    return a ** 2 + b ** 2 + c ** 2 - 2 * a * b - 2 * a * c - 2 * b * c


# =======================
# ---- 2-Body Widths ----
# =======================


def _width_nu_neu_meson(self, m0: float) -> float:
    """
    Compute the partial width for the decay of a right-handed neutrino
    into an active neutrino and neutral meson.
    """
    if self.mx < m0:
        return 0.0
    return (
        pow(fpi, 2)
        * pow(GF, 2)
        * np.sqrt(pow(pow(m0, 2) - pow(self.mx, 2), 2))
        * (-pow(m0, 2) + pow(self.mx, 2))
        * pow(np.sin(2 * self.theta), 2)
    ) / (32.0 * pow(cw, 4) * cw * self.mx)


def width_nu_pi0(self):
    """
    Computes the width of the right-handed neutrino into an active neutrino
    and a neutral pion.

    Returns
    -------
    width: float
        Partial width for N -> nu + pi^0.
    """
    return _width_nu_neu_meson(self, mpi0)


def width_nu_eta(self):
    """
    Computes the width of the right-handed neutrino into an active neutrino
    and an eta.

    Returns
    -------
    width: float
        Partial width for N -> nu + eta.
    """
    return _width_nu_neu_meson(self, meta) / 3.0


def _width_l_chg_meson(self, mchg: float, ckm2: float) -> float:
    """
    Compute the partial width for the decay of a right-handed neutrino
    into a charged meson and charged lepton.
    """
    if self.mx < mchg + self.ml:
        return 0.0
    return (
        pow(fpi, 2)
        * pow(GF, 2)
        * ckm2
        * np.sqrt(kallen_lambda(pow(mchg, 2), pow(self.ml, 2), pow(self.mx, 2)))
        * (
            pow(self.ml, 4)
            - pow(mchg, 2) * pow(self.mx, 2)
            + pow(self.mx, 4)
            - pow(self.ml, 2) * (pow(mchg, 2) + 2 * pow(self.mx, 2))
        )
        * pow(np.sin(self.theta), 2)
    ) / (8.0 * np.pi * pow(self.mx, 3))


def width_l_pi(self):
    """
    Computes the width of the right-handed neutrino into a charged pion and
    a lepton. This only includes the contribution from one charge
    configuration (i.e. N -> pi^+ + e^-).

    Returns
    -------
    width: float
        Partial width for N -> pi + l.
    """
    return _width_l_chg_meson(self, mpi, Vud ** 2)


def width_l_k(self):
    """
    Computes the width of the right-handed neutrino into a charged kaon and
    a lepton. This only includes the contribution from one charge
    configuration (i.e. N -> K^+ + e^-).

    Returns
    -------
    width: float
        Partial width for N -> K + l.
    """
    return _width_l_chg_meson(self, mk, Vus ** 2)


def width_nu_gamma(self):
    """
    Compute the width for a right-handed neutrino to decay into an active
    neutrino and a photon at one-loop (via a W and lepton in the loop.)

    Returns
    -------
    width: float
        Partial decay width for N -> nu + gamma.

    """
    return (
        alpha_em
        * self.mx ** 5
        * ((GF * np.sin(2 * self.theta) / (8 * np.pi ** 2)) ** 2)
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
    temp1 = self.mx ** 2
    temp2 = mpi ** 2
    temp3 = temp2 ** 3
    temp4 = temp2 ** 2
    temp5 = temp1 ** 3
    temp6 = -4 * temp2
    temp7 = temp1 + temp6
    temp8 = np.sqrt(temp7)

    return (
        pow(GF, 2)
        * pow(1 - 2 * pow(sw, 2), 2)
        * (
            self.mx
            * (24 * pow(self.mx, 4) * temp2 + 12 * temp3 - 10 * temp1 * temp4 + temp5)
            * temp8
            + 12
            * temp2
            * (-2 * temp3 + 2 * temp1 * temp4 + temp5)
            * np.log((4 * temp2) / pow(self.mx + temp8, 2))
        )
        * pow(np.sin(2 * self.theta), 2)
    ) / (3072.0 * pow(cw, 4) * pow(self.mx, 3) * pow(np.pi, 3))


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
    smix = self.theta

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


def width_nu_nu_nu(self):
    """
    Compute the width for right-handed neutrino to three active neutrinos. This
    result is summed over all three active-neutrino species.

    Returns
    -------
    width: float
        Partial width for N -> 3nu.
    """
    # NOTE: This is summed over all three neutrino species.
    return (
        GF ** 2
        * self.mx ** 5
        * np.sin(2 * self.theta) ** 2
        * (3 + 2 * np.cos(self.theta) ** 4)
        / (768 * np.pi ** 3 * cw ** 4)
    )


def width_nu_l_l(self, gens: Optional[Tuple[int, int, int]] = None) -> float:
    """
    Compute the width for right-handed neutrino to an active neutrino and
    two charged leptons.

    Parameters
    ----------
    gens: Tuple[int,int,int]
        Generations of the active-neutrino and chaged leptons. If None,
        the width is summed over all generations.

    Returns
    -------
    width: float
        Partial width for N -> nu + l + l.
    """
    mx = self.mx
    i = self._gen

    if gens is None:
        res = 0.0
        # NOTE: Only non-zero if pairs of generations match.
        for i in (1, 2, 3):
            res += self.width_nu_l_l((self._gen, i, i))
            res += 2.0 * self.width_nu_l_l((i, self._gen, i))
        return res

    genv = gens[0]
    genl1 = gens[1]
    genl2 = gens[2]

    mlk = lepton_masses[genl1 - 1]
    mll = lepton_masses[genl2 - 1]

    if mx < mlk + mll:
        return 0.0

    pre = (np.sin(self.theta) ** 4 * GF ** 2 * mx ** 5) / (768 * cw ** 4 * np.pi ** 3)

    width = 0.0
    if genv == i and genl1 == genl2:
        width += np.cos(self.theta) ** 4 * pre

    if genl1 == i and genv == genl2:
        width += pre / 2.0

    if genl2 == i and genv == genl1:
        width += pre / 2.0

    if genv == i and genl2 == i and genl1 == i:
        width += pre / 2.0

    return width


def width_nu_g_g(self):
    """
    Compute the width for a right-handed neutrino to decay into an active
    neutrino and an off-shell pion which decays into two photons.
    """
    mx = self.mx

    # Return zero if we can go into an on-shell pion
    if mx > mpi0:
        return 0.0

    smix = self.theta

    return 0.5 * (GF ** 2 * mx ** 11 * qe ** 4 * smix ** 2 * (-1 + smix ** 2)) / (
        245760.0 * mpi0 ** 6 * np.pi ** 7 * (-1 + sw ** 2)
    ) + (GF ** 2 * mx ** 9 * qe ** 4 * smix ** 2 * (-1 + smix ** 2)) / (
        245760.0 * mpi0 ** 4 * np.pi ** 7 * (-1 + sw ** 2)
    )
