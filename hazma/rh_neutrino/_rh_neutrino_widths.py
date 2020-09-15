"""
This file contains the mixin class which implements the partial widths of the
right-handed neutrino.
"""
from hazma.parameters import (
    GF,
    fpi,
    Vud,
    Vus,
    neutral_pion_mass as mpi0,
    charged_pion_mass as mpi,
    charged_kaon_mass as mk,
    sin_theta_weak as sw,
    cos_theta_weak as cw,
)
from scipy.integrate import quad
import numpy as np

# Load data for N -> gamma + nu
import os

_this_dir, _ = os.path.split(__file__)

_data_n_to_g_nue = np.log10(
    np.genfromtxt(
        os.path.join(_this_dir, "msqrd_n_to_g_nue.csv"),
        skip_header=1,
        delimiter=",",
    )
).T
_data_n_to_g_num = np.log10(
    np.genfromtxt(
        os.path.join(_this_dir, "msqrd_n_to_g_numu.csv"),
        skip_header=1,
        delimiter=",",
    )
).T
_data_fit_n_to_g_nu = np.genfromtxt(
    os.path.join(_this_dir, "msqrd_n_to_g_nu_fit.csv"),
    skip_header=1,
    delimiter=",",
)

_mx_min_n_to_g_nu = _data_n_to_g_nue[0][0]
_mx_max_n_to_g_nu = _data_n_to_g_nue[0][-1]

_small_mx_int_nue = _data_fit_n_to_g_nu[0]
_small_mx_slo_nue = _data_fit_n_to_g_nu[1]
_large_mx_int_nue = _data_fit_n_to_g_nu[2]
_large_mx_slo_nue = _data_fit_n_to_g_nu[3]

_small_mx_int_num = _data_fit_n_to_g_nu[4]
_small_mx_slo_num = _data_fit_n_to_g_nu[5]
_large_mx_int_num = _data_fit_n_to_g_nu[6]
_large_mx_slo_num = _data_fit_n_to_g_nu[7]


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

    Returns
    -------
    width: float
        Partial width for N -> pi + l.
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
    mvr = self.mx
    stheta = self.stheta
    ml = self.ml

    if mvr < mk + ml:
        return 0.0

    return -(
        fpi ** 2
        * GF ** 2
        * (-((ml ** 2 - mvr ** 2) ** 2) + mk ** 2 * (ml ** 2 + mvr ** 2))
        * np.sqrt(
            -(ml ** 2)
            + (-(mk ** 2) + ml ** 2 + mvr ** 2) ** 2 / (4.0 * mvr ** 2)
        )
        * stheta ** 2
        * Vus ** 2
    ) / (4.0 * mvr ** 2 * np.pi)


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
    logmx = np.log10(mx)
    stheta = self.stheta
    lepton = self.lepton

    if _mx_min_n_to_g_nu < mx < _mx_max_n_to_g_nu:
        if lepton == "e":
            msqrd = 10 ** np.interp(
                logmx, _data_n_to_g_nue[0], _data_n_to_g_nue[1]
            )
        else:
            msqrd = 10 ** np.interp(
                logmx, _data_n_to_g_num[0], _data_n_to_g_num[1]
            )
    elif mx < _mx_min_n_to_g_nu:
        if lepton == "e":
            msqrd = 10 ** (_small_mx_int_nue + _small_mx_slo_nue * logmx)
        else:
            msqrd = 10 ** (_small_mx_int_num + _small_mx_slo_num * logmx)
    else:
        if lepton == "e":
            msqrd = 10 ** (_large_mx_int_nue + _large_mx_slo_nue * logmx)
        else:
            msqrd = 10 ** (_large_mx_int_num + _large_mx_slo_num * logmx)

    return stheta ** 2 * msqrd / (16.0 * mx * np.pi)


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
    MN = self.mx
    stheta = self.stheta
    if MN < 2.0 * mpi:
        return 0.0
    return (
        GF ** 2
        * stheta ** 2
        * (-1 + stheta ** 2)
        * (1 - 2 * sw ** 2) ** 2
        * (
            np.sqrt(MN ** 4 - 4 * MN ** 2 * mpi ** 2)
            * (
                MN ** 6
                + 24 * MN ** 4 * mpi ** 2
                - 10 * MN ** 2 * mpi ** 4
                + 12 * mpi ** 6
            )
            + 6
            * mpi ** 2
            * (MN ** 6 + 2 * MN ** 2 * mpi ** 4 - 2 * mpi ** 6)
            * np.log(
                (
                    MN ** 4
                    + 2 * mpi ** 4
                    + 2 * mpi ** 2 * np.sqrt(MN ** 4 - 4 * MN ** 2 * mpi ** 2)
                    - MN ** 2
                    * (
                        4 * mpi ** 2
                        + np.sqrt(MN ** 4 - 4 * MN ** 2 * mpi ** 2)
                    )
                )
                / (2.0 * mpi ** 4)
            )
        )
    ) / (768.0 * MN ** 3 * np.pi ** 3 * (-1 + sw ** 2))


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

    MN = self.mx
    Ml = self.ml
    smix = self.stheta

    if MN < Ml + mpi + mpi0:
        return 0.0

    def integrand(s):
        """
        Returns the squared matrix element integrated over the Mandelstam
        variable t.
        """
        return (
            -2
            * GF ** 2
            * np.sqrt(
                Ml ** 4 + (MN ** 2 - s) ** 2 - 2 * Ml ** 2 * (MN ** 2 + s)
            )
            * np.sqrt(
                mpi ** 4
                + (mpi0 ** 2 - s) ** 2
                - 2 * mpi ** 2 * (mpi0 ** 2 + s)
            )
            * (
                MN ** 2
                * s
                * (
                    -2 * mpi ** 4
                    - 2 * mpi0 ** 4
                    + mpi ** 2 * (4 * mpi0 ** 2 - 2 * s)
                    - 2 * mpi0 ** 2 * s
                    + s ** 2
                )
                - 2
                * s ** 2
                * (
                    mpi ** 4
                    + (mpi0 ** 2 - s) ** 2
                    - 2 * mpi ** 2 * (mpi0 ** 2 + s)
                )
                + Ml ** 4
                * (
                    4 * mpi ** 4
                    + 4 * mpi0 ** 4
                    - 2 * mpi0 ** 2 * s
                    + s ** 2
                    - 2 * mpi ** 2 * (4 * mpi0 ** 2 + s)
                )
                + MN ** 4
                * (
                    4 * mpi ** 4
                    + 4 * mpi0 ** 4
                    - 2 * mpi0 ** 2 * s
                    + s ** 2
                    - 2 * mpi ** 2 * (4 * mpi0 ** 2 + s)
                )
                + Ml ** 2
                * (
                    s
                    * (
                        -2 * mpi ** 4
                        - 2 * mpi0 ** 4
                        + mpi ** 2 * (4 * mpi0 ** 2 - 2 * s)
                        - 2 * mpi0 ** 2 * s
                        + s ** 2
                    )
                    - 2
                    * MN ** 2
                    * (
                        4 * mpi ** 4
                        + 4 * mpi0 ** 4
                        - 2 * mpi0 ** 2 * s
                        + s ** 2
                        - 2 * mpi ** 2 * (4 * mpi0 ** 2 + s)
                    )
                )
            )
            * smix ** 2
            * Vud ** 2
        ) / (3.0 * s ** 3)

    # Measure associated with ds*dt
    measure = 1.0 / (16.0 * MN ** 2 * (2.0 * np.pi) ** 3)
    width_pre = 1.0 / (2.0 * MN) * measure
    # Integration bounds
    ub = (MN - Ml) ** 2
    lb = (mpi + mpi0) ** 2

    return -width_pre * quad(integrand, lb, ub)[0]


def width_nu_nu_nu(self):
    """
    Compute the width for right-handed neutrino to three active neutrinos.

    Returns
    -------
    width: float
        Partial width for N -> 3nu.
    """
    MN = self.mx
    stheta = self.stheta
    return -(GF ** 2 * MN ** 5 * stheta ** 2 * (-1 + stheta ** 2) ** 3) / (
        32.0 * np.pi ** 3
    )


def width_nu_l_l(self):
    """
    Compute the width for right-handed neutrino to an active neutrino and
    two charged leptons.

    Returns
    -------
    width: float
        Partial width for N -> nu + l + l.
    """
    MN = self.mx
    Ml = self.ml

    if MN < 2.0 * Ml:
        return 0.0

    stheta = self.stheta

    width = (
        GF ** 2
        * stheta ** 2
        * (-1 + stheta ** 2)
        * (
            MN
            * np.sqrt(-4 * Ml ** 2 + MN ** 2)
            * (
                -(MN ** 6 * (1 + 4 * sw ** 2 + 8 * sw ** 4))
                + 12 * Ml ** 6 * (1 + 12 * sw ** 2 + 24 * sw ** 4)
                + 2 * Ml ** 2 * MN ** 4 * (7 + 20 * sw ** 2 + 40 * sw ** 4)
                - 2 * Ml ** 4 * MN ** 2 * (-1 + 36 * sw ** 2 + 72 * sw ** 4)
            )
            + 12
            * Ml ** 4
            * MN ** 4
            * np.log(
                1
                + (MN ** 3 * (MN - np.sqrt(-4 * Ml ** 2 + MN ** 2)))
                / (2.0 * Ml ** 4)
                + (MN * (-2 * MN + np.sqrt(-4 * Ml ** 2 + MN ** 2))) / Ml ** 2
                + 0.0j
            )
            + 12
            * Ml ** 6
            * (
                -8 * MN ** 2 * sw ** 2 * (1 + 2 * sw ** 2)
                + Ml ** 2 * (1 + 12 * sw ** 2 + 24 * sw ** 4)
            )
            * np.log(
                (2 * Ml ** 4)
                / (
                    2 * Ml ** 4
                    + MN ** 3 * (MN - np.sqrt(-4 * Ml ** 2 + MN ** 2))
                    + 2
                    * Ml ** 2
                    * MN
                    * (-2 * MN + np.sqrt(-4 * Ml ** 2 + MN ** 2))
                )
                + 0.0j
            )
        )
    ) / (384.0 * MN ** 3 * np.pi ** 3)

    return np.real(width)

