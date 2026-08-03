from hazma.spectra import (
    dnde_photon_charged_pion as charged_pion,
    dnde_photon_muon as muon,
    dnde_photon_neutral_pion as neutral_pion,
)

from hazma.field_theory_helper_functions.common_functions import minkowski_dot as MDot

from hazma.parameters import electron_mass as me
from hazma.parameters import muon_mass as mmu
from hazma.parameters import charged_pion_mass as mpi
from hazma.parameters import neutral_pion_mass as mpi0
from hazma.parameters import charged_kaon_mass as mk
from hazma.parameters import neutral_kaon_mass as mk0

from hazma.rambo import generate_energy_histogram as geh

import numpy as np

bfs = {
    "munu": 0.6356,
    "p0": 0.2067,
    "ppm": 0.05583,
    "0enu": 0.0507,
    "0munu": 0.03352,
    "00p": 0.01760,
}


# ############################
# ##### Matrix Elements ######
# ############################

# Constants for weak hadronic matrix elements
alpha1 = 93.16 * 10**-8.0
alpha3 = -6.72 * 10**-8.0
beta1 = -27.06 * 10**-8.0
beta3 = -2.22 * 10**-8.0
gamma3 = 2.95 * 10**-8.0
zeta1 = -0.40 * 10**-8.0
zeta3 = -0.09 * 10**-8.0
xi1 = -1.83 * 10**-8.0
xi3 = -0.17 * 10**-8.0
xi3p = -0.56 * 10**-8.0
A2 = 0.0212 * 10**-3.0
lamp = 0.034
lam0 = 0.025


def amp_L000(moms):
    """
    Amplitude for k_L(k) -> pi0(p1) + pi0(p2) + pi0(p3)

    Parameters
    ----------
    moms : ndarray[dim=2]
        moms[0] = pi0
        moms[1] = pi0
        moms[2] = pi0
    """
    p1 = moms[0]
    p2 = moms[1]
    p3 = moms[2]

    k = p1 + p2 + p3

    s0 = (1.0 / 3.0) * (mk0**2 + 3 * mpi0**2)
    s1 = MDot(k - p1, k - p1)
    s2 = MDot(k - p2, k - p2)
    s3 = MDot(k - p3, k - p3)

    x = (s2 - s1) / mpi**2
    y = (s3 - s0) / mpi**2

    return 3.0 * (alpha1 + alpha3) + -3.0 * (zeta1 - 2.0 * zeta3) * (y**2 + x**2 / 3.0)


def amp_Lpm0(moms):
    """
    Amplitude for k_L(k) -> pi^+(p1) + pi^-(p2) + pi0(p3)

    Parameters
    ----------
    moms : ndarray[dim=2]
        moms[0] = pi
        moms[1] = pi
        moms[2] = pi0
    """
    p1 = moms[0]
    p2 = moms[1]
    p3 = moms[2]

    k = p1 + p2 + p3

    s0 = (1.0 / 3.0) * (mk0**2 + 2 * mpi**2 + mpi0**2)
    s1 = MDot(k - p1, k - p1)
    s2 = MDot(k - p2, k - p2)
    s3 = MDot(k - p3, k - p3)

    x = (s2 - s1) / mpi**2
    y = (s3 - s0) / mpi**2

    return (
        (alpha1 + alpha3)
        - (beta1 + beta3) * y
        + (zeta1 - 2.0 * zeta3) * (y**2 + x**2 / 3.0)
        + (xi1 - 2.0 * xi3) * (y**2 - x**2 / 3.0)
    )


def amp_Spm0(moms):
    """
    Amplitude for k_S(k) -> pi^+(p1) + pi^-(p2) + pi0(p3)

    Parameters
    ----------
    moms : ndarray[dim=2]
        moms[0] = pi
        moms[1] = pi
        moms[2] = pi0
    """
    p1 = moms[0]
    p2 = moms[1]
    p3 = moms[2]

    k = p1 + p2 + p3

    s0 = (1.0 / 3.0) * (mk0**2 + 2 * mpi**2 + mpi0**2)
    s1 = MDot(k - p1, k - p1)
    s2 = MDot(k - p2, k - p2)
    s3 = MDot(k - p3, k - p3)

    x = (s2 - s1) / mpi**2
    y = (s3 - s0) / mpi**2

    return (2.0 / 3.0) * np.sqrt(3) * gamma3 * x - (4.0 / 3.0) * xi3p * x * y


def amp_00p(moms):
    """
    Amplitude for k^+(k) -> pi0(p1) + pi0(p2) + pi^+(p3)

    Parameters
    ----------
    moms : ndarray[dim=2]
        moms[0] = pi0
        moms[1] = pi0
        moms[2] = pi
    """
    p1 = moms[0]
    p2 = moms[1]
    p3 = moms[2]

    k = p1 + p2 + p3

    s0 = (1.0 / 3.0) * (mk0**2 + 2 * mpi0**2 + mpi**2)
    s1 = MDot(k - p1, k - p1)
    s2 = MDot(k - p2, k - p2)
    s3 = MDot(k - p3, k - p3)

    x = (s2 - s1) / mpi**2
    y = (s3 - s0) / mpi**2

    return (
        -0.5 * (2.0 * alpha1 - alpha3)
        + (beta1 - 0.5 * beta3 - np.sqrt(3) * gamma3) * y
        - (zeta1 + zeta3) * (y**2 + x**2 / 3.0)
        - (xi1 + xi3 + xi3p) * (y**2 - x**2 / 3.0)
    )


def amp_ppm(moms):
    """
    Amplitude for k^+(k) -> pi^+(p1) + pi^+(p2) + pi^-(p3)

    Parameters
    ----------
    moms : ndarray[dim=2]
        moms[0] = pi
        moms[1] = pi
        moms[2] = pi
    """
    p1 = moms[0]
    p2 = moms[1]
    p3 = moms[2]

    k = p1 + p2 + p3

    s0 = (1.0 / 3.0) * (mk**2 + 3 * mpi**2)
    s1 = MDot(k - p1, k - p1)
    s2 = MDot(k - p2, k - p2)
    s3 = MDot(k - p3, k - p3)

    x = (s2 - s1) / mpi**2
    y = (s3 - s0) / mpi**2

    return (
        (2.0 * alpha1 - alpha3)
        + (beta1 - 0.5 * beta3 + np.sqrt(3) * gamma3) * y
        - 2.0 * (zeta1 + zeta3) * (y**2 + x**2 / 3.0)
        - (xi1 + xi3 - xi3p) * (y**2 - x**2 / 3.0)
    )


def msqrd_L000(moms):
    """
    Squared matrix element for k_L(k) -> pi0(p1) + pi0(p2) + pi0(p3)

    Parameters
    ----------
    moms : ndarray[dim=2]
        moms[0] = pi0
        moms[1] = pi0
        moms[2] = pi0
    """
    return abs(amp_L000(moms)) ** 2


def msqrd_Lpm0(moms):
    """
    Squared matrix element for k_L(k) -> pi^+(p1) + pi^-(p2) + pi0(p3)

    Parameters
    ----------
    moms : ndarray[dim=2]
        moms[0] = pi
        moms[1] = pi
        moms[2] = pi0
    """
    return abs(amp_Lpm0(moms)) ** 2


def msqrd_Spm0(moms):
    """
    Squared matrix element for k_S(k) -> pi^+(p1) + pi^-(p2) + pi0(p3)

    Parameters
    ----------
    moms : ndarray[dim=2]
        moms[0] = pi
        moms[1] = pi
        moms[2] = pi0
    """
    return abs(amp_Spm0(moms)) ** 2


def msqrd_00p(moms):
    """
    Squared matrix element for k^+(k) -> pi0(p1) + pi0(p2) + pi^+(p3)

    Parameters
    ----------
    moms : ndarray[dim=2]
        moms[0] = pi0
        moms[1] = pi0
        moms[2] = pi
    """
    return abs(amp_00p(moms)) ** 2


def msqrd_ppm(moms):
    """
    Squared matrix element for k^+(k) -> pi^+(p1) + pi^+(p2) + pi^-(p3)

    Parameters
    ----------
    moms : ndarray[dim=2]
        moms[0] = pi
        moms[1] = pi
        moms[2] = pi
    """
    return abs(amp_ppm(moms)) ** 2


def msqrd_pilnu(moms, ml):
    pp = moms[0]
    pl = moms[1]
    pn = moms[2]

    return (
        (lam0 - lamp) ** 2
        * (mk - mpi) ** 2
        * (mk + mpi) ** 2
        * MDot(pl, pn)
        * (
            -(mk**2)
            + 2 * ml**2
            + mpi**2
            + 2 * MDot(pl, pn)
            + 2 * MDot(pl, pp)
            + 2 * MDot(pp, pn)
        )
        - (
            lamp * mk**2
            + mpi**2
            - lamp * mpi**2
            - 2 * lamp * MDot(pl, pp)
            - 2 * lamp * MDot(pp, pn)
        )
        ** 2
        * (
            -2 * MDot(pl, pn) ** 2
            + MDot(pl, pn)
            * (mk**2 - 2 * ml**2 + 3 * mpi**2 - 2 * MDot(pl, pp) - 2 * MDot(pp, pn))
            - 4 * (ml**2 + 2 * MDot(pl, pp)) * MDot(pp, pn)
        )
        - (lam0 - lamp)
        * (mk - mpi)
        * (mk + mpi)
        * (
            -2 * MDot(pl, pn) ** 2
            + MDot(pl, pn)
            * (mk**2 - 2 * ml**2 - mpi**2 - 2 * MDot(pl, pp) - 2 * MDot(pp, pn))
            - 2 * ml**2 * MDot(pp, pn)
        )
        * (
            lamp * mk**2
            + (1 + lamp) * mpi**2
            - 2 * lamp * (mpi**2 + MDot(pl, pp) + MDot(pp, pn))
        )
        - (lam0 - lamp)
        * (-mk + mpi)
        * (mk + mpi)
        * (
            lamp * mk**2
            + (1 + lamp) * mpi**2
            - 2 * lamp * (mpi**2 + MDot(pl, pp) + MDot(pp, pn))
        )
        * (
            2 * MDot(pl, pn) ** 2
            + 2 * ml**2 * MDot(pp, pn)
            + MDot(pl, pn)
            * (-(mk**2) + 2 * ml**2 + mpi**2 + 2 * MDot(pl, pp) + 2 * MDot(pp, pn))
        )
    ) / mpi**4


def msqrd_pienu(moms):
    return msqrd_pilnu(moms, me)


def msqrd_pimunu(moms):
    return msqrd_pilnu(moms, mmu)


# ####################################
# #### Create Probability Arrays #####
# ####################################

npts = 10**6
nbins = 25

m_vecs = {
    "ppm": np.array([mpi, mpi, mpi]),
    "0enu": np.array([mpi0, me, 0.0]),
    "0munu": np.array([mpi0, mmu, 0.0]),
    "00p": np.array([mpi0, mpi0, mpi]),
}

prob_dists_nome = {
    "ppm": geh(npts, m_vecs["ppm"], mk, num_bins=nbins, density=True)[0],
    "0enu": geh(npts, m_vecs["0enu"], mk, num_bins=nbins, density=True)[0],
    "0munu": geh(npts, m_vecs["0munu"], mk, num_bins=nbins, density=True)[0],
    "00p": geh(npts, m_vecs["00p"], mk, num_bins=nbins, density=True)[0],
}

prob_dists = {
    "ppm": geh(
        npts, m_vecs["ppm"], mk, num_bins=nbins, density=True, mat_elem_sqrd=msqrd_ppm
    )[0],
    "0enu": geh(
        npts,
        m_vecs["0enu"],
        mk,
        num_bins=nbins,
        density=True,
        mat_elem_sqrd=msqrd_pienu,
    )[0],
    "0munu": geh(
        npts,
        m_vecs["0munu"],
        mk,
        num_bins=nbins,
        density=True,
        mat_elem_sqrd=msqrd_pimunu,
    )[0],
    "00p": geh(
        npts, m_vecs["00p"], mk, num_bins=nbins, density=True, mat_elem_sqrd=msqrd_00p
    )[0],
}


# ##########################
# #### Compute Spectra #####
# ##########################

neng_gams = 1000

eng_gams = np.logspace(-5.0, 4.0, num=1000, dtype=np.float64)

# Two body decays
spec_munu = muon(eng_gams, (mk**2 - mmu**2) / (2.0 * mk))
spec_p0 = charged_pion(eng_gams, (mk**2 - mpi**2 + mpi0**2) / (2.0 * mk))
spec_p0 += neutral_pion(eng_gams, (mk**2 + mpi**2 - mpi0**2) / (2.0 * mk))

# Three body decays
spec_ppm = np.zeros(neng_gams, dtype=np.float64)
spec_0enu = np.zeros(neng_gams, dtype=np.float64)
spec_0munu = np.zeros(neng_gams, dtype=np.float64)
spec_00p = np.zeros(neng_gams, dtype=np.float64)

for k in range(nbins):
    spec_ppm += prob_dists["ppm"][0, 1, k] * charged_pion(
        eng_gams, prob_dists["ppm"][0, 0, k]
    )
    spec_ppm += prob_dists["ppm"][1, 1, k] * charged_pion(
        eng_gams, prob_dists["ppm"][1, 0, k]
    )
    spec_ppm += prob_dists["ppm"][2, 1, k] * charged_pion(
        eng_gams, prob_dists["ppm"][2, 0, k]
    )

    spec_0enu += prob_dists["0enu"][0, 1, k] * neutral_pion(
        eng_gams, prob_dists["0enu"][0, 0, k]
    )

    spec_0munu += prob_dists["0munu"][0, 1, k] * neutral_pion(
        eng_gams, prob_dists["0munu"][0, 0, k]
    )
    spec_0munu += prob_dists["0munu"][1, 1, k] * muon(
        eng_gams, prob_dists["0munu"][1, 0, k]
    )

    spec_00p += prob_dists["00p"][0, 1, k] * neutral_pion(
        eng_gams, prob_dists["00p"][0, 0, k]
    )
    spec_00p += prob_dists["00p"][1, 1, k] * neutral_pion(
        eng_gams, prob_dists["00p"][1, 0, k]
    )
    spec_00p += prob_dists["00p"][2, 1, k] * charged_pion(
        eng_gams, prob_dists["00p"][2, 0, k]
    )


spec = (
    bfs["munu"] * spec_munu
    + bfs["p0"] * spec_p0
    + bfs["ppm"] * spec_ppm
    + bfs["0enu"] * spec_0enu
    + bfs["0munu"] * spec_0munu
    + bfs["00p"] * spec_00p
)


np.savetxt("charged_kaon_interp.dat", zip(eng_gams, spec), delimiter=",")
