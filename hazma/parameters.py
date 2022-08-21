from typing import Dict, List

import numpy as np
from scipy.integrate import trapz
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d

"""
Physics constants and utility functions.
"""

# ==============================================================================
# ---- Conversion Factors ------------------------------------------------------
# ==============================================================================

#: :math:`c` - Speed of light in m/s
speed_of_light = 299792458

_hbar_joule_sec = 6.62607015e-34 / (2 * np.pi)  # J.s
_joule_to_mev = 1 / (1.602176634e-19) * 1e-6  # MeV

# hbar [eV * s] = 1 => s = 1 / (hbar [eV]) = 1 / (hbar [1e-6 * MeV])

#: Convert :math:`\mathrm{s}` to :math:`\mathrm{MeV}^{-1}`
s_to_inv_MeV = 1.0 / (_hbar_joule_sec * _joule_to_mev)  # MeV^-1
#: Convert :math:`\mathrm{m}` to :math:`\mathrm{MeV}^{-1}`
m_to_inv_MeV = s_to_inv_MeV / speed_of_light  # MeV^-1
#: Convert :math:`\mathrm{cm}` to :math:`\mathrm{MeV}^{-1}`
cm_to_inv_MeV = m_to_inv_MeV * 1e-2  # MeV^-1 cm^-1
#: Convert :math:`\expval{\sigma v}` from
#: :math:`\mathrm{MeV}^{-2}` to :math:`\mathrm{cm}^{3}/\mathrm{s}`
sv_inv_MeV_to_cm3_per_s = 1.0 / cm_to_inv_MeV**2 * speed_of_light * 1e2
#: Convert :math:`\mathrm{grams}` to :math:`\mathrm{MeV}`
g_to_MeV = s_to_inv_MeV / m_to_inv_MeV**2 / _hbar_joule_sec * 1e-3
#: Convert :math:`\mathrm{MeV}` to :math:`\mathrm{grams}`
MeV_to_g = 1 / g_to_MeV
#: Solar mass to grams
Msun_to_g = 1.988e33
#: Gram to solar mass
g_to_Msun = 1 / Msun_to_g
#: Barn to MeV^-2
barn_to_inv_MeV2 = 0.00256819

# ==============================================================================
# ---- Masses (MeV) ------------------------------------------------------------
# ==============================================================================

#: Higgs mass in MeV
higgs_mass: float = 125.1e3
#: Electron mass in MeV
electron_mass: float = 0.5109989461  # m[e-] = 0.5109989461 ± 3.1e-09
#: Muon mass in MeV
muon_mass: float = 105.6583745  # m[mu-] = 105.6583745 ± 2.4e-06
#: Tau mass in MeV
tau_mass: float = 1776.86  # m[tau-] = 1776.86 ± 0.12
#: Neutral pion mass in MeV
neutral_pion_mass: float = 134.9768  # m[pi0] = 134.9768 ± 0.0005
#: Charged pion mass in MeV
charged_pion_mass: float = 139.57039  # m[pi+] = 139.57039 ± 0.00018
#: Eta mass in MeV
eta_mass: float = 547.862  # m[eta] = 547.862 ± 0.017
#: Eta' mass in MeV
eta_prime_mass: float = 957.78  # m[eta'(958)] = 957.78 ± 0.06
#: Charged kaon mass in MeV
charged_kaon_mass: float = 493.677  # m[K+] = 493.677 ± 0.016
#: Neutral kaon mass in MeV
neutral_kaon_mass: float = 497.611  # m[K0] = 497.611 ± 0.013
#: Long kaon mass in MeV
long_kaon_mass: float = 497.611  # m[K(L)0] = 497.611 ± 0.013
#: Short kaon mass in MeV
short_kaon_mass: float = 497.611  # m[K(S)0] = 497.611 ± 0.013
#: Rho mass in MeV
rho_mass: float = 775.26  # m[rho(770)0] = 775.26 ± 0.23
#: Omega mass in MeV
omega_mass: float = 782.66  # m[omega(782)] = 782.66 ± 0.13
#: Phi mass in MeV
phi_mass: float = 1019.461  # m[phi(1020)] = 1019.461 ± 0.016
#: Charged B mass in MeV
charged_B_mass: float = 5279.29  # B^+ meson
#: Pion mass in MeV (chiral-limit)
pion_mass_chiral_limit: float = (neutral_pion_mass + charged_pion_mass) / 2.0
#: Kaon mass in MeV (chiral-limit)
kaon_mass_chiral_limit: float = (neutral_kaon_mass + charged_kaon_mass) / 2.0

#: Up-quark mass in MeV
up_quark_mass: float = 2.16
#: Down-quark mass in MeV
down_quark_mass: float = 4.67
#: Strange-quark mass in MeV
strange_quark_mass: float = 95.0
#: Charm-quark mass in MeV
charm_quark_mass: float = 1.27e3
#: Bottom-quark mass in MeV
bottom_quark_mass: float = 4.18e3
#: Top-quark mass in MeV
top_quark_mass: float = 172.9e3

# Vector Boson Masses

#: W-Boson mass in MeV
wboson_mass: float = 80.385003e-3
#: Z-Boson mass in MeV
zboson_mass: float = 91.18760e-3

# Collections of masses
lepton_masses: List[float] = [electron_mass, muon_mass]

standard_model_masses: Dict[str, float] = {
    "e": electron_mass,
    "mu": muon_mass,
    "tau": tau_mass,
    "ve": 0.0,
    "vm": 0.0,
    "vt": 0.0,
    "u": up_quark_mass,
    "c": charm_quark_mass,
    "t": top_quark_mass,
    "d": down_quark_mass,
    "s": strange_quark_mass,
    "b": bottom_quark_mass,
    "h": higgs_mass,
    "pi": charged_pion_mass,
    "pi0": neutral_pion_mass,
    "k": charged_kaon_mass,
    "k0": neutral_kaon_mass,
    "kl": long_kaon_mass,
    "ks": short_kaon_mass,
    "eta": eta_mass,
    "etap": eta_prime_mass,
    "rho": rho_mass,
    "rho0": rho_mass,
    "omega": omega_mass,
    "phi": phi_mass,
    "photon": 0.0,
    "a": 0.0,
}

# ==============================================================================
# ---- Widths ------------------------------------------------------------------
# ==============================================================================

# Widths (MeV)
rho_width: float = 147.4
B_width = 4.35e-10
k_width = 5.32e-14
kl_width = 1.29e-14

SM_WIDTHS: Dict[str, float] = {
    "mu": 1 / (2.1969811e-6 * s_to_inv_MeV),
    "tau": 1 / (290.3e-15 * s_to_inv_MeV),
    "t": 1.42e3,
    "z": 2.49520e3,
    "w": 2.08500e3,
    "h": 0.00374e3,
    "pi0": 1 / (8.43e-17 * s_to_inv_MeV),
    "pi": 1 / (2.6033e-8 * s_to_inv_MeV),
    "k": 1 / (1.2380e-8 * s_to_inv_MeV),
    "ks": 1 / (0.8954e-10 * s_to_inv_MeV),
    "kl": 1 / (5.116e-8 * s_to_inv_MeV),
    "eta": 1.31e-3,
    "rho": 147.4,
    "omega": 8.68,
    "eta_prime": 0.188,
    "phi": 4.249,
}

# ==============================================================================
# ---- PDG Codes ---------------------------------------------------------------
# ==============================================================================

PDG_CODES: Dict[str, int] = {
    "d": 1,
    "u": 2,
    "s": 3,
    "c": 4,
    "b": 5,
    "t": 6,
    "e": 11,
    "mu": 13,
    "tau": 15,
    "ve": 12,
    "vm": 14,
    "vt": 16,
    "g": 21,
    "a": 22,
    "z": 23,
    "w": 24,
    "h": 25,
    "pi0": 111,
    "pi": 211,
    "k0": 311,
    "kl": 130,
    "ks": 310,
    "k": 321,
    "eta": 221,
    "etap": 331,
    "rho0": 113,
    "rho": 213,
    "omega": 223,
    "phi": 333,
}


# Miscellaneous
#: Electromagnetic fine structure constant.
alpha_em: float = 1.0 / 137.04
#: Fermi constant in MeV**-2
GF: float = 1.1663787e-11
#: Higg vacuum expectation value in MeV
vh: float = 246.22795e3
#:  :math:`e` - Electromagnetic coupling constant
qe: float = np.sqrt(4.0 * np.pi * alpha_em)
#: :math:`T_{\mathrm{CMB}}` - CMB temperature at formation in MeV
temp_cmb_formation: float = 0.235e-6
#: :math:`M_{\mathrm{pl}}` - Plank mass in MeV
plank_mass: float = 1.22091e22
#: :math:`\rho_{\mathrm{crit}}` - Critical energy density in units of
#: :math:`h^2 \ [\mathrm{MeV} / \mathrm{cm}^3]`
rho_crit: float = 1.05375e-2
#: :math:`s_{0}` - Entropy density today [:math:`\mathrm{cm}^{-3}`]
sm_entropy_density_today: float = 2891.2
#: :math:`\Omega_{\mathrm{CMD}}h^2` - Energy density fraction of dark matter
#: times :math:`h^2`
omega_h2_cdm: float = 0.1198
#: :math:`h = H_{0}/(100 \mathrm{km/s/Mpc})` - Hubble constant scale by
#: 100 km/s/Mpc (Plank 2015)
dimensionless_hubble_constant: float = 0.6774
#: :math:`\sin^2\theta_{W}` - Square of the sine of the Weinberg angle
sin_theta_weak_sqrd: float = 0.22290
#: :math:`\sin\theta_{W}` - Sine of the Weinberg angle
sin_theta_weak: float = np.sqrt(sin_theta_weak_sqrd)
#: :math:`\cos\theta_{W}` - Cosine of the Weinberg angle
cos_theta_weak: float = np.sqrt(1.0 - sin_theta_weak_sqrd)

#: :math:`Q_{u}/e` - Up-type quark charge in units of electron charge.
Qu: float = 2.0 / 3.0
#: :math:`Q_{d}/e` - Down-type quark charge in units of electron charge.
Qd: float = -1.0 / 3.0
#: :math:`Q_{e}/e` - Charged lepton charge in units of electron charge.
Qe: float = -1.0

# ==============================================================================
# ---- CKM matrix elements -----------------------------------------------------
# ==============================================================================


def _ckm_wolfenstein(lam, a, rhob, etab):
    z = rhob + 1j * etab
    alam = a * lam**2

    s12 = lam
    s23 = alam
    s13d = (
        a
        * lam**3
        * z
        * np.sqrt(1 - alam**2)
        / (np.sqrt(1 - lam**2) * (1 - alam**2 * z))
    )

    c12 = np.sqrt(1 - s12**2)
    c23 = np.sqrt(1 - s23**2)
    c13 = np.sqrt(1 - abs(s13d) ** 2)

    m1 = np.array([[1.0, 0.0, 0.0], [0.0, c23, s23], [0.0, -s23, c23]])
    m2 = np.array([[c13, 0.0, s13d.conj()], [0.0, 1.0, 0.0], [-s13d, 0.0, c13]])
    m3 = np.array([[c12, s12, 0.0], [-s12, c12, 0.0], [0.0, 0.0, 1.0]])

    return (m1 @ m2) @ m3


# PDG March 2022
_wolfenstein_lam = 0.22500
_wolfenstein_a = 0.826
_wolfenstein_rhob = 0.159
_wolfenstein_etab = 0.348


#: :math:`V_{\mathrm{CKM}}` - CKM matrix
CKM = _ckm_wolfenstein(
    _wolfenstein_lam, _wolfenstein_a, _wolfenstein_rhob, _wolfenstein_etab
)
#: :math:`V_{ud}` - up-down CKM matrix element
Vud: complex = CKM[0, 0]
#: :math:`V_{us}` - up-strange CKM matrix element
Vus: complex = CKM[0, 1]
#: :math:`V_{ub}` - up-bottom CKM matrix element
Vub: complex = CKM[0, 2]

#: :math:`V_{ud}` - charm-down CKM matrix element
Vcd: complex = CKM[1, 0]
#: :math:`V_{us}` - charm-strange CKM matrix element
Vcs: complex = CKM[1, 1]
#: :math:`V_{ub}` - charm-bottom CKM matrix element
Vcb: complex = CKM[1, 2]

#: :math:`V_{td}` - top-down CKM matrix element
Vtd: complex = CKM[2, 0]
#: :math:`V_{ts}` - top-strange CKM matrix element
Vts: complex = CKM[2, 1]
#: :math:`V_{tb}` - top-bottom CKM matrix element
Vtb: complex = CKM[2, 2]


# ==============================================================================
# ---- ChiPT -------------------------------------------------------------------
# ==============================================================================

# Low Energy constants
#: :math:`f_{\pi^{0}}` - Neutral pion decay constant.
fpi0: float = 91.924
#: :math:`f_{\pi^{\pm}}` - Charged pion decay constant.
fpi: float = 92.2138
#: :math:`f_{K^{\pm}}` - Charged kaon decay constant.
fk: float = 110.379
#: :math:`f_{\eta}` - Eta decay constant.
feta: float = 57.8
#: :math:`B_{0} = \expval{\bar{q}q}/(3f_{\pi}^2)` - ChiPT mass parameter
b0: float = pion_mass_chiral_limit**2 / (up_quark_mass + down_quark_mass)

G8: float = 5.47
G27: float = 0.392
gv: float = 67.0
fv: float = 153.0

# The following low energy constants are for NLO ChiPT, evaluated at mu = mrho.
nlo_lec_mu: float = rho_mass
Lr1: float = 0.56 * 1.0e-3
Lr2: float = 1.21 * 1.0e-3
L3: float = -2.79 * 1.0e-3
Lr4: float = -0.36 * 1.0e-3
Lr5: float = 1.4 * 1.0e-3
Lr6: float = 0.07 * 1.0e-3
L7: float = -0.44 * 1.0e-3
Lr8: float = 0.78 * 1.0e-3

# SU(2) LECs
Er: float = 0.029
Gr: float = 0.0073

LECS: Dict[str, float] = {
    "1": Lr1,
    "2": Lr2,
    "3": L3,
    "4": Lr4,
    "5": Lr5,
    "6": Lr6,
    "7": L7,
    "8": Lr8,
    "SU2_Er": Er,
    "SU2_Gr": Gr,
}


def convert_sigmav(sv, target):
    """Changes the units of <sigma v>.

    Parameters
    ----------
    sv : float
        Cross section in units of MeV^-2 or cm^3 / s.
    target : string
        Units to convert to. Must be "MeV^-2" or "cm^3/s" -- whichever units sv
        is NOT in.

    Returns
    -------
    sv : float
        sv converted to be in the target units.
    """
    # hbar^2 c^3 in units of MeV^2 cm^3 / s
    hbar2_c3 = (3.0e10) ** 3 * (6.58e-22) ** 2

    if target == "cm^3 / s":
        return sv * hbar2_c3
    elif target == "MeV^-2":
        return sv / hbar2_c3


def load_interp(rf_name, bounds_error=False, fill_value=0.0):
    """Creates an interpolator from a data file.

    Parameters
    ----------
    rf_name : resource_filename
        Name of resource file.

    Returns
    -------
    interp : interp1d
        An interpolator created using the first column of the file as the x
        values and second as the y values. interp will not raise a bounds error
        and uses a fill values of 0.0.
    """
    xs, ys = np.loadtxt(rf_name, delimiter=",").T
    return interp1d(xs, ys, bounds_error=bounds_error, fill_value=fill_value)


def spec_res_fn(ep, e, energy_res):
    """Get the spectral resolution function."""
    sigma = e * energy_res(e)

    if sigma == 0:
        if hasattr(ep, "__len__"):
            return np.zeros(ep.shape)
        else:
            return 0.0
    else:
        return (
            1.0
            / np.sqrt(2.0 * np.pi * sigma**2)
            * np.exp(-((ep - e) ** 2) / (2.0 * sigma**2))
        )


def convolved_spectrum_fn(
    e_min, e_max, energy_res, spec_fn=None, lines=None, n_pts=1000
):
    r"""
    Convolves a continuum and line spectrum with a detector's spectral
    resolution function.

    Parameters
    ----------
    e_min : float
        Lower bound of energy range over which to perform convolution.
    e_max : float
        Upper bound of energy range over which to perform convolution.
    energy_res : float -> float
        The detector's energy resolution (Delta E / E) as a function of
        photon energy in MeV.
    spec_fn : np.array -> np.array
        Continuum spectrum function.
    lines : dict
        Information about spectral lines.
    n_pts : float
        Number of points to use to create resulting interpolating function.

    Returns
    -------
    dnde_conv : InterpolatedUnivariateSpline
        An interpolator giving the DM annihilation spectrum as seen by the
        detector. Using photon energies outside the range [e_min, e_max] will
        produce a ``bounds_errors``.
    """
    es = np.geomspace(e_min, e_max, n_pts)
    dnde_conv = np.zeros(es.shape)

    # Pad energy grid to avoid edge effects
    es_padded = np.geomspace(0.1 * e_min, 10 * e_max, n_pts)
    if spec_fn is not None:
        dnde_src = spec_fn(es_padded)
        if not np.all(dnde_src == 0):

            def integral(e):
                """
                Performs the integration at given photon energy.
                """
                spec_res_fn_vals = spec_res_fn(es_padded, e, energy_res)
                integrand_vals = (
                    dnde_src * spec_res_fn_vals / trapz(spec_res_fn_vals, es_padded)
                )

                return trapz(integrand_vals, es_padded)

            dnde_conv += np.vectorize(integral)(es)

    # Line contribution
    if lines is not None:
        for ch, line in lines.items():
            dnde_conv += (
                line["bf"]
                * spec_res_fn(es, line["energy"], energy_res)
                * (2.0 if ch == "g g" else 1.0)
            )

    return InterpolatedUnivariateSpline(es, dnde_conv, k=1, ext="raise")  # type: ignore
