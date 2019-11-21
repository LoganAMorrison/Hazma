import numpy as np
from scipy.integrate import trapz
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d

"""
Physics constants and utility functions.
"""

# Masses (MeV)
electron_mass = 0.510998928  # electron
muon_mass = 105.6583715  # muon
neutral_pion_mass = 134.9766  # neutral pion
charged_pion_mass = 139.57018  # charged pion
neutral_kaon_mass = 497.61  # neutral kaon
long_kaon_mass = 497.614
charged_kaon_mass = 493.68  # charged Kaon
eta_mass = 547.86  # eta
eta_prime_mass = 957.8  # eta prime
rho_mass = 775.3  # rho
omega_mass = 782.7  # omega
charged_B_mass = 5279.29  # B^+ meson
pion_mass_chiral_limit = (neutral_pion_mass + charged_pion_mass) / 2.0
kaon_mass_chiral_limit = (neutral_kaon_mass + charged_kaon_mass) / 2.0

# Quark masses in MS-bar scheme
up_quark_mass = 2.3
down_quark_mass = 4.8
strange_quark_mass = 95.0
charm_quark_mass = 1.275e3
bottom_quark_mass = 4.18e3
top_quark_mass = 160.0e3

# Convert <sigma v> from MeV^2
cm_to_inv_MeV = 5.08e10  # MeV^-1 cm^-1
sv_inv_MeV_to_cm3_per_s = 1.0 / cm_to_inv_MeV ** 2 * 3e10  # cm^3/s * MeV^2

# Miscellaneous
alpha_em = 1.0 / 137.04  # fine structure constant.
GF = 1.1663787e-11  # Fermi constant in MeV**-2
vh = 246.22795e3  # Higgs VEV in MeV
qe = np.sqrt(4.0 * np.pi * alpha_em)
temp_cmb_formation = 0.235e-6  # CMB temperature at formation in MeV
plank_mass = 1.22091e22  # MeV
rho_crit = 1.05375e-2  # Critical energy density in units of h^2 MeV / cm^3
sm_entropy_density_today = 2891.2  # Entropy density today in units of cm^-3
omega_h2_cdm = 0.1198  # Dark matter energy fraction times h^2

# Charges
Qu = 2.0 / 3.0
Qd = -1.0 / 3.0
Qe = -1.0

# Low Energy constants
fpi0 = 91.924  # Neutral pion decay constant
fpi = 92.2138  # Charged pion decay constant
fk = 110.379  # Charged kaon decay constant
b0 = pion_mass_chiral_limit ** 2 / (up_quark_mass + down_quark_mass)
G8 = 5.47
G27 = 0.392
gv = 67.0
fv = 153.0

# The following low energy constants are for NLO ChiPT, evaluated at mu = mrho.
nlo_lec_mu = rho_mass
Lr1 = 0.56 * 1.0e-3
Lr2 = 1.21 * 1.0e-3
L3 = -2.79 * 1.0e-3
Lr4 = -0.36 * 1.0e-3
Lr5 = 1.4 * 1.0e-3
Lr6 = 0.07 * 1.0e-3
L7 = -0.44 * 1.0e-3
Lr8 = 0.78 * 1.0e-3

# SU(2) LECs
Er = 0.029
Gr = 0.0073

LECS = {
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

# SU(2) LECs
Er = 0.029
Gr = 0.0073

# CKM matrix elements
Vud = 0.974267
Vus = 0.2248
Vts = -0.0405 - 0.00075987j
Vtb = 0.999139
Vtd = 0.00823123 - 0.00328487j


# Widths (MeV)
rho_width = 146.2
B_width = 4.35e-10
k_width = 5.32e-14
kl_width = 1.29e-14


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
    """Get the spectral resolution function.
    """
    sigma = e * energy_res(e)

    if sigma == 0:
        if hasattr(ep, "__len__"):
            return np.zeros(ep.shape)
        else:
            return 0.0
    else:
        return (
            1.0
            / np.sqrt(2.0 * np.pi * sigma ** 2)
            * np.exp(-(ep - e) ** 2 / (2.0 * sigma ** 2))
        )


def convolved_spectrum_fn(
    e_min, e_max, e_cm, energy_res, spec_fn=None, line_fn=None, n_pts=1000
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
    e_cm : float
        Center of mass energy for DM annihilation.
    energy_res : float -> float
        The detector's energy resolution (Delta E / E) as a function of
        photon energy in MeV.
    spec_fn : np.array -> np.array
        Continuum spectrum function.
    line_fn : np.array -> np.array
        Function returning a dict with information about spectral lines.
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
        dnde_src = spec_fn(es_padded, e_cm)
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
    if line_fn is not None:
        for ch, line in line_fn(e_cm).items():
            dnde_conv += (
                line["bf"]
                * spec_res_fn(es, line["energy"], energy_res)
                * (2.0 if ch == "g g" else 1.0)
            )

    return InterpolatedUnivariateSpline(es, dnde_conv, k=1, ext="raise")
