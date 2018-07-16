"""
Functions to compute CMB constraints on DM models.

See  for details.
"""
from scipy.interpolate import interp1d
from scipy.integrate import quad
import numpy as np
from pkg_resources import resource_filename

# Get paths to files inside the module
f_eff_ep_rf = resource_filename(__name__, "cmb_data/f_eff_ep.dat")
f_eff_g_rf = resource_filename(__name__, "cmb_data/f_eff_g.dat")

# Load f_eff^{e+ e-}
f_eff_ep_data = np.loadtxt(f_eff_ep_rf, delimiter=",").T
f_eff_ep = interp1d(f_eff_ep_data[0] / 1.0e6, f_eff_ep_data[1])  # eV -> MeV

# Load f_eff^{e+ e-}
f_eff_g_data = np.loadtxt(f_eff_g_rf, delimiter=",").T
f_eff_g = interp1d(f_eff_g_data[0] / 1.0e6, f_eff_g_data[1])  # eV -> MeV


def vx_cmb(mx, x_kd):
    """Computes the DM relative velocity at CMB.

    Notes
    -----
    This is equation 28 from arXiv:1506.03811 [hep-ph].

    Parameters
    ----------
    mx : float
        Dark matter mass in MeV.
    x_kd: float
        T_kd / m_x, where T_kd is the dark matter's kinetic decoupling
        temperature.

    Returns
    -------
    v_x : float
        The DM relative velocity at the time of CMB formation.
    """
    temp_cmb = 0.235  # eV

    return 2.0e-4 * temp_cmb / mx * np.sqrt(1.0e-4 / x_kd)


# TODO: moving this into theory would make it much cleaner.

def f_eff(spec_fn, line_fn, pos_spec_fn, pos_line_fn, mx, x_kd=1.0e-4):
    """Computes f_eff(m_x) for DM annihilations.

    Notes
    -----
    This is equation 2 from arXiv:1506.03811 [hep-ph].

    Parameters
    ----------
    spec_fn : np.array, float -> np.array
        A function taking an array of photon energies and the DM's center of
        mass energy which returns the photon spectrum.
    line_fn : float -> dict(dict)
        A function which takes the DM's center of mass energy and returns a
        dict whose keys are the names of channels producing monochromatic gamma
        ray lines. The values of this dict are dicts with keys "energy" and
        "bf", which are the energy and branching fraction for the line.
    pos_spec_fn : np.array, float -> np.array
        A function taking an array of positron energies and the DM's center of
        mass energy which returns the positron spectrum.
    pos_line_fn : float -> dict(dict)
        A function which takes the DM's center of mass energy and returns a
        dict whose keys are the names of channels producing monochromatic
        positrons. The values of this dict are dicts with keys "energy" and
        "bf", which are the energy and branching fraction for the positron
        line.
    mx : float
        Dark matter mass in MeV.
    x_kd: float
        T_kd / m_x, where T_kd is the dark matter's kinetic decoupling
        temperature.

    Returns
    -------
    f_eff : float
        f_eff for dark matter annihilation.
    """
    # Center of mass energy
    e_cm = 2. * mx * (1. + 0.5 * vx_cmb(mx, x_kd)**2)

    # Lower bound on integrals
    e_min_g = f_eff_g.x[0]
    # Lower bound on integrals
    e_min_ep = f_eff_ep.x[0]

    # Continuum contributions from photons. Create an interpolator to avoid
    # recomputing spectrum.
    e_gams = np.logspace(np.log10(e_min_g), np.log10(mx), 1000)
    dnde_tot = spec_fn(e_gams, e_cm)
    spec_interp = interp1d(e_gams, dnde_tot, bounds_error=False, fill_value=0.)

    def g_integrand(e):
        return e * spec_interp(e) * f_eff_g(e)

    f_eff_g_dm = quad(g_integrand, e_min_g, mx, epsabs=0, epsrel=1e-3)[0] \
        / (2.*mx)

    # Continuum contributions from e+ e-. Create an interpolator to avoid
    # recomputing positron spectrum.
    e_eps = np.logspace(np.log10(e_min_ep), np.log10(mx), 1000)
    dnde_ep_tot = pos_spec_fn(e_eps, e_cm)
    pos_spec_interp = interp1d(e_eps, dnde_ep_tot, bounds_error=False,
                               fill_value=0.)

    def ep_integrand(e):
        # Note the factor of 2
        return e * 2.*pos_spec_interp(e) * f_eff_ep(e)

    f_eff_ep_dm = quad(ep_integrand, e_min_ep, mx, epsabs=0, epsrel=1e-3)[0] \
        / (2.*mx)

    # Line contributions from photons
    lines = line_fn(e_cm)
    f_eff_g_line_dm = 0

    for ch, line in lines.iteritems():
        energy = line["energy"]
        # Make sure f_eff^g is defined at this energy
        if energy > e_min_g:
            bf = line["bf"]
            multiplicity = 2. if ch == "g g" else 1.

            f_eff_g_line_dm += (energy * bf * f_eff_g(energy) * multiplicity /
                                (2.*mx))

    # Line contributions from e+e-
    pos_lines = pos_line_fn(e_cm)
    f_eff_ep_line_dm = 0

    for ch, line in pos_lines.iteritems():
        energy = line["energy"]
        # Make sure f_eff^ep is defined at this energy
        if energy > e_min_ep:
            bf = line["bf"]
            multiplicity = 2. if ch == "e e" else 1.

            f_eff_ep_line_dm += (energy * bf * f_eff_ep(energy) * multiplicity
                                 / (2.*mx))

    return f_eff_g_dm + f_eff_ep_dm + f_eff_g_line_dm + f_eff_ep_line_dm


def cmb_limit(mx, f_eff):
    """Computes the CMB limit on <sigma v>.

    Notes
    -----
    We use the constraint from the Planck collaboration:
        f_eff <sigma v> / m_x < 4.1e-31 cm^3 s^-1 MeV^-1

    Parameters
    ----------
    mx : float
        Dark matter mass in MeV.
    f_eff : float
        Efficiency with which energy is deposited into the CMB by DM
        annihilations.

    Returns
    -------
    <sigma v> : float
        Upper bound on <sigma v>.
    """
    return 4.1e-31 * mx / f_eff
