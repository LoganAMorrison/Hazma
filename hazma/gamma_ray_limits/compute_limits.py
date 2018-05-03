from gamma_ray_limit_parameters import (A_eff_e_astrogam, T_obs_e_astrogam,
                                        dSph_params, dPhi_dEdOmega_B_default)
from scipy import optimize
from scipy.integrate import quad
import numpy as np


def __I_S(e_a, e_b, dN_dE_DM, A_eff, T_obs):
    """Integrand required to compute number of photons from DM annihilations.
    """
    def integrand_S(e):
        return dN_dE_DM(e) * A_eff(e)

    return quad(integrand_S, e_a, e_b)[0]


def __I_B(e_a, e_b, A_eff, T_obs, target_params, dPhi_dEdOmega_B):
    """Integrand required to compute number of background photons"""
    def integrand_B(e):
        return dPhi_dEdOmega_B(e) * A_eff(e)

    return quad(integrand_B, e_a, e_b)[0]


def __f_lim(e_ab, dN_dE_DM, A_eff, T_obs, target_params, dPhi_dEdOmega_B):
    """Objective function for selecting energy window.
    """
    e_a = min(e_ab)
    e_b = max(e_ab)

    if e_a == e_b:
        return 0.
    else:
        return -__I_S(e_a, e_b, dN_dE_DM, A_eff, T_obs) / \
                np.sqrt(__I_B(e_a, e_b, A_eff, T_obs, target_params,
                              dPhi_dEdOmega_B))


def compute_limit(dN_dE_DM, mx, self_conjugate=False, n_sigma=5.,
                  A_eff=A_eff_e_astrogam, T_obs=T_obs_e_astrogam,
                  target_params=dSph_params,
                  dPhi_dEdOmega_B=dPhi_dEdOmega_B_default):
    """Computes smallest value of <sigma v> detectable for given target and
    experiment parameters.

    Notes
    -----
    We define a signal to be detectable if

    .. math:: N_S / sqrt(N_B) >= n_\sigma,

    where :math:`N_S` and :math:`N_B` are the number of signal and background
    photons in the energy window of interest and :math:`n_\sigma` is the
    significance in number of standard deviations. Note that :math:`N_S \propto
    \langle \sigma v \rangle`. While the photon count statistics are properly
    taken to be Poissonian and using a confidence interval would be more
    rigorous, this procedure provides a good estimate and is simple to compute.

    Parameters
    ----------
    dN_dE_DM : float -> float
        Photon spectrum per dark matter annihilation as a function of photon
        energy
    mx : float
        Dark matter mass
    dPhi_dEdOmega_B : float -> float
        Background photon spectrum per solid angle as a function of photon
        energy
    self_conjugate : bool
        True if DM is its own antiparticle; false otherwise
    n_sigma : float
        Number of standard deviations the signal must be above the background
        to be considered detectable
    dOmega : float
        Angular size of observation region in sr
    J_factor : float
        J factor for target in MeV^2 / cm^5
    A_eff : float -> float
        Effective area of experiment in cm^2 as a function of photon energy
    T_obs : float
        Experiment's observation time in s

    Returns
    -------
    <sigma v>_tot : float
        Smallest detectable thermally averaged total cross section in cm^3 / s
    """
    # Make sure not to go outside the interpolators' ranges
    e_a_min = max([dN_dE_DM.x[0], dPhi_dEdOmega_B.x[0]])
    e_b_max = min([mx, dPhi_dEdOmega_B.x[-1]])

    # Allowed range for energy window bounds
    e_bounds = [e_a_min, e_b_max]

    # Initial guesses for energy window lower bound
    e_a_0 = 0.5 * (e_b_max - e_a_min)
    e_b_0 = 0.75 * (e_b_max - e_a_min)

    # Optimize upper and lower bounds for energy window
    limit_obj = optimize.minimize(__f_lim,
                                  [e_a_0, e_b_0],
                                  bounds=2*[e_bounds],
                                  args=(dN_dE_DM, A_eff, T_obs, target_params,
                                        dPhi_dEdOmega_B),
                                  method="L-BFGS-B",
                                  options={"ftol": 1e-3})

    # Factor to avoid double counting pairs of DM particles
    if self_conjugate:
        dm_factor = 1.
    else:
        dm_factor = 2.

    # Insert appropriate prefactors to convert result to <sigma v>_tot
    prefactor = 2. * 4. * np.pi * dm_factor * mx**2 / \
        (np.sqrt(T_obs * target_params.dOmega) *
         target_params.J_factor)

    return prefactor * n_sigma / (-limit_obj.fun)
