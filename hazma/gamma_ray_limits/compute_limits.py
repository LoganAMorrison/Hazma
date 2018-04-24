from gamma_ray_limit_parameters import eASTROGAM_params, dSph_params
from ..parameters import neutral_pion_mass as mpi0
from scipy import optimize
from scipy.integrate import quad
import numpy as np


def __I_S(e_a, e_b, dN_dE_DM, exp_params):
    """Integrand required to compute number of photons from DM annihilations.
    """
    def integrand_S(e):
        return dN_dE_DM(e) * exp_params.A_eff(e)

    return quad(integrand_S, e_a, e_b)[0]


def __I_B(e_a, e_b, exp_params, target_params):
    """Integrand required to compute number of background photons"""
    def integrand_B(e):
        return target_params.dPhi_dEdOmega_B(e) * exp_params.A_eff(e)

    return quad(integrand_B, e_a, e_b)[0]


def __f_lim(e_ab, dN_dE_DM, exp_params, target_params):
    """Objective function for selecting energy window.
    """
    e_a = min(e_ab)
    e_b = max(e_ab)

    if e_a == e_b:
        return 0.
    else:
        return -__I_S(e_a, e_b, dN_dE_DM, exp_params) / \
                np.sqrt(__I_B(e_a, e_b, exp_params, target_params))


def compute_limit(dN_dE_DM, mx, self_conjugate=False, n_sigma=5.,
                  exp_params=eASTROGAM_params, target_params=dSph_params):
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
    delta_Omega : float
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
    e_a_min = max([dN_dE_DM.x[0], target_params.dPhi_dEdOmega_B.x[0]])
    e_b_max = min([mx, target_params.dPhi_dEdOmega_B.x[-1]])

    # Allowed range for energy window bounds
    e_bounds = [e_a_min, e_b_max]

    # Initial guesses for energy window lower bound
    e_a_0 = 0.5 * (e_b_max - e_a_min)
    e_b_0 = 0.75 * (e_b_max - e_a_min)

    # Optimize upper and lower bounds for energy window
    limit_obj = optimize.minimize(__f_lim,
                                  [e_a_0, e_b_0],
                                  bounds=2*[e_bounds],
                                  args=(dN_dE_DM, exp_params,
                                        target_params),
                                  method="L-BFGS-B",
                                  options={"ftol": 1e-3})

    # Factor to avoid double counting pairs of DM particles
    if self_conjugate:
        dm_factor = 2.
    else:
        dm_factor = 4.

    # Insert appropriate prefactors to convert result to <sigma v>_tot
    prefactor = 4. * np.pi * dm_factor * mx**2 / \
        (np.sqrt(exp_params.T_obs * target_params.delta_Omega) *
         target_params.J_factor)

    return prefactor * n_sigma / (-limit_obj.fun)
