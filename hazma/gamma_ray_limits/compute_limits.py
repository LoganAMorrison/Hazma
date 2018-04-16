from gamma_ray_limit_parameters import (ExperimentParams, TargetParams,
                                        eASTROGAM_params, dSph_params)
from scipy.integrate import quad
import numpy as np


def compute_limit(dN_dE_DM, mx, e_gam_min, e_gam_max, self_conjugate=False,
                  n_sigma=5., exp_params=eASTROGAM_params,
                  target_params=dSph_params):
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
    e_gam_min : float
        Lower bound for energy window used to set limit
    e_gam_max : float
        Upper bound for energy window used to set limit
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
    A_eff : float
        Effective area of experiment in cm^2
    T_obs : float
        Experiment's observation time in s

    Returns
    -------
    <sigma v> : float
        Smallest detectable thermally averaged total cross section in cm^3 / s
    """
    # Prefactor for converting integrated spectrum to photon counts
    prefactor = exp_params.T_obs * exp_params.A_eff * target_params.delta_Omega

    # Factor to avoid double counting pairs of DM particles
    if self_conjugate:
        dm_factor = 2.
    else:
        dm_factor = 4.

    # Number of background photons
    N_gam_B = prefactor * quad(target_params.dPhi_dEdOmega_B, e_gam_min,
                               e_gam_max)[0]

    # Number of signal photons
    dm_prefactor = prefactor * target_params.J_factor / (4. * np.pi * dm_factor
                                                         * mx**2)
    N_gam_S = dm_prefactor * quad(dN_dE_DM, e_gam_min, e_gam_max)[0]

    return n_sigma * np.sqrt(N_gam_B) / N_gam_S
