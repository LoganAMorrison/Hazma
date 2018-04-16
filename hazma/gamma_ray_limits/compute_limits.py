from gamma_ray_limit_parameters import (J_factor_draco, delta_Omega_dSph,
                                        A_eff_e_ASTROGAM, T_obs_e_ASTROGAM)
from gamma_ray_background import dPhi_dEdOmega_B_default
from scipy.integrate import quad
import numpy as np


def compute_limit(dN_dE_DM,
                  mx,
                  e_gam_min,
                  e_gam_max,
                  dPhi_dEdOmega_B=dPhi_dEdOmega_B_default,
                  self_conjugate=False,
                  n_sigma=5.,
                  delta_Omega=delta_Omega_dSph,
                  J_factor=J_factor_draco,
                  A_eff=A_eff_e_ASTROGAM,
                  T_obs=T_obs_e_ASTROGAM):
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
    prefactor = T_obs * A_eff * delta_Omega

    # Factor to avoid double counting pairs of DM particles
    if self_conjugate:
        dm_factor = 2.
    else:
        dm_factor = 4.

    # Number of background photons
    N_gam_B = prefactor * quad(dPhi_dEdOmega_B, e_gam_min, e_gam_max)[0]

    # Number of signal photons
    dm_prefactor = prefactor * J_factor / (4. * np.pi * dm_factor * mx**2)
    N_gam_S = dm_prefactor * quad(dN_dE_DM, e_gam_min, e_gam_max)[0]

    return n_sigma * np.sqrt(N_gam_B) / N_gam_S
