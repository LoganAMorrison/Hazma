from gamma_ray_limit_parameters import eASTROGAM_params, dSph_params
from ..parameters import neutral_pion_mass as mpi0
from scipy.interpolate import interp1d
from scipy import optimize
from scipy.integrate import quad
import numpy as np


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
    # Initial guesses for energy window bounds
    e_a_0 = mpi0 / 2.
    e_b_0 = mx

    # Bounds on upper and lower limits for energy window
    e_a_bounds = [dN_dE_DM.x[0], mpi0 / 2.00001]
    e_b_bounds = [mpi0 / 1.99999, mx]

    # Integrand required to compute number of photons from DM annihilations
    def I_S(e_a, e_b):
        def integrand_S(e):
            return dN_dE_DM(e) * exp_params.A_eff(e)

        return quad(integrand_S, e_a, e_b)[0]

    # Integrand required to compute number of background photons
    def I_B(e_a, e_b):
        def integrand_B(e):
            return target_params.dPhi_dEdOmega_B(e) * exp_params.A_eff(e)

        return quad(integrand_B, e_a, e_b)[0]

    # Objective function for selecting energy window
    def f(e_ab):
        return -I_S(*e_ab) / np.sqrt(I_B(*e_ab))

    # Computes Jacobian of the objective function
    def df_dE(e_ab, bound):
        # df/dE_b and df/dE_a have the same form, up to a minus sign
        if bound == "a":
            e = e_ab[0]
            sign = -1.0
        elif bound == "b":
            e = e_ab[1]
            sign = 1.0

        I_S_val = I_S(*e_ab)
        I_B_val = I_B(*e_ab)
        # Evaluate spectra at upper or lower bound
        dN_dE_DM_val = dN_dE_DM(e)
        dPhi_dEdOmega_B_val = target_params.dPhi_dEdOmega_B(e)

        prefactor = exp_params.A_eff(e) / np.sqrt(I_B_val)

        return sign * prefactor * (dN_dE_DM_val - 0.5 * I_S_val / I_B_val *
                                   dPhi_dEdOmega_B_val)

    def jac(e_ab):
        return np.array([df_dE(e_ab, "a"), df_dE(e_ab, "b")])

    # Minimize the objective function to compute the limit
    limit_obj = optimize.minimize(f, [e_a_0, e_b_0], jac=jac,
                                  bounds=[e_a_bounds, e_b_bounds])

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
