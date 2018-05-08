from gamma_ray_limit_parameters import background_model_range
from scipy import optimize
from scipy.integrate import quad
import numpy as np


def __I_S(e_a, e_b, dnde, A_eff, T_obs):
    """Integrand required to compute number of photons from DM annihilations.
    """
    def integrand_S(e):
        return dnde(e) * A_eff(e)

    return quad(integrand_S, e_a, e_b)[0]


def __I_B(e_a, e_b, A_eff, T_obs, target_params, dPhi_dEdOmega_B):
    """Integrand required to compute number of background photons"""
    def integrand_B(e):
        return dPhi_dEdOmega_B(e) * A_eff(e)

    return quad(integrand_B, e_a, e_b)[0]


def __f_lim(e_ab, dnde, A_eff, T_obs, target_params, dPhi_dEdOmega_B):
    """Objective function for selecting energy window.
    """
    e_a = min(e_ab)
    e_b = max(e_ab)

    if e_a == e_b:
        return 0.
    else:
        return -__I_S(e_a, e_b, dnde, A_eff, T_obs) / \
                np.sqrt(__I_B(e_a, e_b, A_eff, T_obs, target_params,
                              dPhi_dEdOmega_B))


def unbinned_limit(dnde, mx, self_conjugate, A_eff, T_obs,
                   target_params, dPhi_dEdOmega_B, n_sigma=5.):
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
    dnde : float -> float
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
    e_a_min = max([dnde.x[0], background_model_range[0]])
    e_b_max = min([mx, background_model_range[1]])

    # Allowed range for energy window bounds
    e_bounds = [e_a_min, e_b_max]

    # Initial guesses for energy window lower bound
    e_a_0 = 0.5 * (e_b_max - e_a_min)
    e_b_0 = 0.75 * (e_b_max - e_a_min)

    # Optimize upper and lower bounds for energy window
    limit_obj = optimize.minimize(__f_lim,
                                  [e_a_0, e_b_0],
                                  bounds=2*[e_bounds],
                                  args=(dnde, A_eff, T_obs, target_params,
                                        dPhi_dEdOmega_B),
                                  method="L-BFGS-B",
                                  options={"ftol": 1e-3})

    # Factor to avoid double counting pairs of DM particles
    if self_conjugate:
        f_dm = 1.
    else:
        f_dm = 2.

    # Insert appropriate prefactors to convert result to <sigma v>_tot
    prefactor = 2. * 4. * np.pi * f_dm * mx**2 / \
        (np.sqrt(T_obs * target_params.dOmega) *
         target_params.J)

    return prefactor * n_sigma / (-limit_obj.fun)


def binned_limit(dnde, mx, self_conjugate, measurement, n_sigma=2.):
    """Determines the limit on <sigma v> from data for a given DM spectrum.

    Notes
    -----
    We define a signal to be in conflict for the measured flux for the
    :math:`i`th bin for an experiment if

    .. math:: \Phi_\chi^{(i)} > n_\sigma \sigma^{(i)} + \Phi^{(i)},

    where :math:`\Phi_\chi^{(i)}` is the flux due to DM annihilations for the
    bin, :math:`\Phi^{(i)}` is the measured flux in the bin,
    :math:`\sigma^{(i)}` is size of the upper error bar for the bin and
    :math:`n_\sigma = 2` is the significance. The overall limit on
    :math:`\langle\sigma v\rangle` is computed by minimizing over the limits
    determined for each bin.

    Parameters
    ----------
    dnde : float -> float
        Photon spectrum per dark matter annihilation as a function of photon
        energy
    mx : float
        Dark matter mass
    self_conjugate : bool
        True if DM is its own antiparticle; false otherwise
    measurement : FluxMeasurement
        Information about the flux measurement and target.
    n_sigma : float
        See the notes for this function.

    Returns
    -------
    <sigma v>_tot : float
        Largest allowed thermally averaged total cross section in cm^3 / s
    """
    # Factor to avoid double counting pairs of DM particles
    if self_conjugate:
        f_dm = 1.
    else:
        f_dm = 2.

    # Factor to convert dN/dE to Phi
    dm_flux_factor = measurement.target.J * measurement.target.dOmega / \
        (2. * 4. * np.pi * f_dm * mx**2)

    # Keep track of <sigma v> limit for each bin
    sv_lims = [np.inf]  # make sure to return SOMETHING

    # Check whether interpolator and bins have any overlap
    if (dnde.x[0] > measurement.bins[-1][-1] or
            dnde.x[-1] < measurement.bins[0][0]):
        return np.inf

    # Loop over experiment's bins
    for i, ((bin_low, bin_high), phi, sigma) in \
        enumerate(zip(measurement.bins, measurement.fluxes,
                      measurement.upper_errors)):
        # Make sure not to go out of the DM spectrum interpolator's range
        if bin_low > dnde.x[0] and bin_high < dnde.x[-1]:
            # Integrate DM spectrum to compute flux in this bin
            phi_dm = dm_flux_factor * quad(dnde, bin_low, bin_high)[0]

            assert phi_dm >= 0

            if phi_dm != 0:
                # Compute maximum allow flux
                phi_max = measurement.target.dOmega * (bin_high - bin_low) * \
                    (n_sigma * sigma + phi)

                assert phi_max > 0

                # Find limit on <sigma v>
                sv_lims.append(phi_max / phi_dm)
            else:
                sv_lims.append(np.inf)

    return np.min(sv_lims)
