from scipy import optimize
from scipy.integrate import quad
from scipy.interpolate import interp1d
import numpy as np


def __I_S(e_a, e_b, dnde, A_eff, T_obs):
    """Integrand required to compute number of photons from DM annihilations.
    """
    def integrand_S(e):
        return dnde(e) * A_eff(e)

    return quad(integrand_S, e_a, e_b)[0]


def __I_B(e_a, e_b, A_eff, T_obs, target_params, bg_model):
    """Integrand required to compute number of background photons"""
    def integrand_B(e):
        return bg_model.dPhi_dEdOmega(e) * A_eff(e)

    return quad(integrand_B, e_a, e_b)[0]


def __f_lim(e_ab, dnde, A_eff, T_obs, target_params, bg_model):
    """Objective function for selecting energy window.
    """
    e_a = min(e_ab)
    e_b = max(e_ab)

    if e_a == e_b:
        return 0.
    else:
        return -__I_S(e_a, e_b, dnde, A_eff, T_obs) / \
                np.sqrt(__I_B(e_a, e_b, A_eff, T_obs, target_params, bg_model))


def unbinned_limit(e_gams, dndes, line_es, line_bfs, mx, self_conjugate, A_eff,
                   energy_res, T_obs, target_params, bg_model, n_sigma=5.):
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
        energy. Note that this spectrum must be defined over the whole domain
        of A_eff.
    mx : float
        Dark matter mass
    dPhi_dEdOmega_B : float -> float
        Background photon spectrum per solid angle as a function of photon
        energy. Note that the model must be defined over the whole domain of
        A_eff.
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
    # Convolve the spectrum with the detector's spectral resolution
    dnde_det = get_detected_spectrum(e_gams, dndes, line_es, line_bfs,
                                     energy_res)

    # Allowed range for energy window bounds
    e_min, e_max = A_eff.x[[0, -1]]

    # Initial guesses for energy window lower bound
    e_a_0 = 0.5 * (e_max - e_min)
    e_b_0 = 0.75 * (e_max - e_min)

    # Optimize upper and lower bounds for energy window
    limit_obj = optimize.minimize(__f_lim,
                                  [e_a_0, e_b_0],
                                  bounds=2*[[e_min, e_max]],
                                  args=(dnde_det, A_eff, T_obs, target_params,
                                        bg_model),
                                  method="L-BFGS-B",
                                  options={"ftol": 1e-3})

    # Insert appropriate prefactors to convert result to <sigma v>_tot
    prefactor = (2.*4.*np.pi * (1. if self_conjugate else 2.) * mx**2 /
                 (np.sqrt(T_obs * target_params.dOmega) * target_params.J))

    return prefactor * n_sigma / (-limit_obj.fun)


def binned_limit(e_gams, dndes, line_es, line_bfs, mx, self_conjugate,
                 measurement, n_sigma=2.):
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
        energy. Note that this spectrum must be defined over the whole energy
        range covered by measurement.bins.
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
    # Factor to convert dN/dE to Phi
    dm_flux_factor = (measurement.target.J * measurement.target.dOmega /
                      (2.*4.*np.pi * (1. if self_conjugate else 2.) * mx**2))

    # Keep track of <sigma v> limit for each bin
    sv_lims = [np.inf]

    # Convolve the spectrum with the detector's spectral resolution
    dnde_det = get_detected_spectrum(e_gams, dndes, line_es, line_bfs,
                                     measurement.energy_res)

    # Loop over experiment's bins
    for i, ((bin_low, bin_high), phi, sigma) in \
        enumerate(zip(measurement.bins, measurement.fluxes,
                      measurement.upper_errors)):
        # Integrate DM spectrum to compute flux in this bin
        phi_dm = dm_flux_factor * quad(dnde_det, bin_low, bin_high)[0]

        if not np.isnan(phi_dm):
            assert phi_dm >= 0

            # Compute maximum allow flux
            phi_max = (measurement.target.dOmega * (bin_high - bin_low) *
                       (n_sigma * sigma + phi))

            assert phi_max > 0

            # Find limit on <sigma v>
            sv_lims.append(phi_max / phi_dm)
        else:
            sv_lims.append(np.inf)

    return np.min(sv_lims)


def get_detected_spectrum(e_gams, dndes, line_es, line_bfs, energy_res):
    """Convolves a DM annihilation spectrum with a detector's spectral
    resolution function.

    Parameters
    ----------
    e_gams : np.array
        An array of photon energies, in MeV.
    dndes : np.array
        The source spectrum at the energies in e_gams, in MeV^-1.
    line_es : np.array
        An array of energies at which the DM spectrum has monochromatic gamma
        ray lines, in MeV.
    line_bfs : np.array
        The branching fraction for DM to annihilate to the states producing
        lines with energies line_es.
    energy_res : float -> float
        The detector's energy resolution (Delta E / E) as a function of photon
        energy in MeV.

    Returns
    -------
    dnde_det : interp1d
        An interpolator giving the DM annihilation spectrum as seen by the
        detector. Given photon energies outside the range covered by e_gams,
        the interpolator will produce bounds_errors.
    """
    # Get the spectral resolution function, normalized to one
    def spec_res_fn(e):
        eps = energy_res(e)

        if eps == 0.:
            return np.zeros(e_gams.shape)
        else:
            srf = np.exp(-(e_gams - e)**2 / (2. * (energy_res(e) * e)**2))
            return srf / srf.sum()

    # Continuum contribution
    dndes_cont_det = np.array([np.dot(spec_res_fn(e), dndes) for e in e_gams])

    # Line contribution
    dndes_line_det = np.array([spec_res_fn(e) * bf for
                               e, bf in zip(line_es, line_bfs)]).sum(axis=0)

    return interp1d(e_gams, dndes_cont_det + dndes_line_det)
