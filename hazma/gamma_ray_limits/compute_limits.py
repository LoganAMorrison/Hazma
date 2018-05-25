from scipy import optimize
from scipy.integrate import quad, cumtrapz
from scipy.interpolate import interp1d
from scipy.stats import norm
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


def __f_jac(e_ab, dnde, A_eff, T_obs, target_params, bg_model):
    e_a = min(e_ab)
    e_b = max(e_ab)

    if e_a == e_b:
        return 0.
    else:
        I_S_val = __I_S(e_a, e_b, dnde, A_eff, T_obs)
        I_B_val = __I_B(e_a, e_b, A_eff, T_obs, target_params, bg_model)

        # Function value
        f_val = -I_S_val / np.sqrt(I_B_val)

        # Jacobian
        df_de_a = A_eff(e_a) * (-dnde(e_a) / np.sqrt(I_B_val) +
                                0.5 * I_S_val / I_B_val**1.5 *
                                bg_model.dPhi_dEdOmega(e_a))
        df_de_b = A_eff(e_b) * (dnde(e_b) / np.sqrt(I_B_val) -
                                0.5 * I_S_val / I_B_val**1.5 *
                                bg_model.dPhi_dEdOmega(e_b))
        jac_val = np.array([df_de_a, df_de_b]).T

        return f_val, jac_val


def unbinned_limit(spec_fn, line_fn, mx, self_conjugate, A_eff,
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
    # Compute the spectrum over a grid over photon energies so it's not
    # recomputed every time it's needed. Pad the grid to avoid edge effects
    # from the convolution.
    e_min, e_max = A_eff.x[[0, -1]]

    # TODO: this should depend on the target!
    e_cm = 2.*mx*(1. + 0.5*1e-6)  # v_x = Milky Way velocity dispersion

    # Convolve the spectrum with the detector's spectral resolution
    dnde_det = get_detected_spectrum(spec_fn, line_fn, e_min, e_max, e_cm,
                                     energy_res)

    # Energy at which spectrum peaks
    e_dnde_max = dnde_det.x[np.argmax(dnde_det.y)]

    # Choose initial guess for energy window bounds
    if e_dnde_max != e_min and e_dnde_max != e_max:
        # If there is a peak in the spectrum, include it in the initial
        # energy window
        e_a_0 = 10.**(0.5 * (np.log10(e_dnde_max) + np.log10(e_min)))
        e_b_0 = 10.**(0.5 * (np.log10(e_max) + np.log10(e_dnde_max)))
    else:
        # If the spectrum has no prominent features, the initial window
        # matters less
        e_a_0 = 0.25 * 10.**(0.5 * (np.log10(e_max) + np.log10(e_min)))
        e_b_0 = 0.75 * 10.**(0.5 * (np.log10(e_max) + np.log10(e_min)))

    # Optimize upper and lower bounds for energy window
    limit_obj = optimize.minimize(__f_lim,
                                  [e_a_0, e_b_0],
                                  bounds=2 * [[e_min, e_max]],
                                  args=(dnde_det, A_eff, T_obs,
                                        target_params, bg_model),
                                  method="L-BFGS-B",
                                  options={"ftol": 1e-3})

    # Insert appropriate prefactors to convert result to <sigma v>_tot
    prefactor = (2. * 4. * np.pi * (1. if self_conjugate else 2.) * mx**2 /
                 (np.sqrt(T_obs * target_params.dOmega) * target_params.J))

    assert -limit_obj.fun >= 0

    return prefactor * n_sigma / (-limit_obj.fun)


def binned_limit(spec_fn, line_fn, mx, self_conjugate, measurement,
                 n_sigma=2.):
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
                      (2. * 4. * np.pi * (1. if self_conjugate else 2.) *
                       mx**2))

    # TODO: this should depend on the target!
    e_cm = 2.*mx*(1. + 0.5*1e-6)  # v_x = Milky Way velocity dispersion

    # Keep track of <sigma v> limit for each bin
    sv_lims = [np.inf]

    # Convolve the spectrum with the detector's spectral resolution
    e_bin_min, e_bin_max = measurement.bins[0][0], measurement.bins[-1][1]
    dnde_det = get_detected_spectrum(spec_fn, line_fn, e_bin_min, e_bin_max,
                                     e_cm, measurement.energy_res)

    # Loop over experiment's bins
    for i, ((bin_low, bin_high), phi, sigma) in \
        enumerate(zip(measurement.bins, measurement.fluxes,
                      measurement.upper_errors)):
        # Integrate DM spectrum to compute flux in this bin
        phi_dm = dm_flux_factor * quad(dnde_det, bin_low, bin_high)[0]

        # If flux is finite and nonzero, set a limit using this bin
        if not np.isnan(phi_dm) and phi_dm > 0:
            # Compute maximum allow flux
            phi_max = (measurement.target.dOmega * (bin_high - bin_low) *
                       (n_sigma * sigma + phi))

            assert phi_max > 0

            # Find limit on <sigma v>
            sv_lims.append(phi_max / phi_dm)

    return np.min(sv_lims)


# Get the spectral resolution function
def spec_res_fn(ep, e, energy_res):
    sigma = e * energy_res(e)

    if sigma == 0:
        if hasattr(ep, '__len__'):
            return np.zeros(ep.shape)
        else:
            return 0.
    else:
        return norm.pdf(ep, loc=e, scale=sigma)


def find_nearest_idx(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def get_detected_spectrum(spec_fn, line_fn, e_min, e_max, e_cm, energy_res,
                          n_energies=250):
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
    # Compute source spectrum over a wide grid to avoid edge effects from the
    # convolution
    e_gams_padded = np.logspace(np.log10(e_min) - 1, np.log10(e_max) + 1,
                                n_energies)
    dnde_src = spec_fn(e_gams_padded, e_cm)

    # Energies at which to compute detected spectrum
    e_gams = np.logspace(np.log10(e_min), np.log10(e_max), n_energies)
    dnde_cont_det = np.zeros(e_gams.shape)

    # If continuum spectrum is zero, don't waste time on the convolution
    if not np.all(dnde_src == 0):
        def integral(e):  # performs the integration at the given photon energy
            integrand_vals = dnde_src*spec_res_fn(e_gams_padded, e, energy_res)
            return cumtrapz(integrand_vals, e_gams_padded)[-1]

        dnde_cont_det = np.vectorize(integral)(e_gams)

    # Line contribution
    lines = line_fn(e_cm)
    dnde_line_det = np.array([line["bf"] *
                              spec_res_fn(e_gams, line["energy"], energy_res) *
                              (2. if ch == "g g" else 1.)
                              for ch, line in lines.iteritems()]).sum(axis=0)

    return interp1d(e_gams, dnde_cont_det + dnde_line_det)
