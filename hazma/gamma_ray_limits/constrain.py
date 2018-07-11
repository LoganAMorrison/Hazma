from scipy.integrate import trapz
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.stats import norm
import numpy as np


def __get_product_spline(f1, f2, grid, k=1, ext="raise"):
    """Returns a spline representing the product of two functions.

    Parameters
    ----------
    f1, f2 : float -> float
        The two functions to multiply.
    grid : numpy.array
        The grid used to create the product spline.
    k : int
        Degree of returned spline, 1 <= k <= 5.
    ext : string or int
        Extrapolation method. See documentation for
        `InterpolatedUnivariateSpline`.

    Returns
    -------
    spl : InterpolatedUnivariateSpline
        A degree k spline created using grid for the x array and
        f1(grid)*f2(grid) for the y array, with the specified extrapolation
        method.
    """
    return InterpolatedUnivariateSpline(grid, f1(grid)*f2(grid), k=k, ext=ext)


def binned_limit(theory, measurement, n_sigma=2.):
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
    # TODO: this should depend on the target!
    v_x = 1.0e-3  # v_x = Milky Way velocity dispersion
    e_cm = 2.*mx*(1. + 0.5*v_x**2)

    # Factor to convert dN/dE to Phi
    sv = theory.cross_sections(e_cm)["total"] * v_x
    dm_flux_factor = (measurement.target.J * measurement.target.dOmega /
                      (2. * 4. * np.pi * (1. if self_conjugate else 2.) *
                       mx**2))

    # Convolve the spectrum with the detector's spectral resolution
    e_bin_min, e_bin_max = measurement.bins[0][0], measurement.bins[-1][1]
    dnde_det = get_detected_spectrum(spec_fn, line_fn, e_bin_min, e_bin_max,
                                     e_cm, measurement.energy_res, 1000)

    def bin_lim(e_bin, phi, sigma):  # computes limit in a bin
        bin_low, bin_high = e_bin

        # Integrate DM spectrum to compute flux in this bin
        phi_dm = sv * dm_flux_factor * dnde_det.integral(bin_low, bin_high)

        # If flux is finite and nonzero, set a limit using this bin
        if not np.isnan(phi_dm) and phi_dm > 0:
            # Compute maximum allow flux
            phi_max = (measurement.target.dOmega * (bin_high - bin_low) *
                       (n_sigma * sigma + phi))

            assert phi_max > 0

            # Find limit on <sigma v>
            return phi_max / phi_dm
        else:
            return np.inf

    sv_lims = [bin_lim(e_bin, phi, sigma)
               for (e_bin, phi, sigma) in zip(measurement.bins,
                                              measurement.fluxes,
                                              measurement.upper_errors)]

    return np.min(sv_lims)


def spec_res_fn(ep, e, energy_res):
    # Get the spectral resolution function
    sigma = e * energy_res(e)

    if sigma == 0:
        if hasattr(ep, '__len__'):
            return np.zeros(ep.shape)
        else:
            return 0.
    else:
        return norm.pdf(ep, loc=e, scale=sigma)


def get_detected_spectrum(spec_fn, line_fn, e_min, e_max, e_cm, energy_res,
                          n_pts=1000):
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
                                n_pts)
    dnde_src = spec_fn(e_gams_padded, e_cm)

    # Energies at which to compute detected spectrum
    e_gams = np.logspace(np.log10(e_min), np.log10(e_max), n_pts)
    dnde_cont_det = np.zeros(e_gams.shape)

    # If continuum spectrum is zero, don't waste time on the convolution
    if not np.all(dnde_src == 0):
        def integral(e):  # performs the integration at the given photon energy
            spec_res_fn_vals = spec_res_fn(e_gams_padded, e, energy_res)
            integrand_vals = dnde_src * spec_res_fn_vals / \
                trapz(spec_res_fn_vals, e_gams_padded)

            return trapz(integrand_vals, e_gams_padded)

        dnde_cont_det = np.vectorize(integral)(e_gams)

    # Line contribution
    lines = line_fn(e_cm)
    dnde_line_det = np.array([line["bf"] *
                              spec_res_fn(e_gams, line["energy"], energy_res) *
                              (2. if ch == "g g" else 1.)
                              for ch, line in lines.iteritems()]).sum(axis=0)

    # return interp1d(e_gams, dnde_cont_det + dnde_line_det)
    return InterpolatedUnivariateSpline(e_gams, dnde_cont_det + dnde_line_det,
                                        k=1, ext="raise")
