from math import sqrt, pi
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.stats import chi2, norm

# from scipy.optimize import root_scalar


class TheoryGammaRayLimits:
    def _get_product_spline(self, f1, f2, grid, k=1, ext="raise"):
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
        return InterpolatedUnivariateSpline(grid, f1(grid) * f2(grid), k=k, ext=ext)

    def binned_limit(self, measurement, n_sigma=2.0, method="1bin"):
        r"""
        Determines the limit on :math:`<sigma v>` from gamma-ray data.

        We define a signal to be in conflict with the measured flux for bin
        :math:`i` for an experiment if

        .. math::

            \Phi_{\chi}^{(i)} > n_{\sigma} \sigma^{(i)} + \Phi^{(i)},

        where :math:`\Phi_\chi^{(i)}` is the integrated flux due to DM
        annihilations for the bin, :math:`\Phi^{(i)}` is the measured flux in
        the bin, :math:`\sigma^{(i)}` is size of the upper error bar for the
        bin and :math:`n_{\sigma} = 2` is the significance. The overall limit on
        :math:`\langle\sigma v\rangle` is computed by minimizing over the
        limits determined for each bin.

        Parameters
        ----------
        measurement : FluxMeasurement
            Information about the flux measurement and target.
        n_sigma : float
            See the notes for this function.

        Returns
        -------
        <sigma v>_tot : float
            Largest allowed thermally averaged total cross section in cm^3 / s

        """
        e_min, e_max = measurement.e_lows[0], measurement.e_highs[-1]

        # Factor to convert dN/dE to Phi. Factor of 2 comes from DM not being
        # self-conjugate.
        if self.kind == "ann":  # type: ignore
            f_dm = 2.0
            dm_flux_factor = (
                measurement.target.J
                * measurement.target.dOmega
                / (2.0 * f_dm * self.mx ** 2 * 4.0 * pi)  # type: ignore
            )

            # TODO: this should depend on the target!
            e_cm = (
                2.0 * self.mx * (1.0 + 0.5 * measurement.target.vx ** 2)  # type: ignore
            )
            dnde_conv = self.total_conv_spectrum_fn(  # type: ignore
                e_min, e_max, e_cm, measurement.energy_res
            )
        elif self.kind == "dec":  # type: ignore
            # e_cm = self.mx
            dm_flux_factor = (
                measurement.target.D
                * measurement.target.dOmega
                / (self.mx * 4.0 * pi)  # type: ignore
            )
            dnde_conv = self.total_conv_spectrum_fn(  # type: ignore
                e_min, e_max, measurement.energy_res
            )

        # Integrated flux (excluding <sigma v>) from DM processes in each bin
        Phi_dms_un = []
        for e_low, e_high in zip(measurement.e_lows, measurement.e_highs):
            Phi_dms_un.append(
                dm_flux_factor * dnde_conv.integral(e_low, e_high)  # type: ignore
            )
        Phi_dms_un = np.array(Phi_dms_un)

        if method == "1bin":
            # Maximum allowed integrated flux in each bin
            Phi_maxs = (
                measurement.target.dOmega
                * (measurement.e_highs - measurement.e_lows)
                * (n_sigma * measurement.upper_errors + measurement.fluxes)
            )

            # Return the most stringent limit
            sv_lims = Phi_maxs / Phi_dms_un
            sv_lims[np.isnan(sv_lims) | (Phi_dms_un <= 0)] = np.inf
            return np.min(sv_lims)
        elif method == "chi2":
            # Observed integrated fluxes
            Phi_obss = (
                measurement.target.dOmega
                * (measurement.e_highs - measurement.e_lows)
                * measurement.fluxes
            )
            # Errors on integrated fluxes
            Sigmas = (
                measurement.target.dOmega
                * (measurement.e_highs - measurement.e_lows)
                * measurement.upper_errors
            )

            chi2_obs = np.sum(np.maximum(Phi_dms_un - Phi_obss, 0) ** 2 / Sigmas ** 2)

            if chi2_obs == 0:
                return np.inf
            else:
                # Convert n_sigma to chi^2 critical value
                p_val = norm.cdf(n_sigma)
                chi2_crit = chi2.ppf(p_val, df=len(Phi_dms_un))
                return sqrt(chi2_crit / chi2_obs)
        else:
            raise NotImplementedError()

    def unbinned_limit(
        self,
        A_eff,
        energy_res,
        T_obs,
        target,
        bg_model,
        e_grid=None,
        n_grid=20,
        n_sigma=5.0,
        debug_msgs=False,
    ):
        r"""
        Computes smallest-detectable value of <sigma v> for given target and
        experiment parameters.

        We define a signal to be detectable if

        .. math::

            N_S / \sqrt{N_B} \geq n_{\sigma},

        where :math:`N_S` and :math:`N_B` are the number of signal and
        background photons in the energy window of interest and
        :math:`n_\sigma` is the significance in number of standard deviations.
        Note that :math:`N_S \propto \langle \sigma v \rangle`. While the
        photon count statistics are properly taken to be Poissonian and using a
        confidence interval would be more rigorous, this procedure provides a
        good estimate and is simple to compute. The energy window is chosen to
        maximize N_S/sqrt(N_B).

        Parameters
        ----------
        A_eff : float -> float
            Effective area of experiment in cm^2 as a function of photon
            energy.
        energy_res : float -> float
            The detector's energy resolution (:math:`\Delta E / E`) as a
            function of photon energy in MeV.
        T_obs : float
            Experiment's observation time in s
        target : TargetParams
            Object containing information about the observation target.
        bg_model : BackgroundModel
            Object representing a gamma ray background model.
        n_sigma : float
            Number of standard deviations the signal must be above the
            background to be considered detectable
        debug_msgs : bool
            If True, the energy window found by the optimizer will be printed.

        Returns
        -------
        <sigma v> : float
            Smallest-detectable thermally averaged total cross section in units
            of cm^3 / s.
        """
        debug_msgs  # To make linter shut up

        # Convolve the spectrum with the detector's spectral resolution
        e_min, e_max = A_eff.x[[0, -1]]

        if self.kind == "ann":  # type: ignore
            # TODO: this should depend on the target!
            e_cm = 2.0 * self.mx * (1.0 + 0.5 * target.vx ** 2)  # type: ignore
            dnde_conv = self.total_conv_spectrum_fn(  # type: ignore
                e_min, e_max, e_cm, energy_res
            )
        elif self.kind == "dec":  # type: ignore
            dnde_conv = self.total_conv_spectrum_fn(  # type: ignore
                e_min, e_max, energy_res
            )

        # Insert appropriate prefactors to convert result to <sigma v>_tot. The
        # factor of 2 is from the DM not being self-conjugate.
        if self.kind == "ann":  # type: ignore
            f_dm = 2.0  # TODO: refactor
            prefactor = 2.0 * f_dm * self.mx ** 2  # type: ignore
        else:
            prefactor = self.mx  # type: ignore

        prefactor *= (
            4.0
            * pi
            / (
                np.sqrt(T_obs * target.dOmega)
                * (target.J if self.kind == "ann" else target.D)  # type: ignore
            )
        )

        # Integrating these gives the number of signal and background photons,
        # up to normalization
        integrand_S = self._get_product_spline(
            dnde_conv,  # type: ignore
            A_eff,
            dnde_conv.get_knots(),  # type: ignore
        )
        integrand_B = self._get_product_spline(
            bg_model.dPhi_dEdOmega, A_eff, dnde_conv.get_knots()  # type: ignore
        )

        # Optimize energy window
        if e_grid is None:
            e_grid = np.geomspace(e_min, e_max, n_grid)

        snrs = []
        for i, e_low in enumerate(e_grid[:-1]):
            j = i + 1
            for e_high in e_grid[j:]:
                assert e_low < e_high, (e_low, ", ", e_high)
                I_S_val = integrand_S.integral(e_low, e_high)
                I_B_val = integrand_B.integral(e_low, e_high)
                snrs.append([e_low, e_high, I_S_val / np.sqrt(I_B_val)])

        snrs = np.stack(snrs)

        # Find best energy window
        try:
            bound = prefactor * n_sigma / np.nanmax(snrs[:, 2])
            return bound
        except Exception as e:
            print(f"Error in unbinned_limit: {repr(e)}")
            return np.inf
