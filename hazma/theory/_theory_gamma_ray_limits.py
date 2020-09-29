import numpy as np
from scipy import optimize
from scipy.interpolate import InterpolatedUnivariateSpline


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
        # Convolve the spectrum with the detector's spectral resolution
        e_min, e_max = measurement.e_lows[0], measurement.e_highs[-1]

        # Factor to convert dN/dE to Phi. Factor of 2 comes from DM not being
        # self-conjugate.
        if self.kind == "ann":
            f_dm = 2.0
            dm_flux_factor = (
                measurement.target.J
                * measurement.target.dOmega
                / (2.0 * f_dm * self.mx ** 2 * 4.0 * np.pi)
            )

            # TODO: this should depend on the target!
            e_cm = 2.0 * self.mx * (1.0 + 0.5 * measurement.target.vx ** 2)
            dnde_conv = self.total_conv_spectrum_fn(
                e_min, e_max, e_cm, measurement.energy_res
            )
        elif self.kind == "dec":
            # e_cm = self.mx
            dm_flux_factor = (
                measurement.target.D
                * measurement.target.dOmega
                / (self.mx * 4.0 * np.pi)
            )
            dnde_conv = self.total_conv_spectrum_fn(
                e_min, e_max, measurement.energy_res
            )

        def bin_lim(e_low, e_high, phi, sigma):
            """Subroutine to compute limit in a single bin."""
            # Compute integrated flux from DM annihilation
            phi_dm = dm_flux_factor * dnde_conv.integral(e_low, e_high)

            # If flux is finite and nonzero, set a limit
            if not np.isnan(phi_dm) and phi_dm > 0:
                # Compute maximum allow integrated flux
                phi_max = (
                    measurement.target.dOmega
                    * (e_high - e_low)
                    * (n_sigma * sigma + phi)
                )

                assert phi_max > 0

                # Find limit on <sigma v>
                return phi_max / phi_dm
            else:
                return np.inf

        # Compute limits for each bin
        sv_lims = []

        for (e_low, e_high, phi, sigma) in zip(
            measurement.e_lows,
            measurement.e_highs,
            measurement.fluxes,
            measurement.upper_errors,
        ):
            sv_lims.append(bin_lim(e_low, e_high, phi, sigma))

        if method == "1bin":
            # Return the most stringent limit
            return np.min(sv_lims)
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
        # Convolve the spectrum with the detector's spectral resolution
        e_min, e_max = A_eff.x[[0, -1]]

        if self.kind == "ann":
            # TODO: this should depend on the target!
            e_cm = 2.0 * self.mx * (1.0 + 0.5 * target.vx ** 2)
            dnde_conv = self.total_conv_spectrum_fn(e_min, e_max, e_cm, energy_res)
        elif self.kind == "dec":
            dnde_conv = self.total_conv_spectrum_fn(e_min, e_max, energy_res)

        # Insert appropriate prefactors to convert result to <sigma v>_tot. The
        # factor of 2 is from the DM not being self-conjugate.
        if self.kind == "ann":
            f_dm = 2.0  # TODO: refactor
            prefactor = 2.0 * f_dm * self.mx ** 2
        else:
            prefactor = self.mx

        prefactor *= (
            4.0
            * np.pi
            / (
                np.sqrt(T_obs * target.dOmega)
                * (target.J if self.kind == "ann" else target.D)
            )
        )

        # Integrating these gives the number of signal and background photons,
        # up to normalization
        integrand_S = self._get_product_spline(dnde_conv, A_eff, dnde_conv.get_knots())
        integrand_B = self._get_product_spline(
            bg_model.dPhi_dEdOmega, A_eff, dnde_conv.get_knots()
        )

        # Optimize energy window
        if e_grid is None:
            e_grid = np.geomspace(e_min, e_max, n_grid)

        snrs = []
        for i, e_low in enumerate(e_grid[:-1]):
            for e_high in e_grid[i + 1 :]:
                assert e_low < e_high, (e_low, ", ", e_high)
                I_S_val = integrand_S.integral(e_low, e_high)
                I_B_val = integrand_B.integral(e_low, e_high)
                snrs.append([e_low, e_high, I_S_val / np.sqrt(I_B_val)])

        snrs = np.stack(snrs)

        # Find best energy window
        try:
            bound = prefactor * n_sigma / np.nanmax(snrs[:, 2])
            return bound
        except:
            return np.inf
