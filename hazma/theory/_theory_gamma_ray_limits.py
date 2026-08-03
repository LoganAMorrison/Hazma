from typing import Callable, Any, Optional, List, NamedTuple
from collections import OrderedDict
from math import pi, sqrt
import logging
import functools

import numpy as np
import numpy.typing as npt
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.stats import chi2, norm

from hazma.flux_measurement import FluxMeasurement
from hazma.background_model import ParametricBackgroundModel, BackgroundModel
from hazma.target_params import TargetParams

EnergyResolution = Callable[[Any], Any]
EffectiveArea = Callable[[Any], Any]


def _get_product_spline(f1, f2, grid, k=1, ext="raise"):
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


def constrain_one_bin(phi, measurement: FluxMeasurement, n_sigma: float):
    """Compute test statistic using maximum flux in single bin.

    Parameters
    ----------
    phi: array
        Array of the predicted fluxes in each bin.
    measurement: FluxMeasurement
        Measurement to compare with.
    n_sigma:
        Number of standard deviations.

    Returns
    -------
    sv: float
        Maximum allowed annihilation cross section.
    """
    dw = measurement.target.dOmega
    de = measurement.e_highs - measurement.e_lows
    phi_obs = measurement.fluxes
    dphi_obs = n_sigma * measurement.upper_errors

    # Maximum allowed integrated flux in each bin
    phi_max = dw * de * (phi_obs + dphi_obs)

    # Return the most stringent limit
    sv_lims = np.ones_like(phi) * np.inf
    mask = phi > 0.0
    if np.any(mask):
        sv_lims[mask] = phi_max[mask] / phi[mask]

    return np.min(sv_lims)


def constrain_chi_squared(phi, measurement: FluxMeasurement, n_sigma: float):
    dw = measurement.target.dOmega
    de = measurement.e_highs - measurement.e_lows

    # Observed integrated fluxes
    phi_obs = dw * de * measurement.fluxes
    # Errors on integrated fluxes
    uncertainty = dw * de * measurement.upper_errors

    chi2_obs = np.sum(np.maximum(phi - phi_obs, 0) ** 2 / uncertainty**2)

    if chi2_obs == 0:
        return np.inf

    # Convert n_sigma to chi^2 critical value
    p_val = norm.cdf(n_sigma)
    chi2_crit = chi2.ppf(p_val, df=len(phi))
    return sqrt(chi2_crit / chi2_obs)


def _differential_background_flux(
    background_model: ParametricBackgroundModel,
    energies,
    target: TargetParams,
    k: int = 1,
    ext: str = "raise",
):
    """
    Compute the expected differential background flux that would be observed by
    a given telescope looking a a target assuming a background model.

    Parameters
    ----------
    background_model: BackgroundModel
        Model for the background flux.
    energies: array
        Array of energies used to construct the interpolating function.
    effective_area: Callable
        Effective area of the observing telescope.
    target: TargetParams
        Target the telescope is observing.
    tobs: float
        Observing time in seconds.
    k: int, optional
        Order of the underlying interpolating function. Default is 1 (linear).
    ext: str, optional
        String specifying how the interpolating function should handle energies
        outside the interval of the input energies. Default is 'raise'.

    Returns
    -------
    dphi/dE: InterpolatedUnivariateSpline
        Interpolating function used to compute the expected differential
        background flux.
    """

    def phi_b(e):
        return background_model.dPhi_dEdOmega(e) * target.dOmega

    return InterpolatedUnivariateSpline(energies, phi_b(energies), k=k, ext=ext)


def _differential_signal_flux_prefactor(
    model, target: TargetParams, self_conjugate: bool = False
) -> float:
    def errmsg(name, var):
        return (
            f"Found None for {name} `{var}` of input target."
            f"The {name} is required to compute DM flux."
        )

    mx = model.mx
    rfac: float = 1.0  # rate prefactor

    assert target.dOmega is not None, errmsg("solid angle", "dOmega")
    dw: float = target.dOmega

    if model.kind == "ann":
        assert target.J is not None, errmsg("J-factor", "J")
        f_dm = 1.0 if self_conjugate else 2.0
        jfac = target.J
        a = 2
        rfac = 1.0 / (2.0 * f_dm)

    elif model.kind == "dec":
        assert target.D is not None, errmsg("J-factor", "D")
        jfac = target.D
        a = 1
    else:
        raise ValueError(f"Encountered model with invalid `kind`: {model.kind}")

    return dw / (4.0 * np.pi * mx**a) * jfac * rfac


def _differential_signal_flux(
    model,
    energies,
    energy_res: EnergyResolution,
    target: TargetParams,
    self_conjugate: bool = False,
    scale: Optional[float] = None,
    vx: float = 1e-3,
    k: int = 1,
    ext: str = "raise",
):
    """
    Compute the expected differential signal flux that would be observed by
    a given telescope looking a a target assuming a dark matter model.

    Parameters
    ----------
    model: Theory
        Dark matter model.
    energies: array
        Array of energies used to construct the interpolating function.
    effective_area: Callable
        Effective area of the observing telescope.
    energy_res: Callable
        Energy resolution of the telescope.
    target: TargetParams
        Target the telescope is observing.
    tobs: float
        Observing time in seconds.
    self_conjugate: bool, optional
        If True, DM is assumed to be self-conjugate. Default is False.
    scale: float
        Central value of the annihilation cross-section. Default is 3e-26.
    vx: float, optional
        Dark matter velocity. Default is 1e-3.
    k: int, optional
        Order of the underlying interpolating function. Default is 1 (linear).
    ext: str, optional
        String specifying how the interpolating function should handle energies
        outside the interval of the input energies. Default is 'raise'.

    Returns
    -------
    dphi/dE: InterpolatedUnivariateSpline
        Interpolating function used to compute the expected differential
        signal flux.
    """
    emin = np.min(energies)
    emax = np.max(energies)

    if model.kind == "ann":
        cme = 2.0 * model.mx * (1.0 + 0.5 * vx**2)
        dnde_conv = model.total_conv_spectrum_fn(emin, emax, cme, energy_res)
    elif model.kind == "dec":
        dnde_conv = model.total_conv_spectrum_fn(emin, emax, energy_res)
    else:
        raise ValueError(f"Encountered model with invalid `kind`: {model.kind}")

    scale = 1.0 if scale is None else scale
    prefactor = scale * _differential_signal_flux_prefactor(
        model=model, target=target, self_conjugate=self_conjugate
    )

    def phi_s(e):
        return prefactor * dnde_conv(e)

    return InterpolatedUnivariateSpline(energies, phi_s(energies), k=k, ext=ext)


class FisherResults(NamedTuple):
    fisher_matrix: np.matrix
    params: OrderedDict[str, float]

    def limit(self, sigma, key: str = "rate", log: bool = False) -> float:
        """Compute the limit on the parameter corresponding to the specified key."""
        idx = {k: i for i, k in enumerate(self.params.keys())}[key]
        try:
            inv = np.linalg.inv(self.fisher_matrix)
        except np.linalg.LinAlgError as e:
            if log:
                logging.warning(f"LinAlgError: {str(e)}.")
            n = len(self.params)
            inv = np.full((n, n), np.nan)

        return norm.ppf(norm.cdf(sigma)) * np.sqrt(inv[idx, idx])


def fisher(
    model,
    effective_area: EffectiveArea,
    energy_res: EnergyResolution,
    target: TargetParams,
    background_model: ParametricBackgroundModel,
    tobs: float,
    vx: float = 1e-3,
    n_grid: int = 20,
    e_grid: Optional[npt.NDArray[np.float64]] = None,
    k: int = 1,
    ext: str = "raise",
) -> FisherResults:
    """Compute the Fisher matrix.

    Parameters
    ----------
    model: Theory
        Dark matter model.
    effective_area: Callable
        Effective area of the observing telescope.
    energy_res: Callable
        Energy resolution of the telescope.
    target: TargetParams
        Target the telescope is observing.
    background_model: BackgroundModel
        Model for the background flux.
    tobs: float
        Observing time in seconds.
    vx: float, optional
        Dark matter velocity. Default is 1e-3.
    n_sigma: int, optional
        Number of standard deviations used for confidence interval. Default is
        5.
    n_grid: int, optional
        Number of grid points used for generation of interpolating splines.
        Default is 20. Ignored if `e_grid` is not None.
    e_grid: array, optional
        Energy grid used for generation of interpolating splines. Default is
        None.
    k: int, optional
        Order of the underlying interpolating function. Default is 1 (linear).
    ext: str, optional
        String specifying how the interpolating function should handle energies
        outside the interval of the input energies. Default is 'raise'.

    Returns
    -------
    results: FisherResults
        NamedTuple containing:

        fisher: np.matrix
            Full Fisher matrix. The limit is obtained using: results.limit().
        params: Dict[str, float]
            Dictionary of the fiducial parameters.
    """
    if model.kind == "ann":
        scale = 1.0  # 3e-26
    elif model.kind == "dec":
        # TODO: Come up with a better natural scale for decaying DM.
        scale = 1.0  # 1e-24
    else:
        raise ValueError(f"Encountered model with invalid `kind`: {model.kind}")

    e_min, e_max = effective_area.x[[0, -1]]
    if e_grid is not None:
        energies = e_grid
    else:
        energies = np.geomspace(e_min, e_max, n_grid)

    phi_s = _differential_signal_flux(
        model=model,
        energies=energies,
        energy_res=energy_res,
        target=target,
        vx=vx,
        scale=scale,
        k=k,
        ext=ext,
    )

    params = OrderedDict(background_model.params.copy())
    params["rate"] = 0.0
    params.move_to_end("rate", last=False)

    assert target.dOmega is not None
    dOmega: float = target.dOmega

    def phi_b(energy):
        return background_model.dPhi_dEdOmega(energy) * dOmega

    def dphi_b(energy, key):
        return background_model.derivatives(energy)[key]

    dphi = {key: functools.partial(dphi_b, key=key) for key in background_model.params}
    dphi["rate"] = lambda e: phi_s(e)  # type: ignore

    def fisher_matrix_element(k1, k2):
        dphi1 = dphi[k1](energies)
        dphi2 = dphi[k2](energies)
        phib = phi_b(energies)
        aeff = effective_area(energies)

        integrand = dphi1 * dphi2 / phib * aeff * tobs
        interp = InterpolatedUnivariateSpline(energies, integrand, ext=ext, k=k)
        return interp.integral(e_min, e_max)

    fisher = np.matrix(
        np.array(
            [
                [fisher_matrix_element(k1, k2) for k1 in params.keys()]
                for k2 in params.keys()
            ]
        )
    )

    return FisherResults(fisher_matrix=fisher, params=params)


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
        return _get_product_spline(f1, f2, grid, k=k, ext=ext)

    def _compute_fluxes(self, measurement: FluxMeasurement):
        """Compute the predicted fluxes given the current measurement."""
        e_min, e_max = measurement.e_lows[0], measurement.e_highs[-1]
        args = (e_min, e_max)
        mx = self.mx  # type: ignore

        # Factor to convert dN/dE to Phi. Factor of 2 comes from DM not being
        # self-conjugate.
        dm_flux_factor = measurement.target.dOmega / (4.0 * np.pi * mx)
        if self.kind == "ann":  # type: ignore
            f_dm = 2.0
            dm_flux_factor *= measurement.target.J / (2.0 * f_dm * mx)
            e_cm = 2.0 * mx * (1.0 + 0.5 * measurement.target.vx**2)  # type: ignore
            args += (e_cm,)

        elif self.kind == "dec":  # type: ignore
            dm_flux_factor *= measurement.target.D

        else:
            raise ValueError(
                "Invalid 'kind' encountered: {kind}. Expected 'ann' or 'dec'."
            )

        args += (measurement.energy_res,)
        dnde_conv = self.total_conv_spectrum_fn(*args)  # type: ignore

        # Integrated flux (excluding <sigma v>) from DM processes in each bin
        bounds = zip(measurement.e_lows, measurement.e_highs)
        return np.array(
            [dm_flux_factor * dnde_conv.integral(el, eh) for el, eh in bounds]
        )

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
        phi = self._compute_fluxes(measurement)

        if method == "1bin":
            constrainer = constrain_one_bin

        elif method == "chi2":
            constrainer = constrain_chi_squared
        else:
            raise NotImplementedError()

        return constrainer(phi, measurement, n_sigma)

    def unbinned_limit(
        self,
        A_eff: EffectiveArea,
        energy_res: EnergyResolution,
        T_obs: float,
        target: TargetParams,
        bg_model: BackgroundModel,
        e_grid=None,
        n_grid: int = 20,
        n_sigma: float = 5.0,
        _: bool = False,  # debug_msgs
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

        if self.kind == "ann":  # type: ignore
            # TODO: this should depend on the target!
            e_cm = 2.0 * self.mx * (1.0 + 0.5 * target.vx**2)  # type: ignore
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
            prefactor = 2.0 * f_dm * self.mx**2  # type: ignore
        else:
            prefactor = self.mx  # type: ignore

        assert target.dOmega is not None
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

    def fisher_limit(
        self,
        effective_area: EffectiveArea,
        energy_res: EnergyResolution,
        target: TargetParams,
        background_model: ParametricBackgroundModel,
        tobs: float,
        vx: float = 1e-3,
        sigma_levels: List[float] = [5.0],
        n_grid: int = 2000,
        e_grid: Optional[npt.NDArray[np.float64]] = None,
        k: int = 1,
        ext: str = "raise",
    ) -> List[float]:
        """Compute prospective constraints on a model using the Fisher information
        matrix.

        Parameters
        ----------
        model: Theory
            Dark matter model.
        effective_area: Callable
            Effective area of the observing telescope.
        energy_res: Callable
            Energy resolution of the telescope.
        target: TargetParams
            Target the telescope is observing.
        background_model: BackgroundModel
            Model for the background flux.
        tobs: float
            Observing time in seconds.
        vx: float, optional
            Dark matter velocity. Default is 1e-3.
        sigma_levels: List[float], optional
            List of the desired number of standard deviations to compute limits at.
            For example, if [3, 5] is used, the returned limits will be evaluated
            at 3 and 4 standard deviations. Default is [5.0].
        n_grid: int, optional
            Number of grid points used for generation of interpolating splines.
            Default is 20. Ignored if `e_grid` is not None.
        e_grid: array, optional
            Energy grid used for generation of interpolating splines. Default is
            None.
        k: int, optional
            Order of the underlying interpolating function. Default is 1 (linear).
        ext: str, optional
            String specifying how the interpolating function should handle energies
            outside the interval of the input energies. Default is 'raise'.

        Returns
        -------
        limits: float
            The limits on the rate at each sigma level specified. The limits have units
            of:
                annihilating DM : [cm^3/s]
                decaying DM: [1/s]
        """
        result = fisher(
            model=self,
            effective_area=effective_area,
            energy_res=energy_res,
            target=target,
            background_model=background_model,
            tobs=tobs,
            vx=vx,
            n_grid=n_grid,
            e_grid=e_grid,
            k=k,
            ext=ext,
        )

        return [result.limit(sigma, key="rate") for sigma in sigma_levels]
