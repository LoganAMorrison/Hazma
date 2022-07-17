import numpy as np


class FluxMeasurement:
    r"""

    Container for all information about a completed gamma ray analysis.

    Attributes
    ----------
    e_lows : np.array
        Lower edges of energy bins.
    e_highs : np.array
        Upper edges of energy bins.
    fluxes : np.array
        Flux measurements for each bin
        (:math:`\mathrm{MeV}^{-1}\mathrm{cm}^{-2} \mathrm{s}^{-1} \mathrm{sr}^{-1}`).
    upper_errors : np.array
        Size of upper error bars on flux measurements
        (:math:`\mathrm{MeV}^{-1}\mathrm{cm}^{-2} \mathrm{s}^{-1} \mathrm{sr}^{-1}`).
    lower_errors : np.array
        Size of lower error bars on flux measurements
        (:math:`\mathrm{MeV}^{-1}\mathrm{cm}^{-2} \mathrm{s}^{-1} \mathrm{sr}^{-1}`).
    energy_res : callable
        Function returning energy resolution (:math:`\Delta E / E`) as a function of
        photon energy.
    target : TargetParams
        Information about the target observed for this measurement.

    """

    def __init__(
        self,
        e_lows,
        e_highs,
        fluxes,
        upper_errors,
        lower_errors,
        energy_res,
        target,
        power=2,
    ):
        r"""Constructor.

        Parameters
        ----------
        fname : str
            Name of file containing observation information. The columns of
            this file must be:

                1. Lower bin edge (MeV)
                2. Upper bin edge (MeV)
                3. :math:`E^2 d^2 \Phi/dE d\Omega`
                   (:math:`\mathrm{MeV}^{-1}\mathrm{cm}^{-2}\mathrm{s}^{-1}\mathrm{sr}^{-1}`).
                4. Upper error bar
                   (:math:`\mathrm{MeV}^{-1}\mathrm{cm}^{-2}\mathrm{s}^{-1}\mathrm{sr}^{-1}`).
                5. Lower error bar
                   (:math:`\mathrm{MeV}^{-1}\mathrm{cm}^{-2}\mathrm{s}^{-1}\mathrm{sr}^{-1}`).

            Note that the error bar values are their y-coordinates, not their
            relative distances from the central flux.
        energy_res : callable
            Energy resolution function.
        target : TargetParams
            The target of the analysis
        """
        self.e_lows = e_lows
        self.e_highs = e_highs
        self.fluxes = fluxes
        self.upper_errors = upper_errors
        self.lower_errors = lower_errors

        # Get bin central values
        self._e_bins = 0.5 * (self.e_lows + self.e_highs)

        # E^2 dN/dE -> dN/dE
        self.fluxes /= self._e_bins**power

        # Compute upper and lower error bars
        self.upper_errors = self.upper_errors / self._e_bins**power - self.fluxes
        self.lower_errors = self.fluxes - self.lower_errors / self._e_bins**power

        # Load energy resolution
        self.energy_res = energy_res

        # Store analysis region
        self.target = target

    @classmethod
    def from_file(cls, fname, energy_res, target, power=2):
        (
            e_lows,
            e_highs,
            fluxes,
            upper_errors,
            lower_errors,
        ) = np.loadtxt(fname, delimiter=",").T
        return cls(
            e_lows,
            e_highs,
            fluxes,
            upper_errors,
            lower_errors,
            energy_res,
            target,
            power,
        )
