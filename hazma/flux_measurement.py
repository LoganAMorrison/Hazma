import numpy as np
from hazma.parameters import load_interp


class FluxMeasurement(object):
    """Container for all information about a completed gamma ray analysis.

    Attributes
    ----------
    target : TargetParams
        Information about the target observed for this measurement.
    bins : 2D np.array
        Bins used for the measurement. This is an Nx2 array where each pair
        indicates the lower and upper edges of the bin (MeV).
    fluxes : np.array
        Flux measurements for each bin (MeV^-1 cm^-2 s^-1 sr^-1).
    upper_errors : np.array
        Size of upper error bars on flux measurements (MeV^-1 cm^-2 s^-1
        sr^-1).
    lower_errors : np.array
        Size of lower error bars on flux measurements (MeV^-1 cm^-2 s^-1
        sr^-1).
    energy_res : interp1d
        Function returning energy resolution (Delta E / E) as a function of
        photon energy.
    """

    def __init__(self, bin_rf, measurement_rf, energy_res_rf, target):
        """Constructor.

        Parameters
        ----------
        bin_rf : resource_filename
            Name of file where bins are defined. The file must contain
            comma-separated pairs. The elements of the pair define the lower
            and upper bounds for each energy bin in MeV.
        measurement_rf : resource_filename
            Name of file containing flux measurements. The rows of the file
            must be sets of three comma-separated values, where the first
            indicated the central value for the measurement of E^2 dN/dE, and
            the second and third indicate the absolute positions of the upper
            and lower error bars respectively. All values are in MeV cm^-2 s^-1
            sr^-1
        energy_res_rf : resource_filename
            Name of file containing energy resolution data. The rows of the
            file must be comma-separated pairs indicating an energy in MeV and
            energy resolution Delta E / E. The energy resolution must be
            defined over the full range of energy bins used for this
            measurement.
        target : TargetParams
            The target of the analysis
        """
        # Store analysis region
        self.target = target

        # Load bin info
        self.bins = np.loadtxt(bin_rf, delimiter=",")

        # Load flux data
        # Get bin central values
        bin_centers = np.mean(self.bins, axis=1)

        # Load E^2 dN/dE and convert to dN/dE
        raw_fluxes = np.loadtxt(measurement_rf, delimiter=",")
        self.fluxes = raw_fluxes[:, 0] / bin_centers**2

        # Compute upper and lower error bars
        self.upper_errors = raw_fluxes[:, 1] / bin_centers**2 - self.fluxes
        self.lower_errors = self.fluxes - raw_fluxes[:, 2] / bin_centers**2

        # Load energy resolution
        self.energy_res = load_interp(energy_res_rf, fill_value="extrapolate")
