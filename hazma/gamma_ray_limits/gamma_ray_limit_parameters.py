"""
Parameters relevant to computing constraints from gamma ray experiments
"""
from collections import namedtuple
import numpy as np
from pkg_resources import resource_filename
from scipy.interpolate import interp1d


# Get paths to files inside the module
A_eff_e_astrogam_rf = resource_filename(__name__,
                                        "e-astrogam_effective_area.dat")
A_eff_comptel_rf = resource_filename(__name__, "comptel_effective_area.dat")
A_eff_egret_rf = resource_filename(__name__, "egret_effective_area.dat")
A_eff_fermi_rf = resource_filename(__name__, "fermi_effective_area.dat")
egret_bins_rf = resource_filename(__name__, "egret_bins.dat")
egret_diffuse_rf = resource_filename(__name__, "egret_diffuse.dat")
comptel_bins_rf = resource_filename(__name__, "comptel_bins.dat")
comptel_diffuse_rf = resource_filename(__name__, "comptel_diffuse.dat")
fermi_bins_rf = resource_filename(__name__, "fermi_bins.dat")
fermi_diffuse_rf = resource_filename(__name__, "fermi_diffuse.dat")
comptel_energy_res_rf = resource_filename(__name__,
                                          "comptel_energy_resolution.dat")
egret_energy_res_rf = resource_filename(__name__,
                                        "egret_energy_resolution.dat")
fermi_energy_res_rf = resource_filename(__name__,
                                        "fermi_energy_resolution.dat")
e_astrogam_energy_res_rf = resource_filename(__name__,
                                             "e-astrogam_energy_resolution.dat")
gc_bg_model_rf = resource_filename(__name__, "gc_bg_model.dat")


def load_interp(rf_name, bounds_error=False, fill_value=0.0):
    """Creates an interpolator from a data file.

    Parameters
    ----------
    rf_name : resource_filename
        Name of resource file.

    Returns
    -------
    interp : interp1d
        An interpolator created using the first column of the file as the x
        values and second as the y values. interp will not raise a bounds error
        and uses a fill values of 0.0.
    """
    xs, ys = np.loadtxt(rf_name, delimiter=",").T
    return interp1d(xs, ys, bounds_error=bounds_error, fill_value=fill_value)


class BackgroundModel(object):
    """Represents a gamma ray background model.

    Attributes
    ----------
    e_range : [float, float]
        Minimum and maximum photon energies for which this model is valid, in
        MeV.
    """

    def __init__(self, e_range, flux_fn):
        """
        Parameters
        ----------
        e_range : [float, float]
            Photon energies (MeV) between which this background model is valid.
            Note that these bounds are inclusive.
        flux_fn : np.array
            Background gamma ray flux (MeV^-1 sr^-1 m^-2 s^-1) as a function of
            photon energy (MeV).
        """
        self.e_range = e_range
        self.__dPhi_dEdOmega = flux_fn

    @classmethod
    def from_file(cls, rf_name):
        """Factory method to create a BackgroundModel from a data file.

        Parameters
        ----------
        rf_name : resource_filename
            Name of file whose lines are comma-separated pairs of photon
            energies and background fluxes, in MeV and MeV^-1 sr^-1 m^-2 s^-1.

        Returns
        -------
        bg_model : BackgroundModel
        """
        flux_fn = load_interp(rf_name, bounds_error=True, fill_value=np.nan)
        e_range = flux_fn.x[[0, -1]]

        return cls(e_range, flux_fn)

    @classmethod
    def from_vals(cls, es, fluxes):
        """Factory method to create a BackgroundModel from flux data points.

        Parameters
        ----------
        es : np.array
            Photon energies (MeV).
        fluxes : np.array
            Background gamma ray flux (MeV^-1 sr^-1 m^-2 s^-1) at the energies
            in es.

        Returns
        -------
        bg_model : BackgroundModel
        """
        return cls(es[[0, -1]], interp1d(es, fluxes))

    def dPhi_dEdOmega(self, es):
        """Computes this background model's gamma ray flux.

        Parameters
        ----------
        es : float or np.array
            Photon energy/energies at which to compute

        Returns
        -------
        d^2Phi / dE dOmega
            Background gamma ray flux, in MeV^-1 sr^-1 m^-2 s^-1. For any
            energies outside of self.e_range, np.nan is returned.
        """
        if hasattr(es, "__len__"):
            return np.array([self.dPhi_dEdOmega(e) for e in es])
        else:
            if es >= self.e_range[0] and es <= self.e_range[1]:
                return np.array([self.__dPhi_dEdOmega(es)])
            else:
                raise ValueError("The gamma ray background model is not "
                                 "applicable for energy %f MeV." % es)
                return np.array([np.nan])


def solid_angle(l_max, b_min, b_max):
    """Returns solid angle subtended for a target region.

    Parameters
    ----------
    l_max : float
        Maximum value of galactic longitude in deg. Note that l must lie in the
        interval [-180, 180].
    b_min, b_max : float, float
        Minimum and maximum values for |b| in deg. Note that b must lie in the
        interval [-90, 90], with the equator at b = 0.

    Returns
    -------
    Omega : float
        Solid angle subtended by the region in sr.
    """
    deg_to_rad = np.pi / 180.
    return 4. * l_max * deg_to_rad * (np.sin(b_max * deg_to_rad) -
                                      np.sin(b_min * deg_to_rad))


# Angular sizes (in sr) and J factors (in MeV^2 cm^-5) for various objects
TargetParams = namedtuple("TargetParams", ["J", "dOmega"])
# Dwarf with high J factor
draco_params = TargetParams(1.0e29, 1.0e-3)

# This is the background model from arXiv:1504.04024, eq. 14. It was derived
# by performing a simple power law fit to COMPTEL data from 0.8 - 30 MeV and
# EGRET data from 30 MeV - 10 GeV. We take the lower range of validity to be
# the lowest energy for which e-ASTROGAM has nonzero effective area.
default_bg_model = BackgroundModel([0.3, 10.0e3], lambda e: 2.74e-3 / e**2)

# This is the more complex background model from arXiv:1703.02546. Note that it
# is only applicable to the inner 10deg x 10deg region of the Milky Way.
gc_bg_model = BackgroundModel.from_file(gc_bg_model_rf)
gc_bg_model_target = TargetParams(1.795e29, solid_angle(10., 0., 10.))

# Effective areas in cm^2
A_eff_e_astrogam = load_interp(A_eff_e_astrogam_rf)
A_eff_fermi = load_interp(A_eff_fermi_rf)
A_eff_comptel = load_interp(A_eff_comptel_rf)
A_eff_egret = load_interp(A_eff_egret_rf)

# Energy resolutions, Delta E / E
energy_res_comptel = load_interp(comptel_energy_res_rf,
                                 fill_value="extrapolate")
energy_res_egret = load_interp(egret_energy_res_rf, fill_value="extrapolate")
energy_res_fermi = load_interp(fermi_energy_res_rf, fill_value="extrapolate")
energy_res_e_astrogam = load_interp(e_astrogam_energy_res_rf,
                                    fill_value="extrapolate")

# Approximate observing time for e-ASTROGAM in seconds
T_obs_e_astrogam = 365. * 24. * 60.**2


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
        self.energy_res = load_interp(energy_res_rf)


# COMPTEL diffuse
comptel_diffuse_target = TargetParams(J=3.725e28,
                                      dOmega=solid_angle(60., 0., 20.))
comptel_diffuse = FluxMeasurement(comptel_bins_rf, comptel_diffuse_rf,
                                  comptel_energy_res_rf,
                                  comptel_diffuse_target)
# EGRET diffuse
egret_diffuse_target = TargetParams(J=3.79e27,
                                    dOmega=solid_angle(180., 20., 60.))
egret_diffuse = FluxMeasurement(egret_bins_rf, egret_diffuse_rf,
                                egret_energy_res_rf, egret_diffuse_target)
# Fermi diffuse
fermi_diffuse_target = TargetParams(J=4.698e27,
                                    dOmega=solid_angle(180., 8., 90.))
fermi_diffuse = FluxMeasurement(fermi_bins_rf, fermi_diffuse_rf,
                                fermi_energy_res_rf, fermi_diffuse_target)
