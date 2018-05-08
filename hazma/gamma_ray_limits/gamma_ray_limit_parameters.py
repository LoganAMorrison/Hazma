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


# Range of validity for the background model. TODO: put this in a class.
background_model_range = [0.8, 10.e3]


def dPhi_dEdOmega_B_default(e):
    """Simple model for the gamma ray background, d^2Phi / dE dOmega.

    Notes
    -----
    This is the background model from arXiv:1504.04024, eq. 14. It was derived
    by performing a simple power law fit to COMPTEL data from 0.8 - 30 MeV and
    EGRET data from 30 MeV - 10 GeV.

    Parameters
    ----------
    e : float
        Photon energy in MeV. Must be between 0.8 MeV and 10 GeV.

    Returns
    -------
    d^2Phi / dE dOmega : float
        The gamma ray background.
    """
    if background_model_range[0] <= e and e <= background_model_range[1]:
        return 2.74e-3 / e**2
    else:
        raise ValueError("The gamma ray background model is not applicable " +
                         "for energy %f MeV." % e)


def load_interp(rf_name):
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
    xs, ys = np.transpose(np.loadtxt(rf_name, delimiter=","))
    return interp1d(xs, ys, bounds_error=False, fill_value=0.0)


# Effective areas in cm^2
A_eff_e_astrogam = load_interp(A_eff_e_astrogam_rf)
A_eff_fermi = load_interp(A_eff_fermi_rf)
A_eff_comptel = load_interp(A_eff_comptel_rf)
A_eff_egret = load_interp(A_eff_egret_rf)

# Energy resolutions, Delta E / E
energy_res_comptel = load_interp(comptel_energy_res_rf)
energy_res_egret = load_interp(egret_energy_res_rf)
energy_res_fermi = load_interp(fermi_energy_res_rf)
energy_res_e_astrogam = load_interp(e_astrogam_energy_res_rf)

# Approximate observing time for e-ASTROGAM in seconds
T_obs_e_astrogam = 365. * 24. * 60.**2


# Angular sizes (in sr) and J factors (in MeV^2 cm^-5) for various objects
TargetParams = namedtuple("TargetParams", ["J", "dOmega"])
# Dwarf with high J factor
draco_params = TargetParams(1.0e29, 1.0e-3)


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
    return 4. * l_max*deg_to_rad * (np.sin(b_max*deg_to_rad) -
                                    np.sin(b_min*deg_to_rad))


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
    def __init__(self, bin_rf, measurement_rf, energy_res_rf, J, dOmega):
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
        J : float
            The J-factor for the target region in MeV^2 cm^-5 sr^-1.
        dOmega : float
            Solid angle subtended by the target region in sr.
        """
        # Store analysis region
        self.target = TargetParams(J, dOmega)

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
comptel_diffuse = FluxMeasurement(comptel_bins_rf, comptel_diffuse_rf,
                                  comptel_energy_res_rf, J=3.725e28,
                                  dOmega=solid_angle(60., 0., 20.))
# EGRET diffuse
egret_diffuse = FluxMeasurement(egret_bins_rf, egret_diffuse_rf,
                                egret_energy_res_rf, J=3.79e27,
                                dOmega=solid_angle(180., 20., 60.))
# Fermi diffuse
fermi_diffuse = FluxMeasurement(fermi_bins_rf, fermi_diffuse_rf,
                                fermi_energy_res_rf, J=4.698e27,
                                dOmega=solid_angle(180., 8., 90.))
