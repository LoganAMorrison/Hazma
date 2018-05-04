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
egret_bins_rf = resource_filename(__name__, "egret_bins.dat")
egret_diffuse_rf = resource_filename(__name__, "egret_diffuse.dat")
comptel_bins_rf = resource_filename(__name__, "comptel_bins.dat")
comptel_diffuse_rf = resource_filename(__name__, "comptel_diffuse.dat")
fermi_bins_rf = resource_filename(__name__, "fermi_bins.dat")
fermi_diffuse_rf = resource_filename(__name__, "fermi_diffuse.dat")


# Range of validity for the background model
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


# Effective areas in cm^2
# e-ASTROGAM
e_gams_e_a, A_effs_e_a = np.transpose(np.loadtxt(A_eff_e_astrogam_rf,
                                                 delimiter=","))
A_eff_e_astrogam = interp1d(e_gams_e_a, A_effs_e_a, bounds_error=False,
                            fill_value=0.0)

# Approximate observing time for e-ASTROGAM in seconds
T_obs_e_astrogam = 365. * 24. * 60.**2


# Angular sizes (in sr) and J factors (in MeV^2 cm^-5) for various objects
TargetParams = namedtuple("TargetParams", ["J", "dOmega"])
# Dwarf with high J factor
draco_params = TargetParams(1.0e29, 1.0e-3)


def load_flux_data(fn, bins):
    """Parses a flux data file.

    Parameters
    ----------
    fn : string
        Name of file containing comma-separated data. The columns in the file
        must be E^2 d^2Phi / dE dOmega, the upper and lower error bar's
        positions, all in MeV cm^-2 s^-1 sr^-1.
    bins : np.array
        A 2D array where each element is the lower and upper bound of the
        energy bin corresponding to the fluxes in fn.

    Returns
    -------
    central_fluxes, upper_error_bars, lower_error_bars : np.array, np.array,
    np.array
        Central flux values, size of upper error bar and size of lower error
        bar (a positive number), all in MeV cm^-2 s^-1 sr^-1.
    """
    # Get bin central values
    bin_centers = np.mean(bins, axis=1)

    # Load E^2 dN/dE and convert to dN/dE
    raw_fluxes = np.loadtxt(fn, delimiter=",")
    central_fluxes = raw_fluxes[:, 0] / bin_centers**2
    upper_fluxes = raw_fluxes[:, 1] / bin_centers**2
    lower_fluxes = raw_fluxes[:, 2] / bin_centers**2

    return (central_fluxes, upper_fluxes - central_fluxes, central_fluxes -
            lower_fluxes)


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


# Container for all information about an analysis
FluxMeasurement = namedtuple("AnalysisInfo", ["target", "bins", "fluxes",
                                              "flux_upper_errors",
                                              "flux_lower_errors"])

# EGRET diffuse
egret_diffuse_region = TargetParams(3.79e27, solid_angle(180., 20., 60.))
egret_bins = np.loadtxt(egret_bins_rf, delimiter=",")
egret_diffuse = FluxMeasurement(egret_diffuse_region, egret_bins,
                                *load_flux_data(egret_diffuse_rf, egret_bins))

# Fermi diffuse
fermi_diffuse_region = TargetParams(4.698e27, solid_angle(180., 8., 90.))
fermi_bins = np.loadtxt(fermi_bins_rf, delimiter=",")
fermi_diffuse = FluxMeasurement(fermi_diffuse_region, fermi_bins,
                                *load_flux_data(fermi_diffuse_rf, fermi_bins))

# COMPTEL diffuse
comptel_diffuse_region = TargetParams(3.725e28, solid_angle(60., 0., 20.))
comptel_bins = np.loadtxt(comptel_bins_rf, delimiter=",")
comptel_diffuse = FluxMeasurement(comptel_diffuse_region, comptel_bins,
                                  *load_flux_data(comptel_diffuse_rf,
                                                  comptel_bins))
