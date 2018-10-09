"""
Parameters relevant to computing constraints from gamma ray experiments.
"""
from collections import namedtuple
import numpy as np
from pkg_resources import resource_filename
from hazma.background_model import BackgroundModel
from hazma.flux_measurement import FluxMeasurement
from hazma.parameters import load_interp

# Get paths to files inside the module
gr_data_dir = "gamma_ray_data/"
A_eff_e_astrogam_rf = resource_filename(__name__,
                                        gr_data_dir +
                                        "e-astrogam_effective_area.dat")
A_eff_comptel_rf = resource_filename(__name__,
                                     gr_data_dir +
                                     "comptel_effective_area.dat")
A_eff_egret_rf = resource_filename(__name__,
                                   gr_data_dir + "egret_effective_area.dat")
A_eff_fermi_rf = resource_filename(__name__,
                                   gr_data_dir + "fermi_effective_area.dat")
egret_bins_rf = resource_filename(__name__, gr_data_dir + "egret_bins.dat")
egret_diffuse_rf = resource_filename(__name__,
                                     gr_data_dir + "egret_diffuse.dat")
comptel_bins_rf = resource_filename(__name__, gr_data_dir + "comptel_bins.dat")
comptel_diffuse_rf = resource_filename(__name__,
                                       gr_data_dir + "comptel_diffuse.dat")
fermi_bins_rf = resource_filename(__name__, gr_data_dir + "fermi_bins.dat")
fermi_diffuse_rf = resource_filename(__name__,
                                     gr_data_dir + "fermi_diffuse.dat")
comptel_energy_res_rf = resource_filename(__name__,
                                          gr_data_dir +
                                          "comptel_energy_resolution.dat")
egret_energy_res_rf = resource_filename(__name__,
                                        gr_data_dir +
                                        "egret_energy_resolution.dat")
fermi_energy_res_rf = resource_filename(__name__,
                                        gr_data_dir +
                                        "fermi_energy_resolution.dat")
e_astrogam_energy_res_rf = resource_filename(__name__,
                                             gr_data_dir + ("e-astrogam_energy"
                                                            "_resolution.dat"))
gc_bg_model_rf = resource_filename(__name__, gr_data_dir + "gc_bg_model.dat")


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


"""Angular sizes (in sr) and J factors (in MeV^2 cm^-5) for various targets"""
TargetParams = namedtuple("TargetParams", ["J", "dOmega"])
# Dwarf with high J factor
draco_params = TargetParams(6.94e27, 1.62e-3)

# This is the background model from arXiv:1504.04024, eq. 14. It was derived
# by performing a simple power law fit to COMPTEL data from 0.8 - 30 MeV and
# EGRET data from 30 MeV - 10 GeV. We take the lower range of validity to be
# the lowest energy for which e-ASTROGAM has nonzero effective area.
default_bg_model = BackgroundModel([0.3, 10.0e3], lambda e: 2.74e-3 / e**2)

# This is the more complex background model from arXiv:1703.02546. Note that it
# is only applicable to the inner 10deg x 10deg region of the Milky Way.
gc_bg_model = BackgroundModel.from_file(gc_bg_model_rf)
gc_target = TargetParams(1.795e29, solid_angle(10., 0., 10.))

"""Effective areas, cm^2"""
A_eff_e_astrogam = load_interp(A_eff_e_astrogam_rf)
A_eff_fermi = load_interp(A_eff_fermi_rf)
A_eff_comptel = load_interp(A_eff_comptel_rf)
A_eff_egret = load_interp(A_eff_egret_rf)

"""Energy resolutions, Delta E / E"""
energy_res_comptel = load_interp(comptel_energy_res_rf,
                                 fill_value="extrapolate")
energy_res_egret = load_interp(egret_energy_res_rf, fill_value="extrapolate")
energy_res_fermi = load_interp(fermi_energy_res_rf, fill_value="extrapolate")
energy_res_e_astrogam = load_interp(e_astrogam_energy_res_rf,
                                    fill_value="extrapolate")

# Approximate observing time for e-ASTROGAM in seconds
T_obs_e_astrogam = 365. * 24. * 60.**2


"""Target parameters"""
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
