from collections import namedtuple
import numpy as np
from pkg_resources import resource_filename
from hazma.background_model import BackgroundModel
from hazma.flux_measurement import FluxMeasurement
from hazma.parameters import load_interp

"""
Parameters relevant to computing constraints from gamma ray experiments.
"""

# Get paths to files inside the module
gr_data_dir = "gamma_ray_data/"
# Gamma-ray observations
egret_obs_rf = resource_filename(__name__, gr_data_dir + "egret_obs.dat")
comptel_obs_rf = resource_filename(__name__, gr_data_dir + "comptel_obs.dat")
fermi_obs_rf = resource_filename(__name__, gr_data_dir + "fermi_obs.dat")
# Effective areas
A_eff_e_astrogam_rf = resource_filename(
    __name__, gr_data_dir + "e-astrogam_effective_area.dat"
)
A_eff_comptel_rf = resource_filename(
    __name__, gr_data_dir + "comptel_effective_area.dat"
)
A_eff_egret_rf = resource_filename(__name__, gr_data_dir + "egret_effective_area.dat")
A_eff_fermi_rf = resource_filename(__name__, gr_data_dir + "fermi_effective_area.dat")

e_astrogam_energy_res_rf = resource_filename(
    __name__, gr_data_dir + ("e-astrogam_energy" "_resolution.dat")
)
# Complex background model
gc_bg_model_rf = resource_filename(__name__, gr_data_dir + "gc_bg_model.dat")


def solid_angle(l_max, b_min, b_max):
    """
    Returns solid angle subtended for a rectangular target region centered on
    the galactic center.

    Parameters
    ----------
    l_max : float
        Maximum value of galactic longitude in deg. Note that :math:`l` must
        lie in the interval :math:`[-180, 180]`.
    b_min, b_max : float, float
        Minimum and maximum values for :math:`b` in deg. Note that :math:`b`
        must lie in the interval :math:`[-90, 90]`, with the equator at
        :math:`b = 0`.

    Returns
    -------
    Omega : float
        Solid angle subtended by the region in sr.
    """
    deg_to_rad = np.pi / 180.0
    return (
        4.0
        * l_max
        * deg_to_rad
        * (np.sin(b_max * deg_to_rad) - np.sin(b_min * deg_to_rad))
    )


class TargetParams:
    """
    Container for information about a target region.

    Currently implemented for the Draco dwarf galaxy and :math:`10^\circ \times
    10^\circ` region around the galactic center, which can be imported using::

        from hazma.gamma_ray_parameters import draco_params, gc_target

    Parameters
    ----------
    J : float
        J-factor in MeV^2 cm^-5
    dOmega : float
        Angular size in sr
    """

    def __init__(self, J, dOmega):
        self.J = J
        self.dOmega = dOmega


# Dwarf with high J factor
draco_params = TargetParams(6.94e27, 1.62e-3)

# This is the background model from arXiv:1504.04024, eq. 14. It was derived
# by performing a simple power law fit to COMPTEL data from 0.8 - 30 MeV and
# EGRET data from 30 MeV - 10 GeV. We take the lower range of validity to be
# the lowest energy for which e-ASTROGAM has nonzero effective area.
default_bg_model = BackgroundModel([0.3, 10.0e3], lambda e: 2.74e-3 / e ** 2)

# This is the more complex background model from arXiv:1703.02546. Note that it
# is only applicable to the inner 10deg x 10deg region of the Milky Way.
gc_bg_flux_fn = load_interp(gc_bg_model_rf, bounds_error=True, fill_value=np.nan)
gc_bg_e_range = gc_bg_flux_fn.x[[0, -1]]
gc_bg_model = BackgroundModel(gc_bg_e_range, gc_bg_flux_fn)
gc_target = TargetParams(1.795e29, solid_angle(10.0, 0.0, 10.0))

# # # Effective areas, cm^2
A_eff_e_astrogam = load_interp(
    A_eff_e_astrogam_rf
)  #: e-ASTROGAM effective area function
A_eff_fermi = load_interp(A_eff_fermi_rf)  #: Fermi-LAT effective area function
A_eff_comptel = load_interp(A_eff_comptel_rf)  #: COMPTEL effective area function
A_eff_egret = load_interp(A_eff_egret_rf)  #: EGRET effective area function


def energy_res_comptel(e):
    """COMPTEL energy resolution :math:`\Delta E / E`.

    Taken from `ch. II, page 11
    <http://wwwgro.unh.edu/users/ckappada/thesis_stuff/thesis.html>`_.
    """
    return 0.05


def energy_res_egret(e):
    """EGRET's energy resolution :math:`\Delta E / E`.

    This is the most optimistic value, taken from
    `sec. 4.3.3 <http://adsabs.harvard.edu/doi/10.1086/191793>`_.
    """
    return 0.18


def energy_res_fermi(e):
    """Fermi-LAT's energy resolution :math:`\Delta E / E`.

    This is the average of the most optimistic normal and 60deg off-axis values
    from `fig. 18 <https://arxiv.org/abs/0902.1089>`_.
    """
    return 0.075


#: e-ASTROGAM energy resolution function. From table 1 of the `e-ASTROGAM
#: whitebook <https://arxiv.org/abs/1711.01265>`_.
energy_res_e_astrogam = load_interp(e_astrogam_energy_res_rf, fill_value="extrapolate")

# Approximate observing time for e-ASTROGAM in seconds
T_obs_e_astrogam = 365.0 * 24.0 * 60.0 ** 2

# # # Target parameters
comptel_diffuse_target = TargetParams(J=3.725e28, dOmega=solid_angle(60.0, 0.0, 20.0))
egret_diffuse_target = TargetParams(J=3.79e27, dOmega=solid_angle(180.0, 20.0, 60.0))
fermi_diffuse_target = TargetParams(J=4.698e27, dOmega=solid_angle(180.0, 8.0, 90.0))
#: COMPTEL diffuse gamma-ray flux measurements
comptel_diffuse = FluxMeasurement(
    comptel_obs_rf, energy_res_comptel, comptel_diffuse_target
)
#: EGRET diffuse gamma-ray flux measurements
egret_diffuse = FluxMeasurement(egret_obs_rf, energy_res_egret, egret_diffuse_target)
#: Fermi diffuse gamma-ray flux measurements
fermi_diffuse = FluxMeasurement(fermi_obs_rf, energy_res_fermi, fermi_diffuse_target)
