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

# Resources
A_eff_e_astrogam_rf = resource_filename(
    __name__, gr_data_dir + "e-astrogam_effective_area.dat"
)
A_eff_comptel_rf = resource_filename(
    __name__, gr_data_dir + "comptel_effective_area.dat"
)
A_eff_egret_rf = resource_filename(__name__, gr_data_dir + "egret_effective_area.dat")
A_eff_fermi_rf = resource_filename(__name__, gr_data_dir + "fermi_effective_area.dat")
A_eff_gecco_rf = resource_filename(__name__, gr_data_dir + "gecco_effective_area.dat")
A_eff_adept_rf = resource_filename(__name__, gr_data_dir + "adept_effective_area.dat")
A_eff_amego_rf = resource_filename(__name__, gr_data_dir + "amego_effective_area.dat")
A_eff_mast_rf = resource_filename(__name__, gr_data_dir + "mast_effective_area.dat")
A_eff_pangu_rf = resource_filename(__name__, gr_data_dir + "pangu_effective_area.dat")
A_eff_grams_rf = resource_filename(__name__, gr_data_dir + "grams_effective_area.dat")
A_eff_grams_upgrade_rf = resource_filename(
    __name__, gr_data_dir + "grams_upgrade_effective_area.dat"
)

e_astrogam_energy_res_rf = resource_filename(
    __name__, gr_data_dir + ("e-astrogam_energy_resolution.dat")
)
gecco_energy_res_rf = resource_filename(
    __name__, gr_data_dir + "gecco_energy_resolution.dat"
)
amego_energy_res_rf = resource_filename(
    __name__, gr_data_dir + "amego_energy_resolution.dat"
)
mast_energy_res_rf = resource_filename(
    __name__, gr_data_dir + "mast_energy_resolution.dat"
)
gecco_large_energy_res_rf = resource_filename(
    __name__, gr_data_dir + "gecco_large_energy_resolution.dat"
)

# Complex background model
gc_bg_model_rf = resource_filename(__name__, gr_data_dir + "gc_bg_model.dat")


def solid_angle_cone(radius):
    """
    Returns solid angle subtended for a circular/conical target region.

    Parameters
    ----------
    radius : float
        Cone radius in degrees.

    Returns
    -------
    Omega : float
        Solid angle subtended by the region in sr.
    """
    return 4 * np.pi * np.sin(radius * np.pi / 180 / 2) ** 2


def solid_angle_rect(l_max, b_min, b_max):
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

    Parameters
    ----------
    J : float
        (Averaged) J-factor for DM annihilation in MeV^2 cm^-5.
    D : float
        (Averaged) D-factor for DM decay in MeV cm^-2.
    dOmega : float
        Angular size in sr.
    vx : float
        Average DM velocity in target in units of c. Defaults to 1e-3, the
        Milky Way velocity dispersion.
    """

    def __init__(self, J=None, D=None, dOmega=None, vx=1e-3):
        self.J = J
        self.D = D
        self.dOmega = dOmega
        self.vx = vx


# From Alex Moiseev's slides. Ref: G. Weidenspointner et al, AIP 510, 467, 2000.
# Additional factor of two due to uncertainty about radioactive and
# instrumental backgrounds.
gecco_bg_model = BackgroundModel([0.2, 4e3], lambda e_gam: 2 * 4e-3 / e_gam ** 2)

# This is the background model from arXiv:1504.04024, eq. 14. It was derived
# by performing a simple power law fit to COMPTEL data from 0.8 - 30 MeV and
# EGRET data from 30 MeV - 10 GeV. We take the lower range of validity to be
# the lowest energy for which e-ASTROGAM has nonzero effective area.
default_bg_model = BackgroundModel([0.0, 10.0e3], lambda e: 2.74e-3 / e ** 2)

# This is the more complex background model from arXiv:1703.02546. Note that it
# is only applicable to the inner 10deg x 10deg region of the Milky Way.
gc_bg_flux_fn = load_interp(gc_bg_model_rf, bounds_error=True, fill_value=np.nan)
gc_bg_e_range = gc_bg_flux_fn.x[[0, -1]]
gc_bg_model = BackgroundModel(gc_bg_e_range, gc_bg_flux_fn)

# Several different GC targets. Halo parameters are from the DM fit using
# baryonic model B2 in https://arxiv.org/abs/1906.06133 (table III).
gc_targets = {
    "nfw": {
        "1 arcmin cone": TargetParams(J=6.972e32, D=4.84e26, dOmega=2.66e-7),
        "5 deg cone": TargetParams(J=1.782e30, D=1.597e26, dOmega=0.0239),
        "10 deg cone": TargetParams(J=7.458e29, D=1.214e26, dOmega=0.0955),
    },
    "ein": {
        "1 arcmin cone": TargetParams(J=1.724e32, D=5.413e26, dOmega=2.66e-7),
        "5 deg cone": TargetParams(J=4.442e30, D=2.269e26, dOmega=0.0239),
        "10 deg cone": TargetParams(J=1.706e30, D=1.615e26, dOmega=0.0955),
    },
}
# +/- 1 sigma
gc_targets_optimistic = {
    "nfw": {
        "1 arcmin cone": TargetParams(J=1.415e33, D=6.666e26, dOmega=2.66e-7),
        "5 deg cone": TargetParams(J=3.372e30, D=2.058e26, dOmega=0.0239),
        "10 deg cone": TargetParams(J=1.355e30, D=1.522e26, dOmega=0.0955),
    },
    "ein": {
        "1 arcmin cone": TargetParams(J=5.987e34, D=4.179e27, dOmega=2.66e-7),
        "5 deg cone": TargetParams(J=4.965e31, D=4.345e26, dOmega=0.0239),
        "10 deg cone": TargetParams(J=1.404e31, D=2.62e26, dOmega=0.0955),
    },
}


# Observing regions for various experiments. Same NFW profile as above.
comptel_diffuse_target = TargetParams(J=9.308e28, D=4.866e25, dOmega=1.433)
comptel_diffuse_target_optimistic = TargetParams(J=1.751e29, D=5.541e25, dOmega=1.433)
egret_diffuse_target = TargetParams(J=1.253e28, D=3.42e25, dOmega=6.585)
fermi_diffuse_target = TargetParams(J=1.695e28, D=3.563e25, dOmega=10.82)

# Draco dwarf.
draco_targets = {
    "nfw": {
        "1 arcmin cone": TargetParams(J=3.418e30, D=5.949e25, dOmega=2.66e-7),
        "5 deg cone": TargetParams(J=8.058e26, D=1.986e24, dOmega=0.0239),
    },
}

# Andromeda. See https://arxiv.org/abs/1009.5988.
m31_targets = {
    "nfw": {
        "1 arcmin cone": TargetParams(J=1.496e31, D=3.297e26, dOmega=2.66e-7),
        "5 deg cone": TargetParams(J=1.479e27, D=4.017e24, dOmega=0.0239),
    },
}

# Fornax cluster. See https://arxiv.org/abs/1009.5988.
fornax_targets = {
    "nfw": {
        "1 arcmin cone": TargetParams(J=5.316e29, D=2.898e26, dOmega=2.66e-7),
        "2 deg cone": TargetParams(J=2.558e26, D=9.081e24, dOmega=0.00383),
    },
}

# # # Effective areas, cm^2
A_eff_e_astrogam = load_interp(
    A_eff_e_astrogam_rf
)  #: e-ASTROGAM effective area function
A_eff_fermi = load_interp(A_eff_fermi_rf)  #: Fermi-LAT effective area function
A_eff_comptel = load_interp(A_eff_comptel_rf)  #: COMPTEL effective area function
A_eff_egret = load_interp(A_eff_egret_rf)  #: EGRET effective area function
A_eff_gecco = load_interp(A_eff_gecco_rf)  #: GECCO effective area function
A_eff_adept = load_interp(A_eff_adept_rf)  #: AdEPT effective area function
A_eff_amego = load_interp(A_eff_amego_rf)  #: AMEGO effective area function
A_eff_mast = load_interp(A_eff_mast_rf)  #: MAST effective area function
A_eff_pangu = load_interp(A_eff_pangu_rf)  #: PANGU effective area function
A_eff_grams = load_interp(A_eff_grams_rf)  #: GRAMS effective area function
A_eff_grams_upgrade = load_interp(
    A_eff_grams_upgrade_rf
)  #: Upgraded GRAMS effective area function

# Multiplicative factor to convert FWHM into standard deviations, assuming
# energy resolution function is a Gaussian
fwhm_factor = 1 / (2 * np.sqrt(2 * np.log(2)))

def energy_res_grams_upgrade(e):
    """
    GRAMS upgrade approximate energy resolution. See https://arxiv.org/abs/1901.03430.
    """

    def _res(e):
        return 0.05

    return np.vectorize(_res)(e)


def energy_res_grams(e):
    """
    GRAMS approximate energy resolution. See https://arxiv.org/abs/1901.03430.
    """

    def _res(e):
        return 0.05

    return np.vectorize(_res)(e)


def energy_res_adept(e):
    """
    AdEPT energy resolution. See arXiv1311.2059. The energy dependence is not
    given.
    """

    def _res(e):
        return 0.3 * fwhm_factor

    return np.vectorize(_res)(e)


def energy_res_pangu(e):
    """
    PANGU energy resolution. https://doi.org/10.22323/1.246.0069. There is not
    much energy dependence.
    """

    def _res(e):
        return 0.4

    return np.vectorize(_res)(e)


def energy_res_comptel(e):
    """COMPTEL energy resolution :math:`\Delta E / E`.

    This is the most optimistic value, taken from `ch. II, page 11
    <https://scholars.unh.edu/dissertation/2045/>`_. The
    energy resolution at 1 MeV is 10% (FWHM).
    """

    def _res(e):
        return 0.05 * fwhm_factor

    return np.vectorize(_res)(e)


def energy_res_egret(e):
    """EGRET's energy resolution :math:`\Delta E / E`.

    This is the most optimistic value, taken from
    `sec. 4.3.3 <http://adsabs.harvard.edu/doi/10.1086/191793>`_.
    """

    def _res(e):
        # Convert FWHM into standard deviation
        return 0.18 * fwhm_factor

    return np.vectorize(_res)(e)


def energy_res_fermi(e):
    """Fermi-LAT's energy resolution :math:`\Delta E / E`.

    This is the average of the most optimistic normal and 60deg off-axis values
    from `fig. 18 <https://arxiv.org/abs/0902.1089>`_.
    """

    def _res(e):
        return 0.075

    return np.vectorize(_res)(e)


#: e-ASTROGAM energy resolution function. From table 1 of the `e-ASTROGAM
#: whitebook <https://arxiv.org/abs/1711.01265>`_.
energy_res_e_astrogam = load_interp(e_astrogam_energy_res_rf, fill_value="extrapolate")
energy_res_gecco = load_interp(gecco_energy_res_rf, fill_value="extrapolate")
energy_res_gecco_large = load_interp(
    gecco_large_energy_res_rf, fill_value="extrapolate"
)
energy_res_amego = load_interp(amego_energy_res_rf, fill_value="extrapolate")
energy_res_mast = load_interp(mast_energy_res_rf, fill_value="extrapolate")

# Approximate observing time for e-ASTROGAM in seconds
T_obs_e_astrogam = 365.0 * 24.0 * 60.0 ** 2

#: COMPTEL diffuse gamma-ray flux measurements
comptel_diffuse = FluxMeasurement(
    comptel_obs_rf, energy_res_comptel, comptel_diffuse_target
)
comptel_diffuse_optimistic = FluxMeasurement(
    comptel_obs_rf, energy_res_comptel, comptel_diffuse_target_optimistic
)
#: EGRET diffuse gamma-ray flux measurements
egret_diffuse = FluxMeasurement(egret_obs_rf, energy_res_egret, egret_diffuse_target)
#: Fermi diffuse gamma-ray flux measurements
fermi_diffuse = FluxMeasurement(fermi_obs_rf, energy_res_fermi, fermi_diffuse_target)
