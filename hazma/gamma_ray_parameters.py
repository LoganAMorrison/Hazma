import importlib.resources as pkg_resources
import os
from pathlib import Path

import numpy as np
from hazma.background_model import BackgroundModel
from hazma.flux_measurement import FluxMeasurement
from hazma.target_params import TargetParams
from scipy.interpolate import interp1d

"""

Parameters relevant to computing constraints from gamma ray experiments.

"""

# Directory of this file
_dir = Path(__file__).parent.absolute()
# Directory to gamma_ray_data
grd_dir = os.path.join(_dir, "gamma_ray_data")


def _generate_interp(subdir, filename, fill_value=np.nan, bounds_error=True):
    path = os.path.join(grd_dir, subdir, filename)
    data = np.genfromtxt(path, delimiter=",", unpack=True)
    return interp1d(*data, bounds_error=bounds_error, fill_value=fill_value)


# From Alex Moiseev's slides. Ref: G. Weidenspointner et al, AIP 510, 467, 2000.
# Additional factor of two due to uncertainty about radioactive and
# instrumental backgrounds.
gecco_bg_model = BackgroundModel([0.2, 4e3], lambda e_gam: 2 * 4e-3 / e_gam ** 2)

# This is the background model from arXiv:1504.04024, eq. 14. It was derived
# by performing a simple power law fit to COMPTEL data from 0.8 - 30 MeV and
# EGRET data from 30 MeV - 10 GeV. We take the lower range of validity to be
# the lowest energy for which e-ASTROGAM has nonzero effective area.
default_bg_model = BackgroundModel([0.0, 10.0e3], lambda e: 2.74e-3 / e ** 2)


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


# ==================
# ---- Targets -----
# ==================

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
# For backwards compatability
comptel_diffuse_target = TargetParams(J=9.308e28, D=4.866e25, dOmega=1.433)
comptel_diffuse_target_optimistic = TargetParams(J=1.751e29, D=5.541e25, dOmega=1.433)
egret_diffuse_target = TargetParams(J=1.253e28, D=3.42e25, dOmega=6.585)
fermi_diffuse_target = TargetParams(J=1.695e28, D=3.563e25, dOmega=10.82)
integral_diffuse_target = TargetParams(J=2.086e29, D=7.301e25, dOmega=0.5421)

# New interface
comptel_diffuse_targets = {
    "nfw": TargetParams(J=9.308e28, D=4.866e25, dOmega=1.433),
    "ein": TargetParams(J=7.412e27, D=1.292e25, dOmega=1.433),
}
comptel_diffuse_targets_optimistic = {
    "nfw": TargetParams(J=1.53e29, D=5.571e25, dOmega=1.433),
    "ein": TargetParams(J=1.387e28, D=1.544e25, dOmega=1.433),
}
egret_diffuse_targets = {
    "nfw": TargetParams(J=1.253e28, D=3.42e25, dOmega=6.585),
    "ein": TargetParams(J=8.426e26, D=8.559e24, dOmega=6.585),
}
egret_diffuse_targets_optimistic = {
    "nfw": TargetParams(J=1.511e28, D=3.523e25, dOmega=6.585),
    "ein": TargetParams(J=1.084e27, D=9.065e24, dOmega=6.585),
}
fermi_diffuse_targets = {
    "nfw": TargetParams(J=1.695e28, D=3.563e25, dOmega=10.82),
    "ein": TargetParams(J=1.189e27, D=8.967e24, dOmega=10.82),
}
fermi_diffuse_targets_optimistic = {
    "nfw": TargetParams(J=2.213e28, D=3.708e25, dOmega=10.82),
    "ein": TargetParams(J=1.691e27, D=9.618e24, dOmega=10.82),
}
integral_diffuse_targets = {
    "nfw": TargetParams(J=2.086e29, D=7.301e25, dOmega=0.5421),
    "ein": TargetParams(J=1.694e28, D=1.979e25, dOmega=0.5421),
}
integral_diffuse_targets_optimistic = {
    "nfw": TargetParams(J=2.086e29, D=7.301e25, dOmega=0.5421),
    "ein": TargetParams(J=1.694e28, D=1.979e25, dOmega=0.5421),
}


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


# =========================
# ---- Effective Areas ----
# =========================

# Construct interpolating functions for effective areas
effective_area_adept = _generate_interp(
    "A_eff", "adept.dat", fill_value=0.0, bounds_error=False
)
effective_area_amego = _generate_interp(
    "A_eff", "amego.dat", fill_value=0.0, bounds_error=False
)
effective_area_comptel = _generate_interp(
    "A_eff", "comptel.dat", fill_value=0.0, bounds_error=False
)
effective_area_all_sky_astrogam = _generate_interp(
    "A_eff", "all_sky_astrogam.dat", fill_value=0.0, bounds_error=False
)
effective_area_e_astrogam = _generate_interp(
    "A_eff", "e_astrogam.dat", fill_value=0.0, bounds_error=False
)
effective_area_egret = _generate_interp(
    "A_eff", "egret.dat", fill_value=0.0, bounds_error=False
)
effective_area_fermi = _generate_interp(
    "A_eff", "fermi.dat", fill_value=0.0, bounds_error=False
)
effective_area_gecco = _generate_interp(
    "A_eff", "gecco.dat", fill_value=0.0, bounds_error=False
)
effective_area_grams = _generate_interp(
    "A_eff", "grams.dat", fill_value=0.0, bounds_error=False
)
effective_area_grams_upgrade = _generate_interp(
    "A_eff", "grams_upgrade.dat", fill_value=0.0, bounds_error=False
)
effective_area_mast = _generate_interp(
    "A_eff", "mast.dat", fill_value=0.0, bounds_error=False
)
effective_area_pangu = _generate_interp(
    "A_eff", "pangu.dat", fill_value=0.0, bounds_error=False
)


# These are for backwards compatability
A_eff_adept = effective_area_adept
A_eff_amego = effective_area_amego
A_eff_comptel = effective_area_comptel
A_eff_all_sky_astrogam = effective_area_all_sky_astrogam
A_eff_e_astrogam = effective_area_e_astrogam
A_eff_egret = effective_area_egret
A_eff_fermi = effective_area_fermi
A_eff_gecco = effective_area_gecco
A_eff_grams = effective_area_grams
A_eff_grams_upgrade = effective_area_grams_upgrade
A_eff_mast = effective_area_mast
A_eff_pangu = effective_area_pangu


# ============================
# ---- Energy Resolutions ----
# ============================

# Multiplicative factor to convert FWHM into standard deviations, assuming
# energy resolution function is a Gaussian
fwhm_factor = 1 / (2 * np.sqrt(2 * np.log(2)))

# Construct interpolating functions for energy resolutions
_e_res_amego_interp = _generate_interp(
    "energy_res", "amego.dat", fill_value="extrapolate", bounds_error=False
)
_e_res_all_sky_astrogam_interp = _generate_interp(
    "energy_res", "e_astrogam.dat", fill_value="extrapolate", bounds_error=False
)
_e_res_e_astrogam_interp = _generate_interp(
    "energy_res", "e_astrogam.dat", fill_value="extrapolate", bounds_error=False
)
_e_res_gecco_large_interp = _generate_interp(
    "energy_res", "gecco_large.dat", fill_value="extrapolate", bounds_error=False
)
_e_res_gecco_interp = _generate_interp(
    "energy_res", "gecco.dat", fill_value="extrapolate", bounds_error=False
)
_e_res_integral_interp = _generate_interp(
    "energy_res", "integral.dat", fill_value="extrapolate", bounds_error=False
)
_e_res_mast_interp = _generate_interp(
    "energy_res", "mast.dat", fill_value="extrapolate", bounds_error=False
)


def energy_res_adept(energy):
    """
    AdEPT energy resolution. See arXiv1311.2059. The energy dependence is not
    given.
    """
    return np.vectorize(lambda e: 0.3 * fwhm_factor)(energy)


def energy_res_amego(energy):
    """
    Energy resolution of AMEGO.
    """
    return _e_res_amego_interp(energy)


def energy_res_comptel(energy):
    r"""
    COMPTEL energy resolution :math:`\Delta E / E`.

    This is the most optimistic value, taken from `ch. II, page 11
    <https://scholars.unh.edu/dissertation/2045/>`_. The
    energy resolution at 1 MeV is 10% (FWHM).
    """
    return np.vectorize(lambda e: 0.05 * fwhm_factor)(energy)


def energy_res_all_sky_astrogam(energy):
    """
    Energy resolution of E-Astrogam.
    """
    return _e_res_all_sky_astrogam_interp(energy)


def energy_res_e_astrogam(energy):
    """
    Energy resolution of E-Astrogam.
    """
    return _e_res_e_astrogam_interp(energy)


def energy_res_egret(energy):
    """
    EGRET's energy resolution :math:`\\Delta E / E`.

    This is the most optimistic value, taken from
    `sec. 4.3.3 <http://adsabs.harvard.edu/doi/10.1086/191793>`_.
    """
    return np.vectorize(lambda e: 0.18 * fwhm_factor)(energy)


def energy_res_fermi(energy):
    """Fermi-LAT's energy resolution :math:`\Delta E / E`.

    This is the average of the most optimistic normal and 60deg off-axis values
    from `fig. 18 <https://arxiv.org/abs/0902.1089>`_.
    """
    return np.vectorize(lambda e: 0.075)(energy)


def energy_res_gecco(energy):
    """
    Energy resolution of E-Astrogam.
    """
    return _e_res_gecco_interp(energy)


def energy_res_gecco_large(energy):
    """
    Energy resolution of E-Astrogam.
    """
    return _e_res_gecco_large_interp(energy)


def energy_res_grams_upgrade(energy):
    """
    GRAMS upgrade approximate energy resolution. See https://arxiv.org/abs/1901.03430.
    """
    return np.vectorize(lambda e: 0.05)(energy)


def energy_res_grams(energy):
    """
    GRAMS approximate energy resolution. See https://arxiv.org/abs/1901.03430.
    """
    return np.vectorize(lambda e: 0.05)(energy)


def energy_res_integral(energy):
    """
    Energy resolution of integral.
    """
    return _e_res_integral_interp(energy)


def energy_res_mast(energy):
    """
    Energy resolution of E-Astrogam.
    """
    return _e_res_mast_interp(energy)


def energy_res_pangu(energy):
    """
    PANGU energy resolution. https://doi.org/10.22323/1.246.0069. There is not
    much energy dependence.
    """
    return np.vectorize(lambda e: 0.4)(energy)


# ==========================
# ---- Flux Measurments ----
# ==========================


def _generate_flux_measurement(subdir, filename, energy_res, target):
    path = os.path.join(grd_dir, subdir, filename)
    return FluxMeasurement.from_file(path, energy_res, target)


comptel_diffuse = _generate_flux_measurement(
    "obs",
    "comptel_diffuse.dat",
    energy_res_comptel,
    comptel_diffuse_target_optimistic,
)
egret_diffuse = _generate_flux_measurement(
    "obs", "egret_diffuse.dat", energy_res_egret, egret_diffuse_target
)
fermi_diffuse = _generate_flux_measurement(
    "obs", "fermi_diffuse.dat", energy_res_fermi, fermi_diffuse_target
)
integral_diffuse = _generate_flux_measurement(
    "obs", "integral_diffuse.dat", energy_res_integral, integral_diffuse_target
)

# ===========================
# ---- Background Models ----
# ===========================


def _generate_background_model(subdir, filename):
    interp = _generate_interp(subdir, filename)
    return BackgroundModel.from_interp(interp)


# This is the more complex background model from arXiv:1703.02546. Note that it
# is only applicable to the inner 10deg x 10deg region of the Milky Way.
gc_bg_model = _generate_background_model("bg_model", "gc.dat")
