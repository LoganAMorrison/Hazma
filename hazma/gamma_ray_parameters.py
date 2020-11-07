import os
import importlib.resources as pkg_resources
from collections import namedtuple

import numpy as np
from scipy.interpolate import interp1d

from hazma.background_model import BackgroundModel
from hazma.flux_measurement import FluxMeasurement
from hazma.target_params import TargetParams

"""

Parameters relevant to computing constraints from gamma ray experiments.

"""

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
integral_diffuse_target = TargetParams(J=2.086e29, D=7.301e25, dOmega=0.5421)

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

# Multiplicative factor to convert FWHM into standard deviations, assuming
# energy resolution function is a Gaussian
fwhm_factor = 1 / (2 * np.sqrt(2 * np.log(2)))


def energy_res_grams_upgrade(e):
    """
    GRAMS upgrade approximate energy resolution. See https://arxiv.org/abs/1901.03430.
    """
    return np.vectorize(lambda e: 0.05)(e)


def energy_res_grams(e):
    """
    GRAMS approximate energy resolution. See https://arxiv.org/abs/1901.03430.
    """
    return np.vectorize(lambda e: 0.05)(e)


def energy_res_adept(e):
    """
    AdEPT energy resolution. See arXiv1311.2059. The energy dependence is not
    given.
    """
    return np.vectorize(lambda e: 0.3 * fwhm_factor)(e)


def energy_res_pangu(e):
    """
    PANGU energy resolution. https://doi.org/10.22323/1.246.0069. There is not
    much energy dependence.
    """
    return np.vectorize(lambda e: 0.4)(e)


def energy_res_comptel(e):
    """COMPTEL energy resolution :math:`\Delta E / E`.

    This is the most optimistic value, taken from `ch. II, page 11
    <https://scholars.unh.edu/dissertation/2045/>`_. The
    energy resolution at 1 MeV is 10% (FWHM).
    """
    return np.vectorize(lambda e: 0.05 * fwhm_factor)(e)


def energy_res_egret(e):
    """EGRET's energy resolution :math:`\Delta E / E`.

    This is the most optimistic value, taken from
    `sec. 4.3.3 <http://adsabs.harvard.edu/doi/10.1086/191793>`_.
    """
    return np.vectorize(lambda e: 0.18 * fwhm_factor)(e)


def energy_res_fermi(e):
    """Fermi-LAT's energy resolution :math:`\Delta E / E`.

    This is the average of the most optimistic normal and 60deg off-axis values
    from `fig. 18 <https://arxiv.org/abs/0902.1089>`_.
    """
    return np.vectorize(lambda e: 0.075)(e)


# Effective areas, cm^2
a_eff_prefix = "A_eff"
a_eff_pkg = "hazma.gamma_ray_data." + a_eff_prefix
a_eff_rf_names = [n for n in pkg_resources.contents(a_eff_pkg) if n.endswith(".dat")]
for name in a_eff_rf_names:
    with pkg_resources.path(a_eff_pkg, name) as path:
        var_name = a_eff_prefix + "_" + os.path.splitext(name)[0]
        var_val = interp1d(
            *np.loadtxt(path, delimiter=",", unpack=True),
            bounds_error=False,
            fill_value=0.0
        )
        globals()[var_name] = var_val

# Energy resolutions, Delta E / E
e_res_prefix = "energy_res"
e_res_pkg = "hazma.gamma_ray_data." + e_res_prefix
e_res_rf_names = [n for n in pkg_resources.contents(e_res_pkg) if n.endswith(".dat")]
for name in e_res_rf_names:
    with pkg_resources.path(e_res_pkg, name) as path:
        var_name = e_res_prefix + "_" + os.path.splitext(name)[0]
        var_val = interp1d(
            *np.loadtxt(path, delimiter=",", unpack=True),
            fill_value="extrapolate"
        )
        globals()[var_name] = var_val

# Package the measurements
obs_pkg = "hazma.gamma_ray_data.obs"
obs_rf_names = [n for n in pkg_resources.contents(obs_pkg) if n.endswith(".dat")]
for name in obs_rf_names:
    with pkg_resources.path(obs_pkg, name) as path:
        obs = os.path.splitext(name)[0]
        telescope = "_".join(obs.split("_")[:-1])
        var_val = FluxMeasurement.from_file(
            path, eval("energy_res_" + telescope), eval(obs + "_target")
        )
        globals()[obs] = var_val

        if obs == "comptel_diffuse":
            comptel_diffuse_optimistic = FluxMeasurement.from_file(
                path, energy_res_comptel, comptel_diffuse_target_optimistic
            )

# This is the more complex background model from arXiv:1703.02546. Note that it
# is only applicable to the inner 10deg x 10deg region of the Milky Way.
bg_suffix = "bg_model"
bg_pkg = "hazma.gamma_ray_data." + bg_suffix
bg_rf_names = [n for n in pkg_resources.contents(bg_pkg) if n.endswith(".dat")]
for name in bg_rf_names:
    with pkg_resources.path(bg_pkg, name) as path:
        var_name = os.path.splitext(name)[0] + "_" + bg_suffix
        var_val = BackgroundModel.from_interp(
            interp1d(
                *np.loadtxt(path, delimiter=",", unpack=True),
                bounds_error=True,
                fill_value=np.nan
            )
        )
        globals()[var_name] = var_val
