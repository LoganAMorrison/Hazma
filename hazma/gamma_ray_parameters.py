"""
Parameters relevant to computing constraints from gamma ray experiments.


"""

import os
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
from scipy import interpolate

from hazma.background_model import BackgroundModel, ParametricBackgroundModel
from hazma.flux_measurement import FluxMeasurement
from hazma.target_params import TargetParams

# Directory of this file
_dir = Path(__file__).parent.absolute()
# Directory to gamma_ray_data
grd_dir = os.path.join(_dir, "gamma_ray_data")


def _generate_interp(subdir, filename, fill_value=np.nan, bounds_error=True):
    path = os.path.join(grd_dir, subdir, filename)
    data = np.genfromtxt(path, delimiter=",", unpack=True)
    return interpolate.interp1d(*data, bounds_error=bounds_error, fill_value=fill_value)


# From Alex Moiseev's slides. Ref: G. Weidenspointner et al, AIP 510, 467, 2000.
# Additional factor of two due to uncertainty about radioactive and
# instrumental backgrounds.
gecco_bg_model = BackgroundModel([0.2, 4e3], lambda e_gam: 2 * 4e-3 / e_gam**2)

# This is the background model from arXiv:1504.04024, eq. 14. It was derived
# by performing a simple power law fit to COMPTEL data from 0.8 - 30 MeV and
# EGRET data from 30 MeV - 10 GeV. We take the lower range of validity to be
# the lowest energy for which e-ASTROGAM has nonzero effective area.
default_bg_model = BackgroundModel([0.0, 10.0e3], lambda e: 2.74e-3 / e**2)


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


# def solid_angle_rect(l_max, b_min, b_max):
#     """
#     Returns solid angle subtended for a rectangular target region centered on
#     the galactic center.
#
#     Parameters
#     ----------
#     l_max : float
#         Maximum value of galactic longitude in deg. Note that :math:`l` must
#         lie in the interval :math:`[-180, 180]`.
#     b_min, b_max : float, float
#         Minimum and maximum values for :math:`b` in deg. Note that :math:`b`
#         must lie in the interval :math:`[-90, 90]`, with the equator at
#         :math:`b = 0`.
#
#     Returns
#     -------
#     Omega : float
#         Solid angle subtended by the region in sr.
#     """
#     deg_to_rad = np.pi / 180.0
#     return (
#         4.0
#         * l_max
#         * deg_to_rad
#         * (np.sin(b_max * deg_to_rad) - np.sin(b_min * deg_to_rad))
#     )


# ==================
# ---- Targets -----
# ==================

#: Several different GC targets. Halo parameters are from the DM fit using
#: baryonic model B2 in https://arxiv.org/abs/1906.06133 (table III).
gc_targets = {
    "nfw": {
        "1 arcmin cone": TargetParams(J=6.972e32, D=4.84e26, dOmega=2.66e-7),
        "5 deg cone": TargetParams(J=1.782e30, D=1.597e26, dOmega=0.0239),
        "10 deg cone": TargetParams(J=7.458e29, D=1.214e26, dOmega=0.0955),
        "10x10 deg box": TargetParams(J=2.879e30, D=1.955e26, dOmega=0.03042),
    },
    "ein": {
        "1 arcmin cone": TargetParams(J=1.724e32, D=5.413e26, dOmega=2.66e-7),
        "5 deg cone": TargetParams(J=4.442e30, D=2.269e26, dOmega=0.0239),
        "10 deg cone": TargetParams(J=1.706e30, D=1.615e26, dOmega=0.0955),
        "10x10 deg box": TargetParams(J=3.99e31, D=3.974e26, dOmega=0.03042),
    },
}
# +/- 1 sigma
gc_targets_optimistic = {
    "nfw": {
        "1 arcmin cone": TargetParams(J=1.415e33, D=6.666e26, dOmega=2.66e-7),
        "5 deg cone": TargetParams(J=3.372e30, D=2.058e26, dOmega=0.0239),
        "10 deg cone": TargetParams(J=1.355e30, D=1.522e26, dOmega=0.0955),
        "10x10 deg box": TargetParams(J=2.879e30, D=1.955e26, dOmega=0.03042),
    },
    "ein": {
        "1 arcmin cone": TargetParams(J=5.987e34, D=4.179e27, dOmega=2.66e-7),
        "5 deg cone": TargetParams(J=4.965e31, D=4.345e26, dOmega=0.0239),
        "10 deg cone": TargetParams(J=1.404e31, D=2.62e26, dOmega=0.0955),
        "10x10 deg box": TargetParams(J=3.99e31, D=3.974e26, dOmega=0.03042),
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

#: COMPTEL diffuse targets
comptel_diffuse_targets = {
    "nfw": TargetParams(J=9.308e28, D=4.866e25, dOmega=1.433),
    "ein": TargetParams(J=1.751e29, D=5.541e25, dOmega=1.433),
}
#: COMPTEL diffuse targets with optimistic parameters
comptel_diffuse_targets_optimistic = {
    "nfw": TargetParams(J=1.53e29, D=5.571e25, dOmega=1.433),
    "ein": TargetParams(J=1.04e30, D=7.098e25, dOmega=1.433),
}
#: EGRET diffuse targets
egret_diffuse_targets = {
    "nfw": TargetParams(J=6.265e27, D=1.71e25, dOmega=6.585),
    "ein": TargetParams(J=6.994e27, D=1.738e25, dOmega=6.585),
}
#: EGRET diffuse targets with optimistic parameters
egret_diffuse_targets_optimistic = {
    "nfw": TargetParams(J=7.556e27, D=1.761e25, dOmega=6.585),
    "ein": TargetParams(J=9.062e27, D=1.952e25, dOmega=6.585),
}
#: Fermi-LAT diffuse targets
fermi_diffuse_targets = {
    "nfw": TargetParams(J=8.475e27, D=1.782e25, dOmega=10.82),
    "ein": TargetParams(J=1.058e28, D=1.832e25, dOmega=10.82),
}
#: Fermi-LAT diffuse targets with optimistic parameters
fermi_diffuse_targets_optimistic = {
    "nfw": TargetParams(J=1.106e28, D=1.854e25, dOmega=10.82),
    "ein": TargetParams(J=1.601e28, D=2.084e25, dOmega=10.82),
}
#: INTEGRAL diffuse targets
integral_diffuse_targets = {
    "nfw": TargetParams(J=2.086e29, D=7.301e25, dOmega=0.5421),
    "ein": TargetParams(J=4.166e29, D=8.76e25, dOmega=0.5421),
}
#: INTEGRAL diffuse targets with optimistic parameters
integral_diffuse_targets_optimistic = {
    "nfw": TargetParams(J=2.086e29, D=7.301e25, dOmega=0.5421),
    "ein": TargetParams(J=4.166e29, D=8.76e25, dOmega=0.5421),
}


#: Draco dwarf
draco_targets = {
    "nfw": {
        "1 arcmin cone": TargetParams(J=3.418e30, D=5.949e25, dOmega=2.66e-7),
        "5 deg cone": TargetParams(J=8.058e26, D=1.986e24, dOmega=0.0239),
    },
}

#: Andromeda target.
#: See Sofue 2015, https://arxiv.org/abs/1504.05368
m31_targets = {
    "nfw": {
        "1 arcmin cone": TargetParams(J=7.116e29, D=9.449e25, dOmega=2.66e-7),
        "5 deg cone": TargetParams(J=2.639e26, D=5.507e24, dOmega=0.0239),
    },
}

#: Fornax cluster.
#: See https://arxiv.org/abs/1009.5988.
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
__effective_area_adept = _generate_interp(
    "A_eff", "adept.dat", fill_value=0.0, bounds_error=False
)
__effective_area_amego = _generate_interp(
    "A_eff", "amego.dat", fill_value=0.0, bounds_error=False
)
__effective_area_comptel = _generate_interp(
    "A_eff", "comptel.dat", fill_value=0.0, bounds_error=False
)
__effective_area_all_sky_astrogam = _generate_interp(
    "A_eff", "all_sky_astrogam.dat", fill_value=0.0, bounds_error=False
)
__effective_area_e_astrogam = _generate_interp(
    "A_eff", "e_astrogam.dat", fill_value=0.0, bounds_error=False
)
__effective_area_egret = _generate_interp(
    "A_eff", "egret.dat", fill_value=0.0, bounds_error=False
)
__effective_area_fermi = _generate_interp(
    "A_eff", "fermi.dat", fill_value=0.0, bounds_error=False
)
__effective_area_gecco = _generate_interp(
    "A_eff", "gecco.dat", fill_value=0.0, bounds_error=False
)
__effective_area_grams = _generate_interp(
    "A_eff", "grams.dat", fill_value=0.0, bounds_error=False
)
__effective_area_grams_upgrade = _generate_interp(
    "A_eff", "grams_upgrade.dat", fill_value=0.0, bounds_error=False
)
__effective_area_mast = _generate_interp(
    "A_eff", "mast.dat", fill_value=0.0, bounds_error=False
)
__effective_area_pangu = _generate_interp(
    "A_eff", "pangu.dat", fill_value=0.0, bounds_error=False
)

# https://arxiv.org/abs/1107.0200


def effective_area_amego(energy):
    """
    Compute the effective area of the All-sky Medium Energy Gamma-ray
    Observatory (AMEGO) [1]_.

    Parameters
    ----------
    energy: array-like
        Energy where the effective area should be evaluated.

    Returns
    -------
    a_eff: array-like
        Effective area Aeff(E).

    References
    ----------
    .. [1] McEnery, Julie, et al. "All-sky medium energy gamma-ray observatory:
    exploring the extreme multimessenger universe." arXiv preprint
    arXiv:1907.07558 (2019).
    """
    return __effective_area_amego(energy)


def effective_area_comptel(energy):
    """
    Compute the effective area of the COMPTEL telescope [1]_.

    Parameters
    ----------
    energy: array-like
        Energy where the effective area should be evaluated.

    Returns
    -------
    a_eff: array-like
        Effective area Aeff(E).

    References
    ----------
    .. [1] Denherder, J. W., et al. "COMPTEL: Instrument description and
    performance." NASA. Goddard Space Flight Center, The Compton Observatory
    Science Workshop. 1992.
    """
    return __effective_area_comptel(energy)


def effective_area_egret(energy):
    """
    Compute the effective area of the EGRET telescope [1]_.

    Parameters
    ----------
    energy: array-like
        Energy where the effective area should be evaluated.

    Returns
    -------
    a_eff: array-like
        Effective area Aeff(E).

    References
    ----------
    .. [1] Strong, Andrew W., Igor V. Moskalenko, and Olaf Reimer. "Diffuse
    galactic continuum gamma rays: a model compatible with EGRET data and
    cosmic-ray measurements." The Astrophysical Journal 613.2 (2004): 962.
    """
    return __effective_area_egret(energy)


def effective_area_fermi(energy):
    """
    Compute the effective area of the Fermi telescope [1]_.

    Parameters
    ----------
    energy: array-like
        Energy where the effective area should be evaluated.

    Returns
    -------
    a_eff: array-like
        Effective area Aeff(E).

    References
    ----------
    .. [1] Ackermann, Markus, et al. "Fermi-LAT observations of the diffuse
    γ-ray emission: implications for cosmic rays and the interstellar medium."
    The Astrophysical Journal 750.1 (2012): 3.
    """
    return __effective_area_fermi(energy)


def effective_area_adept(energy):
    """
    Compute the effective area of the proposed Advanced Energetic Pair
    Telescope (AdEPT) [1]_.

    Parameters
    ----------
    energy: array-like
        Energy where the effective area should be evaluated.

    Returns
    -------
    a_eff: array-like
        Effective area Aeff(E).

    References
    ----------
    .. [1] Hunter, Stanley D., et al. "Development of the Advance Energetic
    Pair Telescope (AdEPT) for medium-energy gamma-ray astronomy." Space
    Telescopes and Instrumentation 2010: Ultraviolet to Gamma Ray. Vol. 7732.
    SPIE, 2010.
    """
    return __effective_area_adept(energy)


def effective_area_all_sky_astrogam(energy):
    """
    Compute the effective area of the proposed All-Sky-ASTROGAM telescope [1]_.

    Parameters
    ----------
    energy: array-like
        Energy where the effective area should be evaluated.

    Returns
    -------
    a_eff: array-like
        Effective area Aeff(E).

    References
    ----------
    .. [1] Mallamaci, Manuela, et al. "All-Sky-ASTROGAM: a MeV Companion for
    Multimessenger Astrophysics." 36th International Cosmic Ray Conference
    (ICRC2019). Vol. 36. 2019.
    """
    return __effective_area_all_sky_astrogam(energy)


def effective_area_e_astrogam(energy):
    """
    Compute the effective area of the proposed e-ASTROGAM telescope [1]_.

    Parameters
    ----------
    energy: array-like
        Energy where the effective area should be evaluated.

    Returns
    -------
    a_eff: array-like
        Effective area Aeff(E).

    References
    ----------
    .. [1] De Angelis, Alejandro, et al. "Science with e-ASTROGAM: A space
    mission for MeV–GeV gamma-ray astrophysics." Journal of High Energy
    Astrophysics 19 (2018): 1-106.
    """
    return __effective_area_e_astrogam(energy)


def effective_area_gecco(energy):
    """
    Compute the effective area of proposed Galactic Explorer with a Coded
    Aperture Mask Compton Telescope (GECCO) [1]_.

    Parameters
    ----------
    energy: array-like
        Energy where the effective area should be evaluated.

    Returns
    -------
    a_eff: array-like
        Effective area Aeff(E).

    References
    ----------
    .. [1] Orlando, Elena, et al. "Exploring the MeV Sky with a Combined Coded
    Mask and Compton Telescope: The Galactic Explorer with a Coded Aperture
    Mask Compton Telescope (GECCO)." arXiv preprint arXiv:2112.07190 (2021).
    """
    return __effective_area_gecco(energy)


def effective_area_grams(energy):
    """
    Compute the effective area of the proposed Dual MeV Gamma-Ray and Dark
    Matter Observator (GRAMS) [1]_.

    Parameters
    ----------
    energy: array-like
        Energy where the effective area should be evaluated.

    Returns
    -------
    a_eff: array-like
        Effective area Aeff(E).

    References
    ----------
    .. [1] Aramaki, Tsuguo, et al. "Dual MeV gamma-ray and dark matter
    observatory-GRAMS Project." Astroparticle Physics 114 (2020): 107-114.
    """
    return __effective_area_grams(energy)


def effective_area_grams_upgrade(energy):
    """
    Compute the effective area of the proposed Dual MeV Gamma-Ray and Dark
    Matter Observator (GRAMS) [1]_.

    Parameters
    ----------
    energy: array-like
        Energy where the effective area should be evaluated.

    Returns
    -------
    a_eff: array-like
        Effective area Aeff(E).

    References
    ----------
    .. [1] Aramaki, Tsuguo, et al. "Snowmass 2021 Letter of Interest: The GRAMS
    Project: MeV Gamma-Ray Observations and Antimatter-Based Dark Matter
    Searches." arXiv preprint arXiv:2009.03754 (2020).
    """
    return __effective_area_grams_upgrade(energy)


def effective_area_mast(energy):
    """
    Compute the effective area of the proposed Massive Argon Space Telescope
    (MAST) [1]_.

    Parameters
    ----------
    energy: array-like
        Energy where the effective area should be evaluated.

    Returns
    -------
    a_eff: array-like
        Effective area Aeff(E).

    References
    ----------
    .. [1] Dzhatdoev, Timur, and Egor Podlesnyi. "Massive Argon Space Telescope
    (MAST): A concept of heavy time projection chamber for γ-ray astronomy in
    the 100 MeV–1 TeV energy range." Astroparticle Physics 112 (2019): 1-7.
    """
    return __effective_area_mast(energy)


def effective_area_pangu(energy):
    """
    Compute the effective area of proposed PAir-productioN Gamma-ray Unit
    (PANGU) [1]_.

    Parameters
    ----------
    energy: array-like
        Energy where the effective area should be evaluated.

    Returns
    -------
    a_eff: array-like
        Effective area Aeff(E).

    References
    ----------
    .. [1] Wu, Xin, et al. "PANGU: a high resolution gamma-ray space
    telescope." Space Telescopes and Instrumentation 2014: Ultraviolet to Gamma
    Ray. Vol. 9144. International Society for Optics and Photonics, 2014.
    """
    return __effective_area_pangu(energy)


# These are for backwards compatability
effective_area_adept.x = __effective_area_adept.x
effective_area_amego.x = __effective_area_amego.x
effective_area_comptel.x = __effective_area_comptel.x
effective_area_all_sky_astrogam.x = __effective_area_all_sky_astrogam.x
effective_area_e_astrogam.x = __effective_area_e_astrogam.x
effective_area_egret.x = __effective_area_egret.x
effective_area_fermi.x = __effective_area_fermi.x
effective_area_gecco.x = __effective_area_gecco.x
effective_area_grams.x = __effective_area_grams.x
effective_area_grams_upgrade.x = __effective_area_grams_upgrade.x
effective_area_mast.x = __effective_area_mast.x
effective_area_pangu.x = __effective_area_pangu.x


# These are for backwards compatability
A_eff_adept = __effective_area_adept
A_eff_amego = __effective_area_amego
A_eff_comptel = __effective_area_comptel
A_eff_all_sky_astrogam = __effective_area_all_sky_astrogam
A_eff_e_astrogam = __effective_area_e_astrogam
A_eff_egret = __effective_area_egret
A_eff_fermi = __effective_area_fermi
A_eff_gecco = __effective_area_gecco
A_eff_grams = __effective_area_grams
A_eff_grams_upgrade = __effective_area_grams_upgrade
A_eff_mast = __effective_area_mast
A_eff_pangu = __effective_area_pangu


# ============================
# ---- Energy Resolutions ----
# ============================

# Multiplicative factor to convert FWHM into standard deviations, assuming
# energy resolution function is a Gaussian
fwhm_factor = 1 / (2 * np.sqrt(2 * np.log(2)))

# Construct interpolating functions for energy resolutions
_e_res_amego_interp = _generate_interp(
    "energy_res",
    "amego.dat",
    fill_value="extrapolate",  # type: ignore
    bounds_error=False,
)
_e_res_all_sky_astrogam_interp = _generate_interp(
    "energy_res",
    "e_astrogam.dat",
    fill_value="extrapolate",  # type: ignore
    bounds_error=False,
)
_e_res_e_astrogam_interp = _generate_interp(
    "energy_res",
    "e_astrogam.dat",
    fill_value="extrapolate",  # type: ignore
    bounds_error=False,
)
_e_res_gecco_large_interp = _generate_interp(
    "energy_res",
    "gecco_large.dat",
    fill_value="extrapolate",  # type: ignore
    bounds_error=False,
)
_e_res_gecco_interp = _generate_interp(
    "energy_res",
    "gecco.dat",
    fill_value="extrapolate",  # type: ignore
    bounds_error=False,
)
_e_res_integral_interp = _generate_interp(
    "energy_res",
    "integral.dat",
    fill_value="extrapolate",  # type: ignore
    bounds_error=False,
)
_e_res_mast_interp = _generate_interp(
    "energy_res",
    "mast.dat",
    fill_value="extrapolate",  # type: ignore
    bounds_error=False,
)


def energy_res_adept(energy):
    r"""Energy resolution for the AdEPT telescope [1]_ [2]_.

    Note that the energy dependence fro AdEPT was not specified. We thus take
    it to be constant.

    Parameters
    ----------
    energy: array-like
        Energy where the energy resoltuon should be evaluated.

    Returns
    -------
    e_res: array-like
        Energy resoluton delta_e(e) / e.

    References
    ----------
    .. [1] Hunter, Stanley D., et al. "Development of the Advance Energetic
    Pair Telescope (AdEPT) for medium-energy gamma-ray astronomy." Space
    Telescopes and Instrumentation 2010: Ultraviolet to Gamma Ray. Vol. 7732.
    SPIE, 2010.

    .. [2] Hunter, Stanley D., et al. "A pair production telescope for
    medium-energy gamma-ray polarimetry." Astroparticle physics 59 (2014):
    18-28.
    """
    return np.vectorize(lambda _: 0.3 * fwhm_factor)(energy)


def energy_res_amego(energy):
    r"""Compute the energy resolution of the All-sky Medium Energy Gamma-ray
    Observatory (AMEGO) [1]_.

    Parameters
    ----------
    energy: array-like
        Energy where the energy resoltuon should be evaluated.

    Returns
    -------
    e_res: array-like
        Energy resoluton delta_e(e) / e.

    References
    ----------
    .. [1] McEnery, Julie, et al. "All-sky medium energy gamma-ray observatory:
    exploring the extreme multimessenger universe." arXiv preprint
    arXiv:1907.07558 (2019).
    """
    return _e_res_amego_interp(energy)


def energy_res_comptel(energy):
    r"""Compute the energy resolution :math:`\Delta E / E` of the COMPTEL [1]_.

    This is the most optimistic value, taken from chapter II, page 11 of [2]_.
    The energy resolution at 1 MeV is 10% (FWHM).

    Parameters
    ----------
    energy: array-like
        Energy where the energy resoltuon should be evaluated.

    Returns
    -------
    e_res: array-like
        Energy resoluton delta_e(e) / e.

    References
    ----------
    .. [1] Denherder, J. W., et al. "COMPTEL: Instrument description and
    performance." NASA. Goddard Space Flight Center, The Compton Observatory
    Science Workshop. 1992.

    .. [2] Kappadath, Srinivas Cheenu. Measurement of the cosmic diffuse
    gamma-ray spectrum from 800 keV to 30 MeV. University of New Hampshire,
    1998.
    """
    return np.vectorize(lambda _: 0.05 * fwhm_factor)(energy)


def energy_res_all_sky_astrogam(energy):
    r"""Compute the energy resolution of the proposed All-Sky-ASTROGAM
    telescope [1]_.

    Parameters
    ----------
    energy: array-like
        Energy where the energy resoltuon should be evaluated.

    Returns
    -------
    e_res: array-like
        Energy resoluton delta_e(e) / e.

    References
    ----------
    .. [1] Mallamaci, Manuela, et al. "All-Sky-ASTROGAM: a MeV Companion for
    Multimessenger Astrophysics." 36th International Cosmic Ray Conference
    (ICRC2019). Vol. 36. 2019.
    """
    return _e_res_all_sky_astrogam_interp(energy)


def energy_res_e_astrogam(energy):
    r"""Compute the energy resoluton of the proposed e-ASTROGAM telescope [1]_.

    Parameters
    ----------
    energy: array-like
        Energy where the energy resoltuon should be evaluated.

    Returns
    -------
    e_res: array-like
        Energy resoluton delta_e(e) / e.

    References
    ----------
    .. [1] De Angelis, Alejandro, et al. "Science with e-ASTROGAM: A space
    mission for MeV–GeV gamma-ray astrophysics." Journal of High Energy
    Astrophysics 19 (2018): 1-106.
    """
    return _e_res_e_astrogam_interp(energy)


def energy_res_egret(energy):
    r"""Compute the energy resoluton :math:`\Delta E / E` of the EGRET
    telescope [1]_.

    This is the most optimistic value, taken from section 4.3.3 of [2]_.

    Parameters
    ----------
    energy: array-like
        Energy where the energy resoltuon should be evaluated.

    Returns
    -------
    e_res: array-like
        Energy resoluton delta_e(e) / e.

    References
    ----------
    .. [1] Strong, Andrew W., Igor V. Moskalenko, and Olaf Reimer. "Diffuse
    galactic continuum gamma rays: a model compatible with EGRET data and
    cosmic-ray measurements." The Astrophysical Journal 613.2 (2004): 962.

    .. [2] Thompson, D. J., et al. "Calibration of the energetic gamma-ray
    experiment telescope (EGRET) for the Compton gamma-ray observatory." The
    astrophysical Journal supplement series 86 (1993): 629-656.
    """
    return np.vectorize(lambda _: 0.18 * fwhm_factor)(energy)


def energy_res_fermi(energy):
    r"""Compute the energy resolution :math:`\Delta E / E` of the Fermi-LAT
    telescope.

    This is the average of the most optimistic normal and 60deg off-axis values
    from Fig. (18) of [2]_.

    Parameters
    ----------
    energy: array-like
        Energy where the energy resoltuon should be evaluated.

    Returns
    -------
    e_res: array-like
        Energy resoluton delta_e(e) / e.

    References
    ----------
    .. [1] Ackermann, Markus, et al. "Fermi-LAT observations of the diffuse
    γ-ray emission: implications for cosmic rays and the interstellar medium."
    The Astrophysical Journal 750.1 (2012): 3.

    .. [2] Atwood, W. B., et al. "The large area telescope on the Fermi
    gamma-ray space telescope mission." The Astrophysical Journal 697.2 (2009):
    1071.
    """
    return np.vectorize(lambda _: 0.075)(energy)


def energy_res_gecco(energy):
    r"""
    Compute the energy resolution :math:`\Delta E / E` of proposed Galactic
    Explorer with a Coded Aperture Mask Compton Telescope (GECCO) [1]_.

    Parameters
    ----------
    energy: array-like
        Energy where the energy resoltuon should be evaluated.

    Returns
    -------
    e_res: array-like
        Energy resoluton delta_e(e) / e.

    References
    ----------
    .. [1] Orlando, Elena, et al. "Exploring the MeV Sky with a Combined Coded
    Mask and Compton Telescope: The Galactic Explorer with a Coded Aperture
    Mask Compton Telescope (GECCO)." arXiv preprint arXiv:2112.07190 (2021).
    """
    return _e_res_gecco_interp(energy) * fwhm_factor


def energy_res_gecco_large(energy):
    r"""
    Compute the energy resolution :math:`\Delta E / E` of proposed Galactic
    Explorer with a Coded Aperture Mask Compton Telescope (GECCO).

    Parameters
    ----------
    energy: array-like
        Energy where the energy resoltuon should be evaluated.

    Returns
    -------
    e_res: array-like
        Energy resoluton delta_e(e) / e.

    References
    ----------
    .. [1] Orlando, Elena, et al. "Exploring the MeV Sky with a Combined Coded
    Mask and Compton Telescope: The Galactic Explorer with a Coded Aperture
    Mask Compton Telescope (GECCO)." arXiv preprint arXiv:2112.07190 (2021).
    """
    return _e_res_gecco_large_interp(energy)


def energy_res_grams_upgrade(energy):
    r"""
    Compute the energy resolution :math:`\Delta E / E` of the proposed GRAMS
    upgrade [1]_.

    Parameters
    ----------
    energy: array-like
        Energy where the energy resoltuon should be evaluated.

    Returns
    -------
    e_res: array-like
        Energy resoluton delta_e(e) / e.

    References
    ----------
    .. [1] Aramaki, Tsuguo, et al. "Snowmass 2021 Letter of Interest: The GRAMS
    Project: MeV Gamma-Ray Observations and Antimatter-Based Dark Matter
    Searches." arXiv preprint arXiv:2009.03754 (2020).
    """
    return np.vectorize(lambda _: 0.05)(energy)


def energy_res_grams(energy):
    r"""
    Compute the energy resolution :math:`\Delta E / E` of GRAMS [1]_.

    Parameters
    ----------
    energy: array-like
        Energy where the energy resoltuon should be evaluated.

    Returns
    -------
    e_res: array-like
        Energy resoluton delta_e(e) / e.

    References
    ----------
    .. [1] Aramaki, Tsuguo, et al. "Dual MeV gamma-ray and dark matter
    observatory-GRAMS Project." Astroparticle Physics 114 (2020): 107-114.
    """
    return np.vectorize(lambda _: 0.05)(energy)


def energy_res_integral(energy):
    r"""
    Compute the energy resolution :math:`\Delta E / E` of INTEGRAL.

    Parameters
    ----------
    energy: array-like
        Energy where the energy resoltuon should be evaluated.

    Returns
    -------
    e_res: array-like
        Energy resoluton delta_e(e) / e.

    References
    ----------
    .. [1] Bouchet, Laurent, et al. "Diffuse emission measurement with the
    spectrometer on INTEGRAL as an indirect probe of cosmic-ray electrons and
    positrons." The Astrophysical Journal 739.1 (2011): 29.
    """
    return _e_res_integral_interp(energy)


def energy_res_mast(energy):
    r"""
    Compute the energy resolution :math:`\Delta E / E` of the proposed Massive
    Argon Space Telescope (MAST) [1]_.

    Parameters
    ----------
    energy: array-like
        Energy where the energy resoltuon should be evaluated.

    Returns
    -------
    e_res: array-like
        Energy resoluton delta_e(e) / e.

    References
    ----------
    .. [1] Dzhatdoev, Timur, and Egor Podlesnyi. "Massive Argon Space Telescope
    (MAST): A concept of heavy time projection chamber for γ-ray astronomy in
    the 100 MeV–1 TeV energy range." Astroparticle Physics 112 (2019): 1-7.
    """
    return _e_res_mast_interp(energy)


def energy_res_pangu(energy):
    r"""
    Compute the energy resolution :math:`\Delta E / E` of proposed
    PAir-productioN Gamma-ray Unit (PANGU) [1]_.

    Parameters
    ----------
    energy: array-like
        Energy where the energy resoltuon should be evaluated.

    Returns
    -------
    e_res: array-like
        Energy resoluton delta_e(e) / e.

    References
    ----------
    .. [1] Wu, Xin, et al. "PANGU: a high resolution gamma-ray space telescope."
    Space Telescopes and Instrumentation 2014: Ultraviolet to Gamma Ray. Vol.
    9144. International Society for Optics and Photonics, 2014.
    """
    return np.vectorize(lambda _: 0.4)(energy)


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
    comptel_diffuse_target,
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


#: This is the more complex background model from arXiv:1703.02546. Note that it
#: is only applicable to the inner 10deg x 10deg region of the Milky Way.
gc_bg_model = _generate_background_model("bg_model", "gc.dat")


class GalacticCenterBackgroundModel(ParametricBackgroundModel):
    """Model for the Galactic astrophysical background.

    The total background flux is modeled using:
        ɸ = Ag * (E / 1 MeV)^(-ɑ_g) * exp(-(E/Ec)^Ɣ) + Aeg * (E / 1 MeV)^(-ɑ_eg)
    where the parameters of the model are:
        - Ag: Amplitude of the Galactic astrophysical background
          [MeV⁻¹ cm⁻² s⁻¹ sr⁻¹]
        - ɑ_g: Power-law index of the Galactic astrophysical background,
        - Ec: Exponential cutoff energy of the Galactic astrophysical background
          [MeV],
        - gam: Exponential power-law index of the Galactic astrophysical background,
        - Aeg: Amplitude of the extra-Galactic astrophysical background
          [MeV⁻¹ cm⁻² s⁻¹ sr⁻¹],
        - ɑ_eg: Power-law index of the extra-Galactic astrophysical background.
    This model is valid for energies between [0.15 MeV, 5.0 MeV] and
    for |l| <= 5 deg, |b| <= 5 deg.

    See arXiv: 2102.06714 for details.

    """

    def __init__(
        self,
        galactic_amplitude: float = 0.013,
        galactic_power_law_index: float = 1.8,
        galactic_exponential_cutoff: float = 2.0,
        galactic_exponential_index: float = 2.0,
        extra_galactic_amplitude: float = 0.004135,
        extra_galactic_power_law_index: float = 2.8956,
    ):
        """
        Parameters
        ----------
        galactic_amplitude: float
            Amplitude of the Galactic astrophysical background.
            Default is 0.013 [MeV⁻¹ cm⁻² s⁻¹ sr⁻¹].
        galactic_power_law_index: float
            Power-law index of the Galactic astrophysical background.
            Default is 1.8.
        galactic_exponential_cutoff: float
            Exponential cutoff energy of the Galactic astrophysical background.
            Default is 2.0 [MeV].
        galactic_exponential_index: float
            Exponential power-law index of the Galactic astrophysical background.
            Default is 2.0.
        extra_galactic_amplitude: float
            Amplitude of the extra-Galactic astrophysical background.
            Default is 0.004135 [MeV⁻¹ cm⁻² s⁻¹ sr⁻¹].
        extra_galactic_power_law_index: float
            Power-law index of the extra-Galactic astrophysical background.
            Default is 2.8956.
        """
        super().__init__()
        self._galactic_amplitude = galactic_amplitude
        self._galactic_power_law_index = galactic_power_law_index
        self._galactic_exponential_cutoff = galactic_exponential_cutoff
        self._galactic_exponential_index = galactic_exponential_index
        self._extra_galactic_amplitude = extra_galactic_amplitude
        self._extra_galactic_power_law_index = extra_galactic_power_law_index
        self._energy_bounds = (0.15, 5.0)

    @property
    def galactic_amplitude(self) -> float:
        """Amplitude of the Galactic astrophysical background in
        units of [MeV⁻¹ cm⁻² s⁻¹ sr⁻¹].

        The total background flux is
            ɸ = Ag * (E / 1 MeV)^(-ɑ_g) * exp(-(E/Ec)^Ɣ) + Aeg * (E / 1 MeV)^(-ɑ_eg)
               ^--^
        with `Ag` the galactic amplitude.
        """
        return self._galactic_amplitude

    @property
    def galactic_power_law_index(self) -> float:
        """Power-law index of the Galactic astrophysical background.

        The total background flux is
            ɸ = Ag * (E / 1 MeV)^(-ɑ_g) * exp(-(E/Ec)^Ɣ) + Aeg * (E / 1 MeV)^(-ɑ_eg)
                     ^----------------^
        with `ɑ_g` the power-law index.
        """
        return self._galactic_power_law_index

    @property
    def galactic_exponential_cutoff(self) -> float:
        """Exponential cutoff of the Galactic astrophysical background.

        The total background flux is
            ɸ = Ag * (E / 1 MeV)^(-ɑ_g) * exp(-(E/Ec)^Ɣ) + Aeg * (E / 1 MeV)^(-ɑ_eg)
                                          ^------------^
        with `Ec` the exponential cutoff.
        """
        return self._galactic_exponential_cutoff

    @property
    def galactic_exponential_index(self) -> float:
        """Power-law index of the exponential cutoff of the
        Galactic astrophysical background.

        The total background flux is
            ɸ = Ag * (E / 1 MeV)^(-ɑ_g) * exp(-(E/Ec)^Ɣ) + Aeg * (E / 1 MeV)^(-ɑ_eg)
                                          ^------------^
        with `Ɣ` the exponential index.
        """
        return self._galactic_exponential_index

    @property
    def extra_galactic_amplitude(self) -> float:
        """Amplitude of the Extra-Galactic astrophysical background in
        units of [MeV⁻¹ cm⁻² s⁻¹ sr⁻¹].

        The total background flux is
            ɸ = Ag * (E / 1 MeV)^(-ɑ_g) * exp(-(E/Ec)^Ɣ) + Aeg * (E / 1 MeV)^(-ɑ_eg)
                                                           ^-^
        with `Aeg` the extra-galactic amplitude.
        """
        return self._extra_galactic_amplitude

    @property
    def extra_galactic_power_law_index(self) -> float:
        """Power-law index of the Extra-Galactic astrophysical background.

        The total background flux is
            ɸ = Ag * (E / 1 MeV)^(-ɑ_g) * exp(-(E/Ec)^Ɣ) + Aeg * (E / 1 MeV)^(-ɑ_eg)
                                                                 ^-----------------^
        with `ɑ_eg` the extra-galactic power-law index.
        """
        return self._extra_galactic_power_law_index

    @property
    def e_bounds(self) -> Tuple[float, float]:
        """Energy bounds of the model."""
        return self._energy_bounds

    @property
    def energy_bounds(self) -> Tuple[float, float]:
        """Energy bounds of the model."""
        return self._energy_bounds

    @property
    def params(self) -> Dict[str, float]:
        """Model parameters."""
        return {
            "galactic_amplitude": self._galactic_amplitude,
            "galactic_power_law_index": self._galactic_power_law_index,
            "galactic_exponential_cutoff": self._galactic_exponential_cutoff,
            "galactic_exponential_index": self._galactic_exponential_cutoff,
            "extra_galactic_amplitude": self._extra_galactic_amplitude,
            "extra_galactic_power_law_index": self._extra_galactic_power_law_index,
        }

    def dPhi_dEdOmega(self, energy):
        """Compute the differential flux from the background model.

        Parameters
        ----------
        energy: float or array-like
           Energy (in MeV).

        Returns
        -------
        d²ɸ/dEdΩ : dict
            Differential flux.
        """
        amp_g = self.galactic_amplitude
        alpha_g = self.galactic_power_law_index
        gamma = self.galactic_exponential_index
        ec = self.galactic_exponential_cutoff

        amp_eg = self.extra_galactic_amplitude
        alpha_eg = self.extra_galactic_power_law_index

        f_g = amp_g * energy ** (-alpha_g) * np.exp(-((energy / ec) ** gamma))
        f_eg = amp_eg * energy ** (-alpha_eg)

        return f_g + f_eg

    def derivatives(self, energy) -> Dict[str, Any]:
        """Compute the derivatives of the background model with respect to the
        parameters of the model.

        Parameters
        ----------
        energy: float or array-like
           Energy (in MeV).

        Returns
        -------
        deriv: dict
            Dictionary of containing the derivatives.
        """
        amp_g = self.galactic_amplitude
        alpha_g = self.galactic_power_law_index
        gamma = self.galactic_exponential_index
        ec = self.galactic_exponential_cutoff

        amp_eg = self.extra_galactic_amplitude
        alpha_eg = self.extra_galactic_power_law_index

        eec = energy / ec
        f_g = amp_g * energy ** (-alpha_g) * np.exp(-((eec) ** gamma))
        f_eg = amp_eg * energy ** (-alpha_eg)

        dn_g = f_g / amp_g
        da_g = -f_g * np.log(energy)
        dg_g = -f_g * np.log(eec) * eec**gamma
        dec_g = f_g * (eec**gamma) * gamma / ec

        dn_eg = f_eg / amp_eg
        da_eg = -f_eg * np.log(energy)

        return {
            "galactic_amplitude": dn_g,
            "galactic_power_law_index": da_g,
            "galactic_exponential_cutoff": dec_g,
            "galactic_exponential_index": dg_g,
            "extra_galactic_amplitude": dn_eg,
            "extra_galactic_power_law_index": da_eg,
        }


class GeccoBackgroundModel:
    r"""Fiducial extra-galactic background model used for projecting GECCO's
    sensitivity.
    """

    def __init__(self, amplitude: float = 2 * 4e-3, power_law_index: float = 2.0):
        """
        Parameters
        ----------
        amplitude: float
            Amplitude of the Galactic astrophysical background.

            Default is 0.013 [MeV⁻¹ cm⁻² s⁻¹ sr⁻¹].
        power_law_index: float
            Power-law index of the Galactic astrophysical background.
            Default is 2.0.
        """
        self._amplitude = amplitude
        self._power_law_index = power_law_index
        self._energy_bounds = (0.2, 4e3)

    @property
    def amplitude(self) -> float:
        """Amplitude of the Galactic astrophysical background in
        units of [MeV⁻¹ cm⁻² s⁻¹ sr⁻¹].

        The total background flux is
            ɸ = A * (E / 1 MeV)^(-ɑ)
               ^-^
        with `A` the galactic amplitude.
        """
        return self._amplitude

    @property
    def power_law_index(self) -> float:
        """Power-law index of the background model.

        The total background flux is
            ɸ = A * (E / 1 MeV)^(-ɑ)
                   ^----------------^
        with `ɑ` the power-law index.
        """
        return self._power_law_index

    @property
    def e_bounds(self) -> Tuple[float, float]:
        """Energy bounds of the model."""
        return self._energy_bounds

    @property
    def energy_bounds(self) -> Tuple[float, float]:
        """Energy bounds of the model."""
        return self._energy_bounds

    @property
    def params(self) -> Dict[str, float]:
        """Model parameters."""
        return {
            "amplitude": self.amplitude,
            "power_law_index": self.power_law_index,
        }

    def dPhi_dEdOmega(self, energy):
        """Compute the differential flux from the background model.

        Parameters
        ----------
        energy: float or array-like
           Energy (in MeV).

        Returns
        -------
        d²ɸ/dEdΩ : dict
            Differential flux.
        """
        amp = self.amplitude
        alpha = self.power_law_index
        f = amp * energy ** (-alpha)
        return f

    def derivatives(self, energy):
        """Compute the derivatives of the background model with respect to the
        parameters of the model.

        Parameters
        ----------
        energy: float or array-like
           Energy (in MeV).

        Returns
        -------
        deriv: dict
            Dictionary of containing the derivatives.
        """
        amp = self.amplitude
        alpha = self.power_law_index

        dn = energy ** (-alpha)
        da = amp * energy ** (-alpha) * np.log(energy)

        return {"amplitude": dn, "power_law_index": da}
