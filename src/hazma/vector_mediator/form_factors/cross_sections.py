import numpy as np


def width_to_cs(*, cme, mx, mv, wv):
    """
    Computes the factor needed to transform a width V -> X into a
    cross-section x+xbar -> X.

    (
        gvxx**2 * (2 * mx**2 + s)
    )/(
        np.sqrt(1 - (4 * mx**2)/s) * cme * ((M**2 - s)**2 + mv**2 * widthv**2)
    )
    """
    mux2 = (mx / cme) ** 2
    mug2 = (wv / cme) ** 2
    muv2 = (mv / cme) ** 2

    num = 1 + 2 * mux2
    den = (1.0 + (mug2 - 2.0) * muv2 + muv2**2) * np.sqrt(1.0 - 4.0 * mux2) * cme**3
    return num / den


def cross_section_x_x_to_p_p(s, mx, mp, ff, mv, gamv):
    prop = (s - mv**2) ** 2 + (mv * gamv) ** 2
    return ((-4 * mp**2 + s) ** 1.5 * (2 * mx**2 + s) * np.abs(ff) ** 2) / (
        48.0 * np.pi * s * np.sqrt(-4 * mx**2 + s) * prop
    )


def cross_section_x_x_to_p_a(s, mx, mp, ff, mv, gamv):
    prop = (s - mv**2) ** 2 + mv**2 * gamv**2
    return ((-(mp**2) + s) ** 3 * (2 * mx**2 + s) * np.abs(ff) ** 2) / (
        96.0 * np.pi * s * np.sqrt(s * (-4 * mx**2 + s)) * prop
    )


def cross_section_x_x_to_p_v(s, mx, mp, mvec, ff, mv, gamv):
    prop = (s - mv**2) ** 2 + mv**2 * gamv**2
    return (
        (2 * mx**2 + s)
        * ((mp**4 + (mvec**2 - s) ** 2 - 2 * mp**2 * (mvec**2 + s)) / s) ** 1.5
        * np.abs(ff) ** 2
    ) / (96.0 * np.pi * np.sqrt(-4 * mx**2 + s) * prop * s)


def cross_section(cme, mx, mv, gamv, ff):
    s = cme**2
    prop = (s - mv**2) ** 2 + mv**2 * gamv**2
    return s * ff * (s + 2 * mx**2) / (prop * np.sqrt(s * (s - 4 * mx**2)))
