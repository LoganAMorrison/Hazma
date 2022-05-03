import numpy as np


def cross_section_x_x_to_p_p(s, mx, mp, ff, mv, gamv):
    prop = mv**4 + s**2 + mv**2 * (gamv**2 - 2 * s)
    return ((-4 * mp**2 + s) ** 1.5 * (2 * mx**2 + s) * np.abs(ff) ** 2) / (
        48.0 * np.pi * s * np.sqrt(-4 * mx**2 + s) * prop
    )


def cross_section_x_x_to_p_a(s, mx, mp, ff, mv, gamv):
    prop = mv**4 + s**2 + mv**2 * (gamv**2 - 2 * s)
    return ((-(mp**2) + s) ** 3 * (2 * mx**2 + s) * np.abs(ff) ** 2) / (
        96.0 * np.pi * s * np.sqrt(s * (-4 * mx**2 + s)) * prop
    )


def cross_section_x_x_to_p_v(s, mx, mp, mvec, ff, mv, gamv):
    prop = mv**4 + s**2 + mv**2 * (gamv**2 - 2 * s)
    return (
        (2 * mx**2 + s)
        * ((mp**4 + (mvec**2 - s) ** 2 - 2 * mp**2 * (mvec**2 + s)) / s) ** 1.5
        * np.abs(ff) ** 2
    ) / (96.0 * np.pi * np.sqrt(-4 * mx**2 + s) * prop)
