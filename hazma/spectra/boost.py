import numpy as np


def boost_delta_function(e, e0: float, m: float, beta: float):
    """
    Boost a delta function of the form δ(e - e0) of a product of mass `m`
    with a boost parameter `beta`.

    Parameters
    ----------
    e: double
        Energy of the product in the lab frame.
    e0: double
        Center of the dirac-delta spectrum in rest-frame
    m: double
        Mass of the product
    beta: double
        Boost velocity of the decaying particle
    """
    dnde = np.zeros_like(e)

    if beta > 1.0 or beta <= 0.0:
        return dnde

    gamma = 1.0 / np.sqrt(1.0 - beta * beta)
    k = np.sqrt(e**2 - m**2)
    eminus = gamma * (e - beta * k)
    eplus = gamma * (e + beta * k)

    # - b * k0 < (e/g) - e0 < b * k0
    mask = np.logical_and(eminus < e0, e0 < eplus)
    dnde[mask] = 1.0 / (2.0 * gamma * beta * np.sqrt(e0 * e0 - m * m))

    return dnde


def double_boost_delta_function(e2, e0: float, m: float, beta1: float, beta2: float):
    """
    Perform a double-boost of a delta function of the form δ(e - e0) of a
    product.

    Parameters
    ----------
    e: double
        Energy of the product in the lab frame.
    e0: double
        Center of the dirac-δ spectrum in original rest-frame.
    m: double
        Mass of the product.
    beta1, beta2: double
        1st and 2nd boost velocities of the decaying particle.
    """
    gamma1 = 1.0 / np.sqrt(1.0 - beta1**2)
    gamma2 = 1.0 / np.sqrt(1.0 - beta2**2)

    eps_m = gamma1 * (e0 - beta1 * np.sqrt(e0**2 - m**2))
    eps_p = gamma1 * (e0 + beta1 * np.sqrt(e0**2 - m**2))
    e_m = gamma2 * (e2 - beta2 * np.sqrt(e2**2 - m**2))
    e_p = gamma2 * (e2 + beta2 * np.sqrt(e2**2 - m**2))

    mask = np.logical_and(e_p > eps_m, e_m < eps_p)
    b = np.minimum(eps_p, e_p)[mask]
    a = np.maximum(eps_m, e_m)[mask]

    if m > 0.0:
        num = (a - np.sqrt(a**2 - m**2)) * (b + np.sqrt(b**2 - m**2))
        den = (a + np.sqrt(a**2 - m**2)) * (b - np.sqrt(b**2 - m**2))
        pre = 0.5
    else:
        num = b
        den = a
        pre = 1.0

    res = np.zeros_like(e2)
    res[mask] = pre * np.log(num / den)
    res = res / (4.0 * gamma1 * gamma2 * beta1 * beta2 * e0)

    return res
