import numpy as np

m_e = 0.510998928e-3
m_mu = 0.1056583715
m_tau = 1.77682
m_top = 173.21

CS0 = (0.0, 0.00835, 1.000)
CS1 = (0.0, 0.00238, 3.927)
CS2 = (0.00165, 0.00299, 1.000)
CS3 = (0.00221, 0.00293, 1.000)


def hadronic(abc):
    a, b, c = abc

    def f(s):
        return a + b * np.log1p(c * s)

    return f


def real_pi(r):
    fvthr = 1.666666666666667e0
    rmax = 1.0e6

    m1 = r < 1e-3
    m2 = r > rmax
    m3 = 4.0 * r > 1.0
    m4 = ~(m1 | m2 | m3)

    results = np.zeros_like(r)

    # use assymptotic formula
    if np.any(m1):
        results[m1] = -fvthr - np.log(r[m1])
    if np.any(m2):
        beta = np.sqrt(4.0 * r[m2] - 1.0)
        results[m2] = 1.0 / 3.0 - (1.0 + 2.0 * r[m2]) * (
            2.0 - beta * np.arccos(1.0 - 1.0 / (2.0 * r[m2]))
        )
    if np.any(m4):
        beta = np.sqrt(1.0 - 4.0 * r[m4])
        results[m4] = 1.0 / 3.0 - (1.0 + 2.0 * r[m4]) * (
            2.0 + beta * np.log(abs((beta - 1.0) / (beta + 1.0)))
        )
    return results


def alpha_em(s):
    eps = 1e-6

    # alpha_EM at Q^2=0
    alem = 7.2973525698e-3
    aempi = alem / (3.0 * np.pi)

    single = np.isscalar(s)
    ss = np.atleast_1d(s).astype(np.float64)

    # return q^2=0 value for small scales
    mask = s < eps
    repigg = np.zeros_like(ss)
    sm = ss[mask]

    # leptonic component
    repigg[mask] = aempi * (
        real_pi(m_e**2 / sm) + real_pi(m_mu**2 / sm) + real_pi(m_tau**2 / sm)
    )

    # Hadronic component from light quarks
    repigg[mask] += np.piecewise(
        sm,
        [sm < 9e-2, sm < 9.0, sm < 1.0e4],
        [
            hadronic(CS0),
            hadronic(CS1),
            hadronic(CS2),
            hadronic(CS3),
        ],
    )

    # Top Contribution
    repigg[mask] += aempi * real_pi(m_top**2 / sm)

    result = alem / (1.0 - repigg)

    if single:
        return result[0]
    return result
