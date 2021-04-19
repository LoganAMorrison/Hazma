import numpy as np


def beta2(s, m1, m2):
    return max(0.0, (1.0 - (m1 + m2) ** 2 / s) * (1.0 - (m1 - m2) ** 2 / s))


def beta(s, m1, m2):
    return np.sqrt(beta2(s, m1, m2))


def dhhatds(mRes, gamma, m1, m2):
    v2 = beta2(mRes ** 2, m1, m2)
    v = np.sqrt(v2)
    r = (m1 ** 2 + m2 ** 2) / mRes ** 2
    return (
        gamma
        / np.pi
        / mRes
        / v2
        * (
            (3.0 - 2.0 * v2 - 3.0 * r) * np.log((1.0 + v) / (1.0 - v))
            + 2.0 * v * (1.0 - r / (1.0 - v2))
        )
    )


def hhat(s, mRes, gamma, m1, m2):
    vR = beta(mRes ** 2, m1, m2)
    v = beta(s, m1, m2)
    return (
        gamma
        / mRes / np.pi * s * (v / vR) ** 3 * np.log((1.0 + v) / (1.0 - v)))


def h(s, mRes, gamma, m1, m2, dH, Hres):
    if s != 0.0:
        return hhat(s, mRes, gamma, m1, m2) - Hres - (s - mRes ** 2) * dH
    else:
        return (
            -2.0 * (m1 + m2) ** 2 / np.pi * gamma /
            mRes / beta(mRes ** 2, m1, m2) ** 3
            - Hres
            + mRes ** 2 * dH
        )


def gamma_p(s, mRes, gamma, m1, m2):
    v2 = beta2(s, m1, m2)
    if v2 <= 0.0:
        return 0.0
    vR2 = beta2(mRes ** 2, m1, m2)
    if vR2 == 0.0:
        rp = 0.0
    else:
        rp = np.sqrt(max(0.0, v2 / vR2))
    return np.sqrt(s) / mRes * rp ** 3 * gamma


def breit_wigner_gs(s, mRes, gamma, m1, m2, H0, dH, Hres):
    mR2 = mRes ** 2
    return (mR2 + H0) / (
        mR2
        - s
        + h(s, mRes, gamma, m1, m2, dH, Hres)
        - 1j * np.sqrt(s) * gamma_p(s, mRes, gamma, m1, m2)
    )


def breit_wigner_fw(s, mRes, gamma):
    mR2 = mRes ** 2
    return mR2 / (mR2 - s - 1j * mRes * gamma)
