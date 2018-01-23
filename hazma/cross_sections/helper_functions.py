"""Module containing commonly used particle physics factors.

@author - Logan Morrison and Adam Coogan
@date - December 2017

"""


def cross_section_prefactor(m1, m2, cme):

    E1 = (cme**2 + m1**2 - m2**2) / (2 * cme)
    E2 = (cme**2 + m2**2 - m1**2) / (2 * cme)

    p = (cme**2 - m1**2 - m2**2) / (2 * cme)

    v1, v2 = p / E1, p / E2

    vrel = v1 + v2

    return 1.0 / (2.0 * E1) / (2.0 * E2) / vrel
