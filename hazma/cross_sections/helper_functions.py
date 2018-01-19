"""
Module containing commonly used particle physics factors.

@author - Logan Morrison and Adam Coogan
@date - December 2017
"""

import numpy as np


def cross_section_prefactor(m, Q):
    return 1.0 / (2.0 * Q**2 * np.sqrt(1. - 4.0 * m**2 / Q**2))
