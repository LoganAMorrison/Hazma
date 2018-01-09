"""
Module containing squared matrix elements.

@author - Logan Morrison and Adam Coogan
@date - December 2017
"""

import warnings


def kl_to_pienu(kList):
    """
    Matrix element squared for kl -> pi  + e  + nu.
    """
    warnings.warn("""
                  kl -> pi  + e  + nu matrix element not yet available.
                  Currently this returns 1.0.""")
    return 1.0


def kl_to_pimunu(kList):
    """
    Matrix element squared for kl -> pi  + mu  + nu.
    """
    warnings.warn("""kl -> pi  + mu  + nu matrix element not yet available.
                  Currently this returns 1.0.""")
    return 1.0


def kl_to_pi0pi0pi0(kList):
    """
    Matrix element squared for kl -> pi0 + pi0  + pi0.
    """
    warnings.warn("""kl -> pi0 + pi0  + pi0 matrix element not yet available.
                  Currently this returns 1.0.""")
    return 1.0


def kl_to_pipipi0(kList):
    """
    Matrix element squared for kl -> pi  + pi  + pi0.
    """
    warnings.warn("""kl -> pi  + pi  + pi0 matrix element not yet available.
                  Currently this returns 1.0.""")
    return 1.0
