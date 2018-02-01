import os
import numpy as np
from .parameters import neutral_pion_mass as mpi0
from .parameters import fpi

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "unitarization_data",
                         "final_state_int_unitarizated.dat")

unitarizated_data = np.genfromtxt(DATA_PATH, delimiter=',')


def unit_matrix_elem_sqrd(cme):
    """
    Returns the unitarized squared matrix element for :math:`\pi\pi\to\pi\pi`
    divided by the leading order, ununitarized squared matrix element for
    :math:`\pi\pi\to\pi\pi`.

    Parameters
    ----------
    cme : double
        Invariant mass of the two charged pions.

    Results
    -------
    __unit_matrix_elem_sqrd : double
        The unitarized matrix element for :math:`\pi\pi\to\pi\pi`, |t_u|^2,
        divided by the un-unitarized squared matrix element for
        :math:`\pi\pi\to\pi\pi`, |t|^2; |t_u|^2 / |t|^2.
    """
    t_mod_sqrd = np.interp(cme, unitarizated_data[0], unitarizated_data[1])
    additional_factor = (2 * cme**2 - mpi0**2) / (32. * fpi**2 * np.pi)
    return t_mod_sqrd / additional_factor**2
