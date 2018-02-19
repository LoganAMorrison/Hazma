import os
import numpy as np
from .parameters import charged_pion_mass as mpi
from .parameters import charged_kaon_mass as mk
from .parameters import neutral_pion_mass as mpi0
from .parameters import neutral_kaon_mass as mk0
from .parameters import fpi

q_max = 1.1 * 10**3  # cut-off scale taken to be 1.2 GeV
mPI = (mpi + mpi0) / 2.
mK = (mk + mk0) / 2.

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "unitarization_data",
                         "final_state_int_unitarizated.dat")

unitarizated_data = np.loadtxt(
    DATA_PATH, delimiter=',', dtype=complex)
unitarizated_data_x = np.array(unitarizated_data[:, 0], dtype=float)
unitarizated_data_y = unitarizated_data[:, 1]


def bubble_loop(Q):
    return ((np.sqrt(4 * mpi**2 - Q**2 + 0j) *
             np.arctan((np.sqrt(Q**2) * q_max) /
                       np.sqrt((4 * mpi**2 - Q**2 + 0j) *
                               (mpi**2 + q_max**2)) + 0j)) /
            np.sqrt(Q**2) +
            np.log(mpi / (q_max + np.sqrt(mpi**2 + q_max**2))) + 0j) / \
        (8. * np.pi**2)


def amp_pipi_to_pipi_I0(cme):
    """
    Lowest order pion scattering squared matrix element in the isospin I=0
    channel.

    Parameters
    ----------
    cme : double
        Invariant mass of the two charged pions.
    """
    return (2 * cme**2 - mpi0**2) / (32. * fpi**2 * np.pi)


def amp_inverse_amplitude_pipi_to_pipi(cme):
    """
    Unitarized pion scattering squared matrix element in the isopin I = 0
    channel.

    Unitarization was computed using the inverse amplitude method (IAM) with
    only pion contributions.

    Parameters
    ----------
    cme : double
        Invariant mass of the two charged pions.
    """
    return np.interp(cme, unitarizated_data_x, unitarizated_data_y)


def ratio_pipi_to_pipi_unitarized_tree(cme):
    """
    Returns the unitarized squared matrix element for :math:`\pi\pi\to\pi\pi`
    divided by the leading order, ununitarized squared matrix element for
    :math:`\pi\pi\to\pi\pi`.

    This was computed using the inverse amplitude method (IAM) with only

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
    return amp_inverse_amplitude_pipi_to_pipi(cme) / \
        amp_pipi_to_pipi_I0(cme)


def unit_matrix_elem_sqrd(cme):
    """
    Returns the unitarized squared matrix element for :math:`\pi\pi\to\pi\pi`
    divided by the leading order, ununitarized squared matrix element for
    :math:`\pi\pi\to\pi\pi`.

    This was computed using the inverse amplitude method (IAM) with only

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
    t_mod_sqrd = np.interp(
        cme, unitarizated_data_x, unitarizated_data_y)
    additional_factor = (2 * cme**2 - mpi0**2) / (32. * fpi**2 * np.pi)
    return t_mod_sqrd / additional_factor**2


def amp_bethe_salpeter_pipi_to_pipi(Q):
    """
    Returns the unitarized matrix element for pipi -> pipi in the zero isospin
    channel.

    This is computed by resumming an infinite number of bubble
    intermediate states. The intermediate states used were pions and kaons
    in the I=0 channel.

    Parameters
    ----------
    Q : float
        Invariant mass of the pions.

    Returns
    -------
    Mu : float
        Unitarized matrix element for pipi -> pipi in the zero isospin
        channel.
    """
    s = Q**2 + 0j

    return (8 * np.pi**2 * np.sqrt(s) *
            (64 * fpi**2 * np.pi**2 * (mPI**2 - 2 * s) -
             3 * np.sqrt(4 * mK**2 - s) * np.sqrt(s) *
             (-2 * mPI**2 + 3 * s) *
             np.arctan((q_max * np.sqrt(s)) /
                       np.sqrt((mK**2 + q_max**2) * (4 * mK**2 - s))) +
             3 * (2 * mPI**2 - 3 * s) * s *
             np.log(mK / (q_max + np.sqrt(mK**2 + q_max**2))))) / \
        (-(np.sqrt(4 * mPI**2 - s) *
           np.arctan((q_max * np.sqrt(s)) /
                     np.sqrt((mPI**2 + q_max**2) * (4 * mPI**2 - s))) *
           (64 * fpi**2 * np.pi**2 * (mPI**2 - 2 * s) +
            3 * (2 * mPI**2 - 3 * s) * s *
            np.log(mK / (q_max + np.sqrt(mK**2 + q_max**2))))) +
         3 * np.sqrt(4 * mK**2 - s) * np.sqrt(s) *
         np.arctan((q_max * np.sqrt(s)) /
                   np.sqrt((mK**2 + q_max**2) * (4 * mK**2 - s))) *
         (-((2 * mPI**2 - 3 * s) * np.sqrt(4 * mPI**2 - s) *
            np.arctan((q_max * np.sqrt(s)) /
                      np.sqrt((mPI**2 + q_max**2) * (4 * mPI**2 - s)))) +
          np.sqrt(s) *
          (32 * fpi**2 * np.pi**2 + (-2 * mPI**2 + 3 * s) *
           np.log(mPI / (q_max + np.sqrt(mPI**2 + q_max**2))))) +
         np.sqrt(s) * (64 * fpi**2 * np.pi**2 *
                       (16 * fpi**2 * np.pi**2 -
                        (mPI**2 - 2 * s) *
                        np.log(mPI / (q_max + np.sqrt(mPI**2 + q_max**2)))) +
                       3 * s *
                       np.log(mK / (q_max + np.sqrt(mK**2 + q_max**2))) *
                       (32 * fpi**2 * np.pi**2 +
                        (-2 * mPI**2 + 3 * s) *
                        np.log(mPI / (q_max + np.sqrt(mPI**2 + q_max**2))))))


def msqrd_bethe_salpeter_pipi_to_pipi(Q):
    """
    Returns the square of the unitarized pipi->pipi amplitude.

    Parameters
    ----------
    Q : float
        Invariant mass of the pions.

    Returns
    -------
    Mu2 : float
        Unitarized squared matrix element for pipi -> pipi in the zero isospin
        channel.
    """
    return np.abs(amp_bethe_salpeter_pipi_to_pipi(Q))**2
