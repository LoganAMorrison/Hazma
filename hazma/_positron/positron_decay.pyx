"""Module for computing positron spectra from a many-particle final state.

@author - Logan Morrison and Adam Coogan
@date - December 2017
"""
import numpy as np
cimport numpy as np
import cython
import multiprocessing as mp

from hazma import rambo
from hazma.rambo import compute_annihilation_cross_section
from hazma.rambo import compute_decay_width
from hazma._positron.positron_muon cimport c_muon_positron_spectrum_array
from hazma._positron.positron_muon cimport c_muon_positron_spectrum_point
from hazma._positron.positron_charged_pion cimport c_charged_pion_positron_spectrum_array
from hazma._positron.positron_charged_pion cimport c_charged_pion_positron_spectrum_point

include "../_decay/common.pxd"


cdef int ID_MU = 0
cdef int ID_PI = 1
cdef int ID_PI0 = 2
cdef int ID_E = 3


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef np.ndarray[np.float64_t,ndim=1] id_to_mass(int ident):
    if ident == ID_MU:
        return MASS_MU
    elif ident == ID_PI:
        return MASS_PI
    elif ident == ID_PI0:
        return MASS_PI0
    elif ident == ID_E:
        return MASS_E
    else:
        return 0.0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef np.ndarray[np.float64_t,ndim=1] id_to_masses(np.ndarray[np.int_t,ndim=1] ids):
    cdef int size = ids.shape[0]
    cdef np.ndarray[np.float64_t,ndim=1] masses = np.zeros(size, dtype=np.float64)
    for i in range(size):
        masses[i] = id_to_mass(ids[i])
    return masses


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef np.ndarray[np.float64_t,ndim=1] c_positron_array_two_body(int id1, int id2, double cme, np.ndarray[np.int_t,ndim=1] eng_ps):
    cdef np.ndarray[np.float64_t,ndim=1] spec = np.zeros_like(eng_ps)
    cdef int npts = eng_ps.shape[0]
    cdef double m1 = id_to_mass(id1)
    cdef double m2 = id_to_mass(id2)
    cdef double E1 = (cme * cme + m1 * m1 - m2 * m2) / (2 * cme)
    cdef double E2 = (cme * cme - m1 * m1 + m2 * m2) / (2 * cme)
   
    if id1 == ID_MU:
        for i in range(npts):
            spec[i] += c_muon_positron_spectrum_point(eng_ps[i], E1)
    elif id1 == ID_PI:
        for i in range(npts):
            spec[i] += c_charged_pion_positron_spectrum_point(eng_ps[i], E1)

    if id2 == ID_MU:
        for i in range(npts):
            spec[i] += c_muon_positron_spectrum_point(eng_ps[i], E2)
    elif id2 == ID_PI:
        for i in range(npts):
            spec[i] += c_charged_pion_positron_spectrum_point(eng_ps[i], E2)

    return spec


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef np.ndarray[np.float64_t,ndim=1] c_positron_array(
    np.ndarray[np.int_t,ndim=1] ids,
    double cme,
    np.ndarray[np.float64_t,ndim=1] eng_ps,
    mat_elem_sqrd,
    int num_ps_pts,
    int num_bins
):
    cdef np.ndarray[np.float64_t,ndim=1] masses = id_to_masses(ids)
    cdef int num_fsp  = ids.shape[0]
    cdef int num_engs = eng_ps.shape[0]
    cdef np.ndarray hist
    cdef np.ndarray[np.float64_t,ndim=1] spec = np.zeros_like(eng_ps)

    if num_fsp == 1:
        if ids[0] == ID_MU:
            return c_muon_positron_spectrum_array(eng_ps, cme)
        elif ids[0] == ID_PI:
            return c_charged_pion_positron_spectrum_array(eng_ps, cme)
        else:
            return np.zeros_like(eng_ps)
    elif num_fsp == 2:
        return c_positron_array_two_body(ids[0], ids[1], cme, eng_ps)

    hist = rambo.generate_energy_histogram(masses, cme, num_ps_pts, mat_elem_sqrd, num_bins, density=True)[0]

    for i in range(num_bins):
        for j in range(num_fsp):
            eng = hist[j, 0, i]
            # Normalize spectrum: Need to multiply by the probability of
            # particle having energy part_eng. Since we are essentially
            # integrating over the energy probability distribution, we need to
            # multiply by (b - a) / N, where a = e_min, b = e_max and
            # N = num_bins.
            norm = (hist[j, 0, num_bins-1] - hist[j, 0, 0]) / num_bins * hist[j, 1, i]

            if ids[j] == ID_MU:
                for k in range(num_engs):
                    spec[i] += norm * c_muon_positron_spectrum_point(eng_ps[k], eng)

            elif ids[j] == ID_PI:
                for k in range(num_engs):
                    spec[i] += norm * c_charged_pion_positron_spectrum_point(eng_ps[k], eng)

    return spec


@cython.boundscheck(False)
@cython.wraparound(False)
def positron(particles, cme, eng_ps, mat_elem_sqrd=lambda k_list: 1.0, num_ps_pts=10000, num_bins=25):
    """Returns total gamma ray spectrum from final state particles.

    Parameters
    ----------
    particles : array-like
        List of strings containing the final state particle names. The
        accepted particle names are: "muon", and "charged_pion".
    cme : double
        Center of mass energy of the final state.
    eng_ps : array-like
        List of positron energies to compute spectra at.
    mat_elem_sqrd : callable
        Function for the matrix element squared of the proccess. Must be
        a function taking in a list of four momenta of size (num_fsp, 4).
        Default value is a flat matrix element.
    num_ps_pts : int
        Number of phase space points to use.
    num_bins : int
        Number of bins to use.

    Returns
    -------
    spec : np.ndarray
        1-D array of total gamma ray spectrum from all final state particles.
    """
    cdef np.ndarray[np.int_t,ndim=1] ids = np.zeros((len(particles),), dtype=int)
    cdef np.ndarray[np.float64_t,ndim=1] spec
    for i, particle in enumerate(particles):
        if particle == "muon":
            ids[i] = ID_MU
        elif particle == "charged_pion":
            ids[i] = ID_PI
        elif particle == "electron":
            ids[i] = ID_E
        elif particle == "neutral_pion":
            ids[i] = ID_PI0
        else:
            raise ValueError("Invalid particle " + particle + ".")
        
    energies = np.array(eng_ps)
    assert len(energies.shape) == 1, "Positron energies must be 0 or 1-dimensional."

    if len(energies) > 1:
        return c_positron_array(ids, cme, energies, mat_elem_sqrd, num_ps_pts, num_bins)
    else:
        spec = c_positron_array(ids, cme, eng_ps, mat_elem_sqrd, num_ps_pts, num_bins)
        return spec[0]


