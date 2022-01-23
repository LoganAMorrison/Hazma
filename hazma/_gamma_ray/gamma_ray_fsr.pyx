import numpy as np
cimport numpy as np
import cython
from hazma._phase_space.generator cimport c_generate_space
from libc.math cimport M_PI, sqrt, cos, sin
from libcpp.vector cimport vector
from libcpp.pair cimport pair


cdef extern from "<random>" namespace "std":
    cdef cppclass mt19937 nogil:
        mt19937() nogil
        mt19937(unsigned int seed) nogil

    cdef cppclass random_device nogil:
        random_device() except +
        unsigned int operator()()

    cdef cppclass uniform_real_distribution[T] nogil:
        uniform_real_distribution() nogil
        uniform_real_distribution(T a, T b) nogil
        T operator()(mt19937 gen) nogil

cdef uniform_real_distribution[double] uniform \
    = uniform_real_distribution[double](0., 1.)

cdef mt19937 rng

cdef extern from "random" namespace "std":
    cdef cppclass random


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double c_cross_section_prefactor(double m1, double m2, double cme):
    cdef double E1 = (cme**2 + m1**2 - m2**2) / (2. * cme)
    cdef double E2 = (cme**2 + m2**2 - m1**2) / (2. * cme)

    cdef double p = sqrt((m1 - m2 - cme) * (m1 + m2 - cme) *
                         (m1 - m2 + cme) * (m1 + m2 + cme)) / (2. * cme)

    cdef double v1 = p / E1
    cdef double v2 = p / E2

    cdef double vrel = v1 + v2

    return 1.0 / (2.0 * E1) / (2.0 * E2) / vrel


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef pair[double,double] c_gamma_ray_fsr_point(
    double photon_energy,
    double cme,
    vector[double] &isp_masses,
    vector[double] &fsp_masses,
    double non_rad,
    msqrd_type_vector msqrd,
    int nevents,
    vector[double] &params,
):
    cdef int nfsp = fsp_masses.size()
    cdef int i
    cdef int j
    cdef int k

    cdef double _cme
    cdef double _cme_rf
    cdef double e_gamma
    cdef double res
    cdef double std
    cdef double pre
    cdef double ct
    cdef double phi
    cdef double weight
    cdef double mass_sum

    cdef vector[double] phis = vector[double](nevents)
    cdef vector[double] cts = vector[double](nevents)
    cdef vector[vector[double]] momenta = vector[vector[double]](nfsp + 1)
    cdef vector[vector[double]] events

    # If we have a decay, set cme to mass of decaying particle.
    if isp_masses.size() == 2:
        _cme = cme
    else:
        _cme = isp_masses[0]

    # Check if process is kinematically possible
    mass_sum = 0.0
    for i in range(nfsp):
        mass_sum = mass_sum + fsp_masses[i]
    if _cme * (_cme - 2 * photon_energy) < mass_sum ** 2:
        return pair[double,double](0.0, 0.0)

    # Energy of the photon in the rest frame where final state particles
    # (excluding the photon)
    e_gamma = (photon_energy * _cme) / sqrt(
        _cme * (-2 * photon_energy + _cme)
    )
    # Total energy of the final state particles (excluding the photon) in their
    # rest frame
    _cme_rf = sqrt(_cme * (-2 * photon_energy + _cme))
    # Generate events for the final state particles in their rest frame
    events = c_generate_space(nevents, fsp_masses, _cme_rf, nfsp)

    res = 0.0
    std = 0.0
    for i in range(nevents):
        phi = 2.0 * M_PI * uniform(rng)
        ct = 2.0 * uniform(rng) - 1.0

        # Copy the four-momenta from the event
        for j in range(nfsp):
            momenta[j] = vector[double](4)
            for k in range(4):
                momenta[j][k] = events[i][4 * j + k]

        # Fill in the photon four-momentum
        momenta[nfsp] = vector[double](4)
        momenta[nfsp][0] = e_gamma
        momenta[nfsp][1] = e_gamma * cos(phi) * sqrt(1 - ct ** 2)
        momenta[nfsp][2] = e_gamma * sin(phi) * sqrt(1 - ct ** 2)
        momenta[nfsp][3] = e_gamma * ct

        weight = events[i][4 * nfsp] * msqrd(momenta, params)

        res = res + weight
        std = std + weight * weight

    res = res / nevents
    std = std / (nevents * sqrt(nevents))

    pre = 1.0 / non_rad * photon_energy / (16 * M_PI ** 3) * (4.0 * M_PI)

    if isp_masses.size() == 1:
        pre *= 1.0 / (2.0 * isp_masses[0])
    else:
        pre *= c_cross_section_prefactor(isp_masses[0], isp_masses[1], cme)

    return pair[double, double](pre * res, pre * std)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef pair[vector[double], vector[double]] c_gamma_ray_fsr(
    vector[double] &photon_energies,
    double cme,
    vector[double] &isp_masses,
    vector[double] &fsp_masses,
    double non_rad,
    msqrd_type_vector msqrd,
    int nevents,
    vector[double] &params,
):
    cdef int i
    cdef int npts = photon_energies.size()
    cdef pair[double,double] result
    cdef vector[double] results = vector[double](npts)
    cdef vector[double] errors = vector[double](npts)

    for i in range(npts):
        result = c_gamma_ray_fsr_point(
            photon_energies[i],
            cme,
            isp_masses,
            fsp_masses,
            non_rad,
            msqrd,
            nevents,
            params,
        )
        results[i] = result.first
        errors[i] = result.second

    return pair[vector[double], vector[double]](results, errors)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef pair[double, double] py_gamma_ray_fsr_point(
    double photon_energy,
    double cme,
    vector[double] &isp_masses,
    vector[double] &fsp_masses,
    double non_rad,
    msqrd,
    int nevents,
):
    cdef int nfsp = fsp_masses.size()
    cdef int i
    cdef int j
    cdef int k

    cdef double _cme
    cdef double _cme_rf
    cdef double e_gamma
    cdef double res
    cdef double std
    cdef double pre
    cdef double ct
    cdef double phi
    cdef double weight
    cdef double mass_sum

    cdef vector[double] phis = vector[double](nevents)
    cdef vector[double] cts = vector[double](nevents)
    cdef vector[vector[double]] momenta = vector[vector[double]](nfsp + 1)
    cdef vector[vector[double]] events

    # If we have a decay, set cme to mass of decaying particle.
    if isp_masses.size() == 2:
        _cme = cme
    else:
        _cme = isp_masses[0]

    # Check if process is kinematically possible
    mass_sum = 0.0
    for i in range(nfsp):
        mass_sum = mass_sum + fsp_masses[i]
    if _cme * (_cme - 2 * photon_energy) < mass_sum ** 2:
        return pair[double,double](0.0, 0.0)

    # Energy of the photon in the rest frame where final state particles
    # (excluding the photon)
    e_gamma = (photon_energy * _cme) / sqrt(
        _cme * (-2 * photon_energy + _cme)
    )
    # Total energy of the final state particles (excluding the photon) in their
    # rest frame
    _cme_rf = sqrt(_cme * (-2 * photon_energy + _cme))
    # Generate events for the final state particles in their rest frame
    events = c_generate_space(nevents, fsp_masses, _cme_rf, nfsp)

    res = 0.0
    std = 0.0
    for i in range(nevents):
        phi = 2.0 * M_PI * uniform(rng)
        ct = 2.0 * uniform(rng) - 1.0

        # Copy the four-momenta from the event
        for j in range(nfsp):
            for k in range(4):
                momenta[j].push_back(events[i][4 * j + k])

        # Fill in the photon four-momentum
        momenta[nfsp][0] = e_gamma
        momenta[nfsp][1] = e_gamma * cos(phi) * sqrt(1 - ct ** 2)
        momenta[nfsp][2] = e_gamma * sin(phi) * sqrt(1 - ct ** 2)
        momenta[nfsp][3] = e_gamma * ct

        weight = events[i][4 * nfsp] * msqrd(momenta)

        res = res + weight
        std = std + weight * weight

    res = res / nevents
    std = std / (nevents * sqrt(nevents))

    pre = 1.0 / non_rad * photon_energy / (16 * M_PI ** 3) * (4.0 * M_PI)

    if isp_masses.size() == 1:
        pre *= 1.0 / (2.0 * isp_masses[0])
    else:
        pre *= c_cross_section_prefactor(isp_masses[0], isp_masses[1], cme)

    return pair[double, double](pre * res, pre * std)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef pair[vector[double], vector[double]] py_gamma_ray_fsr(
    vector[double] &photon_energies,
    double cme,
    vector[double] &isp_masses,
    vector[double] &fsp_masses,
    double non_rad,
    msqrd,
    int nevents,
):
    cdef int i
    cdef int npts = photon_energies.size()
    cdef pair[double,double] result
    cdef vector[double] results = vector[double](npts)
    cdef vector[double] errors = vector[double](npts)

    for i in range(npts):
        result = py_gamma_ray_fsr_point(
            photon_energies[i],
            cme,
            isp_masses,
            fsp_masses,
            non_rad,
            msqrd,
            nevents,
        )
        results[i] = result.first
        errors[i] = result.second

    return pair[vector[double], vector[double]](results, errors)


def gamma_ray_fsr(
    photon_energies,
    cme,
    isp_masses,
    fsp_masses,
    non_rad,
    msqrd,
    nevents,
):
    """
    Compute the gamma-ray spectrum for a given process at specified photon
    energies.

    Parameters
    ----------
    photon_energy: float
        Energy of the photon.
    cme: float
        Center-of-mass energy. This will be ignored in the case where
        `len(isp_masses) == 1` (i.e. for the decay of a particle.)
    isp_masses: array
        List of the initial state particle masses.
    fsp_masses: array
        List of the final state particle masses excluding the photon.
    non_rad: float
        The non-radiative cross-section or width.
    msqrd: callable
        Function to compute the squared and averaged cross-section or width.
        The signature must be `msqrd(momenta)`, where `momenta` is a list of
        four-momenta of the final state particles. `momenta` must be ordered
        such that the momentum of particle `i` has mass equal to
        `fsp_masses[i]` (except the photon.) The photon momentum must be the
        last momentum in the list, i.e. at `momentum[len(fsp_masses)]`.
    nevents: int, optional
        Number of events to use for computing the dnde.

    Returns
    -------
    dnde: tuple of floats
        The photon spectrum at `photon_energy` and the error estimate.
    """
    if hasattr(photon_energies, '__len__'):
        return py_gamma_ray_fsr(
            np.array(photon_energies),
            cme,
            np.array(isp_masses),
            np.array(fsp_masses),
            non_rad,
            msqrd,
            nevents,
        )
    return py_gamma_ray_fsr_point(
            photon_energies,
            cme,
            np.array(isp_masses),
            np.array(fsp_masses),
            non_rad,
            msqrd,
            nevents,
        )


