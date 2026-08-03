"""Module for computing gamma ray spectra from a many-particle final state.

"""

# author : Logan Morrison and Adam Coogan
# date : December 2017

from typing import Union, List, Callable, Optional, overload

import numpy as np
from numpy.typing import NDArray

from hazma import rambo
from hazma._gamma_ray.gamma_ray_generator import (
    gamma,
    gamma_point,
)
from hazma.field_theory_helper_functions.common_functions import (
    cross_section_prefactor,
)
from hazma.utils import RealArray, RealOrRealArray

SquaredMatrixElement = Callable[[RealArray], float]


def __flat_squared_matrix_element(_: RealArray) -> float:
    return 1.0


@overload
def gamma_ray_decay(
    particles: Union[List[str], NDArray[np.str_]],
    cme: float,
    photon_energies: float,
    mat_elem_sqrd: Optional[SquaredMatrixElement] = None,
    num_ps_pts: int = 1000,
    num_bins: int = 25,
    verbose: bool = False,
) -> float: ...


@overload
def gamma_ray_decay(
    particles: Union[List[str], NDArray[np.str_]],
    cme: float,
    photon_energies: RealArray,
    mat_elem_sqrd: Optional[SquaredMatrixElement] = None,
    num_ps_pts: int = 1000,
    num_bins: int = 25,
    verbose: bool = False,
) -> RealArray: ...


def gamma_ray_decay(
    particles: Union[List[str], NDArray[np.str_]],
    cme: float,
    photon_energies: RealOrRealArray,
    mat_elem_sqrd: Optional[SquaredMatrixElement] = None,
    num_ps_pts: int = 1000,
    num_bins: int = 25,
    verbose: bool = False,
) -> RealOrRealArray:
    r"""Returns gamma ray spectrum from the decay of a set of particles.

    This function works by running the Monte-Carlo algorithm RAMBO on the
    final state particles to obtain energies distributions for each final state
    particle. The energies distributions are then convolved with the decay
    spectra in order to obtain the gamma-ray spectrum.

    Parameters
    ----------
    particles : `array_like`
        List of particle names. Available particles are 'muon', 'electron'
        'charged_pion', 'neutral pion', 'charged_kaon', 'long_kaon', 'short_kaon'.
    cme : double
        Center of mass energy of the final state in MeV.
    photon_energies : np.ndarray[double, ndim=1]
        List of photon energies in MeV to evaluate spectra at.
    mat_elem_sqrd : double(\*func)(np.ndarray, )
        Function for the matrix element squared of the process. Must be a
        function taking in a list of four momenta of shape (len(particles), 4).
        Default value is a flat matrix element with a value of 1.
    num_ps_pts : int {1000}, optional
        Number of phase space points to use.
    num_bins : int {25}, optional
        Number of bins to use.
    verbose: Bool
        If true, additional output is displayed while the function is computing
        spectra.

    Returns
    -------
    spec : np.ndarray
        Total gamma ray spectrum from all final state particles.

    Notes
    -----
    The total spectrum is computed using

    .. math::

        \frac{dN}{dE}(E_{\gamma})
        =\sum_{i,j}P_{i}(E_{j})\frac{dN_i}{dE}(E_{\gamma}, E_{j})

    where :math:`i` runs over the final state particles, :math:`j` runs over
    energies sampled from probability distributions. :math:`P_{i}(E_{j})` is
    the probability that particle :math:`i` has energy :math:`E_{j}`. The
    probabilities are computed using ``hazma.phase_space_generator.rambo``. The
    total number of energies used is ``num_bins``.

    Examples
    --------

    Example of generating a spectrum from a muon, charged kaon and long kaon
    with total energy of 5000 MeV::

        from hazma.gamma_ray import gamma_ray_decay
        import numpy as np
        particles = np.array(['muon', 'charged_kaon', 'long_kaon'])
        cme = 5000.
        photon_energies = np.logspace(0., np.log10(cme), num=200, dtype=np.float64)
        gamma_ray_decay(particles, cme, photon_energies)

    """
    if mat_elem_sqrd is None:
        msqrd = __flat_squared_matrix_element
    else:
        msqrd = mat_elem_sqrd

    if isinstance(particles, str):
        particles = [particles]

    particles = np.array(particles)

    if hasattr(photon_energies, "__len__"):
        return gamma(
            particles,
            cme,
            np.array(photon_energies),
            mat_elem_sqrd=msqrd,
            num_ps_pts=num_ps_pts,
            num_bins=num_bins,
            verbose=verbose,
        )
    elif isinstance(photon_energies, float):
        return gamma_point(particles, cme, photon_energies, msqrd, num_ps_pts, num_bins)
    else:
        raise ValueError(
            f"Invalid type for 'photon_energies': {type(photon_energies)}."
        )


def __gamma_ray_fsr(
    photon_energy: float,
    cme: float,
    isp_masses: Union[List[float], RealArray],
    fsp_masses: Union[List[float], RealArray],
    non_rad: float,
    msqrd: Callable[[RealArray], float],
    nevents: int = 1000,
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
    _cme = cme if len(isp_masses) == 2 else isp_masses[0]

    if _cme * (_cme - 2 * photon_energy) < np.sum(fsp_masses) ** 2:
        return (0.0, 0.0)

    # Energy of the photon in the rest frame where final state particles
    # (excluding the photon)
    e_gamma = (photon_energy * _cme) / np.sqrt(_cme * (-2 * photon_energy + _cme))
    # Total energy of the final state particles (excluding the photon) in their
    # rest frame
    _cme_rf = np.sqrt(_cme * (-2 * photon_energy + _cme))
    # Number of final state particles
    nfsp = len(fsp_masses)
    # Generate events for the final state particles in their rest frame
    events = rambo.generate_phase_space(fsp_masses, _cme_rf, nevents)

    # Photon momenta in N + photon rest frame
    phis = np.random.rand(nevents) * 2.0 * np.pi
    cts = 2.0 * np.random.rand(nevents) - 1.0
    g_momenta = [
        np.array(
            [
                e_gamma,
                e_gamma * np.cos(phi) * np.sqrt(1 - ct**2),
                e_gamma * np.sin(phi) * np.sqrt(1 - ct**2),
                e_gamma * ct,
            ]
        )
        for phi, ct in zip(phis, cts)
    ]

    # momenta in the rest frame of N + photon
    fsp_momenta = [np.append(event[:-1], pg) for event, pg in zip(events, g_momenta)]

    weights = [event[-1] for event in events]

    terms = [
        weight * msqrd(ps_fps.reshape((nfsp + 1, 4)))
        for ps_fps, weight in zip(fsp_momenta, weights)
    ]
    res = np.average(terms)
    std = np.std(terms) / np.sqrt(nevents)
    pre = 1.0 / non_rad * photon_energy / (16 * np.pi**3) * (4.0 * np.pi)

    if len(isp_masses) == 1:
        pre *= 1.0 / (2.0 * isp_masses[0])
    else:
        pre *= cross_section_prefactor(isp_masses[0], isp_masses[1], cme)

    return pre * res, pre * std


def gamma_ray_fsr(
    photon_energies: Union[float, List[float], RealArray],
    cme: float,
    isp_masses: Union[List[float], RealArray],
    fsp_masses: Union[List[float], RealArray],
    non_rad: float,
    msqrd: Callable[[RealArray], float],
    nevents: int = 1000,
):
    """
    Compute the gamma-ray spectrum for a given process at specified photon
    energies.

    Parameters
    ----------
    photon_energies: float or np.array
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
    dnde: array of (float, float)
        The photon spectrum at `photon_energy` and the error estimate.
    """
    assert hasattr(isp_masses, "__len__"), "`isp_masses` must be a list."
    assert hasattr(fsp_masses, "__len__"), "`fsp_masses` must be a list."
    assert len(isp_masses) in [1, 2], "`isp_masses` must be of length 1 or 2."

    if isinstance(photon_energies, np.ndarray) or isinstance(photon_energies, list):
        return np.array(
            [
                __gamma_ray_fsr(
                    e,
                    cme,
                    isp_masses,
                    fsp_masses,
                    non_rad,
                    msqrd,
                    nevents=nevents,
                )
                for e in photon_energies
            ]
        )
    elif isinstance(photon_energies, float):
        return __gamma_ray_fsr(
            photon_energies,
            cme,
            isp_masses,
            fsp_masses,
            non_rad,
            msqrd,
            nevents=nevents,
        )
    else:
        raise ValueError(
            f"Invalid type for 'photon_energies': {type(photon_energies)}."
        )
