"""
Module for computing positron spectra.
@author: Logan Morrison and Adam Coogan
"""

from typing import overload, Union, Callable, List, Optional

# from hazma._positron.positron_decay import positron
from hazma import parameters
from hazma.utils import RealArray, RealOrRealArray

from . import _muon, _pion
from ._utils import (
    load_interp as _load_interp,
    dnde_positron as _dnde_positron,
)

SquaredMatrixElement = Callable[[RealArray], float]


_eta_interp = _load_interp("eta_positron.csv")
_charged_kaon_integrand_interp = _load_interp("charged_kaon_positron.csv")
_long_kaon_integrand_interp = _load_interp("long_kaon_positron.csv")
_short_kaon_integrand_interp = _load_interp("short_kaon_positron.csv")
_omega_integrand_interp = _load_interp("omega_positron.csv")
_rho_integrand_interp = _load_interp("rho_positron.csv")
_eta_prime_integrand_interp = _load_interp("eta_prime_positron.csv")
_phi_integrand_interp = _load_interp("phi_positron.csv")


def __flat_squared_matrix_element(_: RealArray) -> float:
    return 1.0


@overload
def dnde_positron_muon(positron_energies: float, muon_energy: float) -> float: ...


@overload
def dnde_positron_muon(
    positron_energies: RealArray, muon_energy: float
) -> RealArray: ...


def dnde_positron_muon(
    positron_energies: RealOrRealArray, muon_energy: float
) -> RealOrRealArray:
    """
    Returns the positron spectrum from muon decay.

    Parameters
    ----------
    positron_energies : float or numpy.array
        Energy(ies) of the positron/electron.
    muon_energy : float or array-like
        Energy of the muon.

    Returns
    -------
    dnde : float or numpy.array
        The value of the spectrum given a positron energy(ies)
        ``positron_energies`` and muon energy ``muon_energy``.
    """
    return _muon.dnde_positron_muon(positron_energies, muon_energy)


@overload
def dnde_positron_charged_pion(
    positron_energies: float, pion_energy: float
) -> float: ...


@overload
def dnde_positron_charged_pion(
    positron_energies: RealArray, pion_energy: float
) -> RealArray: ...


def dnde_positron_charged_pion(
    positron_energies: RealOrRealArray, pion_energy: float
) -> RealOrRealArray:
    """
    Returns the positron spectrum from the decay of a charged pion.

    Parameters
    ----------
    positron_energies : float or numpy.array
        Energy(ies) of the positron/electron.
    pion_energy : float or numpy.array
        Energy of the charged pion.

    Returns
    -------
    dnde : float or numpy.array
        The value of the spectrum given a positron energy(ies)
        ``positron_energies`` and charged pion energy ``pion_energy``.
    """
    return _pion.dnde_positron_charged_pion(positron_energies, pion_energy)


@overload
def dnde_positron_charged_kaon(
    positron_energies: float, kaon_energy: float
) -> float: ...


@overload
def dnde_positron_charged_kaon(
    positron_energies: RealArray, kaon_energy: float
) -> RealArray: ...


def dnde_positron_charged_kaon(
    positron_energies: RealOrRealArray, kaon_energy: float
) -> RealOrRealArray:
    """
    Returns the positron spectrum from the decay of a charged kaon.

    Parameters
    ----------
    positron_energies : float or numpy.array
        Energy(ies) of the positron/electron.
    kaon_energy : float or numpy.array
        Energy of the charged kaon.

    Returns
    -------
    dnde : float or numpy.array
        The value of the spectrum given a positron energy(ies)
        ``positron_energies`` and charged kaon energy ``pion_energy``.
    """
    interp = _charged_kaon_integrand_interp
    parent_mass = parameters.charged_kaon_mass
    parent_energy = kaon_energy

    return _dnde_positron(
        positron_energy=positron_energies,
        parent_energy=parent_energy,
        parent_mass=parent_mass,
        interp=interp,
    )


@overload
def dnde_positron_long_kaon(positron_energies: float, kaon_energy: float) -> float: ...


@overload
def dnde_positron_long_kaon(
    positron_energies: RealArray, kaon_energy: float
) -> RealArray: ...


def dnde_positron_long_kaon(
    positron_energies: RealOrRealArray, kaon_energy: float
) -> RealOrRealArray:
    """
    Returns the positron spectrum from the decay of a long kaon.

    Parameters
    ----------
    positron_energies : float or numpy.array
        Energy(ies) of the positron/electron.
    kaon_energy : float or numpy.array
        Energy of the long kaon.

    Returns
    -------
    dnde : float or numpy.array
        The value of the spectrum given a positron energy(ies)
        ``positron_energies`` and long kaon energy ``kaon_energy``.
    """
    interp = _long_kaon_integrand_interp
    parent_mass = parameters.neutral_kaon_mass
    parent_energy = kaon_energy

    return _dnde_positron(
        positron_energy=positron_energies,
        parent_energy=parent_energy,
        parent_mass=parent_mass,
        interp=interp,
    )


@overload
def dnde_positron_short_kaon(positron_energies: float, kaon_energy: float) -> float: ...


@overload
def dnde_positron_short_kaon(
    positron_energies: RealArray, kaon_energy: float
) -> RealArray: ...


def dnde_positron_short_kaon(
    positron_energies: RealOrRealArray, kaon_energy: float
) -> RealOrRealArray:
    """
    Returns the positron spectrum from the decay of a short kaon.

    Parameters
    ----------
    positron_energies : float or numpy.array
        Energy(ies) of the positron/electron.
    kaon_energy : float or numpy.array
        Energy of the short kaon.

    Returns
    -------
    dnde : float or numpy.array
        The value of the spectrum given a positron energy(ies)
        ``positron_energies`` and short kaon energy ``kaon_energy``.
    """
    interp = _short_kaon_integrand_interp
    parent_mass = parameters.neutral_kaon_mass
    parent_energy = kaon_energy

    return _dnde_positron(
        positron_energy=positron_energies,
        parent_energy=parent_energy,
        parent_mass=parent_mass,
        interp=interp,
    )


@overload
def dnde_positron_eta(positron_energy: float, eta_energy: float) -> float: ...


@overload
def dnde_positron_eta(positron_energy: RealArray, eta_energy: float) -> RealArray: ...


def dnde_positron_eta(
    positron_energy: Union[RealArray, float], eta_energy: float
) -> Union[RealArray, float]:
    interp = _eta_interp
    parent_mass = parameters.eta_mass
    parent_energy = eta_energy

    return _dnde_positron(
        positron_energy=positron_energy,
        parent_energy=parent_energy,
        parent_mass=parent_mass,
        interp=interp,
    )


@overload
def dnde_positron_omega(positron_energy: float, omega_energy: float) -> float: ...


@overload
def dnde_positron_omega(
    positron_energy: RealArray, omega_energy: float
) -> RealArray: ...


def dnde_positron_omega(
    positron_energy: Union[RealArray, float], omega_energy: float
) -> Union[RealArray, float]:
    interp = _omega_integrand_interp
    parent_mass = parameters.omega_mass
    parent_energy = omega_energy

    return _dnde_positron(
        positron_energy=positron_energy,
        parent_energy=parent_energy,
        parent_mass=parent_mass,
        interp=interp,
    )


@overload
def dnde_positron_neutral_rho(positron_energy: float, rho_energy: float) -> float: ...


@overload
def dnde_positron_neutral_rho(
    positron_energy: RealArray, rho_energy: float
) -> RealArray: ...


def dnde_positron_neutral_rho(
    positron_energy: Union[RealArray, float], rho_energy: float
) -> Union[RealArray, float]:
    interp = _rho_integrand_interp
    parent_mass = parameters.rho_mass
    parent_energy = rho_energy

    return _dnde_positron(
        positron_energy=positron_energy,
        parent_energy=parent_energy,
        parent_mass=parent_mass,
        interp=interp,
    )


@overload
def dnde_positron_charged_rho(positron_energy: float, rho_energy: float) -> float: ...


@overload
def dnde_positron_charged_rho(
    positron_energy: RealArray, rho_energy: float
) -> RealArray: ...


def dnde_positron_charged_rho(
    positron_energy: Union[RealArray, float], rho_energy: float
) -> Union[RealArray, float]:
    return dnde_positron_neutral_rho(positron_energy, rho_energy)


@overload
def dnde_positron_eta_prime(
    positron_energy: float, eta_prime_energy: float
) -> float: ...


@overload
def dnde_positron_eta_prime(
    positron_energy: RealArray, eta_prime_energy: float
) -> RealArray: ...


def dnde_positron_eta_prime(
    positron_energy: Union[RealArray, float], eta_prime_energy: float
) -> Union[RealArray, float]:
    interp = _eta_prime_integrand_interp
    parent_mass = parameters.eta_prime_mass
    parent_energy = eta_prime_energy

    return _dnde_positron(
        positron_energy=positron_energy,
        parent_energy=parent_energy,
        parent_mass=parent_mass,
        interp=interp,
    )


@overload
def dnde_positron_phi(positron_energy: float, phi_energy: float) -> float: ...


@overload
def dnde_positron_phi(positron_energy: RealArray, phi_energy: float) -> RealArray: ...


def dnde_positron_phi(
    positron_energy: Union[RealArray, float], phi_energy: float
) -> Union[RealArray, float]:
    interp = _phi_integrand_interp
    parent_mass = parameters.phi_mass
    parent_energy = phi_energy

    return _dnde_positron(
        positron_energy=positron_energy,
        parent_energy=parent_energy,
        parent_mass=parent_mass,
        interp=interp,
    )


def positron_decay(
    particles: List[str],
    cme: float,
    positron_energies: Union[List[float], RealArray],
    mat_elem_sqrd: Optional[SquaredMatrixElement] = None,
    num_ps_pts: int = 1000,
    num_bins: int = 25,
) -> RealArray:
    r"""Returns total gamma ray spectrum from a set of particles.

    Parameters
    ----------
    particles : array_like
        List of particle names. Available particles are 'muon', and
        'charged_pion'.
    cme : double
        Center of mass energy of the final state in MeV.
    positron_energies : np.ndarray[double, ndim=1]
        List of positron energies in MeV to evaluate spectra at.
    mat_elem_sqrd : double(\*func)(np.ndarray)
        Function for the matrix element squared of the process. Must be a
        function taking in a list of four momenta of size (num_fsp, 4).
        Default value is a flat matrix element returning 1..
    num_ps_pts : int {1000}, optional
        Number of phase space points to use.
    num_bins : int {25}, optional
        Number of bins to use.

    Returns
    -------
    spec : np.ndarray
        Total gamma ray spectrum from all final state particles.

    Notes
    -----
    The total spectrum is computed using

    .. math::
        \frac{dN}{dE}(E_{e^{\pm}}) = \sum_{i,j}P_{i}(E_{j})
        \frac{dN_i}{dE}(E_{e^{\pm}}, E_{j})

    where :math:`i` runs over the final state particles, :math:`j` runs over
    energies sampled from probability distributions. :math:`P_{i}(E_{j})` is
    the probability that particle :math:`i` has energy :math:`E_{j}`. The
    probabilities are computed using ``hazma.phase_space_generator.rambo``. The
    total number of energies used is ``num_bins``.

    Examples
    --------

    Generate spectrum from a muon, and two charged pions
    with total energy of 5 GeV::

        from hazma.positron_spectra import positron_decay
        from hazma.parameters import electron_mass as me
        import numpy as np
        particles = np.array(['muon', 'charged_pion', 'charged_pion'])
        cme = 5000.
        positron_energies = np.logspace(np.log10(me), np.log10(cme),
                                        num=200, dtype=np.float64)
        positron_decay(particles, cme, positron_energies)

    """
    if mat_elem_sqrd is None:
        msqrd = __flat_squared_matrix_element
    else:
        msqrd = mat_elem_sqrd

    if isinstance(particles, str):
        particles = [particles]
    # return positron(
    #     particles,
    #     cme,
    #     positron_energies,
    #     mat_elem_sqrd=msqrd,
    #     num_ps_pts=num_ps_pts,
    #     num_bins=num_bins,
    # )
    raise NotImplementedError()
