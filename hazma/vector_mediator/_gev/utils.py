from typing import Dict, List, Union, overload

import numpy as np
import numpy.typing as npt

from hazma import parameters

RealArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

# Number of rest-frame energies on which the `v v` positron and neutrino
# spectra tabulate a mediator's decay spectrum before boosting it. At
# m_V = 1 GeV the boosted spectra agree with a 32,000-point tabulation to
# 1.2e-5 (positrons) and 5.7e-4 (neutrinos) wherever they exceed 1e-3 of
# their peak, and conserve particle number to 2e-5.
V_V_REST_FRAME_POINTS = 2000


def _two_body_energy(m: float, m1: float, m2: float) -> float:
    """Energy of the first daughter of `m -> m1 m2` at rest, in MeV."""
    return (m * m + m1 * m1 - m2 * m2) / (2.0 * m)


# The delta-function lines in the charged pion's positron and neutrino
# spectra, as (rest-frame energy, daughter mass) in MeV: the neutrinos of
# `pi -> mu nu` and `pi -> e nu`, and the positron of `pi -> e nu`. They are
# the only lines in the decay spectra the `v v` sums read.
_CHARGED_PION_LINES = [
    (_two_body_energy(parameters.charged_pion_mass, 0.0, parameters.muon_mass), 0.0),
    (
        _two_body_energy(parameters.charged_pion_mass, 0.0, parameters.electron_mass),
        0.0,
    ),
    (
        _two_body_energy(parameters.charged_pion_mass, parameters.electron_mass, 0.0),
        parameters.electron_mass,
    ),
]

# Relative offset, in units of a box's width, of the grid points that
# bracket each of its edges.
_EDGE_OFFSET = 1e-6


def v_v_rest_energies(mv: float, emin: float) -> RealArray:
    """Rest-frame energies on which to tabulate a mediator's decay spectrum.

    The grid has `V_V_REST_FRAME_POINTS` log-spaced energies from `emin` to
    `mv / 2`, in MeV. A pion from `V -> pi pi` carries `mv / 2` and boosts
    each of its lines into a box whose relative width, about twice the
    pion's velocity, vanishes at threshold, so the log spacing alone can
    step over it. The grid therefore also brackets each box edge from both
    sides, which makes the linear interpolant carry the box's full area.
    """
    emax = 0.5 * mv
    energies = [np.geomspace(emin, emax, V_V_REST_FRAME_POINTS)]

    mpi = parameters.charged_pion_mass
    if emax > mpi:
        gamma = emax / mpi
        beta = np.sqrt(1.0 - gamma**-2)
        for e0, m in _CHARGED_PION_LINES:
            p0 = np.sqrt(e0 * e0 - m * m)
            lo = gamma * (e0 - beta * p0)
            hi = gamma * (e0 + beta * p0)
            offset = _EDGE_OFFSET * (hi - lo)
            energies.append(
                np.array([lo - offset, lo + offset, hi - offset, hi + offset])
            )

    grid = np.unique(np.concatenate(energies))
    return grid[(grid >= emin) & (grid <= emax)]


def float_zeros_like(energies: float | RealArray) -> float | RealArray:
    """Floating-point zeros shaped like `energies`; a float for a scalar."""
    # Indexing with `()` unwraps a 0-d array and leaves any other untouched.
    return np.zeros(np.shape(energies))[()]


def call_with_kinematic_threshold(f, x, thresholds: List[float]):
    """
    Call a unary function
    """
    if hasattr(x, "__len__"):
        mask = np.array([True] * len(x))
        for t in thresholds:
            mask = np.logical_and(mask, x > t)

        result = np.zeros_like(x)
        if not np.all(~mask):
            result[mask] = f(x[mask])
        return result

    accessible = np.all([x > t for t in thresholds])
    result = 0.0
    if accessible:
        result = f(x)
    return 0.0


STR_TO_MASS: Dict[str, float] = {
    "e": parameters.electron_mass,
    "mu": parameters.muon_mass,
    "pi": parameters.charged_pion_mass,
    "pi0": parameters.neutral_pion_mass,
    "k": parameters.charged_kaon_mass,
    "k0": parameters.neutral_kaon_mass,
    "eta": parameters.eta_mass,
    "etap": parameters.eta_prime_mass,
    "omega": parameters.omega_mass,
    "phi": parameters.phi_mass,
    "ve": 0.0,
    "vm": 0.0,
    "vt": 0.0,
    "gamma": 0.0,
}


@overload
def channel_open(
    cme: float,
    state: str,
    extra_masses: Dict[str, float] = dict(),
    delimiter: str = " ",
) -> bool: ...


@overload
def channel_open(
    cme: RealArray,
    state: str,
    extra_masses: Dict[str, float] = dict(),
    delimiter: str = " ",
) -> BoolArray: ...


def channel_open(
    cme: Union[float, RealArray],
    state: str,
    extra_masses: Dict[str, float] = dict(),
    delimiter: str = " ",
) -> Union[bool, BoolArray]:
    """Return True if the channel is kinematically accessible.

    Parameters
    ----------
    cme: float
        Center-of-mass energy.
    state: str
        String containing the states with the specified delimiter.
    extra_masses: dict[str,float], optional
        Extra masses aside from the SM particles.
    delimiter: str, optional
        Delimiter of the states in state string. Default is a single space `' '`.

    Returns
    -------
    accessible: bool
        True if the channel is open.
    """
    mass_dict: Dict[str, float] = {**STR_TO_MASS, **extra_masses}
    states: List[str] = state.split(delimiter)
    return cme > sum(map(lambda s: mass_dict[s], states))
