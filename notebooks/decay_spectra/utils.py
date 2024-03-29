import abc
import dataclasses
import functools
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import numpy.typing as npt
from scipy import integrate, interpolate

from hazma import parameters, spectra
from hazma.rambo import PhaseSpace
from hazma.utils import kallen_lambda, lnorm_sqr

T = TypeVar("T")

BinType = Union[npt.NDArray[np.float64], int]

ME = 0.5109989461  # m[e-] = 0.5109989461 ± 3.1e-09
MMU = 105.6583745  # m[mu-] = 105.6583745 ± 2.4e-06
MTAU = 1776.86  # m[tau-] = 1776.86 ± 0.12
MPI0 = 134.9768  # m[pi0] = 134.9768 ± 0.0005
MPI = 139.57039  # m[pi+] = 139.57039 ± 0.00018
META = 547.862  # m[eta] = 547.862 ± 0.017
METAP = 957.78  # m[eta'(958)] = 957.78 ± 0.06
MK = 493.677  # m[K+] = 493.677 ± 0.016
MK0 = 497.611  # m[K0] = 497.611 ± 0.013
MKL = 497.611  # m[K(L)0] = 497.611 ± 0.013
MKS = 497.611  # m[K(S)0] = 497.611 ± 0.013
MRHO = 775.26  # m[rho(770)0] = 775.26 ± 0.23
MOMEGA = 782.66  # m[omega(782)] = 782.66 ± 0.13
MPHI = 1019.461  # m[phi(1020)] = 1019.461 ± 0.016

qualitative = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#a65628",
    "#f781bf",
    "#999999",
    "#ffff33",
]


def two_body_three_momentum(cme, m1, m2):
    return np.sqrt(kallen_lambda(cme**2, m1**2, m2**2)) / (2 * cme)


def energy_one_cme(cme, m1, m2):
    return (cme**2 + m1**2 - m2**2) / (2 * cme)


class ParticleType(Enum):
    Scalar = 0
    Fermion = 1
    Vector = 2


def _dndx_photon_fsr(x, s, m, q=1.0, ty: ParticleType = ParticleType.Fermion):
    pre = q**2 * parameters.alpha_em / (2.0 * np.pi)
    xm = 1.0 - x
    kernel = np.zeros_like(x)

    if ty == ParticleType.Scalar:
        mask = s * xm / m**2 > np.e
        kernel[mask] = 2.0 * xm[mask] / x[mask] * (np.log(s * xm[mask] / m**2) - 1.0)
    elif ty == ParticleType.Fermion:
        mask = s * xm / m**2 > np.e
        kernel[mask] = (
            (1.0 + xm[mask] ** 2) / x[mask] * (np.log(s * xm[mask] / m**2) - 1.0)
        )
    else:
        mask = x < 1 - 2 * m / np.sqrt(s)

        y = s * xm[mask] ** 2 / (4.0 * m**2)

        x_mask = x[mask]
        xm_mask = xm[mask]

        lf1 = np.log(y) + 2.0 * np.log(1.0 - np.sqrt(1.0 - 1.0 / y))
        lf2 = np.log(s / m**2)

        kernel[mask] = 2.0 * (
            x_mask / xm_mask * lf1 + xm_mask / x_mask * lf2 + x_mask * xm_mask * lf2
        )
        kernel = np.clip(kernel, 0.0, None)

    return pre * kernel


def _dnde_photon_fsr(e, s, m, q=1.0, ty: ParticleType = ParticleType.Fermion):
    e_to_x = 2.0 / np.sqrt(s)
    x = e * e_to_x
    return _dndx_photon_fsr(x, s, m, q, ty) * e_to_x


def dndx_photon_fsr_fermion(x, s, m, q=1.0):
    """
    Compute dndx from the FSR off a charged fermion f from a
    process of the form X -> (f + Y + gamma) + Z.

    Parameters
    ----------
    x:
        Scaled energies of the fermion, x = 2E/sqrt(s).
    s:
        Squared center-of-mass energy flowing through of the radiating
        fermion and Y in the process X -> (f + Y) + Z.
    m:
        Mass of the radiating fermion.
    q:
        Charge of the radiating fermion. Default is 1.
    """
    return _dndx_photon_fsr(x, s, m, q=q, ty=ParticleType.Fermion)


def dndx_photon_fsr_scalar(x, s, m, q=1.0):
    return _dndx_photon_fsr(x, s, m, q=q, ty=ParticleType.Scalar)


def dndx_photon_fsr_vector(x, s, m, q=1.0):
    return _dndx_photon_fsr(x, s, m, q=q, ty=ParticleType.Vector)


def dnde_photon_fsr_fermion(e, s, m, q=1.0):
    return _dnde_photon_fsr(e, s, m, q=q, ty=ParticleType.Fermion)


def dnde_photon_fsr_scalar(e, s, m, q=1.0):
    return _dnde_photon_fsr(e, s, m, q=q, ty=ParticleType.Scalar)


def dnde_photon_fsr_vector(e, s, m, q=1.0):
    return _dnde_photon_fsr(e, s, m, q=q, ty=ParticleType.Vector)


def convolve(e, dnde, energies, probabilities):
    integrand = np.array([p * dnde(e, ep) for ep, p in zip(energies, probabilities)])
    if len(integrand) == 1:
        return np.sum(integrand, axis=0)
    return np.trapz(integrand, energies, axis=0)


def invariant_mass_distribution_analytic(
    m0: float, m1: float, m2: float, m3: float
) -> Callable[[T], T]:
    def unnormalized(s):
        p1 = kallen_lambda(s, m0**2, m1**2)
        p2 = kallen_lambda(s, m2**2, m3**2)
        return np.sqrt(p1 * p2) / s

    lb = (m2 + m3) ** 2
    ub = (m0 - m1) ** 2
    norm: float = integrate.quad(unnormalized, lb, ub)[0]

    def dist(s):
        return unnormalized(s) / norm

    return dist


def energy_distribution_analytic(
    m0: float, m1: float, m2: float, m3: float
) -> Callable[[T], T]:
    def unnormalized(e):
        s = m0**2 + m1**2 - 2 * m0 * e
        p1 = kallen_lambda(s, m0**2, m1**2)
        p2 = kallen_lambda(s, m2**2, m3**2)
        return np.sqrt(p1 * p2) / s

    lb = m1
    ub = (m0**2 + m1**2 - (m2 + m3) ** 2) / (2 * m0)
    norm: float = integrate.quad(unnormalized, lb, ub)[0]

    def dist(e):
        return unnormalized(e) / norm

    return dist


@dataclasses.dataclass(frozen=True)
class Particle:
    name: str
    mass: float
    ty: ParticleType
    charge: float

    _dnde_photon_fsr_fn: Callable[[Any, Any], Any] = dataclasses.field(
        init=False, repr=False
    )

    def __post_init__(self):
        if not self.charge == 0.0:
            _dnde_photon_fsr_fn = functools.partial(
                _dnde_photon_fsr, m=self.mass, q=self.charge, ty=self.ty
            )

            object.__setattr__(self, "_dnde_photon_fsr_fn", _dnde_photon_fsr_fn)
        else:
            object.__setattr__(
                self, "_dnde_photon_fsr_fn", lambda e, *_: np.zeros_like(e)
            )

    def dnde_photon_decay(self, photon_energy, _):
        return np.zeros_like(photon_energy)

    def dnde_positron_decay(self, positron_energy, _):
        return np.zeros_like(positron_energy)

    def dnde_neutrino_decay(self, neutrino_energy, _):
        return np.zeros_like(neutrino_energy)

    def dnde_photon_fsr(self, photon_energy, s):
        return self._dnde_photon_fsr_fn(photon_energy, s)


@dataclasses.dataclass(frozen=True)
class _NeutralPion(Particle):
    name: str = "pi0"
    mass: float = MPI0
    ty: ParticleType = ParticleType.Scalar
    charge: float = 0.0

    def dnde_photon_decay(self, photon_energy, self_energy):
        return spectra.dnde_photon_neutral_pion(photon_energy, self_energy)


@dataclasses.dataclass(frozen=True)
class _ChargedPion(Particle):
    name: str = "pi"
    mass: float = MPI
    ty: ParticleType = ParticleType.Scalar
    charge: float = 1.0

    def dnde_photon_decay(self, photon_energy, self_energy):
        return spectra.dnde_photon_charged_pion(photon_energy, self_energy)

    def dnde_positron_decay(self, positron_energy, self_energy):
        return spectra.dnde_positron_charged_pion(positron_energy, self_energy)

    def dnde_neutrino_decay(self, positron_energy, self_energy):
        return spectra.dnde_neutrino_charged_pion(positron_energy, self_energy)


@dataclasses.dataclass(frozen=True)
class _Muon(Particle):
    name: str = "mu"
    mass: float = MMU
    ty: ParticleType = ParticleType.Fermion
    charge: float = -1.0

    def dnde_photon_decay(self, photon_energy, self_energy):
        return spectra.dnde_photon_muon(photon_energy, self_energy)

    def dnde_positron_decay(self, photon_energy, self_energy):
        return spectra.dnde_positron_muon(photon_energy, self_energy)

    def dnde_neutrino_decay(self, photon_energy, self_energy):
        return spectra.dnde_neutrino_muon(photon_energy, self_energy)


@dataclasses.dataclass(frozen=True)
class _Electron(Particle):
    name: str = "e"
    mass: float = ME
    ty: ParticleType = ParticleType.Fermion
    charge: float = -1.0


@dataclasses.dataclass(frozen=True)
class _ChargedKaon(Particle):
    name: str = "k"
    mass: float = MK
    ty: ParticleType = ParticleType.Scalar
    charge: float = 1.0

    def dnde_photon_decay(self, photon_energy, self_energy):
        return spectra.dnde_photon_charged_kaon(photon_energy, self_energy)

    def dnde_positron_decay(self, positron_energy, self_energy):
        return spectra.dnde_positron_charged_kaon(positron_energy, self_energy)


@dataclasses.dataclass(frozen=True)
class _LongKaon(Particle):
    name: str = "kL"
    mass: float = MK0
    ty: ParticleType = ParticleType.Scalar
    charge: float = 0.0

    def dnde_photon_decay(self, photon_energy, self_energy):
        return spectra.dnde_photon_long_kaon(photon_energy, self_energy)

    def dnde_positron_decay(self, positron_energy, self_energy):
        return spectra.dnde_positron_long_kaon(positron_energy, self_energy)


@dataclasses.dataclass(frozen=True)
class _ShortKaon(Particle):
    name: str = "kS"
    mass: float = MK0
    ty: ParticleType = ParticleType.Scalar
    charge: float = 0.0

    def dnde_photon_decay(self, photon_energy, self_energy):
        return spectra.dnde_photon_short_kaon(photon_energy, self_energy)

    def dnde_positron_decay(self, positron_energy, self_energy):
        return spectra.dnde_positron_short_kaon(positron_energy, self_energy)


@dataclasses.dataclass(frozen=True)
class _Eta(Particle):
    name: str = "eta"
    mass: float = META
    ty: ParticleType = ParticleType.Scalar
    charge: float = 0.0

    def dnde_photon_decay(self, photon_energy, self_energy):
        return spectra.dnde_photon_eta(photon_energy, self_energy)

    def dnde_positron_decay(self, positron_energy, self_energy):
        return spectra.dnde_positron_eta(positron_energy, self_energy)


@dataclasses.dataclass(frozen=True)
class _EtaPrime(Particle):
    name: str = "etap"
    mass: float = METAP
    ty: ParticleType = ParticleType.Scalar
    charge: float = 0.0


@dataclasses.dataclass(frozen=True)
class _Omega(Particle):
    name: str = "omega"
    mass: float = MOMEGA
    ty: ParticleType = ParticleType.Vector
    charge: float = 0.0

    def dnde_photon_decay(self, photon_energy, self_energy):
        return spectra.dnde_photon_omega(photon_energy, self_energy)

    def dnde_positron_decay(self, positron_energy, self_energy):
        return spectra.dnde_positron_omega(positron_energy, self_energy)


@dataclasses.dataclass(frozen=True)
class _NeutralRho(Particle):
    name: str = "rho0"
    mass: float = MRHO
    ty: ParticleType = ParticleType.Vector
    charge: float = 0.0

    def dnde_photon_decay(self, photon_energy, self_energy):
        return spectra.dnde_photon_neutral_rho(photon_energy, self_energy)

    def dnde_positron_decay(self, positron_energy, self_energy):
        return spectra.dnde_positron_neutral_rho(positron_energy, self_energy)


@dataclasses.dataclass(frozen=True)
class _ChargedRho(Particle):
    name: str = "rho"
    mass: float = MRHO
    ty: ParticleType = ParticleType.Vector
    charge: float = 1.0

    def dnde_photon_decay(self, photon_energy, self_energy):
        return spectra.dnde_photon_charged_rho(photon_energy, self_energy)

    def dnde_positron_decay(self, positron_energy, self_energy):
        return spectra.dnde_positron_charged_rho(positron_energy, self_energy)

    def dnde_photon_fsr(self, photon_energy, _):
        return np.zeros_like(photon_energy)


@dataclasses.dataclass(frozen=True)
class _Phi(Particle):
    name: str = "phi"
    mass: float = MPHI
    ty: ParticleType = ParticleType.Vector
    charge: float = 0.0


@dataclasses.dataclass(frozen=True)
class _Photon(Particle):
    name: str = "a"
    mass: float = 0.0
    ty: ParticleType = ParticleType.Vector
    charge: float = 0.0


electron = _Electron()
muon = _Muon()
charged_pion = _ChargedPion()
neutral_pion = _NeutralPion()
charged_kaon = _ChargedKaon()
long_kaon = _LongKaon()
short_kaon = _ShortKaon()
eta = _Eta()
eta_prime = _EtaPrime()
omega = _Omega()
neutral_rho = _NeutralRho()
charged_rho = _ChargedRho()
neutrino = Particle(mass=0.0, ty=ParticleType.Fermion, charge=0.0, name="nu")
photon = _Photon()
phi = _Phi()


class ThreeBodyPhaseSpace:
    def __init__(
        self,
        m0: float,
        m1: float,
        m2: float,
        m3: float,
        msqrd: Optional[Callable],
    ) -> None:
        self.m0 = m0
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.has_msqrd = msqrd is not None
        self.phase_space = PhaseSpace(m0, np.array([m1, m2, m3]), msqrd=msqrd)

    def _energy_bounds(self) -> List[Tuple[float, float]]:
        m0 = self.m0

        def bounds(m1, m2, m3):
            emin = m1
            emax = (m0**2 + m1**2 - (m2 + m3) ** 2) / (2 * m0)
            return emin, emax

        return [
            bounds(self.m1, self.m2, self.m3),
            bounds(self.m2, self.m1, self.m3),
            bounds(self.m3, self.m1, self.m2),
        ]

    def __energy_probabilities(
        self, bins: tuple[BinType, BinType, BinType], npts=10000
    ) -> npt.NDArray[np.float64]:
        momenta, weights = self.phase_space.generate(npts)

        p1 = momenta[:, 0]
        p2 = momenta[:, 1]
        p3 = momenta[:, 2]

        e1s = p1[0]
        e2s = p2[0]
        e3s = p3[0]

        p1s, _ = np.histogram(e1s, bins=bins[0], weights=weights, density=True)
        p2s, _ = np.histogram(e2s, bins=bins[1], weights=weights, density=True)
        p3s, _ = np.histogram(e3s, bins=bins[2], weights=weights, density=True)

        return np.array([p1s, p2s, p3s])

    def __invariant_mass_probabilities(
        self,
        i: int,
        j: int,
        bins: BinType,
        npts: int = 10000,
    ) -> npt.NDArray[np.float64]:
        assert i in [0, 1, 2], f"Invalid index i={i}. Must be 0, 1, or 2"
        assert j in [0, 1, 2], f"Invalid index j={j}. Must be 0, 1, or 2"

        momenta, weights = self.phase_space.generate(npts)

        sij = lnorm_sqr(momenta[:, i] + momenta[:, j])
        probs = np.histogram(sij, bins=bins, weights=weights, density=True)[0]

        return probs

    @staticmethod
    def __update_distribution(
        dist: npt.NDArray[np.float64],
        ctor: Callable[[], npt.NDArray[np.float64]],
        batchsize: int,
    ) -> Tuple[npt.NDArray[np.float64], float]:
        new_dist: npt.NDArray[np.float64] = np.zeros_like(dist)
        for _ in range(batchsize):
            new_dist += ctor()
        max_delta = np.max(np.abs(new_dist / dist))

        return new_dist + dist, max_delta

    def __build_distribution(
        self,
        dist: npt.NDArray[np.float64],
        ctor: Callable[[], npt.NDArray[np.float64]],
        eps: float,
        maxiter: int,
        batchsize: int,
    ) -> npt.NDArray[np.float64]:
        # warm up
        for _ in range(batchsize):
            dist += ctor()
        count = batchsize

        max_delta = np.inf
        converged = False
        for _ in range(maxiter):
            dist, max_delta = self.__update_distribution(dist, ctor, batchsize)
            count += batchsize

            if max_delta < eps:
                converged = True
                break

        if not converged:
            print(
                "warning: popability distributions did not converge."
                + f"max error: {max_delta}"
            )
        return dist / count

    def energy_distributions(
        self, nbins: int = 20, eps: float = 1e-2, maxiter: int = 100, batchsize: int = 5
    ) -> List[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
        ebounds = self._energy_bounds()
        bins = tuple(
            np.linspace(ebounds[i][0], ebounds[i][1], nbins + 1) for i in range(3)
        )
        cs = tuple((bins[i][1:] + bins[i][:-1]) / 2 for i in range(3))

        if not self.has_msqrd:
            m0 = self.m0
            dndes = [
                energy_distribution_analytic(m0, self.m1, self.m2, self.m3),
                energy_distribution_analytic(m0, self.m2, self.m1, self.m3),
                energy_distribution_analytic(m0, self.m3, self.m1, self.m2),
            ]

            return [(c, dnde(c)) for c, dnde in zip(cs, dndes)]

        def ctor():
            return self.__energy_probabilities(bins)

        ps = np.zeros((3, nbins), dtype=np.float64)
        ps = self.__build_distribution(ps, ctor, eps, maxiter, batchsize)

        return [(c, p) for c, p in zip(cs, ps)]

    def invariant_mass_distribution(
        self, i: int, j: int, nbins: int = 20, eps=1e-2, maxiter=100, batchsize=5
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        ms = [self.m1, self.m2, self.m3]
        # s = (pi + pj)^2 = (P - pk)^2
        k = {0, 1, 2}.difference({i, j}).pop()

        smin = (ms[i] + ms[j]) ** 2
        smax = (self.m0 - ms[k]) ** 2
        bins: npt.NDArray[np.float64] = np.linspace(smin, smax, nbins + 1)
        cs = (bins[1:] + bins[:-1]) / 2

        if not self.has_msqrd:
            m0 = self.m0
            m1 = ms[k]
            m2 = ms[i]
            m3 = ms[j]
            dist = invariant_mass_distribution_analytic(m0, m1, m2, m3)
            ps = dist(cs)
            return cs, ps

        def ctor():
            return self.__invariant_mass_probabilities(i, j, nbins)

        ps = np.zeros(nbins, dtype=np.float64)
        ps = self.__build_distribution(ps, ctor, eps, maxiter, batchsize)

        return cs, ps


@dataclasses.dataclass
class DecayProcess:
    parent: Particle = dataclasses.field()
    final_states: List[Particle] = dataclasses.field()
    branching_fraction: float = dataclasses.field()
    msqrd: Optional[Callable] = dataclasses.field(default=None, repr=False)
    nbins: int = dataclasses.field(default=25, repr=False)

    energy_distributions: List[Tuple[np.ndarray, np.ndarray]] = dataclasses.field(
        init=False, repr=False
    )

    invariant_mass_distributions: Dict[
        Tuple[int, int], Tuple[np.ndarray, np.ndarray]
    ] = dataclasses.field(init=False, repr=False)

    def __post_init__(self):
        if len(self.final_states) == 2:
            m1 = self.final_states[0].mass
            m2 = self.final_states[1].mass
            e1 = energy_one_cme(self.parent.mass, m1, m2)
            e2 = energy_one_cme(self.parent.mass, m2, m1)
            self.energy_distributions = [
                (np.array([e1]), np.array([1.0])),
                (np.array([e2]), np.array([1.0])),
            ]
            self.invariant_mass_distributions = {
                (0, 1): (np.array([self.parent.mass**2]), np.array([1.0]))
            }
        elif len(self.final_states) == 3:
            m1 = self.final_states[0].mass
            m2 = self.final_states[1].mass
            m3 = self.final_states[2].mass
            tbps = ThreeBodyPhaseSpace(self.parent.mass, m1, m2, m3, self.msqrd)

            self.energy_distributions = tbps.energy_distributions(
                self.nbins, maxiter=1000
            )

            self.invariant_mass_distributions = {
                (1, 2): tbps.invariant_mass_distribution(
                    1, 2, self.nbins, maxiter=1000
                ),
                (0, 2): tbps.invariant_mass_distribution(
                    0, 2, self.nbins, maxiter=1000
                ),
                (0, 1): tbps.invariant_mass_distribution(
                    0, 1, self.nbins, maxiter=1000
                ),
            }
        else:
            raise NotImplementedError()


@dataclasses.dataclass
class DecayProcesses:
    parent: Particle
    processes: Dict[str, DecayProcess]


@dataclasses.dataclass
class DecayProcessInfo:
    final_states: List[Particle] = dataclasses.field()
    branching_fraction: float = dataclasses.field()
    msqrd: Optional[Callable] = dataclasses.field(default=None, repr=False)


def __dnde_photon_process(
    photon_energy, process: DecayProcess, apply_branching_fraction: bool = True
):
    # Decay spectrum
    dnde = np.zeros_like(photon_energy)
    for particle, dist in zip(process.final_states, process.energy_distributions):
        dnde += convolve(photon_energy, particle.dnde_photon_decay, dist[0], dist[1])

    # FSR spectrum
    dnde_fsr = {
        i: {"dnde": np.zeros_like(photon_energy), "count": 0}
        for i in range(len(process.final_states))
    }
    for pair in process.invariant_mass_distributions.keys():
        s, prob = process.invariant_mass_distributions[pair]
        for i in pair:
            particle = process.final_states[i]
            dnde_fsr[i]["count"] += 1
            dnde_fsr[i]["dnde"] += convolve(
                photon_energy, particle.dnde_photon_fsr, s, prob
            )

    # average over counts
    for val in dnde_fsr.values():
        dnde += val["dnde"] / val["count"]

    if apply_branching_fraction:
        dnde *= process.branching_fraction

    return dnde


def dnde_photon(
    photon_energy,
    process: Union[DecayProcess, DecayProcesses],
    apply_branching_fraction: bool = True,
):
    if isinstance(process, DecayProcess):
        return __dnde_photon_process(photon_energy, process)
    elif isinstance(process, DecayProcesses):
        return {
            name: __dnde_photon_process(photon_energy, proc, apply_branching_fraction)
            for name, proc in process.processes.items()
        }


def __dnde_positron_process(
    positron_energy, process: DecayProcess, apply_branching_fraction: bool = True
):
    # Decay spectrum
    dnde = np.zeros_like(positron_energy)
    for particle, dist in zip(process.final_states, process.energy_distributions):
        dnde += convolve(
            positron_energy, particle.dnde_positron_decay, dist[0], dist[1]
        )

    if apply_branching_fraction:
        dnde *= process.branching_fraction

    return dnde


def dnde_positron(
    positron_energy,
    process: Union[DecayProcess, DecayProcesses],
    apply_branching_fraction: bool = True,
):
    if isinstance(process, DecayProcess):
        return __dnde_positron_process(positron_energy, process)
    elif isinstance(process, DecayProcesses):
        return {
            name: __dnde_positron_process(
                positron_energy, proc, apply_branching_fraction
            )
            for name, proc in process.processes.items()
        }


def make_processes(
    parent,
    processes: List[DecayProcessInfo],
    include_all_two_body: bool = True,
    threshold=0.01,
    nbins=25,
) -> DecayProcesses:
    procs: Dict[str, DecayProcess] = dict()
    for process in processes:
        name = " ".join(map(lambda s: s.name, process.final_states))
        include = process.branching_fraction > threshold
        include = include or (len(process.final_states) == 2 and include_all_two_body)
        if include:
            procs[name] = DecayProcess(
                parent=parent,
                final_states=process.final_states,
                msqrd=process.msqrd,
                branching_fraction=process.branching_fraction,
                nbins=nbins,
            )

    return DecayProcesses(parent=parent, processes=procs)


_KAON = Union[_ChargedKaon, _ShortKaon, _LongKaon]


def make_msqrd_k_to_pi_l_nu(
    kaon: _KAON,
    ml: float,
    lam_p: float,
    lam_0: float,
    pi: Union[_NeutralPion, _ChargedPion],
):
    mk2 = kaon.mass**2
    mpi2 = charged_pion.mass**2

    def fp(s):
        return 1.0 + lam_p * s / mpi2

    def fm(_):
        return (mk2 - mpi2) / mpi2 * (lam_0 - lam_p)

    def msqrd(momenta):
        s = lnorm_sqr(momenta[:, 1] + momenta[:, 2])
        t = lnorm_sqr(momenta[:, 0] + momenta[:, 2])
        return -4 * (
            (ml**4 - ml**2 * s) * fm(s) ** 2
            + 2 * ml**2 * (ml**2 + 2 * pi.mass**2 - s - 2 * t) * fm(s) * fp(s)
            + (
                ml**4
                + 4 * mk2 * (pi.mass**2 - t)
                + 4 * t * (-(pi.mass**2) + s + t)
                - ml**2 * (s + 4 * t)
            )
            * fp(s) ** 2
        )

    return msqrd


def make_msqrd_k_to_ppp(
    kaon: _KAON,
    masses: List[float],
    g: float,
    h: float,
    k: float,
):
    """
    Construct the squared matrix element for K -> pi_1 + pi_2 + pi_3.

    Parameters
    ----------
    kaon:
        The decaying kaon.
    masses:
        Final state particle masses.
    g: float
        Linear coefficient. See particle listing for charged kaon.
    h, k: float
        Quadratic coefficent.  See particle listing for charged kaon.

    Returns
    -------
    msqrd: Callable[[Any,Any], Any]
        Squared matrix element.
    """
    s0 = (kaon.mass**2 + sum(map(lambda m: m**2, masses))) / 3.0

    def msqrd(momenta):
        s1 = lnorm_sqr(momenta[:, 1] + momenta[:, 2])
        s2 = lnorm_sqr(momenta[:, 0] + momenta[:, 2])
        s3 = 3.0 * s0 - s1 - s2
        x = (s3 - s0) / charged_pion.mass**2
        y = (s2 - s1) / charged_pion.mass**2  # type: ignore
        return 1.0 + g * x + h * x**2 + k * y**2

    return msqrd


STATE_TO_LATEX = {
    "mu": r"$\mu^{\mp}$",
    "mubar": r"$\mu^{\pm}$",
    "e": r"$e^{\mp}$",
    "ebar": r"$e^{\pm}$",
    "nu": r"$\nu$",
    "pi": r"$\pi^{\pm}$",
    "pibar": r"$\pi^{\mp}$",
    "pi0": r"$\pi^{0}$",
    "a": r"$\gamma$",
    "k": r"$K^{\pm}$",
    "kbar": r"$K^{\mp}$",
    "kL": r"$K_{L}$",
    "kS": r"$K_{S}$",
    "k0": r"$K^{0}$",
    "k0bar": r"$\bar{K}^{0}$",
    "eta": r"$\eta$",
    "etap": r"$\eta'$",
    "rho": r"$\rho^{\pm}$",
    "rhobar": r"$\rho^{\mp}$",
    "rho0": r"$\rho^{0}$",
    "omega": r"$\omega$",
    "phi": r"$\phi$",
}


def process_string_to_latex(process: str):
    states: List[str] = process.split(" ")
    assert (
        len(states) > 1
    ), f"Invalid process string {process}. Found less than 2 states."

    latex = map(lambda s: STATE_TO_LATEX[s], states)
    return " + ".join(latex)
