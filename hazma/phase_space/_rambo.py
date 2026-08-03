"""Module for working with Lorentz-invariant phase-space using the RAMBO algorithm."""

# pylint: disable=invalid-name,too-many-arguments,too-many-locals
# pylint: disable=too-many-instance-attributes


# TODO: Code up specific functions for cross-section
#       functions for 2->2 processes.
# TODO: Code up specific functions for cross-section
#       functions for 2->3 processes.


import math
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
from scipy import integrate

from hazma.utils import RealArray, cross_section_prefactor, kallen_lambda, lnorm_sqr

from ._base import AbstractPhaseSpaceGenerator, AbstractPhaseSpaceIntegrator
from ._dist import PhaseSpaceDistribution1D
from ._utils import energy_limits, invariant_mass_limits

MassList = Union[List[float], RealArray]
SquaredMatrixElement = Callable[[RealArray], float]


def __flat_squared_matrix_element(_: RealArray) -> float:
    return 1.0


class Rambo(AbstractPhaseSpaceIntegrator, AbstractPhaseSpaceGenerator):
    """
    Phase space generator and integrator using the RAMBO algorithm.
    """

    def __init__(
        self,
        cme: float,
        masses: Union[np.ndarray, Sequence[float]],
        msqrd: Optional[SquaredMatrixElement] = None,
    ) -> None:
        """
        Parameters
        ----------
        cme: float
            Center-of-mass energy of the process.
        masses: array-like
            Array of the final state particle masses.
        msqrd: Callable[[ndarray], float], optional
            Squared matrix element of the process.
        """
        ms = np.array(masses)
        n = len(ms)

        self.__cme = cme
        self.__masses = ms.reshape((n, 1))
        self.__msqrd: Optional[SquaredMatrixElement] = msqrd

        self.__n = n
        self.__xi0 = np.sqrt(1.0 - (np.sum(ms) / cme) ** 2)
        self.__base_wgt = self._compute_base_weight()

        self.__rng = np.random.default_rng()

        self.__eps = 2 ** (-52)

    @property
    def cme(self) -> float:
        """
        Center-of-mass energy of the process.
        """
        return self.__cme

    @cme.setter
    def cme(self, val) -> None:
        self.__cme = val
        self.__base_wgt = self._compute_base_weight()

    @property
    def masses(self) -> np.ndarray:
        """
        Masses of the final state particles.
        """
        return self.__masses.squeeze()

    @masses.setter
    def masses(self, masses: npt.NDArray) -> None:
        self.__n = len(masses)
        self.__masses = np.array(masses).reshape((self.__n, 1))
        self.__base_wgt = self._compute_base_weight()

    @property
    def msqrd(self) -> Optional[SquaredMatrixElement]:
        """
        Squared matrix element of the process.
        """
        return self.__msqrd

    def _compute_base_weight(self) -> float:
        """
        Compute the starting weight for each phase space point.
        """
        cme = self.__cme
        n = self.__n

        fact_nm2 = math.factorial(n - 2)
        fact = 1.0 / ((n - 1) * fact_nm2**2)

        return (
            fact
            * (0.5 * np.pi) ** (n - 1)
            * cme ** (2 * n - 4)
            * (0.5 / np.pi) ** (3 * n - 4)
        )

    def _initialize_momenta(self, batch_size: int, dtype=np.float64) -> np.ndarray:
        """
        Initialize isotropic and massless momenta with energyies
        distributed according to E exp(-E).

        Parameters
        ----------
        batch_size: int
            Number of phase space points to generate.

        Returns
        -------
        momenta: ndarray
            Array containing the initialized four momenta with shape
            (4, # final state particles, batch_size). The 1st dimension
            contains the energy, x-, y- and z-components of the 3-momenta. The
            2nd dimension runs over all final state particles. The 3rd
            dimension runs over all events.
        """
        n = self.__n

        rho1 = self.__rng.random(size=(n, batch_size), dtype=dtype)
        rho2 = self.__rng.random(size=(n, batch_size), dtype=dtype)
        rho3 = self.__rng.random(size=(n, batch_size), dtype=dtype)
        rho4 = self.__rng.random(size=(n, batch_size), dtype=dtype)

        ctheta = 2 * rho1 - 1.0
        stheta = np.sqrt(1.0 - ctheta**2)
        phi = 2.0 * np.pi * rho2
        e = -np.log(rho3 * rho4)

        return np.array(
            [e, e * stheta * np.cos(phi), e * stheta * np.sin(phi), e * ctheta]
        )

    def _boost(self, ps: np.ndarray) -> np.ndarray:
        """
        Boost momenta into the center-of-mass frame. Input should be the
        result of `_initialize_momenta`.

        Parameters
        ----------
        ps: ndarray
            Momenta generated from `_initialize_momenta`.

        Returns
        -------
        momenta: ndarray
            Momenta in the center-of-mass frame. The shape is identical to
            input shape.
        """
        sum_ps = np.sum(ps, axis=1)
        inv_mass = np.sqrt(
            sum_ps[0] ** 2 - sum_ps[1] ** 2 - sum_ps[2] ** 2 - sum_ps[3] ** 2
        )
        inv_mass = 1.0 / inv_mass

        bx = -inv_mass * sum_ps[1]
        by = -inv_mass * sum_ps[2]
        bz = -inv_mass * sum_ps[3]

        x = self.__cme * inv_mass
        g = sum_ps[0] * inv_mass
        a = 1.0 / (1.0 + g)

        bdotp = bx * ps[1] + by * ps[2] + bz * ps[3]
        fact = a * bdotp + ps[0]

        ps[0] = x * (g * ps[0] + bdotp)
        ps[1] = x * (fact * bx + ps[1])
        ps[2] = x * (fact * by + ps[2])
        ps[3] = x * (fact * bz + ps[3])

        return ps

    def _compute_scale_factor(self, ps: np.ndarray, iterations=5) -> np.ndarray:
        """
        Compute the scale factor needed to correct the masses of the final
        state particle momenta.

        Parameters
        ----------
        ps: ndarray
            Momenta returned from `_boost`.

        Returns
        -------
        xi: ndarray
            Scale factors for each phase-space point. The shape is
            (batch_size,) where `batch_size` is the size of the last
            dimension of `ps`.
        """
        cme = self.__cme
        xi0 = self.__xi0

        shape = ps.shape[1:]
        e = ps[0]
        xi = xi0 * np.ones((shape[-1],))
        it = 0
        last = False

        while not last:
            deltaf = np.hypot(e * xi, self.__masses)
            f = np.sum(deltaf, axis=0) - cme
            df = np.sum(xi * e**2 / deltaf, axis=0)
            xi = xi - f / df

            # Newton iterations have quadratic convergence. If we are below sqrt(eps),
            # do only one more iteration
            if np.max(f) < np.sqrt(self.__eps) or it == iterations - 1:
                last = True

            it = it + 1

        return xi

    def _correct_masses(self, ps: np.ndarray) -> np.ndarray:
        """
        Correct the masses of the final state particles.

        Parameters
        ----------
        ps: ndarray
            Momenta returned from `_boost`.

        Returns
        -------
        ps: ndarray
            Momenta with corrected masses. The shape is identical to input.
        """
        xi = self._compute_scale_factor(ps)
        ps[0] = np.hypot(xi * ps[0], self.__masses)
        ps[1:] = xi * ps[1:]
        return ps

    def _compute_weights(self, ps) -> np.ndarray:
        """
        Compute the weights of phase space points.

        Parameters
        ----------
        ps: ndarray
            Momenta returned from `_correct_masses`.

        Returns
        -------
        weights: ndarray
            Weights of the phase space points. The shape is (batch_size, )
            where `batch_size` is the size of the last dimension of `ps`.
        """
        cme = self.__cme
        n = self.__n

        modsqr = np.sum(ps[1:] ** 2, axis=0)
        mod = np.sqrt(modsqr)

        t1 = np.sum(mod / cme, axis=0) ** (2 * n - 3)
        t2 = np.sum(modsqr / ps[0], axis=0)
        t3 = np.prod(mod / ps[0], axis=0)

        return t1 / t2 * t3 * self.__cme * self.__base_wgt

    def generator(
        self, n, batch_size: int, seed: Optional[int] = None, dtype=np.float64
    ) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        """
        Create a generator the yields four-momenta and weights distributed
        according to Lorentz-invariant phase space.

        Parameters
        ----------
        n: int
            Number of phase space points to generate.
        batch_size: int
            Number of phase space points to generate each step.
        seed: int, optional
            Seed used for numpy random number generator.
        dtype: DTypeLike, optional
            Type used for generation of momenta and weights.

        Yields
        -------
        momenta: ndarray
            Batch of momenta containing the initialized four momenta with shape
            (4, # final state particles, batch_size). The 1st dimension
            contains the energy, x-, y- and z-components of the 3-momenta. The
            2nd dimension runs over all final state particles. The 3rd
            dimension runs over all events.
        weights: ndarray
            Batch of weights of the phase space points. Shape is (batch_size,).

        Examples
        --------
        Integrating over phase-space in batches:

        >>> phase_space = PhaseSpace(cme=10.0, masses=[1.0, 2.0, 3.0])
        >>> integrals = []
        >>> for _, weights in phase_space.generator(n=100, batch_size=10, seed=1234):
        ...     integrals.append(np.nanmean(integrals))
        >>> np.average(integrals)
        0.003632349411041629
        """
        if dtype == np.float64:
            self.__eps = 2 ** (-52)
        elif dtype == np.float32:
            self.__eps = 2 ** (-23)

        self.__rng = np.random.default_rng(seed)

        niters = n // batch_size
        niters += 0 if batch_size * niters == n else 1

        for _ in range(niters):
            ps = self._initialize_momenta(batch_size, dtype=dtype)
            ps = self._boost(ps)
            ps = self._correct_masses(ps)
            ws = self._compute_weights(ps)

            if self.__msqrd is not None:
                ws *= self.__msqrd(ps)
            yield (ps, ws)

    def generate(
        self, n: int, seed: Optional[int] = None, dtype=np.float64
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate four-momenta and weights distributed according to
        Lorentz-invariant phase space.

        Parameters
        ----------
        n: int
            Number of phase space points to generate.
        seed: int, optional
            Seed used for numpy random number generator.
        dtype: DTypeLike, optional
            Type used for generation of momenta and weights.

        Returns
        -------
        momenta: ndarray
            Array containing the initialized four momenta with shape
            (4, # final state particles, n). The 1st dimension
            contains the energy, x-, y- and z-components of the 3-momenta. The
            2nd dimension runs over all final state particles. The 3rd
            dimension runs over all events.
        weights: ndarray
            Array containing weights of the phase space points. Shape is given by
            (n,).

        Examples
        --------
        Integrating over phase-space:

        >>> phase_space = PhaseSpace(cme=10.0, masses=[1.0, 2.0, 3.0])
        >>> integrals = []
        >>> momenta, weights = phase_space.generator(n=100, seed=1234)
        >>> np.average(weights)
        0.0037198111038192366
        """
        if dtype == np.float64:
            self.__eps = 2 ** (-52)
        elif dtype == np.float32:
            self.__eps = 2 ** (-23)

        self.__rng = np.random.default_rng(seed)

        ps = self._initialize_momenta(n, dtype=dtype)
        ps = self._boost(ps)
        ps = self._correct_masses(ps)
        ws = self._compute_weights(ps)

        if self.__msqrd is not None:
            ws *= self.__msqrd(ps)

        return ps, ws

    def _integrate_two_body(self) -> Tuple[float, float]:
        cme = self.cme
        m1, m2 = self.masses
        p = np.sqrt(kallen_lambda(cme**2, m1**2, m2**2)) / (2 * cme)
        e1 = np.hypot(m1, p)
        e2 = np.hypot(m2, p)
        ps = np.zeros((4, 2), dtype=np.float64)

        msqrd = __flat_squared_matrix_element if self.msqrd is None else self.msqrd

        def integrand(z):
            sin = np.sqrt(1 - z**2)
            ps[:, 0] = np.array([e1, sin * p, 0.0, z * p])
            ps[:, 1] = np.array([e2, -sin * p, 0.0, -z * p])
            return msqrd(ps)

        pre = 1.0 / (8.0 * np.pi) * p / cme

        integral, error = integrate.quad(integrand, -1.0, 1.0)

        return integral * pre, error * pre

    def integrate(
        self,
        n: int,
        batch_size: Optional[int] = None,
        seed: Optional[int] = None,
        dtype=np.float64,
    ) -> Tuple[float, float]:
        """
        Integrate over phase space.

        Parameters
        ----------
        n: int
            Number of phase space points used in integration.
        batch_size: int, optional
            If not None, the phase-space integration will be broken up into
            batches, processing `batch_size` points at a time. Default is None.
        seed: int, optional
            Seed used for numpy random number generator.
        dtype: DTypeLike, optional
            Type used for generation of momenta and weights.

        Returns
        -------
        integral: float
            Value of the phase space integration.
        error_estimate: float
            Estimation of the error.
        """
        if batch_size is not None and not batch_size == n:
            integrals = []
            errors = []
            for _, ws in self.generator(n, batch_size, seed, dtype):
                avg = np.nanmean(ws, dtype=np.float64)
                std = np.nanstd(ws, dtype=np.float64, ddof=1) / np.sqrt(batch_size)
                integrals.append(avg)
                errors.append(std)

            # Average of averages okay since all samples are same size
            integral: float = np.nanmean(integrals, dtype=float)
            # Combined error estimate using quadrature
            error: float = np.sqrt(np.nansum(np.square(errors))) / len(errors)

            return integral, error

        _, weights = self.generate(n, seed=seed, dtype=dtype)
        integral = np.nanmean(weights, dtype=float)
        error = np.nanstd(weights, dtype=float, ddof=1) / np.sqrt(n)

        return integral, error

    def decay_width(
        self,
        n,
        batch_size: Optional[int] = None,
        seed: Optional[int] = None,
        dtype=np.float64,
    ) -> Tuple[float, float]:
        """
        Compute the decay width.

        Parameters
        ----------
        n: int
            Number of phase space points used in integration.
        seed: int, optional
            Seed used for numpy random number generator.
        dtype: DTypeLike, optional
            Type used for generation of momenta and weights.

        Returns
        -------
        width: float
            Estimated value of the decay width.
        error_estimate: float
            Estimation of the error.
        """
        integral, std = self.integrate(n, batch_size=batch_size, seed=seed, dtype=dtype)

        pre = 0.5 / self.__cme
        width = pre * integral
        error = pre * std

        return width, error

    def cross_section(
        self,
        m1,
        m2,
        n,
        batch_size: Optional[int] = None,
        seed: Optional[int] = None,
        dtype=np.float64,
    ) -> Tuple[float, float]:
        """
        Compute the cross section given input masses.

        Parameters
        ----------
        m1: float
            Mass of the 1st incoming particle.
        m2: float
            Mass of the 2nd incoming particle.
        n: int
            Number of phase space points used in integration.
        seed: int, optional
            Seed used for numpy random number generator.
        dtype: DTypeLike, optional
            Type used for generation of momenta and weights.

        Returns
        -------
        cross_section: float
            Estimated value of the cross-section.
        error_estimate: float
            Estimation of the error.
        """
        integral, std = self.integrate(n, batch_size=batch_size, seed=seed, dtype=dtype)

        pre = cross_section_prefactor(m1, m2, self.__cme)
        cross_section = pre * integral
        error = pre * std

        return cross_section, error

    def energy_distributions(
        self,
        n: int,
        nbins: int,
        batch_size: Optional[int] = None,
        seed: Optional[int] = None,
        dtype=np.float64,
    ) -> List[PhaseSpaceDistribution1D]:
        """
        Generate energy distributions of the final state particles.

        Parameters
        ----------
        n: int
            Number of phase space points used in generating the distributions.
        nbins: int
            Number of bins to use for the distributions.
        batch_size: int, optional
            If not None, the phase-space integration will be broken up into
            batches, processing `batch_size` points at a time. Default is None.
        seed: int, optional
            Seed used for numpy random number generator.
        dtype: DTypeLike, optional
            Type used for generation of momenta and weights.

        Returns
        -------
        distributions: List[PhaseSpaceDistribution1D]
            List of the energy distributions. The order is the same as the
            masses used to instantiate the class.
        """
        nfsp = len(self.masses)

        ebounds = energy_limits(self.cme, self.masses)  # type: ignore
        bins = [np.linspace(emin, emax, nbins + 1) for emin, emax in ebounds]

        distributions = [
            np.histogram(np.zeros(nbins, dtype=np.float64), bins=b)[0] for b in bins
        ]

        if batch_size is not None and not batch_size == n:
            # Build up the distributions from batches
            for ps, ws in self.generator(n, batch_size, seed, dtype):
                for i in range(len(self.masses)):
                    d = np.histogram(ps[0, i], bins=bins[i], weights=ws)[0]
                    distributions[i] += d
        else:
            ps, ws = self.generate(n, seed=seed, dtype=dtype)
            for i in range(len(self.masses)):
                distributions[i] = np.histogram(ps[0, i], bins=bins[i], weights=ws)[0]

        dists = [
            PhaseSpaceDistribution1D(bins[i], distributions[i]) for i in range(nfsp)
        ]
        return dists

    def invariant_mass_distributions(
        self,
        n: int,
        nbins: int,
        batch_size: Optional[int] = None,
        seed: Optional[int] = None,
        dtype=np.float64,
    ) -> Dict[Tuple[int, int], PhaseSpaceDistribution1D]:
        """
        Generate invariant mass distributions of the final state particles.

        The invariant masses are defined as sqrt((pi + pj)^2), where pi and pj
        are the four-momenta of the ith and jth particles.

        Parameters
        ----------
        n: int
            Number of phase space points used in generating the distributions.
        nbins: int
            Number of bins to use for the distributions.
        batch_size: int, optional
            If not None, the phase-space integration will be broken up into
            batches, processing `batch_size` points at a time. Default is None.
        seed: int, optional
            Seed used for numpy random number generator.
        dtype: DTypeLike, optional
            Type used for generation of momenta and weights.

        Returns
        -------
        distributions: Dict[Tuple[int,int], PhaseSpaceDistribution1D]
            The invariant mass distributions. The keys are pairs of integers
            specifying the pair of particles.
        """
        masses = self.masses
        nfsp = len(masses)
        bounds = invariant_mass_limits(self.cme, masses)  # type: ignore
        pairs = bounds.keys()

        def make_bins(pair):
            mmin, mmax = bounds[pair]
            return np.linspace(mmin, mmax, nbins + 1)

        def inv_mass(ps, pair):
            i, j = pair
            return np.sqrt(np.abs(lnorm_sqr(ps[:, i] + ps[:, j])))

        bins = [make_bins(pair) for pair in pairs]

        distributions = [
            np.histogram(np.zeros(nbins, dtype=np.float64), bins=b)[0] for b in bins
        ]

        if batch_size is not None and not batch_size == n:
            # Build up the distributions from batches
            for ps, ws in self.generator(n, batch_size, seed, dtype):
                for i, pair in enumerate(pairs):
                    d = np.histogram(inv_mass(ps, pair), bins=bins[i], weights=ws)[0]
                    distributions[i] += d
        else:
            ps, ws = self.generate(n, seed=seed, dtype=dtype)
            for i, pair in enumerate(pairs):
                distributions[i] = np.histogram(
                    inv_mass(ps, pair), bins=bins[i], weights=ws
                )[0]

        dists = {
            key: PhaseSpaceDistribution1D(bins[i], distributions[i])
            for i in range(nfsp)
            for i, key in enumerate(pairs)
        }
        return dists
