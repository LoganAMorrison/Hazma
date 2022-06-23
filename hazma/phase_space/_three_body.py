from typing import Callable, Any, Optional, Sequence, Tuple, Dict, List

import numpy as np
from scipy import integrate

from hazma.utils import RealArray, kallen_lambda, lnorm_sqr

from ._rambo import Rambo
from ._utils import energy_limits, invariant_mass_limits
from ._dist import PhaseSpaceDistribution1D
from ._base import AbstractPhaseSpaceIntegrator

ThreeBodyMsqrd = Callable[[Any, Any], Any]
ThreeBodyMasses = Tuple[float, float, float]
InvariantMassDists = Dict[Tuple[int, int], PhaseSpaceDistribution1D]


def _msqrd_flat(s, t):
    scalar_s = np.isscalar(s)
    scalar_t = np.isscalar(t)

    if scalar_s and scalar_t:
        return 1.0

    if scalar_s:
        return np.zeros_like(t)

    return np.zeros_like(s)


class ThreeBody(AbstractPhaseSpaceIntegrator):
    r"""Class for computing various aspects of three-body phase-space."""

    def __init__(
        self,
        cme: float,
        masses: Sequence[float],
        msqrd: Optional[ThreeBodyMsqrd] = None,
    ):
        """
        Parameters
        ----------
        cme: float
            Center-of-mass energy.
        masses: sequence float
            The three final state particle masses.
        msqrd: callable
            Binary function taking in the squared invariant masses s=(p2+p3)^2,
            and t=(p1+p3)^2.
        """
        assert (
            len(masses) == 3
        ), f"Expected 'masses' to have length 3, found {len(masses)}."

        self.__cme = cme
        self.__masses: Tuple[float, float, float] = (masses[0], masses[1], masses[2])

        if msqrd is None:
            self.__msqrd = _msqrd_flat
        else:
            self.__msqrd = msqrd

    @property
    def cme(self) -> float:
        """
        Center-of-mass energy of the proccess.
        """
        return self.__cme

    @cme.setter
    def cme(self, val) -> None:
        self.__cme = val

    @property
    def masses(self) -> Tuple[float, float, float]:
        """
        Masses of the final state particles.
        """
        return self.__masses

    @masses.setter
    def masses(self, masses: Sequence[float]) -> None:
        assert (
            len(masses) == 3
        ), f"Expected 'masses' to have length 3, found {len(masses)}."

        self.__masses = (masses[0], masses[1], masses[2])

    @property
    def msqrd(self) -> Optional[ThreeBodyMsqrd]:
        """
        Squared matrix element of the proccess.
        """
        return self.__msqrd

    def __trapz(self, fn, a: float, b: float, npts: int):
        xs = np.linspace(a, b, npts)
        fs = np.array([fn(x) for x in xs])
        return integrate.trapz(fs, xs)

    def __simps(self, fn, a: float, b: float, npts: int):
        xs = np.linspace(a, b, npts)
        fs = np.array([fn(x) for x in xs])
        return integrate.simps(fs, xs)

    def __quad(self, fn, a: float, b: float, epsabs: float, epsrel: float):
        return integrate.quad(fn, a, b, epsabs=epsabs, epsrel=epsrel)[0]

    def __integrate_fn(
        self,
        fn,
        a: float,
        b: float,
        method: str,
        npts: int,
        epsabs: float,
        epsrel: float,
    ):
        if method == "trapz":
            return self.__trapz(fn, a, b, npts)
        if method == "simps":
            return self.__simps(fn, a, b, npts)
        if method == "quad":
            return self.__quad(fn, a, b, epsabs=epsabs, epsrel=epsrel)
        raise ValueError(f"Invalid method {method}.")

    def _integration_bounds_t(self, s, masses: Optional[Sequence[float]] = None):
        r"""Compute the integrate bounds on the 'Mandelstam' variable t = (p1 + p3)^2
        for a three-body final state.

        Parameters
        ----------
        s: float
            Invariant mass of particles 2 and 3.

        Returns
        -------
        lb, ub: float
            Lower and upper integration bounds.
        """
        if masses is None:
            masses = self.__masses

        m = self.__cme
        m1, m2, m3 = masses

        msqr = m**2
        m12 = m1**2
        m22 = m2**2
        m32 = m3**2

        pfac = -(msqr - m12) * (m22 - m32) + (msqr + m12 + m22 + m32) * s - s**2
        sfac = np.sqrt(kallen_lambda(s, m12, m22) * kallen_lambda(s, m22, m32))

        lb = 0.5 * (pfac - sfac) / s
        ub = 0.5 * (pfac + sfac) / s

        return lb, ub

    def _integration_bounds_s(self, masses: Optional[Sequence[float]] = None):
        r"""Compute the integrate bounds on the 'Mandelstam' variable s = (p2 + p3)^2
        for a three-body final state.

        Returns
        -------
        lb, ub: float
            Lower and upper integration bounds.
        """
        if masses is None:
            masses = self.__masses

        m = self.__cme
        m1, m2, m3 = masses

        lb = (m2 + m3) ** 2
        ub = (m - m1) ** 2
        return lb, ub

    def _integration_bounds_y(self, x, masses: Optional[Sequence[float]] = None):
        r"""Compute the integrate bounds on the scaled energy of particle 2: y =
        2*E2/cme.

        Parameters
        ----------
        s: float
            Invariant mass of particles 2 and 3.

        Returns
        -------
        lb, ub: float
            Lower and upper integration bounds.
        """
        if masses is None:
            masses = self.__masses

        mu1, mu2, mu3 = [mass / self.__cme for mass in masses]

        pfac = (2 - x) * (1 - x + mu1**2 + mu2**2 - mu3**2)
        sfac = np.sqrt(
            kallen_lambda(1, mu1**2, 1 - x + mu1**2)
            * kallen_lambda(1 - x + mu1**2, mu2**2, mu3**2)
        )

        lb = 0.5 * (pfac - sfac) / (1 - x + mu1**2)
        ub = 0.5 * (pfac + sfac) / (1 - x + mu1**2)
        return lb, ub

    def _integration_bounds_x(self, masses: Optional[Sequence[float]] = None):
        r"""Compute the integrate bounds on the scaled energy of particle 1: x =
        2*E1/cme.

        Returns
        -------
        lb, ub: float
            Lower and upper integration bounds.
        """
        if masses is None:
            masses = self.__masses

        mu1, mu2, mu3 = [mass / self.__cme for mass in masses]

        lb = 2 * mu1
        ub = mu1**2 + (1 - (mu2 + mu3) ** 2)
        return lb, ub

    def _partial_integration(
        self,
        e,
        i: int,
        method: str = "quad",
        npts: int = 100,
        epsabs: float = 1.49e-8,
        epsrel: float = 1.49e-8,
    ):
        r"""Compute the integral of the squared matrix element over the scaled
        energy of particle 2: y = 2 * E2 / q.

        Parameters
        ----------
        msqrd: callable
            Binary function taking in the squared invariant masses s=(p2+p3)^2, and
            t=(p1+p3)^2.
            q: float
        Center-of-mass energy.
        e: float or array-like
        Energy of particle 1.
        masses: (float, float, float)
        Masses of the final state particles.
        i: int
        Final state particle to integrate over.
        method: str
        Method used to integrate. Can be 'trapz', 'simps' or 'quad'. Default is
        'quad'.

        Returns
        -------
        dI/de: float
        Differential phase-space density with respect to the energy of particle 1.
        """
        assert 0 <= i <= 2, f"Invalid argument i = {i}. Must be 0,1 or 2."

        q = self.__cme
        m1, m2, m3 = self.__masses
        pre = q / (64.0 * np.pi**3)

        def msqrdxy(x, y):
            s = q**2 * (1.0 - x + (m1 / q) ** 2)
            t = q**2 * (1.0 - y + (m2 / q) ** 2)
            return self.__msqrd(s, t)

        if i == 0:
            new_masses = m1, m2, m3

            def integrand(x, y):
                return msqrdxy(x, y)

        elif i == 1:
            # swap 1 and 2
            # x -> y, y -> x, z -> z
            new_masses = m2, m1, m3

            def integrand(x, y):
                return msqrdxy(y, x)

        elif i == 2:
            # swap 1 and 3
            # x -> z, y -> y, z -> z
            new_masses = m3, m2, m1

            def integrand(x, y):
                return msqrdxy(2 - (x + y), y)

        else:
            raise ValueError(f"Invalid argument i = {i}. Must be 0,1 or 2.")

        xmin, xmax = self._integration_bounds_x(new_masses)

        def bounds(x):
            return self._integration_bounds_y(x, new_masses)

        def integral(x):
            if x < xmin or xmax < x:
                return 0.0
            lb, ub = bounds(x)
            return pre * self.__integrate_fn(
                lambda y: integrand(x, y),
                lb,
                ub,
                method=method,
                npts=npts,
                epsabs=epsabs,
                epsrel=epsrel,
            )

        single = np.isscalar(e)
        es = np.atleast_1d(e)
        res = np.array([integral(2 * e / q) for e in es])

        if single:
            return res[0]
        return res

    def _integrate_quad(
        self, *, epsabs: float = 1.49e-8, epsrel: float = 1.49e-8
    ) -> Tuple[float, float]:
        r"""Compute the integral of the squared matrix element over three-body phase
        space.

        Returns
        -------
        integral: float
            Integral of the squared matrix element.
        error: float
            Error estimate of the integral.
        """
        q = self.__cme
        m1, m2, _ = self.__masses
        mu1, mu2 = m1 / q, m2 / q
        pre = q**2 / (128.0 * np.pi**3)

        def ymin(x):
            return self._integration_bounds_y(x)[0]

        def ymax(x):
            return self._integration_bounds_y(x)[1]

        def integrand(y, x):
            s = q**2 * (1.0 - x + mu1**2)
            t = q**2 * (1.0 - y + mu2**2)
            return self.__msqrd(s, t)

        xmin, xmax = self._integration_bounds_x()
        res = integrate.dblquad(
            integrand, xmin, xmax, ymin, ymax, epsabs=epsabs, epsrel=epsrel
        )
        return pre * res[0], pre * res[1]

    def _msqrd_rambo(self, momenta: RealArray):
        s = lnorm_sqr(momenta[:, 1] + momenta[:, 2])
        t = lnorm_sqr(momenta[:, 0] + momenta[:, 2])
        return self.__msqrd(s, t)

    def _integrate_rambo(self, n: int) -> Tuple[float, float]:
        """Compute the integral of the squared matrix element over the squared
        invariant mass of particles 2 and 3.

        Parameters
        ----------
        msqrd: callable
        Binary function taking in the squared invariant masses s=(p2+p3)^2, and
        t=(p1+p3)^2.
        q: float
        Center-of-mass energy.
        masses: (float, float, float)
        Masses of the final state particles.
        """
        masses = self.__masses
        q = self.__cme

        phase_space = Rambo(
            cme=q,
            masses=np.array(masses),
            msqrd=self._msqrd_rambo,  # type: ignore
        )
        return phase_space.integrate(n=n)

    def integrate(
        self,
        method: str = "quad",
        npts: int = 10000,
        epsrel: float = 1.49e-8,
        epsabs: float = 1.49e-8,
    ) -> Tuple[float, float]:
        r"""Compute the integral of the squared matrix element over the squared
        invariant mass of particles 2 and 3.

        Parameters
        ----------
        method: str
            Method used to integrate over phase space. Can be:
                'quad': Numerical quadrature using scipy's 'dblquad',
                'rambo': Monte-Carlo integration.
        npts: int
            Number of phase-space points to use to integrate squared matrix
            element. Ignored unless method is 'rambo'. Default is 10000.

        Returns
        -------
        integral: float
            Integral of the squared matrix element.
        error: float
            Error estimate.
        """
        masses = self.__masses
        q = self.__cme

        methods = {
            "quad": lambda: self._integrate_quad(epsabs=epsabs, epsrel=epsrel),
            "rambo": lambda: self._integrate_rambo(n=npts),
        }

        if q < sum(masses):
            return (0.0, 0.0)

        meth = methods.get(method)
        if meth is None:
            ms = ",".join(methods.keys())
            raise ValueError(
                f"Invalid method: {method}. Use one of the following: {ms}."
            )

        return meth()

    def _energy_distributions_quad(
        self, nbins: int, *, epsabs: float = 1.49e-8, epsrel: float = 1.49e-8
    ):
        """Compute energy distributions using numerical integration."""

        def integral(e, i):
            return self._partial_integration(e=e, i=i, epsabs=epsabs, epsrel=epsrel)

        elims = energy_limits(self.__cme, self.__masses)

        dists = []
        for i, (emin, emax) in enumerate(elims):
            ebins = np.linspace(emin, emax, nbins)
            es = 0.5 * (ebins[1:] + ebins[:-1])

            dwde = integral(es, i)
            dists.append(PhaseSpaceDistribution1D(ebins, dwde))

        return dists

    def _energy_distributions_rambo(self, nbins: int, npts: int = 10000):
        """Compute energy distributions using Monte-Carlo integration."""
        phase_space = Rambo(
            cme=self.__cme,
            masses=np.array(self.__masses),
            msqrd=self._msqrd_rambo,  # type: ignore
        )
        return phase_space.energy_distributions(n=npts, nbins=nbins)

    def energy_distributions(
        self, nbins: int, method: str = "quad", npts: int = 10000
    ) -> List[PhaseSpaceDistribution1D]:
        """Compute the energy distributions of the three final-state particles.

        Parameters
        ----------
        method: str
            Method used to integrate over phase space. Can be:
                'quad': Numerical quadrature using scipy's 'quad',
                'rambo': Monte-Carlo integration.
        npts: int
            Number of phase-space points to use to integrate squared matrix
            element. Ignored unless method is 'rambo'. Default is 10000.

        Returns
        -------
        dist1, dist2, dist3: (array, array)
            Distributions of the final-state particles.
        """
        methods = {
            "quad": lambda: self._energy_distributions_quad(nbins=nbins),
            "rambo": lambda: self._energy_distributions_rambo(nbins=nbins, npts=npts),
        }

        meth = methods.get(method)
        if meth is None:
            ms = ",".join(methods.keys())
            raise ValueError(
                f"Invalid method: {method}. Use one of the following: {ms}."
            )

        return meth()

    def _invariant_mass_distributions_quad(self, nbins: int) -> InvariantMassDists:
        """Compute invariant-mass distributions using numerical integration."""
        q = self.__cme
        masses = self.__masses

        def integral(e, i):
            return self._partial_integration(e, i)

        invmass_lims = invariant_mass_limits(q, masses)
        dists: Dict[Tuple[int, int], PhaseSpaceDistribution1D] = dict()

        for i, ((j, k), (mmin, mmax)) in enumerate(invmass_lims.items()):
            m1 = masses[i]

            mbins = np.linspace(mmin, mmax, nbins)
            ms = 0.5 * (mbins[1:] + mbins[:-1])
            es = (q**2 - ms**2 + m1**2) / (2 * q)

            dwdm = ms / q * integral(es, i)
            dists[(j, k)] = PhaseSpaceDistribution1D(mbins, dwdm)

        return dists

    def _invariant_mass_distributions_rambo(
        self, nbins: int, npts: int = 10000
    ) -> InvariantMassDists:
        """Compute invariant-mass distributions using Monte-Carlo integration."""
        phase_space = Rambo(
            cme=self.__cme,
            masses=np.array(self.__masses),
            msqrd=self._msqrd_rambo,  # type: ignore
        )
        return phase_space.invariant_mass_distributions(n=npts, nbins=nbins)

    def invariant_mass_distributions(
        self, nbins: int, method: str = "quad", npts: int = 10000
    ) -> InvariantMassDists:
        r"""Compute the invariant-mass distributions of the three pairs of
        final-state particles.

        Parameters
        ----------
        method: str
            Method used to integrate over phase space. Can be:
                'quad': Numerical quadrature using scipy's 'quad',
                'trapz': Trapiziod rule using scipy's 'trapz',
                'simps': Simpson's rule using scipy's 'simps',
                'rambo': Monte-Carlo integration.
        npts: int
            Number of phase-space points to use to integrate squared matrix
            element. Ignored unless method is 'rambo'. Default is 10000.

        Returns
        -------
        dist1, dist2, dist3: (array, array)
            Distributions of the final-state particles.

        Raises
        ------
        ValueError
            If the final-state particles masses are larger than center-of-mass
            energy.
        """
        q = self.__cme
        masses = self.__masses

        methods = {
            "quad": lambda: self._invariant_mass_distributions_quad(nbins=nbins),
            "rambo": lambda: self._invariant_mass_distributions_rambo(
                nbins=nbins, npts=npts
            ),
        }
        if q < sum(masses):
            raise ValueError(
                "Center of mass energy is less than the sum of final-state particle masses."
            )

        meth = methods.get(method)
        if meth is None:
            ms = ",".join(methods.keys())
            raise ValueError(
                f"Invalid method: {method}. Use one of the following: {ms}."
            )

        return meth()
