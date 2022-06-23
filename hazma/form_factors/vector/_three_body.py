from dataclasses import dataclass
import abc
from typing import Tuple, Union, Dict, List

import numpy as np
from scipy import integrate

from hazma.utils import RealArray, kallen_lambda
from hazma.phase_space import ThreeBody
from hazma.phase_space._utils import energy_limits
from hazma.phase_space import PhaseSpaceDistribution1D

from ._base import VectorFormFactor


def __squared_lorentz_structure_ppp(q, s, t, m1, m2, m3):
    q2 = q**2
    num = (
        -(m1**4 * m2**2)
        + m2**2 * (q2 - s) * (m3**2 - t)
        - (m3**2 + q2 - s - t) * (m3**2 * q2 - s * t)
        + m1**2
        * (-(m2**4) + (m3**2 - s) * (q2 - t) + m2**2 * (m3**2 + q2 + s + t))
    )

    den = 12.0 * q**2

    return num / den


def _squared_lorentz_structure_ppp(q, s, t, m1, m2, m3):
    scalar_q = np.isscalar(q)
    scalar_s = np.isscalar(s)
    scalar_t = np.isscalar(t)

    if scalar_q and scalar_s and scalar_t:
        return __squared_lorentz_structure_ppp(q, s, t, m1, m2, m3)

    # q is in first dimension, s/t are in second.
    qq = np.expand_dims(np.atleast_1d(q).astype(np.float64), -1)
    ss = np.expand_dims(np.atleast_1d(s).astype(np.float64), 0)
    tt = np.expand_dims(np.atleast_1d(t).astype(np.float64), 0)

    res = __squared_lorentz_structure_ppp(qq, ss, tt, m1, m2, m3)

    if scalar_q:
        return res[0]
    elif scalar_t and scalar_s:
        return res[:, 0]

    return res


@dataclass
class VectorFormFactorThreeBody(VectorFormFactor):
    r"""Abstract base class for vector form-factors with three final-state
    mesons.

    This class requires the `form_factor` and `squared_lorentz_structure` to be
    implemented.

    Deriving from this class provides access to generic functions for computing
    integrated form factors, energy distributions and invariant mass
    distributions. Provided the the subclass implements `form_factor` and `sq`
    - `_integrated_form_factor`: integrate over three-body phase-space,
    - `_energy_distributions`: generate energy distributions for each final state,
    - `_invariant_mass_distributions`: generate distributions for each invariant
      mass.
    """

    fsp_masses: Tuple[float, float, float]

    @abc.abstractmethod
    def form_factor(self, q, *args, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def squared_lorentz_structure(self, q, s, t):
        raise NotImplementedError()

    @abc.abstractmethod
    def energy_distributions(self, q: float, nbins: int, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def invariant_mass_distributions(self, q: float, nbins: int, **kwargs):
        raise NotImplementedError()


# =============================================================================
# ---- PPP --------------------------------------------------------------------
# =============================================================================


@dataclass
class VectorFormFactorPPP(VectorFormFactorThreeBody):
    r"""Abstract base class for three pseudo-scalar meson form factors.

    The three pseudo-scalar meson currents are given by:

    .. math::

        J_{\mu} = \epsilon_{\mu\nu\alpha\beta}
        p^{\nu}_{1}p^{\alpha}_{2}p^{\beta}_{3}
        F_{P_1P_2P_3}(q^2, s, t)

    where :math:`F_{P_1P_2P_3}(q^2, s, t)` is the form factor,

    .. math::

        q^{\mu} &= p^{\mu}_{1} + p^{\mu}_{2} + p^{\mu}_{3}\\

        s &= (p_{2} + p_{3})^2\\

        t &= (p_{1} + p_{3})^2

    and :math:`p_{1},p_{2},p_{3}` are the momenta.
    """

    fsp_masses: Tuple[float, float, float]

    @abc.abstractmethod
    def form_factor(self, q, s, t, **kwargs):
        """Compute the squared matrix element."""
        raise NotImplementedError()

    def squared_lorentz_structure(self, q, s, t):
        r"""Compute the squared Lorentz structure of a three pseudo-scalar meson
        final-state.

        Parameters
        ----------
        q: float or array-like
            Center-of-mass energy.
        s: float or array-like
            Invariant mass of particles 2 and 3: s = (p2 + p3)^2
        t: float or array-like
            Invariant mass of particles 1 and 3: s = (p1 + p3)^2

        Returns
        -------
        lorentz: float or array-like
            Lorentz structure. If both `q` has shape `(n,)` and `s` or `t` has shape
            `(m,)`, then the result has shape `(n,m)`. If `q` is a scalar, then the
            shape is equal to the shape of `s` or `t`. If `s` and `t` are scalars,
            then the result has shape equal to the shape of `q`.

        Notes
        -----
        The current is defined as:
            J[mu] = eps[mu,nu1,nu2,nu3] p1[nu1] p2[nu2] p3[nu3] FF[p1,p2,p3]
        where:
            `eps`: the Levi-Civita tensor,
            `p1`: momentum of 1st scalar,
            `p2`: momentum of 2nd scalar,
            `p3`: momentum of 3rd scalar,
            `FF`: form-factor.
        The returned values is the square of this expression excluding the
        form-factor and contracting with -g[mu1,mu2]/(3q^2).
        """
        m1, m2, m3 = self.fsp_masses
        return _squared_lorentz_structure_ppp(q, s, t, m1, m2, m3)

    def __phase_space_integrand(self, q, s, t, **kwargs):
        r"""Compute the integrand of the three-body phase-space.

        Parameters
        ----------
        q: float or array-like
            Center-of-mass energy.
        s: float or array-like
            Invariant mass of particles 2 and 3: s = (p2 + p3)^2
        t: float or array-like
            Invariant mass of particles 1 and 3: t = (p1 + p3)^2
        kwargs: dict
            Keyword arguments to pass to derived class's `form_factor` function.

        Returns
        -------
        msqrd: float or array-like
            Squared matrix element.
        """
        ff = self.form_factor(q, s, t, **kwargs)
        lor = self.squared_lorentz_structure(q, s, t)
        return np.abs(ff) ** 2 * lor

    def _integrated_form_factor(
        self,
        *,
        q: Union[float, RealArray],
        method: str = "quad",
        npts: int = 10000,
        **kwargs,
    ) -> Union[float, RealArray]:
        """Compute the integrated from factor for a three pseudo-scalar meson
        final-state.

        Parameters
        ----------
        q: float or array-like
            Center of mass energy.
        method: str, optional
            Method used to integrate. Default is 'quad'. Options are 'quad' or
            'rambo'.
        npts: int, optional
            Number of phase-space points to use in integration. Ignored is
            method isn't 'rambo'. Default is 10_000.

        Returns
        -------
        ff: float or array-like
            Integrated form-factor.
        """

        def integrate_(q_):
            tb = ThreeBody(
                cme=q_,
                masses=self.fsp_masses,
                msqrd=lambda s, t: self.__phase_space_integrand(q_, s, t, **kwargs),
            )
            return tb.integrate(method=method, npts=npts)[0]

        scalar = np.isscalar(q)
        qs = np.atleast_1d(q)
        mask = qs > sum(self.fsp_masses)
        res = np.zeros_like(qs)

        if np.any(mask):
            res[mask] = np.array([integrate_(q_) for q_ in qs[mask]])

        if scalar:
            return res[0]
        return res

    def _energy_distributions(
        self,
        *,
        q: float,
        nbins: int,
        method: str = "quad",
        npts: int = 10000,
        **kwargs,
    ):
        r"""Compute the energy distributions of the final state particles.

        Parameters
        ----------
        q: float
            Center-of-mass energy in MeV.
        method: str
            Method used to generate energy distributions. Can be 'quad' or
            'rambo'. Default is 'quad'.
        nbins: int
            Number of bins for the distributions.
        npts: int
            Number of phase-space points used to generate distributions. Only
            used if method is 'rambo'. Default is 10_000.

        Returns
        -------
        dist1, dist2, dist3: (array, array)
            Three tuples containing the probabilities and energies for each
            final state particle.
        """

        def integrand(s, t):
            return self.__phase_space_integrand(q, s, t, **kwargs)

        return ThreeBody(q, self.fsp_masses, msqrd=integrand).energy_distributions(
            nbins=nbins, npts=npts, method=method
        )

    def _invariant_mass_distributions(
        self, *, q: float, method: str, nbins: int, npts: int = 10000, **kwargs
    ):
        r"""Compute the invariant-mass distributions of the final state
        particles.

        Parameters
        ----------
        q: float
            Center-of-mass energy in MeV.
        method: str
            Method used to generate energy distributions. Can be 'quad' or
            'rambo'. Default is 'quad'.
        nbins: int
            Number of bins for the distributions.
        npts: int
            Number of phase-space points used to generate distributions. Only
            used if method is 'rambo'. Default is 10_000.

        Returns
        -------
        dist1, dist2, dist3: (array, array)
            Three tuples containing the probabilities and invariant massses for
            each final state particle.

        Other Parameters
        ----------------
        """

        def integrand(s, t):
            return self.__phase_space_integrand(q, s, t, **kwargs)

        return ThreeBody(
            q, self.fsp_masses, msqrd=integrand
        ).invariant_mass_distributions(nbins=nbins, npts=npts, method=method)

    def _width(self, *, mv, method: str = "quad", npts: int = 10000, **kwargs):
        """Compute the partial width of a vector decay."""
        return super()._width(mv=mv, method=method, npts=npts, **kwargs)

    def _cross_section(
        self, *, q, mx, mv, gvxx, wv, method="quad", npts: int = 10000, **kwargs
    ):
        """Compute the cross-section of dark matter annihilation."""
        return super()._cross_section(
            q=q, mx=mx, mv=mv, gvxx=gvxx, wv=wv, method=method, npts=npts, **kwargs
        )


# =============================================================================
# ---- Special Case: form factor independent of t -----------------------------
# =============================================================================


@dataclass
class VectorFormFactorPPP2(VectorFormFactorThreeBody):
    r"""Abstract base class for three pseudo-scalar meson form factors for the
    special case where the form factor depends only on one of invariant masses.

    The three pseudo-scalar meson currents are given by:

    .. math::

        J_{\mu} = \epsilon_{\mu\nu\alpha\beta}
        p^{\nu}_{1}p^{\alpha}_{2}p^{\beta}_{3}
        F_{P_1P_2P_3}(q^2, s)

    where :math:`F_{P_1P_2P_3}(q^2, s)` is the form factor,

    .. math::

        q^{\mu} &= p^{\mu}_{1} + p^{\mu}_{2} + p^{\mu}_{3}\\

        s &= (p_{2} + p_{3})^2\\

    and :math:`p_{2},p_{3}` are the momenta of particles 2 and 3.
    """

    fsp_masses: Tuple[float, float, float]

    @abc.abstractmethod
    def form_factor(self, q, s, **kwargs):
        """Compute the squared matrix element."""
        raise NotImplementedError()

    def __phase_space_integrand(self, q, s, t, **kwargs):
        r"""Compute the integrand of the three-body phase-space.

        Parameters
        ----------
        q: float or array-like
            Center-of-mass energy.
        s: float or array-like
            Invariant mass of particles 2 and 3: s = (p2 + p3)^2
        t: float or array-like
            Invariant mass of particles 1 and 3: t = (p1 + p3)^2
        kwargs: dict
            Keyword arguments to pass to derived class's `form_factor` function.

        Returns
        -------
        msqrd: float or array-like
            Squared matrix element.
        """
        ff = self.form_factor(q, s, **kwargs)
        lor = self.squared_lorentz_structure(q, s, t)
        return np.abs(ff) ** 2 * lor

    def squared_lorentz_structure(self, q, s, t):
        r"""Compute the squared Lorentz structure of a three pseudo-scalar meson
        final-state.

        Parameters
        ----------
        q: float or array-like
            Center-of-mass energy.
        s: float or array-like
            Invariant mass of particles 2 and 3: s = (p2 + p3)^2
        t: float or array-like
            Invariant mass of particles 1 and 3: s = (p1 + p3)^2

        Returns
        -------
        lorentz: float or array-like
            Lorentz structure. If both `q` has shape `(n,)` and `s` or `t` has shape
            `(m,)`, then the result has shape `(n,m)`. If `q` is a scalar, then the
            shape is equal to the shape of `s` or `t`. If `s` and `t` are scalars,
            then the result has shape equal to the shape of `q`.

        Notes
        -----
        The current is defined as:
            J[mu] = eps[mu,nu1,nu2,nu3] p1[nu1] p2[nu2] p3[nu3] FF[p1,p2,p3]
        where:
            `eps`: the Levi-Civita tensor,
            `p1`: momentum of 1st scalar,
            `p2`: momentum of 2nd scalar,
            `p3`: momentum of 3rd scalar,
            `FF`: form-factor.
        The returned values is the square of this expression excluding the
        form-factor and contracting with -g[mu1,mu2]/(3q^2).
        """
        m1, m2, m3 = self.fsp_masses
        return _squared_lorentz_structure_ppp(q, s, t, m1, m2, m3)

    def _partially_integrated_form_factor(self, q, s, **kwargs):
        r"""Computes the form-factor integrated over one of the invariant
        masses, namely, t = (p1 + p3)^2.

        Since the form factor only depends on energy and s=(p2+p3), we can
        integrate over t=(p1+p3) without knowing what the form factor is.
        The phase space integral requires two integrations: one over t=(p1 +
        p3)^2 and another over s=(p2+p3)^2. This function integrates over t,
        leaving dJ/ds. The final integration can be performed by integrating s
        from (m2+m3)^2 to (q - m1)^2.
        """
        m1, m2, m3 = self.fsp_masses
        pre = 1.0 / (16.0 * (2 * np.pi) ** 3 * q**2)

        p1 = kallen_lambda(s, q**2, m1**2)
        p2 = kallen_lambda(s, m2**2, m3**2)
        p12 = np.sqrt(p1 * p2)

        ff = self.form_factor(q, s, **kwargs)

        return pre * np.abs(ff) ** 2 * p12**3 / (72.0 * q**2 * s**2)

    def _integrated_form_factor(
        self,
        *,
        q: Union[float, RealArray],
        **kwargs,
    ) -> Union[float, RealArray]:
        """Compute the integrated from factor for a three pseudo-scalar meson
        final-state.

        Parameters
        ----------
        q: float or array-like
            Center of mass energy.
        kwargs: dict
            Keyword arguments to pass to underlying `form_factor`.

        Returns
        -------
        ff: float or array-like
            Integrated form-factor.
        """

        m1, m2, m3 = self.fsp_masses

        def integrand(x, q_):
            s = q_**2 * (1.0 - x + (m1 / q_) ** 2)
            return self._partially_integrated_form_factor(q=q_, s=s, **kwargs)

        def integrate_(q_):
            mu1, mu2, mu3 = m1 / q_, m2 / q_, m3 / q_
            xmin = 2.0 * mu1
            xmax = mu1**2 + (1.0 - (mu2 + mu3) ** 2)
            return q_**2 * integrate.quad(lambda x: integrand(x, q_), xmin, xmax)[0]

        scalar = np.isscalar(q)
        qs = np.atleast_1d(q)
        mask = qs > m1 + m2 + m3
        res = np.zeros_like(qs)

        if np.any(mask):
            res[mask] = np.array([integrate_(q_) for q_ in qs[mask]])

        if scalar:
            return res[0]
        return res

    def _energy_distributions(self, *, q: float, nbins: int, **kwargs):
        r"""Compute the energy distributions of the final state particles.

        Parameters
        ----------
        q: float
            Center-of-mass energy in MeV.
        nbins: int
            Number of bins for the distributions.
        kwargs: dict
            Keyword arguments to pass to underlying `form_factor`.

        Returns
        -------
        dist1, dist2, dist3: (array, array)
            Three tuples containing the probabilities and energies for each
            final state particle.
        """
        # TODO: Fix this, we can get an analytic result of dn/de1, figure out
        # what to do for e2 and e3.

        def integrand(s, t):
            return self.__phase_space_integrand(q, s, t, **kwargs)

        return ThreeBody(q, self.fsp_masses, msqrd=integrand).energy_distributions(
            nbins=nbins, method="quad"
        )

    def _invariant_mass_distributions(self, *, q: float, nbins: int, **kwargs):
        r"""Compute the invariant-mass distributions of the final state
        particles.

        Parameters
        ----------
        q: float
            Center-of-mass energy in MeV.
        nbins: int
            Number of bins for the distributions.
        kwargs: dict
            Keyword arguments to pass to underlying `form_factor`.

        Returns
        -------
        dist1, dist2, dist3: (array, array)
            Three tuples containing the probabilities and invariant massses for
            each final state particle.

        Other Parameters
        ----------------
        """
        # TODO: Fix this, we can get an analytic result of dn/ds, figure out
        # what to do for t and u.
        def integrand(s, t):
            return self.__phase_space_integrand(q, s, t, **kwargs)

        return ThreeBody(
            q, self.fsp_masses, msqrd=integrand
        ).invariant_mass_distributions(nbins=nbins, method="quad")


# =============================================================================
# ---- PPV --------------------------------------------------------------------
# =============================================================================


@dataclass
class VectorFormFactorPPV(VectorFormFactorThreeBody):
    r"""Abstract base class for two pseudo-scalar meson + vector meson form factors.

    The currents are given by:

    .. math::

        J_{\mu} = \left(g_{\mu\nu} - q_{\mu}q_{\nu}/q^2\right)\epsilon^{\nu}(p_{V})
        p^{\nu}_{1}p^{\alpha}_{2}p^{\beta}_{3}
        F_{P_1P_2P_3}(q^2, s, t)

    where :math:`F_{P_1P_2P_3}(q^2, s, t)` is the form factor,

    .. math::

        q^{\mu} &= p^{\mu}_{1} + p^{\mu}_{2} + p^{\mu}_{3}\\

        s &= (p_{2} + p_{3})^2\\

        t &= (p_{1} + p_{3})^2

    and :math:`p_{1},p_{2},p_{3}` are the momenta.
    """

    fsp_masses: Tuple[float, float, float]

    @abc.abstractmethod
    def form_factor(self, q, **kwargs):
        raise NotImplementedError()

    def __squared_lorentz_structure(self, q, s, t):
        m1, m2, mv = self.fsp_masses

        return (2 * mv**2 * q**2 + (-(m1**2) - m2**2 + s + t) ** 2 / 4.0) / (
            3.0 * mv**2 * q**4
        )

    def squared_lorentz_structure(self, q, s, t):
        r"""Compute the Lorentz structure of a three pseudo-scalar meson
        final-state.

        Parameters
        ----------
        q: float or array-like
            Center-of-mass energy.
        s: float or array-like
            Invariant mass of particles 2 and 3: s = (p2 + p3)^2
        t: float or array-like
            Invariant mass of particles 1 and 3: s = (p1 + p3)^2

        Returns
        -------
        lorentz: float or array-like
            Lorentz structure. If both `q` has shape `(n,)` and `s` or `t` has shape
            `(m,)`, then the result has shape `(n,m)`. If `q` is a scalar, then the
            shape is equal to the shape of `s` or `t`. If `s` and `t` are scalars,
            then the result has shape equal to the shape of `q`.

        Notes
        -----
        The current is defined as:
            J[mu] = (g[mu,nu] - q[mu] q[nu] / q.q) polvec[nu,pv,mv] FF[p1,p2,p3]
        where:
            `p1`: momentum of 1st scalar,
            `p2`: momentum of 2nd scalar,
            `pv`: momentum of vector,
            `polvec`: polarization vector of vector,
            `FF`: form-factor.
        The returned values is the square of this expression excluding the
        form-factor and contracting with -g[mu1,mu2]/(3q^2).
        """
        scalar_q = np.isscalar(q)
        scalar_s = np.isscalar(s)
        scalar_t = np.isscalar(t)

        if scalar_q and scalar_s and scalar_t:
            return self.__squared_lorentz_structure(q, s, t)

        # q is in first dimension, s/t are in second.
        qq = np.expand_dims(np.atleast_1d(q).astype(np.float64), -1)
        ss = np.expand_dims(np.atleast_1d(s).astype(np.float64), 0)
        tt = np.expand_dims(np.atleast_1d(t).astype(np.float64), 0)

        res = self.__squared_lorentz_structure(qq, ss, tt)

        if scalar_q:
            return res[0]
        elif scalar_t and scalar_s:
            return res[:, 0]

        return res

    def _partially_integrated_form_factor(self, q, s, **kwargs):
        r"""Computes the form-factor integrated over one of the invariant
        masses, namely, t = (pv + p2)^2.

        Since the form factor only depends on energy, we can integrate over the
        phase-space without knowing what the form factor is. The phase space
        integral requires two integrations: one over t=(pv + p2)^2 and another
        over s=(p1+p2)^2, where pv is the momentum of the vector and p1,p2 are
        the momenta of the scalars. This function integrates over t, leaving
        dJ/ds. The final integration can be performed by integrating s from
        (m1+m2)^2 to (q - mv)^2.
        """
        mv, m1, m2 = self.fsp_masses
        muv, mu1, mu2 = mv / q, m1 / q, m2 / q

        x = 1.0 - s / q**2 + muv**2
        p1 = kallen_lambda(1.0, muv**2, 1 - x + muv**2)
        p2 = kallen_lambda(mu1**2, mu2**2, 1 - x + muv**2)

        num = q**2 * (x**2 + 8 * muv**2) * np.sqrt(np.clip(p1 * p2, 0.0, None))
        den = 1536.0 * np.pi**3 * mv**2 * s

        ff = self.form_factor(q, **kwargs)

        return num / den * np.abs(ff) ** 2

    def _integrated_form_factor(
        self,
        *,
        q: Union[float, RealArray],
        **kwargs,
    ) -> Union[float, RealArray]:
        """Compute the integrated from factor for a three pseudo-scalar meson
        final-state.

        Parameters
        ----------
        q: float or array-like
            Center of mass energy.

        Returns
        -------
        ff: float or array-like
            Integrated form-factor.
        """
        mv, m1, m2 = self.fsp_masses

        def integrand(x, q_):
            mv, m1, m2 = self.fsp_masses
            muv, mu1, mu2 = mv / q_, m1 / q_, m2 / q_

            xx = 1.0 - x + muv**2
            p1 = kallen_lambda(1.0, muv**2, xx)
            p2 = kallen_lambda(mu1**2, mu2**2, xx)

            return (x**2 + 8 * muv**2) * np.sqrt(p1 * p2) / xx

        def integrate_(q_):
            muv, mu1, mu2 = mv / q_, m1 / q_, m2 / q_
            xmin = 2.0 * muv
            xmax = muv**2 + (1.0 - (mu1 + mu2) ** 2)
            ff = self.form_factor(q_, **kwargs)
            pre = np.abs(ff) ** 2 / (1536.0 * np.pi**3 * muv**2)
            return pre * integrate.quad(lambda x: integrand(x, q_), xmin, xmax)[0]

        scalar = np.isscalar(q)
        qs = np.atleast_1d(q)
        mask = qs > mv + m1 + m2
        res = np.zeros_like(qs)

        if np.any(mask):
            res[mask] = np.array([integrate_(q_) for q_ in qs[mask]])

        if scalar:
            return res[0]
        return res

    def _energy_distributions(
        self, *, q: float, nbins: int, **kwargs
    ) -> List[PhaseSpaceDistribution1D]:
        r"""Compute the energy distributions of the final state particles.

        Parameters
        ----------
        q: float
            Center-of-mass energy in MeV.
        nbins: int
            Number of bins for the distributions.
        kwargs: dict
            Keyword arguments to pass to underlying `form_factor` function.

        Returns
        -------
        dist1, dist2, dist3: (array, array)
            Three tuples containing the probabilities and energies for each
            final state particle.
        """
        mv, m1, m2 = self.fsp_masses
        muv, mu1, mu2 = mv / q, m1 / q, m2 / q

        ff = self.form_factor(q, **kwargs)
        pre = np.abs(ff) ** 2 / (1536.0 * np.pi**3 * muv**2)

        def djde1(e):
            x = 2 * e / q
            xx = 1.0 - x + muv**2
            p1 = kallen_lambda(1.0, muv**2, xx)
            p2 = kallen_lambda(mu1**2, mu2**2, xx)
            return pre * (x**2 + 8 * muv**2) * np.sqrt(p1 * p2) / xx

        def djde2(e):
            x = 2 * e / q
            xx = 1.0 - x + mu1**2
            p1 = kallen_lambda(1.0, mu1**2, xx)
            p2 = kallen_lambda(mu2**2, muv**2, xx)
            p12 = p1 * p2
            return (
                np.sqrt(p12)
                * (
                    96.0 * (1.0 - x + mu1**2) ** 2 * muv**2
                    + 3.0
                    * (-2.0 + x) ** 2
                    * (1.0 - x + mu1**2 - mu2**2 + muv**2) ** 2
                    + p12
                )
            ) / (18432.0 * np.pi**3 * xx**3 * muv**2)

        def djde3(e):
            x = 2 * e / q
            xx = 1.0 - x + mu2**2
            p1 = kallen_lambda(1.0, mu2**2, xx)
            p2 = kallen_lambda(mu1**2, muv**2, xx)
            p12 = p1 * p2

            return (
                np.sqrt(p12)
                * (
                    96.0 * (1.0 - x + mu2**2) ** 2 * muv**2
                    + 3.0
                    * (-2.0 + x) ** 2
                    * (1.0 - x - mu1**2 + mu2**2 + muv**2) ** 2
                    + p12
                )
            ) / (1.0 - x + mu2**2) ** 3

        elims = energy_limits(q, self.fsp_masses)
        ebins = [np.linspace(emin, emax, nbins + 1) for emin, emax in elims]
        ecs = [0.5 * (ebs[1:] + ebs[:-1]) for ebs in ebins]
        dists = [
            PhaseSpaceDistribution1D(ebins[0], djde1(ecs[0])),
            PhaseSpaceDistribution1D(ebins[1], djde2(ecs[1])),
            PhaseSpaceDistribution1D(ebins[2], djde3(ecs[2])),
        ]

        return dists

    def _invariant_mass_distributions(
        self, *, q: float, nbins: int, **kwargs
    ) -> Dict[Tuple[int, int], PhaseSpaceDistribution1D]:
        r"""Compute the invariant-mass distributions of the final state
        particles.

        Parameters
        ----------
        q: float
            Center-of-mass energy in MeV.
        nbins: int
            Number of bins for the distributions.
        kwargs: dict
            Keyword arguments to pass to underlying `form_factor` function.

        Returns
        -------
        dist1, dist2, dist3: (array, array)
            Three tuples containing the probabilities and invariant massses for
            each final state particle.
        """
        edists = self._energy_distributions(q=q, nbins=nbins, **kwargs)
        # Convert to inv mass:
        # s = q^2 (1-2*E1/q + m1^2/q^2), de/ds = 1/2q
        # t = q^2 (1-2*E2/q + m2^2/q^2)
        # u = q^2 (1-2*E3/q + m3^2/q^2)

        def etosqrts(e, m):
            return q * np.sqrt(1 - 2 * e / q + (m / q) ** 2)

        keys = [(1, 2), (0, 2), (0, 1)]
        sqrts = [
            etosqrts(edists[i].bins, self.fsp_masses[i]) for i in range(len(edists))
        ]
        invdists = {
            keys[i]: PhaseSpaceDistribution1D(
                sqrts[i],
                0.25 * (sqrts[i][1:] + sqrts[i][:-1]) / q * edists[i].probabilities,
            )
            for i in range(len(edists))
        }

        return invdists
