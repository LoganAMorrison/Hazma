"""Module implementing the pi-k-k form factors."""

import abc
from dataclasses import InitVar, dataclass, field
from typing import overload

import numpy as np

from hazma.phase_space import PhaseSpaceDistribution1D
from hazma.utils import ComplexArray, ComplexOrComplexArray, RealArray, RealOrRealArray

from . import _utils
from ._three_body import Couplings, VectorFormFactorPPP

MK = _utils.MK_GEV * 1e3
MK0 = _utils.MK0_GEV * 1e3
MPI = _utils.MPI_GEV * 1e3
MPI0 = _utils.MPI0_GEV * 1e3

KS_MASS_GEV = 0.8956  # KStar mass
KS_WIDTH_GEV = 0.047  # KStar width


ISO_SCALAR_MASSES = np.array([1019.461e-3, 1633.4e-3, 1957e-3])
ISO_SCALAR_WIDTHS = np.array([4.249e-3, 218e-3, 267e-3])
ISO_SCALAR_AMPS = np.array([0.0, 0.233, 0.0405])
ISO_SCALAR_PHASES = np.array([0, 1.1e-07, 5.19])  # * np.pi / 180.0


ISO_VECTOR_MASSES = np.array([775.26e-3, 1470e-3, 1720e-3])
ISO_VECTOR_WIDTHS = np.array([149.1e-3, 400e-3, 250e-3])
ISO_VECTOR_AMPS = np.array([-2.34, 0.594, -0.0179])
ISO_VECTOR_PHASES = np.array([0, 0.317, 2.57])  # * np.pi / 180.0


@dataclass
class VectorFormFactorPiKKFitData:  # pylint: disable=too-many-instance-attributes
    r"""Storage class for the fit-data needed to compute pion-kaon-kaon
    form-factors.

    Attributes
    ----------
    iso_scalar_masses: RealArray
        VMD iso-scalar resonance masses.
    iso_scalar_widths: RealArray
        VMD iso-scalar resonance widths.
    iso_scalar_amps: RealArray
        VMD iso-scalar resonance amplitudes.
    iso_scalar_phases: RealArray
        VMD iso-scalar resonance phases.
    iso_vector_masses: RealArray
        VMD iso-vector resonance masses.
    iso_vector_widths: RealArray
        VMD iso-vector resonance widths.
    iso_vector_amps: RealArray
        VMD iso-vector resonance amplitudes.
    iso_vector_phases: RealArray
        VMD iso-vector resonance phases.
    g_ks_k_pi: float
        Coupling of K^*, K and pion.
    """

    iso_scalar_masses: RealArray = field(repr=False)
    iso_scalar_widths: RealArray = field(repr=False)
    iso_scalar_amps: RealArray = field(repr=False)
    iso_scalar_phases: RealArray = field(repr=False)

    iso_vector_masses: RealArray = field(repr=False)
    iso_vector_widths: RealArray = field(repr=False)
    iso_vector_amps: RealArray = field(repr=False)
    iso_vector_phases: RealArray = field(repr=False)

    g_ks_k_pi: float = field(repr=False)

    def iso_scalar_amplitudes(self):
        """Compute the iso-scalar amplitudes with phases included."""
        return self.iso_scalar_amps * np.exp(1j * self.iso_scalar_phases)

    def iso_vector_amplitudes(self):
        """Compute the iso-vector amplitudes with phases included."""
        return self.iso_vector_amps * np.exp(1j * self.iso_vector_phases)


@dataclass
class _VectorFormFactorPiKKBase(VectorFormFactorPPP):
    r"""Abstract base class for the pion-kaon-kaon vector form-factors.

    Implements the common functions between the three different final states.
    """

    fsp_masses: tuple[float, float, float] = field(init=False)
    _fsp_masses: tuple[float, float, float] = field(init=False)
    fit_data: VectorFormFactorPiKKFitData = field(init=False)

    iso_scalar_masses: InitVar[RealArray] = field(default=ISO_SCALAR_MASSES)
    iso_scalar_widths: InitVar[RealArray] = field(default=ISO_SCALAR_WIDTHS)
    iso_scalar_amps: InitVar[RealArray] = field(default=ISO_SCALAR_AMPS)
    iso_scalar_phases: InitVar[RealArray] = field(default=ISO_SCALAR_PHASES)

    iso_vector_masses: InitVar[RealArray] = field(default=ISO_VECTOR_MASSES)
    iso_vector_widths: InitVar[RealArray] = field(default=ISO_VECTOR_WIDTHS)
    iso_vector_amps: InitVar[RealArray] = field(default=ISO_VECTOR_AMPS)
    iso_vector_phases: InitVar[RealArray] = field(default=ISO_VECTOR_PHASES)

    g_ks_k_pi: InitVar[float] = field(default=5.37392360229)

    def __post_init__(  # pylint: disable=too-many-arguments
        self,
        iso_scalar_masses: RealArray,
        iso_scalar_widths: RealArray,
        iso_scalar_amps: RealArray,
        iso_scalar_phases: RealArray,
        iso_vector_masses: RealArray,
        iso_vector_widths: RealArray,
        iso_vector_amps: RealArray,
        iso_vector_phases: RealArray,
        g_ks_k_pi: float,
    ):
        self._fsp_masses = tuple(m * 1e-3 for m in self.fsp_masses)
        self.fit_data = VectorFormFactorPiKKFitData(
            iso_scalar_masses=iso_scalar_masses,
            iso_scalar_widths=iso_scalar_widths,
            iso_scalar_amps=iso_scalar_amps,
            iso_scalar_phases=iso_scalar_phases,
            iso_vector_masses=iso_vector_masses,
            iso_vector_widths=iso_vector_widths,
            iso_vector_amps=iso_vector_amps,
            iso_vector_phases=iso_vector_phases,
            g_ks_k_pi=g_ks_k_pi,
        )

    @abc.abstractmethod
    def _form_factor(self, q, s, t, couplings: Couplings) -> float:
        raise NotImplementedError()

    def _iso_spin_amplitudes(
        self, m: RealOrRealArray, couplings: Couplings
    ) -> tuple[ComplexOrComplexArray, ComplexOrComplexArray]:
        r"""Compute the amplitude coefficients grouped in terms of iso-spin."""
        ci1 = couplings[0] - couplings[1]
        cs = -3 * couplings[2]
        s = m**2

        a0 = np.sum(
            cs
            * self.fit_data.iso_scalar_amplitudes()
            * _utils.breit_wigner_fw(
                s, self.fit_data.iso_scalar_masses, self.fit_data.iso_scalar_widths
            )
        )
        a1 = np.sum(
            ci1
            * self.fit_data.iso_vector_amplitudes()
            * _utils.breit_wigner_fw(
                s, self.fit_data.iso_vector_masses, self.fit_data.iso_vector_widths
            )
        )

        return (a0, a1)  # type: ignore

    @staticmethod
    def _kstar_propagator(s, m1, m2):
        r"""Returns the K^* energy-dependent propagator for a K^* transitioning
        into two other particles.

        Parameters
        ----------
        s: float or array-like
            Squared momentum of K^*.
        m1, m2: float
            Masses of the particles the K^* transitions into.
        """
        return (
            _utils.breit_wigner_pwave(s, KS_MASS_GEV, KS_WIDTH_GEV, m1, m2)
            / KS_MASS_GEV**2
        )

    @overload
    def form_factor(  # pylint: disable=arguments-differ
        self, q: float, s: float, t: float, couplings: Couplings
    ) -> complex: ...

    @overload
    def form_factor(  # pylint: disable=arguments-differ
        self, q: float, s: RealArray, t: RealArray, couplings: Couplings
    ) -> ComplexArray: ...

    def form_factor(  # pylint: disable=arguments-differ
        self, q, s: RealOrRealArray, t: RealOrRealArray, couplings: Couplings
    ) -> ComplexOrComplexArray:
        r"""Compute the vector form-factor for a pion and two kaons.

        Parameters
        ----------
        q: float or array-like
            Center-of-mass energy in MeV.
        s: float or array-like
            Squared invariant mass of the kaons s = (p2+p3)^2.
        t: float or array-like
            Squared invariant mass of the pion and last kaon t=(p1+p3)^2.
        gvuu, gvdd, gvss: float
            Couplings of vector to up-, down- and strange-quarks.

        Returns
        -------
        ff: complex or array-like
            The form-factor.
        """
        qq = 1e-3 * q
        ss = 1e-6 * s
        tt = 1e-6 * t
        ff = self._form_factor(qq, ss, tt, couplings)
        return ff * 1e-9

    @overload
    def integrated_form_factor(  # pylint: disable=arguments-differ,too-many-arguments
        self,
        q: float,
        couplings: Couplings,
        *,
        method: str = ...,
        npts: int = ...,
        epsrel: float = ...,
        epsabs: float = ...,
    ) -> float: ...

    @overload
    def integrated_form_factor(  # pylint: disable=arguments-differ,too-many-arguments
        self,
        q: RealArray,
        couplings: Couplings,
        *,
        method: str = ...,
        npts: int = ...,
        epsrel: float = ...,
        epsabs: float = ...,
    ) -> RealArray: ...

    def integrated_form_factor(  # pylint: disable=arguments-differ,too-many-arguments
        self,
        q: RealOrRealArray,
        couplings: Couplings,
        *,
        method: str = "quad",
        npts: int = 10000,
        epsrel: float = 1e-3,
        epsabs: float = 0.0,
    ) -> RealOrRealArray:
        """Compute the pion-kaon-kaon form factor integrated over phase-space.

        Parameters
        ----------
        q: float or array-like
            Center of mass energy.
        gvuu: float
            Coupling of vector to up-quarks.
        gvdd: float
            Coupling of vector to down-quarks.
        gvss: float
            Coupling of vector to strange-quarks.

        Returns
        -------
        iff: float or array-like
            Integrated form-factor.

        Other Parameters
        ----------------
        method: str, optional
            Method used to integrate. Default is 'quad'. Options are 'quad' or
            'rambo'.
        npts: int, optional
            Number of phase-space points to use in integration. Ignored is
            method isn't 'rambo'. Default is 10_000.
        epsrel: float, optional
            Relative error tolerance. Default is 1e-3.
        epsabs: float, optional
            Absolute error tolerance. Default is 0.0.
        """
        return self._integrated_form_factor(
            q=q,
            method=method,
            npts=npts,
            couplings=couplings,
            epsrel=epsrel,
            epsabs=epsabs,
        )

    @overload
    def width(  # pylint: disable=arguments-differ,too-many-arguments
        self,
        mv: float,
        couplings: Couplings,
        *,
        method: str = ...,
        npts: int = ...,
        epsrel: float = ...,
        epsabs: float = ...,
    ) -> float: ...

    @overload
    def width(  # pylint: disable=arguments-differ,too-many-arguments
        self,
        mv: RealArray,
        couplings: Couplings,
        *,
        method: str = ...,
        npts: int = ...,
        epsrel: float = ...,
        epsabs: float = ...,
    ) -> RealArray: ...

    def width(  # pylint: disable=arguments-differ,too-many-arguments
        self,
        mv: RealOrRealArray,
        couplings: Couplings,
        *,
        method: str = "quad",
        npts: int = 1 << 14,
        epsrel: float = 1e-3,
        epsabs: float = 0.0,
    ) -> RealOrRealArray:
        r"""Compute the partial decay width of a massive vector into a pion and
        two kaons.

        Parameters
        ----------
        mv: float or array-like
            Mass of the vector.
        gvuu, gvdd, gvss: float
            Coupling of vector to up-, donw- and strange-quarks.
        nbins: float
            Number of bins used to generate distribution.

        Returns
        -------
        width: float or array-like
            Decay width of vector into a pion and two kaons.

        Other Parameters
        ----------------
        method: str, optional
            Method used to integrate over phase-space.
            See `hazma.phase_space.energy_distributions_three_body`
            for availible methods.
        npts: int, optional
            Number of phase-space points to use in integration. Only used if
            `method='rambo'`. Default is 2^14.
        epsrel: float, optional
            Relative error tolerance. Default is 1e-3.
        epsabs: float, optional
            Absolute error tolerance. Default is 0.0.
        """
        return self._width(
            mv=mv,
            couplings=couplings,
            method=method,
            npts=npts,
            epsrel=epsrel,
            epsabs=epsabs,
        )

    @overload
    def cross_section(  # pylint: disable=arguments-differ,too-many-arguments
        self,
        q: float,
        mx: float,
        mv: float,
        gvxx: float,
        wv: float,
        couplings: Couplings,
        *,
        method: str = ...,
        npts: int = ...,
        epsrel: float = ...,
        epsabs: float = ...,
    ) -> float: ...

    @overload
    def cross_section(  # pylint: disable=arguments-differ,too-many-arguments
        self,
        q: RealArray,
        mx: float,
        mv: float,
        gvxx: float,
        wv: float,
        couplings: Couplings,
        *,
        method: str = ...,
        npts: int = ...,
        epsrel: float = ...,
        epsabs: float = ...,
    ) -> RealArray: ...

    def cross_section(  # pylint: disable=arguments-differ,too-many-arguments
        self,
        q: RealOrRealArray,
        mx: float,
        mv: float,
        gvxx: float,
        wv: float,
        couplings: Couplings,
        *,
        method: str = "quad",
        npts: int = 1 << 14,
        epsrel: float = 1e-3,
        epsabs: float = 0.0,
    ) -> RealOrRealArray:
        r"""Compute the cross section for dark matter annihilating into a pion
        and two kaons.

        Parameters
        ----------
        q: float or array-like
            Center-of-mass energy.
        mx: float
            Mass of the dark matter in MeV.
        mv: float
            Mass of the vector mediator in MeV.
        gvxx: float
            Coupling of vector to dark matter.
        wv: float
            Width of the vector in MeV.
        gvuu, gvdd, gvss: float
            Coupling of vector to up-, donw- and strange-quarks.

        Returns
        -------
        cs: float or array-like
            Annihilation cross section into a pion and two kaons.

        Other Parameters
        ----------------
        method: str, optional
            Method used to integrate over phase-space.
            See `hazma.phase_space.energy_distributions_three_body`
            for availible methods.
        npts: int, optional
            Number of phase-space points to use in integration. Only used if
            `method='rambo'`. Default is 2^14.
        epsrel: float, optional
            Relative error tolerance. Default is 1e-3.
        epsabs: float, optional
            Absolute error tolerance. Default is 0.0.
        """
        return self._cross_section(
            q=q,
            mx=mx,
            mv=mv,
            gvxx=gvxx,
            wv=wv,
            couplings=couplings,
            method=method,
            npts=npts,
            epsrel=epsrel,
            epsabs=epsabs,
        )

    def energy_distributions(  # pylint: disable=arguments-differ,too-many-arguments
        self,
        q: float,
        couplings: Couplings,
        nbins: int,
        method: str = "quad",
        npts: int = 1 << 14,
        epsrel: float = 1e-3,
        epsabs: float = 0.0,
    ) -> list[PhaseSpaceDistribution1D]:
        r"""Compute the energy distributions of the final state pion, and kaons.

        Parameters
        ----------
        q: float
            Center-of-mass energy.
        gvuu, gvdd, gvss: float
            Coupling of vector to up-, donw- and strange-quarks.
        nbins: float
            Number of bins used to generate distribution.

        Returns
        -------
        dists: List[PhaseSpaceDistribution1D]
            List of the energy distributions.

        Other Parameters
        ----------------
        method: str, optional
            Method used to integrate over phase-space.
            See `hazma.phase_space.energy_distributions_three_body`
            for availible methods.
        npts: int, optional
            Number of phase-space points to use in integration. Only used if
            `method='rambo'`.
        epsrel: float, optional
            Relative error tolerance. Default is 1e-3.
        epsabs: float, optional
            Absolute error tolerance. Default is 0.0.
        """
        return self._energy_distributions(
            q=q,
            nbins=nbins,
            couplings=couplings,
            method=method,
            npts=npts,
            epsrel=epsrel,
            epsabs=epsabs,
        )

    # pylint: disable=arguments-differ,too-many-arguments
    def invariant_mass_distributions(
        self,
        q: float,
        couplings: Couplings,
        nbins: int,
        method: str = "quad",
        npts: int = 1 << 14,
        epsrel: float = 1e-3,
        epsabs: float = 0.0,
    ) -> dict[tuple[int, int], PhaseSpaceDistribution1D]:
        r"""Compute the invariant-mass distributions of the all pairs of the
        final-state particles.

        Parameters
        ----------
        q: float
            Center-of-mass energy.
        gvuu, gvdd, gvss: float
            Coupling of vector to up-, donw- and strange-quarks.
        nbins: float
            Number of bins used to generate distribution.

        Returns
        -------
        dists: Dict[(int,int), PhaseSpaceDistribution1D]
            Dictionary of the invariant-mass distributions. Keys specify the
            pair of particles the distribution represents.

        Other Parameters
        ----------------
        method: str, optional
            Method used to integrate over phase-space.
            See `hazma.phase_space.energy_distributions_three_body`
            for availible methods.
        npts: int, optional
            Number of phase-space points to use in integration. Only used if
            `method='rambo'`.
        epsrel: float, optional
            Relative error tolerance. Default is 1e-3.
        epsabs: float, optional
            Absolute error tolerance. Default is 0.0.
        """
        return self._invariant_mass_distributions(
            q=q,
            nbins=nbins,
            couplings=couplings,
            method=method,
            npts=npts,
            epsrel=epsrel,
            epsabs=epsabs,
        )


@dataclass
class VectorFormFactorPi0K0K0(_VectorFormFactorPiKKBase):
    """
    Class for computing the vector form factor for a neutral pion and two
    neutral kaons.

    Attributes
    ----------
    fsp_masses: (float, float, float)
        Masses of the final-state particles.
    fit_data: VectorFormFactorPiKKFitData
        Stored data used to compute form-factor.

    Methods
    -------
    form_factor
        Compute the un-integrated form-factor into a neutral pion and two
        neutral kaons.
    integrated_form_factor
        Compute the form-factor into a neutral pion and two neutral kaons
        integrated over phase-space.
    width
        Compute the decay width of a vector into a neutral pion and two neutral
        kaons.
    cross_section
        Compute the dark matter annihilation cross section into a neutral pion
        and two neutral kaons.
    """

    fsp_masses: tuple[float, float, float] = field(init=False, default=(MK0, MK0, MPI0))

    def _form_factor(self, q, s, t, couplings: Couplings) -> float:
        a0, a1 = self._iso_spin_amplitudes(q, couplings)
        m1, m2, m3 = self._fsp_masses
        coeff = (a0 + a1) / np.sqrt(6.0) * 2 * self.fit_data.g_ks_k_pi
        return coeff * (
            self._kstar_propagator(s, m2, m3) + self._kstar_propagator(t, m1, m3)
        )


@dataclass
class VectorFormFactorPi0KpKm(_VectorFormFactorPiKKBase):
    """
    Class for computing the vector form factor for a neutral pion and two
    charged kaons.

    Attributes
    ----------
    fsp_masses: (float, float, float)
        Masses of the final-state particles.
    fit_data: VectorFormFactorPiKKFitData
        Stored data used to compute form-factor.

    Methods
    -------
    form_factor
        Compute the un-integrated form-factor into a neutral pion and two
        charged kaons.
    integrated_form_factor
        Compute the form-factor into a neutral pion and two charged kaons
        integrated over phase-space.
    width
        Compute the decay width of a vector into a neutral pion and two charged
        kaons.
    cross_section
        Compute the dark matter annihilation cross section into a neutral pion
        and two charged kaons.
    """

    fsp_masses: tuple[float, float, float] = field(init=False, default=(MK, MK, MPI0))

    def _form_factor(self, q, s, t, couplings) -> float:
        a0, a1 = self._iso_spin_amplitudes(q, couplings)
        m1, m2, m3 = self._fsp_masses
        coeff = (a0 - a1) / np.sqrt(6.0) * 2 * self.fit_data.g_ks_k_pi
        return coeff * (
            self._kstar_propagator(s, m2, m3) + self._kstar_propagator(t, m1, m3)
        )


@dataclass
class VectorFormFactorPiKK0(_VectorFormFactorPiKKBase):
    """
    Class for computing the vector form factor for a charged pion, a neutral
    kaon, and a charged kaon.

    Attributes
    ----------
    fsp_masses: (float, float, float)
        Masses of the final-state particles.
    fit_data: VectorFormFactorPiKKFitData
        Stored data used to compute form-factor.

    Methods
    -------
    form_factor
        Compute the un-integrated form-factor into a charged pion, a charged
        kaon and a neutral kaon.
    integrated_form_factor
        Compute the form-factor into a charged pion, a charged kaon, and a
        neutral kaon, integrated over phase-space.
    width
        Compute the decay width of a vector into a charged pion, a charged kaon,
        and a neutral kaon.
    cross_section
        Compute the dark matter annihilation cross section into a charged pion,
        a charged kaon, and a neutral kaon.
    """

    fsp_masses: tuple[float, float, float] = field(init=False, default=(MK0, MK, MPI))

    def _form_factor(self, q, s, t, couplings: Couplings) -> float:
        a0, a1 = self._iso_spin_amplitudes(q, couplings)
        m1, m2, m3 = self._fsp_masses
        cs = (a0 + a1) / np.sqrt(6.0) * 2 * self.fit_data.g_ks_k_pi
        ct = (a0 - a1) / np.sqrt(6.0) * 2 * self.fit_data.g_ks_k_pi
        return ct * self._kstar_propagator(s, m2, m3) + cs * self._kstar_propagator(
            t, m1, m3
        )
