"""
Module for computing the V-K-K form factor.
"""


from dataclasses import InitVar, dataclass, field
from typing import NamedTuple, Optional, Tuple, Union, overload

import numpy as np
from scipy.special import gamma

from hazma.utils import RealOrRealArray


from ._base import vector_couplings_to_isospin
from ._two_body import VectorFormFactorPP, Couplings

from ._utils import (
    MK0_GEV,
    MK_GEV,
    MPI_GEV,
    ComplexArray,
    RealArray,
    breit_wigner_fw,
    breit_wigner_gs,
    breit_wigner_pwave,
    dhhatds,
    gamma_generator,
    h,
    hhat,
)


class VectorMesonParameters(NamedTuple):
    """Parameters of a vector meson"""

    mass: RealArray
    width: RealArray
    coup: ComplexArray


# rho parameters
RHO_MAG = np.array(
    [
        1.1148916618504967,
        -0.050374779737077324,
        -0.014908906283692132,
        -0.03902475997619905,
        -0.038341465215871416,
    ]
)
RHO_PHASE = np.zeros_like(RHO_MAG)
RHO_MASSES = np.array(
    [0.77549, 1.5206995754050117, 1.7409719246639341, 1.9922811314327789]
)
RHO_WIDTHS = np.array(
    [0.1494, 0.21341728317817743, 0.08412224414791908, 0.2899733272437917]
)
# omega parameters
OMEGA_MAG = np.array(
    [
        1.3653229680598022,
        -0.02775156567495144,
        -0.32497165559032715,
        1.3993153161869765,
    ]
)
OMEGA_PHASE = np.zeros_like(OMEGA_MAG)
OMEGA_MASSES = np.array([0.78265, 1.4144344268685891, 1.655375231284883])
OMEGA_WIDTHS = np.array([8.49e-3, 0.0854413887755723, 0.16031760444832305])
# phi parameters
PHI_MAG = np.array(
    [
        0.965842498579515,
        -0.002379766320723148,
        -0.1956211640216197,
        0.16527771485190898,
    ]
)
PHI_PHASE = np.zeros_like(PHI_MAG)
PHI_MASSES = np.array([1.0194209171596993, 1.594759278457624, 2.156971341201067])
PHI_WIDTHS = np.array([4.252653332329334e-3, 0.028741821847408196, 0.6737556174184005])


def _find_beta(c0: float):
    """
    Find the beta parameter.
    """

    def c_0(b0: float):
        ratio = 2.0 / np.sqrt(np.pi)
        b1 = b0
        while b1 > 2.0:
            ratio *= (b1 - 1.5) / (b1 - 2.0)
            b1 -= 1.0
        ratio *= gamma(b1 - 0.5) / gamma(b1 - 1.0)
        return ratio

    betamin = 1.0
    betamid = 5.0
    betamax = 10.0
    eps = 1e-10
    cmid = c_0(betamid)
    while abs(cmid - c0) > eps:
        cmin = c_0(betamin)
        cmax = c_0(betamax)
        cmid = c_0(betamid)
        if cmin < c0 < cmid:
            betamax = betamid
        elif cmid < c0 < cmax:
            betamin = betamid
        elif c0 >= cmax:
            betamax *= 2.0
        else:
            print(f"bisect fails: {betamin}, {betamid}, {betamax}, {c0}")
            print(f"bisect fails: {cmin}, {cmid}, {cmax}, {c0}")
            raise RuntimeError("Failed to find beta.")
        betamid = 0.5 * (betamin + betamax)
    return betamid


def _compute_masses_width_couplings(  # pylint: disable=too-many-arguments
    masses: RealArray,
    widths: RealArray,
    magnitudes: RealArray,
    phases: RealArray,
    n_max: int,
    gam: Optional[float] = None,
) -> VectorMesonParameters:
    # Get beta
    beta = _find_beta(magnitudes[0])

    # weights
    wgt = magnitudes * np.exp(1j * phases)

    # compute the couplings
    ixs = np.arange(n_max)
    gam_b = np.array(list(gamma_generator(beta, n_max)))
    gam_0 = gamma(beta - 0.5)

    coup = (
        gam_0
        / (0.5 + ixs)
        / np.sqrt(np.pi)
        * np.sin(np.pi * (beta - 1.0 - ixs))
        / np.pi
        * gam_b
        + 0j
    )
    coup[1::2] *= -1

    # set up the masses and widths of the resonances
    mass = masses[0] * np.sqrt(1.0 + 2.0 * ixs)
    mass[: len(masses)] = masses

    width = mass * (widths[0] / masses[0] if gam is None else gam)
    width[: len(widths)] = widths

    # reset the parameters for the low lying resonances
    # set the masses and widths
    # couplings
    coup[: len(wgt)] = wgt
    total = np.sum(coup)
    coup[len(wgt)] += 1.0 - total

    return VectorMesonParameters(mass, width, coup)


def _compute_h_parameters(
    mass: RealArray, width: RealArray, m: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Hhat, H(0) and the derivative of Hhat needed to compute the
    Gounaris-Sakurai Breit-Wigner function with pion loop corrections included.
    See ArXiv:1002.0279 Eqn.(2) for details.

    Parameters
    ----------
    mass: np.ndarray
        Masses of the resonances.
    width: np.ndarray
        Widths of the resonances.
    m: float
    """
    hres = np.array(hhat(mass**2, mass, width, m, m))
    dh = np.array(dhhatds(mass, width, m, m))
    h0 = np.array(h(0.0, mass, width, m, m, dh, hres))
    return hres, dh, h0


@dataclass
class VectorFormFactorKKFitData:  # pylint: disable=too-many-instance-attributes
    """
    Class for storing the parameters needed to compute the form factor for
    V-K-K.
    """

    rho_mass: RealArray = field(init=False, repr=False)
    rho_width: RealArray = field(init=False, repr=False)
    rho_coup: ComplexArray = field(init=False, repr=False)

    hres: RealArray = field(init=False, repr=False)
    h0: RealArray = field(init=False, repr=False)
    dh: RealArray = field(init=False, repr=False)

    omega_mass: RealArray = field(init=False, repr=False)
    omega_width: RealArray = field(init=False, repr=False)
    omega_coup: ComplexArray = field(init=False, repr=False)

    phi_mass: RealArray = field(init=False, repr=False)
    phi_width: RealArray = field(init=False, repr=False)
    phi_coup: ComplexArray = field(init=False, repr=False)

    n_max: int = field(default=200)

    rho_mag: InitVar[Optional[RealArray]] = None
    rho_phase: InitVar[Optional[RealArray]] = None
    rho_masses: InitVar[Optional[RealArray]] = None
    rho_widths: InitVar[Optional[RealArray]] = None
    omega_mag: InitVar[Optional[RealArray]] = None
    omega_phase: InitVar[Optional[RealArray]] = None
    omega_masses: InitVar[Optional[RealArray]] = None
    omega_widths: InitVar[Optional[RealArray]] = None
    phi_mag: InitVar[Optional[RealArray]] = None
    phi_phase: InitVar[Optional[RealArray]] = None
    phi_masses: InitVar[Optional[RealArray]] = None
    phi_widths: InitVar[Optional[RealArray]] = None

    def __post_init__(  # pylint: disable=too-many-locals,too-many-arguments
        self,
        rho_mag: Optional[RealArray],
        rho_phase: Optional[RealArray],
        rho_masses: Optional[RealArray],
        rho_widths: Optional[RealArray],
        omega_mag: Optional[RealArray],
        omega_phase: Optional[RealArray],
        omega_masses: Optional[RealArray],
        omega_widths: Optional[RealArray],
        phi_mag: Optional[RealArray],
        phi_phase: Optional[RealArray],
        phi_masses: Optional[RealArray],
        phi_widths: Optional[RealArray],
    ) -> None:
        """
        Compute the parameters needed for computing the V-K-K form factor.

        Parameters
        ----------
        n_max: int
            Number of resonances to include.

        Returns
        -------
        params: FormFactorKKParameters
            Parameters of the resonances for the V-K-K form factor.
        """

        # initial parameters for the model
        # beta_rho = 2.1968
        # beta_omega = 2.6936
        # beta_phi = 1.9452
        gamma_omega = 0.5
        gamma_phi = 0.2

        if rho_mag is None:
            rho_mag = RHO_MAG
        if rho_phase is None:
            rho_phase = RHO_PHASE
        if rho_widths is None:
            rho_widths = RHO_WIDTHS
        if rho_masses is None:
            rho_masses = RHO_MASSES

        if omega_mag is None:
            omega_mag = OMEGA_MAG
        if omega_phase is None:
            omega_phase = OMEGA_PHASE
        if omega_widths is None:
            omega_widths = OMEGA_WIDTHS
        if omega_masses is None:
            omega_masses = OMEGA_MASSES

        if phi_mag is None:
            phi_mag = PHI_MAG
        if phi_phase is None:
            phi_phase = PHI_PHASE
        if phi_widths is None:
            phi_widths = PHI_WIDTHS
        if phi_masses is None:
            phi_masses = PHI_MASSES

        rho_params = _compute_masses_width_couplings(
            rho_masses,
            rho_widths,
            rho_mag,
            rho_phase,
            self.n_max,
        )
        self.rho_mass = rho_params.mass
        self.rho_width = rho_params.width
        self.rho_coup = rho_params.coup

        self.hres, self.dh, self.h0 = _compute_h_parameters(
            rho_params.mass,
            rho_params.width,
            MPI_GEV,
        )

        omega_params = _compute_masses_width_couplings(
            omega_masses,
            omega_widths,
            omega_mag,
            omega_phase,
            self.n_max,
            gam=gamma_omega,
        )
        self.omega_mass = omega_params.mass
        self.omega_width = omega_params.width
        self.omega_coup = omega_params.coup

        phi_params = _compute_masses_width_couplings(
            phi_masses,
            phi_widths,
            phi_mag,
            phi_phase,
            self.n_max,
            gam=gamma_phi,
        )

        self.phi_mass = phi_params.mass
        self.phi_width = phi_params.width
        self.phi_coup = phi_params.coup


@dataclass
class _VectorFormFactorKKBase(VectorFormFactorPP):
    """
    Class for storing the parameters needed to compute the form factor for
    V-K-K.
    """

    _imode: int = field(repr=False)
    fsp_masses: Tuple[float, float] = field(init=False)

    n_max: int = field(default=200)
    fit_data: VectorFormFactorKKFitData = field(init=False)

    def __post_init__(self) -> None:
        """
        Compute the parameters needed for computing the V-K-K form factor.

        Parameters
        ----------
        n_max: int
            Number of resonances to include.

        Returns
        -------
        params: FormFactorKKParameters
            Parameters of the resonances for the V-K-K form factor.
        """
        if self._imode == 0:
            self.fsp_masses = (MK0_GEV * 1e3, MK0_GEV * 1e3)
        else:
            self.fsp_masses = (MK_GEV * 1e3, MK_GEV * 1e3)

        self.fit_data = VectorFormFactorKKFitData(n_max=self.n_max)

    def __form_factor(self, s: RealArray, couplings: Couplings) -> ComplexArray:
        """
        Compute the form factor for the V-K-K interaction.

        Parameters
        ----------
        s: Union[float, np.ndarray]
            Center-of-mass energies squared.
        params: FormFactorKKParameters
            Parameters for computing the V-K-K form factor.
        gvuu: float
            Coupling of the vector to up-quarks.
        gvdd: float
            Coupling of the vector to down-quarks.
        gvss: float
            Coupling of the vector to strange-quarks.
        imode: int
            If imode = 0, the form factor is for V-K0-K0 and if imode = 1 then
            the form factor is for V-Kp-Km.

        Returns
        -------
        fk: Union[complex, np.ndarray]
            Form factor for V-K-K.
        """
        mk = self.fsp_masses[0] * 1e-3
        eta_phi = 1.055
        ci0, ci1, cs = vector_couplings_to_isospin(*couplings)

        # Force s into an array. This makes vectorization in the case where s is
        # an array easier.
        ss = np.array(s)

        # Rho exchange
        rho_pre = -0.5 if self._imode == 0 else 0.5
        fk = rho_pre * (
            ci1
            * self.fit_data.rho_coup
            * breit_wigner_gs(
                ss,
                self.fit_data.rho_mass,
                self.fit_data.rho_width,
                MPI_GEV,
                MPI_GEV,
                self.fit_data.h0,
                self.fit_data.dh,
                self.fit_data.hres,
                reshape=True,
            )
        )

        # Omega exchange
        fk += (
            1.0
            / 6.0
            * ci0
            * self.fit_data.omega_coup
            * breit_wigner_fw(
                ss,
                self.fit_data.omega_mass,
                self.fit_data.omega_width,
                reshape=True,
            )
        )

        # Phi-exchange
        bwp = breit_wigner_pwave(
            ss,
            self.fit_data.phi_mass,
            self.fit_data.phi_width,
            mk,
            mk,
            reshape=True,
        )

        phi_terms = 1.0 / 3.0 * cs * self.fit_data.phi_coup * bwp
        if self._imode == 0:
            phi_terms[0] *= eta_phi
        fk += phi_terms

        return np.sum(fk, axis=1)

    @overload
    def form_factor(  # pylint: disable=arguments-differ
        self, q: float, couplings: Couplings
    ) -> complex:
        ...

    @overload
    def form_factor(  # pylint: disable=arguments-differ
        self, q: RealArray, couplings: Couplings
    ) -> ComplexArray:
        ...

    def form_factor(  # pylint: disable=arguments-differ
        self, q: Union[float, RealArray], couplings: Couplings
    ) -> Union[complex, ComplexArray]:
        """Compute the kaon-kaon form factor.

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
        ff: float or array-like
            Form-factor.
        """

        single = np.isscalar(q)
        qq = np.atleast_1d(q).astype(np.float64) * 1e-3
        mask = qq > sum(self.fsp_masses) * 1e-3
        ff = np.zeros_like(qq, dtype=np.complex128)

        if np.any(mask):
            ff[mask] = self.__form_factor(s=qq[mask] ** 2, couplings=couplings)

        if single:
            return ff[0]
        return ff

    @overload
    def integrated_form_factor(  # pylint: disable=arguments-differ
        self, q: float, couplings: Couplings
    ) -> float:
        ...

    @overload
    def integrated_form_factor(  # pylint: disable=arguments-differ
        self, q: RealArray, couplings: Couplings
    ) -> RealArray:
        ...

    def integrated_form_factor(  # pylint: disable=arguments-differ
        self, q: RealOrRealArray, couplings: Couplings
    ) -> RealOrRealArray:
        r"""Compute kaon-kaon form-factor integrated over phase-space.

        Parameters
        ----------
        q: float
            Center-of-mass energy in MeV.
        gvuu, gvdd, gvss: float
            Coupling of vector to up-, down-, and strange-quarks.

        Returns
        -------
        iff: float
            Form-factor integrated over phase-space.
        """
        return self._integrated_form_factor(q=q, couplings=couplings)

    @overload
    def width(  # pylint: disable=arguments-differ
        self, mv: float, couplings: Couplings
    ) -> float:
        ...

    @overload
    def width(  # pylint: disable=arguments-differ
        self, mv: RealArray, couplings: Couplings
    ) -> RealArray:
        ...

    def width(  # pylint: disable=arguments-differ
        self, mv: RealOrRealArray, couplings: Couplings
    ) -> RealOrRealArray:
        r"""Compute the partial decay width of a massive vector into two kaons.

        Parameters
        ----------
        mv: float
            Mass of the vector.
        gvuu: float
            Coupling of vector to up-quarks.
        gvdd: float
            Coupling of vector to down-quarks.
        gvss: float
            Coupling of vector to strange-quarks.

        Returns
        -------
        width: float
            Decay width of vector into two kaons.
        """
        return self._width(mv=mv, couplings=couplings)

    @overload
    def cross_section(  # pylint: disable=arguments-differ,too-many-arguments
        self,
        q: float,
        mx: float,
        mv: float,
        gvxx: float,
        wv: float,
        couplings: Couplings,
    ) -> float:
        ...

    @overload
    def cross_section(  # pylint: disable=arguments-differ,too-many-arguments
        self,
        q: RealArray,
        mx: float,
        mv: float,
        gvxx: float,
        wv: float,
        couplings: Couplings,
    ) -> RealArray:
        ...

    def cross_section(  # pylint: disable=arguments-differ,too-many-arguments
        self,
        q: RealOrRealArray,
        mx: float,
        mv: float,
        gvxx: float,
        wv: float,
        couplings: Couplings,
    ) -> RealOrRealArray:
        r"""Compute the cross section for dark matter annihilating into two
        kaons.

        Parameters
        ----------
        q: float
            Center-of-mass energy.
        mx: float
            Mass of the dark matter in MeV.
        mv: float
            Mass of the vector mediator in MeV.
        gvxx: float
            Coupling of vector to dark matter.
        wv: float
            Width of the vector in MeV.
        gvuu: float
            Coupling of vector to up-quarks.
        gvdd: float
            Coupling of vector to down-quarks.
        gvss: float
            Coupling of vector to strange-quarks.

        Returns
        -------
        cs: float or array-like
            Annihilation cross section into two kaons.
        """
        return self._cross_section(
            q=q, mx=mx, mv=mv, gvxx=gvxx, wv=wv, couplings=couplings
        )


@dataclass
class VectorFormFactorK0K0(_VectorFormFactorKKBase):
    r"""Class for computing the form factor into two neutral kaons.

    Parameters
    ----------
    n_max: int
        Maximum number of resonances. Default is 2000.

    Attributes
    ----------
    fsp_masses: (float, float)
        Masses of the final-state particles.
    fit_data: VectorFormFactorKKFitData
        Stored data used to compute form-factor.

    Methods
    -------
    form_factor
        Compute the un-integrated form-factor into two neutral kaons.
    integrated_form_factor
        Compute the form-factor into two neutral kaons integrated over
        phase-space.
    width
        Compute the decay width of a vector into two neutral kaons.
    cross_section
        Compute the dark matter annihilation cross section into two neutral kaons.
    """

    _imode: int = field(init=False, repr=False, default=0)


@dataclass
class VectorFormFactorKK(_VectorFormFactorKKBase):
    r"""Class for computing the form factor into two charged kaons.

    Parameters
    ----------
    n_max: int
        Maximum number of resonances. Default is 2000.

    Attributes
    ----------
    fsp_masses: (float, float)
        Masses of the final-state particles.
    fit_data: VectorFormFactorKKFitData
        Stored data used to compute form-factor.

    Methods
    -------
    form_factor
        Compute the un-integrated form-factor into two charged kaons.
    integrated_form_factor
        Compute the form-factor into two charged kaons integrated over
        phase-space.
    width
        Compute the decay width of a vector into two charged kaons.
    cross_section
        Compute the dark matter annihilation cross section into two charged kaons.
    """

    _imode: int = field(init=False, repr=False, default=1)
