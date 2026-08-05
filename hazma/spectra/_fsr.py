r"""Photon spectra from a user-supplied radiative squared matrix element.

This module implements the maintained replacement for the removed
``hazma.gamma_ray.gamma_ray_fsr`` (see ``docs/adrs/ADR-0001``): a
generator that turns the exact tree-level squared matrix elements of a
radiative process ``X -> f1 ... fn + gamma`` and its non-radiative
counterpart ``X -> f1 ... fn`` into the photon spectrum per
non-radiative event,

.. math::

    \frac{dN}{dE_\gamma}
    = \frac{d\Gamma(X \to F\gamma)/dE_\gamma}{\Gamma(X \to F)}
    = \frac{dI_{\mathrm{rad}}/dE_\gamma}{I_0},
    \qquad
    I[|M|^2] = \int |M|^2 \, d\Phi.

Every initial-state factor (flux or :math:`1/2M`, spin averaging,
couplings, propagators at fixed :math:`s`, and symmetry factors of the
non-photon final state) appears identically in numerator and
denominator and cancels in the ratio, so the generator needs only the
two squared matrix elements and the final-state masses.

At a fixed photon energy :math:`E_\gamma` the radiative phase space
factorizes into the photon and the non-photon system at reduced
invariant mass squared :math:`s' = s - 2\sqrt{s}E_\gamma`:

.. math::

    \frac{dI_{\mathrm{rad}}}{dE_\gamma}
    = \frac{E_\gamma}{4\pi^2}
      \left\langle \int |M_{\mathrm{rad}}|^2 \,
      d\Phi_n(s') \right\rangle_{\hat{k}},

which this module evaluates by adaptive quadrature over the single
remaining angle when ``n == 2`` and by RAMBO Monte-Carlo integration at
the reduced energy otherwise — the spectrum is computed exactly at the
requested energies, with no histogram binning.
"""

from collections.abc import Callable, Sequence
from typing import NamedTuple

import numpy as np
from scipy import integrate

from hazma.hazma_errors import RamboCMETooSmall
from hazma.phase_space import Rambo, ThreeBody
from hazma.utils import RealArray, RealOrRealArray, kallen_lambda

SquaredMatrixElement = Callable[[RealArray], RealOrRealArray]

# Adaptive-quadrature subdivision limit for the angular integrals. FSR
# integrands peak sharply where the photon is collinear with a light
# charged particle (the peak width in cos(theta) is ~ 2 m^2/E^2), so the
# scipy default of 50 subintervals is too small for large hierarchies.
_QUAD_LIMIT = 200

# Final-state multiplicities with a dedicated deterministic backend.
_TWO_BODY = 2
_THREE_BODY = 3

# The one-sigma Monte-Carlo error is a ddof=1 sample estimate — undefined
# for a single sample — so the rambo backend needs at least two points.
_MIN_MC_NPTS = 2


class FSRSpectrum(NamedTuple):
    r"""Photon spectrum with its integration-error estimate.

    Attributes
    ----------
    dnde: float or ndarray
        Photon spectrum dN/dE in MeV⁻¹, shaped like the input energies.
    error: float or ndarray
        One-sigma error estimate on ``dnde`` in MeV⁻¹: statistical for
        the Monte-Carlo backend, quadrature-error propagation for the
        deterministic backend.
    """

    dnde: RealOrRealArray
    error: RealOrRealArray


def _dphi2(s: float, m1: float, m2: float) -> float:
    r"""Total two-body phase-space volume \int d\Phi_2 = \sqrt{\lambda}/(8\pi s)."""
    lam = kallen_lambda(s, m1**2, m2**2)
    if lam <= 0.0:
        return 0.0
    return np.sqrt(lam) / (8.0 * np.pi * s)


def _split_quad(
    integrand: Callable[[float], float],
    epsabs: float | None,
    epsrel: float | None,
) -> tuple[float, float]:
    r"""Integrate over cos(theta) in [-1, 1], split at 0.

    The collinear peaks of FSR integrands sit against the endpoints
    cos(theta) -> ±1; splitting gives each QAGS call a single difficult
    endpoint region.
    """
    # Matrix-element magnitudes are arbitrary (common constants cancel in
    # the spectrum ratio), so an absolute tolerance is meaningless: the
    # scipy default epsabs=1.49e-8 would stop the subdivision early — or
    # report a uselessly conservative error — whenever the integrand
    # happens to be numerically small. Default to purely relative
    # convergence instead.
    lo, err_lo = integrate.quad(
        integrand,
        -1.0,
        0.0,
        epsabs=0.0 if epsabs is None else epsabs,
        epsrel=1.49e-8 if epsrel is None else epsrel,
        limit=_QUAD_LIMIT,
    )
    hi, err_hi = integrate.quad(
        integrand,
        0.0,
        1.0,
        epsabs=0.0 if epsabs is None else epsabs,
        epsrel=1.49e-8 if epsrel is None else epsrel,
        limit=_QUAD_LIMIT,
    )
    return lo + hi, err_lo + err_hi


def _two_body_momenta(cme: float, m1: float, m2: float, ct: float) -> RealArray:
    r"""Back-to-back momenta of (m1, m2) at angle theta to the z-axis.

    Constructed in the pair rest frame, with the momentum of particle 1
    in the x-z plane.
    """
    e1 = (cme**2 + m1**2 - m2**2) / (2.0 * cme)
    p = np.sqrt(max(e1**2 - m1**2, 0.0))
    st = np.sqrt(max(1.0 - ct**2, 0.0))
    return np.array(
        [
            [e1, cme - e1],
            [p * st, -p * st],
            [0.0, 0.0],
            [p * ct, -p * ct],
        ]
    )


def _nonrad_integral(  # noqa: PLR0913, PLR0917 — internal, mirrors the public signature
    cme: float,
    masses: RealArray,
    msqrd_nonrad: SquaredMatrixElement,
    npts: int,
    seed_seq: np.random.SeedSequence,
    epsabs: float | None,
    epsrel: float | None,
) -> tuple[float, float]:
    r"""Compute I_0 = \int |M_0|^2 d\Phi_n at the full center-of-mass energy.

    Deterministic for two- and three-body final states (angular
    quadrature and Dalitz double-quadrature respectively), RAMBO
    Monte-Carlo above that.
    """
    n = len(masses)

    if n == _TWO_BODY:
        m1, m2 = masses

        def integrand(ct: float) -> float:
            return float(msqrd_nonrad(_two_body_momenta(cme, m1, m2, ct)))

        # I_0 = dPhi_2 * <|M_0|^2>, with <.> = (1/2) Integral d cos(theta)
        angular, angular_err = _split_quad(integrand, epsabs, epsrel)
        pre = 0.5 * _dphi2(cme**2, m1, m2)
        return pre * angular, pre * angular_err

    if n == _THREE_BODY:
        three_body = ThreeBody(
            cme, tuple(masses), msqrd=msqrd_nonrad, msqrd_signature="momenta"
        )
        # epsabs=0: purely relative convergence, as in _split_quad.
        return three_body.integrate(
            method="quad",
            epsabs=0.0 if epsabs is None else epsabs,
            epsrel=1.49e-8 if epsrel is None else epsrel,
        )

    phase_space = Rambo(cme, masses, msqrd=msqrd_nonrad)
    return phase_space.integrate(n=npts, seed=seed_seq)  # type: ignore[arg-type]


def _dnde_point_quad(  # noqa: PLR0913, PLR0917 — internal, mirrors the public signature
    photon_energy: float,
    cme: float,
    masses: RealArray,
    msqrd: SquaredMatrixElement,
    epsabs: float | None,
    epsrel: float | None,
) -> tuple[float, float]:
    r"""dI_rad/dE at one photon energy for a two-body non-photon system.

    At fixed E the remaining integral is one-dimensional: the
    orientation of the back-to-back pair relative to the photon in the
    pair rest frame.
    """
    m1, m2 = masses
    sp = cme * (cme - 2.0 * photon_energy)

    if not (photon_energy > 0.0 and sp > (m1 + m2) ** 2):
        return 0.0, 0.0

    cme_rf = np.sqrt(sp)
    # Photon energy in the rest frame of the massive pair.
    eg_rf = photon_energy * cme / cme_rf

    def integrand(ct: float) -> float:
        pair = _two_body_momenta(cme_rf, m1, m2, ct)
        photon = np.array([eg_rf, 0.0, 0.0, eg_rf])
        momenta = np.concatenate((pair, photon[:, None]), axis=1)
        return float(msqrd(momenta))

    angular, angular_err = _split_quad(integrand, epsabs, epsrel)
    pre = photon_energy / (4.0 * np.pi**2) * _dphi2(sp, m1, m2) * 0.5
    return pre * angular, pre * angular_err


def _dnde_point_rambo(  # noqa: PLR0913, PLR0917 — internal, mirrors the public signature
    photon_energy: float,
    cme: float,
    masses: RealArray,
    msqrd: SquaredMatrixElement,
    npts: int,
    seed_seq: np.random.SeedSequence,
) -> tuple[float, float]:
    r"""dI_rad/dE at one photon energy via RAMBO at the reduced energy.

    The non-photon system is generated in its own rest frame at
    invariant mass sqrt(s') and the photon appended along the z-axis
    with the energy it has in that frame; because the RAMBO ensemble is
    isotropic this samples the exact fixed-E phase-space measure for
    any Lorentz-invariant integrand.
    """
    sp = cme * (cme - 2.0 * photon_energy)

    if not (photon_energy > 0.0 and sp > np.sum(masses) ** 2):
        return 0.0, 0.0

    cme_rf = np.sqrt(sp)
    eg_rf = photon_energy * cme / cme_rf

    phase_space = Rambo(cme_rf, masses)
    momenta, weights = phase_space.generate(npts, seed=seed_seq)  # type: ignore[arg-type]

    photon = np.zeros((4, 1, npts))
    photon[0] = eg_rf
    photon[3] = eg_rf
    momenta = np.concatenate((momenta, photon), axis=1)

    integrands = weights * np.asarray(msqrd(momenta))
    mean = np.nanmean(integrands)
    std = np.nanstd(integrands, ddof=1) / np.sqrt(npts)

    pre = photon_energy / (4.0 * np.pi**2)
    return pre * mean, pre * std


def dnde_photon_fsr(  # noqa: PLR0913 — the surface fixed by ADR-0001
    photon_energies: RealOrRealArray,
    cme: float,
    final_state_masses: Sequence[float],
    msqrd: SquaredMatrixElement,
    msqrd_nonrad: SquaredMatrixElement,
    *,
    method: str = "auto",
    npts: int = 1 << 14,
    seed: int | None = None,
    epsabs: float | None = None,
    epsrel: float | None = None,
) -> FSRSpectrum:
    r"""Photon spectrum of a radiative process from its squared matrix element.

    The spectrum, normalized per non-radiative event, is the ratio of
    bare phase-space integrals

    .. math::

        \frac{dN}{dE_\gamma} =
        \frac{d I_{\mathrm{rad}}/dE_\gamma}{I_0},
        \qquad I[|M|^2] = \int |M|^2 \, d\Phi,

    in which every initial-state prefactor (flux or :math:`1/2M`, spin
    averaging, couplings and propagators at fixed center-of-mass
    energy, and symmetry factors of the non-photon final state) cancels
    between numerator and denominator. Decays and annihilations are
    therefore the same call: for the decay ``X -> F + gamma`` pass the
    decaying particle's mass as `cme`, and for the annihilation
    ``x xbar -> F + gamma`` pass the center-of-mass energy.

    Parameters
    ----------
    photon_energies: float or array-like
        Photon energy (energies) in MeV where the spectrum is computed.
        Energies outside the open interval ``(0, e_max)`` with
        ``e_max = (cme**2 - sum(final_state_masses)**2) / (2 * cme)``
        yield zero.
    cme: float
        Center-of-mass energy of the process in MeV (the mass of the
        decaying particle for a decay). Must exceed the sum of the
        final-state masses.
    final_state_masses: sequence of float
        Masses in MeV of the final-state particles excluding the
        photon. At least two are required.
    msqrd: callable
        Spin-summed (and, for an annihilation, initial-spin-averaged
        and beam-direction-averaged) squared matrix element of the
        radiative process. Called as ``msqrd(momenta)`` with `momenta`
        of shape ``(4, len(final_state_masses) + 1[, batch])``: rows
        are ``(E, px, py, pz)`` in MeV, one column per particle in the
        order of `final_state_masses` with the photon last. It must be
        a Lorentz-invariant function of the final-state momenta only
        (implementations built from ``hazma.utils.ldot`` /
        ``lnorm_sqr`` handle the batched and unbatched forms for free);
        the momenta are supplied in the rest frame of the non-photon
        system, and no initial-state momenta are provided.
    msqrd_nonrad: callable
        Squared matrix element of the non-radiative process, in the
        same conventions as `msqrd`. Called with `momenta` of shape
        ``(4, len(final_state_masses)[, batch])``. Any multiplicative
        constant common to `msqrd` and `msqrd_nonrad` cancels.
    method: str, optional
        Integration backend: ``"quad"`` (deterministic; only available
        for a two-body non-photon final state, where the fixed-energy
        integral is one-dimensional), ``"rambo"`` (Monte-Carlo at the
        reduced invariant mass, any multiplicity), or ``"auto"`` (the
        default: ``"quad"`` when possible, else ``"rambo"``).
    npts: int, optional
        Monte-Carlo phase-space points per photon energy (and for the
        non-radiative integral when four or more final-state particles
        make it Monte-Carlo too). Must be an integer of at least 2 —
        the returned one-sigma error is a ``ddof=1`` sample estimate,
        undefined for a single sample. Ignored by the quadrature
        backend. Default is ``2**14``.
    seed: int, optional
        Seed for the Monte-Carlo backend. Each photon energy draws an
        independent substream, so results are deterministic for a fixed
        seed and grid. Ignored by the quadrature backend.
    epsabs, epsrel: float, optional
        Absolute and relative tolerances of the quadrature backend (and
        of the deterministic non-radiative integral). Defaults are
        ``epsabs=0`` — purely relative convergence, since the magnitude
        of a squared matrix element is convention-dependent — and the
        scipy default ``epsrel=1.49e-8``.

    Returns
    -------
    spectrum: FSRSpectrum
        NamedTuple ``(dnde, error)``, each a float for scalar input or
        an ndarray shaped like `photon_energies`. ``dnde`` is the
        photon spectrum dN/dE in MeV⁻¹ per non-radiative event;
        ``error`` is its one-sigma integration-error estimate. The
        Monte-Carlo error estimate assumes a finite-variance integrand;
        it degrades for strongly collinear-peaked matrix elements
        (light charged particles at large ``cme``), for which the
        quadrature backend is preferred.

    Raises
    ------
    RamboCMETooSmall
        If `cme` is below the sum of `final_state_masses`.
    ValueError
        If fewer than two final-state masses are given, if `method` is
        unknown or inapplicable, if the Monte-Carlo backend is selected
        with a non-integral `npts` or ``npts < 2``, or if the
        non-radiative integral is not positive.

    Notes
    -----
    The photon spectrum diverges as :math:`1/E_\gamma` for
    :math:`E_\gamma \to 0` (soft divergence); the returned values are
    finite at every requested energy, but the total photon number is
    not.

    For internal consistency the non-radiative integral :math:`I_0` is
    computed with the same machinery: deterministic quadrature for two-
    and three-body final states, RAMBO above that (with its statistical
    error propagated into `error`).

    Examples
    --------
    Photon spectrum for a 3-body process with a constant radiative and
    non-radiative matrix element (pure phase space):

    >>> import numpy as np
    >>> from hazma.spectra import dnde_photon_fsr
    >>> def msqrd(momenta):
    ...     return np.ones(momenta.shape[-1]) if momenta.ndim == 3 else 1.0
    >>> es = np.array([10.0, 50.0, 100.0])
    >>> dnde, error = dnde_photon_fsr(
    ...     es, 300.0, [105.658, 105.658], msqrd, msqrd
    ... )
    """
    masses = np.atleast_1d(np.asarray(final_state_masses, dtype=np.float64))

    if len(masses) < _TWO_BODY:
        raise ValueError(
            "'final_state_masses' must contain at least two masses (the"
            " non-radiative process needs a two-or-more-body final state);"
            f" got {len(masses)}."
        )
    if cme <= np.sum(masses):
        raise RamboCMETooSmall(
            f"Center-of-mass energy {cme} is below the sum of the"
            f" final-state masses {np.sum(masses)}."
        )

    if method == "auto":
        method = "quad" if len(masses) == _TWO_BODY else "rambo"
    if method == "quad" and len(masses) != _TWO_BODY:
        raise ValueError(
            "method='quad' requires exactly two final-state masses; use"
            " method='rambo' (or 'auto') for higher multiplicities."
        )
    if method not in ("quad", "rambo"):
        raise ValueError(f"Invalid method: {method}. Use 'auto', 'quad' or 'rambo'.")
    if method == "rambo" and not (
        isinstance(npts, (int, np.integer)) and npts >= _MIN_MC_NPTS
    ):
        # A single sample would yield a finite spectrum with error=nan,
        # silently voiding the one-sigma contract of FSRSpectrum.error.
        raise ValueError(
            "The Monte-Carlo backend requires an integer 'npts' of at"
            f" least {_MIN_MC_NPTS} (the one-sigma error is a ddof=1"
            f" sample estimate); got npts={npts!r}."
        )

    single = np.isscalar(photon_energies)
    energies = np.atleast_1d(np.asarray(photon_energies, dtype=np.float64))

    # Independent, reproducible substreams: one per photon energy plus
    # one for a Monte-Carlo non-radiative integral.
    seed_seqs = np.random.SeedSequence(seed).spawn(len(energies) + 1)

    nonrad, nonrad_err = _nonrad_integral(
        cme, masses, msqrd_nonrad, npts, seed_seqs[-1], epsabs, epsrel
    )
    if not nonrad > 0.0:
        raise ValueError(
            "The non-radiative phase-space integral is not positive"
            f" (got {nonrad}); check 'msqrd_nonrad'."
        )

    dnde = np.zeros_like(energies)
    error = np.zeros_like(energies)

    for i, energy in enumerate(energies):
        if method == "quad":
            num, num_err = _dnde_point_quad(energy, cme, masses, msqrd, epsabs, epsrel)
        else:
            num, num_err = _dnde_point_rambo(
                energy, cme, masses, msqrd, npts, seed_seqs[i]
            )
        dnde[i] = num / nonrad
        # Uncorrelated ratio propagation: sigma^2 = (dN/D)^2 + (N dD/D^2)^2.
        error[i] = np.hypot(num_err / nonrad, num * nonrad_err / nonrad**2)

    if single:
        return FSRSpectrum(float(dnde[0]), float(error[0]))
    return FSRSpectrum(dnde, error)
