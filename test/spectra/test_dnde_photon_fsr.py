r"""Validation of `hazma.spectra.dnde_photon_fsr` against real theory.

The validation plan (docs/adrs/ADR-0001, and the follow-up that demanded
it be written before the code):

1. **Corpus self-consistency.** The squared matrix elements in
   `msqrd_corpus` are validated on their own terms: the numerical Dirac
   traces reproduce hand-derived closed forms for the non-radiative
   processes; the radiative amplitudes satisfy the photon (and, where
   applicable, mediator-current) Ward identities; the radiative matrix
   elements factorize onto the eikonal soft-photon factor as
   ``E_gamma -> 0``; and the corpus normalization reproduces the
   annihilation cross sections shipped (and separately tested) in the
   mediator models to machine precision.
2. **Closed-form oracles.** The deterministic quadrature backend is
   pinned against the analytic FSR spectra the mediator models provide
   (``dnde_xx_to_v_to_ffg``, ``dnde_xx_to_s_to_ffg``,
   ``dnde_xx_to_v_to_pipig``) at ``rtol=1e-5``. The backend converges
   to ~4e-7 of the closed forms (limited by quadrature tolerance and
   the oracles' own floating-point evaluation); 1e-5 leaves margin
   without admitting any physics-level discrepancy.
3. **Approximate oracles.** The exact electron spectrum matches twice
   the single-particle Altarelli-Parisi formula to 1e-4 (the AP log
   captures the full vector-current result up to O(m^2/s) ~ 1e-6
   corrections for electrons), while for muons the deviation stays
   below 10% (genuine mass corrections at m/sqrt(s) ~ 0.1 — this
   bounds AP's validity rather than testing the generator).
4. **Monte-Carlo backend.** Pinned with a fixed seed against the
   quadrature backend and against analytic flat-matrix-element phase
   space (including a four-body radiative final state), with
   tolerances expressed as pulls against the returned one-sigma error
   estimate (|pull| < 4, i.e. a ~1e-4 false-failure probability if the
   error estimate is honest).
5. **Kinematic edges and the public contract** — thresholds, the
   photon endpoint, scalars vs arrays, determinism, and error paths.
"""

import functools as ft

import msqrd_corpus as corpus
import numpy as np
import pytest
from numpy.testing import assert_allclose

from hazma.hazma_errors import RamboCMETooSmall
from hazma.parameters import charged_pion_mass as mpi
from hazma.parameters import electron_mass as me
from hazma.parameters import muon_mass as mmu
from hazma.parameters import vh
from hazma.phase_space import Rambo, ThreeBody
from hazma.scalar_mediator import ScalarMediator
from hazma.spectra import FSRSpectrum, dnde_photon_ap_fermion, dnde_photon_fsr
from hazma.utils import (
    RealArray,
    RealOrRealArray,
    cross_section_prefactor,
    kallen_lambda,
)
from hazma.vector_mediator import VectorMediator

# One kinematic point for the whole suite: annihilation at 1 GeV with the
# mediators far off shell. mx < Q/2 keeps the models' kinematic gates open.
Q = 1000.0
MX = 200.0
MV = 3000.0
MS = 2500.0
GVXX, GVLL = 1.3, 0.72
GSXX, GSFF_MODEL = 1.1, 0.4
GVPIPI = 0.5
SEED = 1234

# Ward-identity violations are float noise (~1e-12 measured); a dropped
# diagram or wrong seagull produces O(1). Bound with four orders of margin.
WARD_BOUND = 1e-8
# Residual muon-mass corrections to Altarelli-Parisi at m/sqrt(s) ~ 0.1,
# measured up to ~8% at the hard end of the grid.
AP_MUON_BOUND = 0.10

# Photon-energy grids inside the endpoints e_max = (Q^2 - (2m)^2)/(2Q):
# 477.7 MeV (mu), 500.0 MeV (e), 461.0 MeV (pi).
ES_MU = np.array([1.0, 5.0, 25.0, 100.0, 250.0, 400.0, 470.0])
ES_E = np.array([1.0, 25.0, 100.0, 250.0, 400.0, 490.0])
ES_PI = np.array([1.0, 5.0, 25.0, 100.0, 250.0, 400.0, 455.0])

V_MU = dict(mf=mmu, mx=MX, mv=MV, gvxx=GVXX, gvff=GVLL)
V_E = dict(mf=me, mx=MX, mv=MV, gvxx=GVXX, gvff=GVLL)
# The scalar model's Yukawa convention is gsff * mf / vh.
S_MU = dict(mf=mmu, mx=MX, ms=MS, gsxx=GSXX, gsff=GSFF_MODEL * mmu / vh)
PI = dict(mx=MX, mv=MV, gvxx=GVXX, gvpipi=GVPIPI)


def _flat(momenta: RealArray) -> RealOrRealArray:
    batched = momenta.ndim == 3  # noqa: PLR2004 — (4, n, batch) layout
    return np.ones(momenta.shape[-1]) if batched else 1.0


@pytest.fixture(scope="module")
def vector_model() -> VectorMediator:
    return VectorMediator(
        mx=MX,
        mv=MV,
        gvxx=GVXX,
        gvuu=0.0,
        gvdd=0.0,
        gvss=0.0,
        gvee=GVLL,
        gvmumu=GVLL,
    )


@pytest.fixture(scope="module")
def scalar_model() -> ScalarMediator:
    return ScalarMediator(
        mx=MX, ms=MS, gsxx=GSXX, gsff=GSFF_MODEL, gsGG=0.0, gsFF=0.0, lam=vh
    )


@pytest.fixture(scope="module")
def dnde_quad_mumu() -> FSRSpectrum:
    return dnde_photon_fsr(
        ES_MU,
        Q,
        [mmu, mmu],
        ft.partial(corpus.msqrd_xx_to_v_to_ffg, **V_MU),
        ft.partial(corpus.msqrd_xx_to_v_to_ff, **V_MU),
        method="quad",
    )


# ===================================================================
# ---- 1. Corpus self-consistency -----------------------------------
# ===================================================================


class TestCorpusSelfConsistency:
    def test_nonradiative_traces_match_hand_formulas(self) -> None:
        """The numerical Dirac traces equal the textbook closed forms.

        Pure floating-point identity between two exact expressions:
        rtol 5e-12.
        """
        s = Q**2
        momenta, _ = Rambo(Q, [mmu, mmu]).generate(8, seed=SEED)

        num = corpus.msqrd_xx_to_v_to_ff(momenta, **V_MU)
        hand = (
            GVXX**2
            * GVLL**2
            / (s - MV**2) ** 2
            * (s + 2 * MX**2)
            / 3.0
            * 4.0
            * s
            * (1.0 + 2.0 * mmu**2 / s)
        )
        assert_allclose(num, hand, rtol=5e-12)

        num = corpus.msqrd_xx_to_s_to_ff(momenta, **S_MU)
        hand = (
            GSXX**2
            * S_MU["gsff"] ** 2
            / (s - MS**2) ** 2
            * (0.5 * s - 2.0 * MX**2)
            * 2.0
            * s
            * (1.0 - 4.0 * mmu**2 / s)
        )
        assert_allclose(num, hand, rtol=5e-12)

        momenta, _ = Rambo(Q, [mpi, mpi]).generate(8, seed=SEED)
        num = corpus.msqrd_xx_to_v_to_pipi(momenta, **PI)
        hand = (
            GVXX**2
            * GVPIPI**2
            / (s - MV**2) ** 2
            * (s + 2 * MX**2)
            / 3.0
            * (s - 4.0 * mpi**2)
        )
        assert_allclose(num, hand, rtol=5e-12)

    def test_ward_identities(self) -> None:
        """Ward identities annihilate the radiative amplitudes.

        Contracting with the photon momentum (and, for the pion current,
        the mediator momentum) leaves pure float noise; see WARD_BOUND.
        """
        momenta, _ = Rambo(Q, [mmu, mmu, 0.0]).generate(8, seed=SEED)
        assert np.all(corpus.photon_ward_violation_v(momenta, mf=mmu) < WARD_BOUND)

        momenta, _ = Rambo(Q, [mpi, mpi, 0.0]).generate(8, seed=SEED)
        photon, mediator = corpus.pion_ward_violations(momenta)
        assert np.all(photon < WARD_BOUND)
        assert np.all(mediator < WARD_BOUND)

    @pytest.mark.parametrize(
        ("egam", "rtol"),
        [
            # Soft-theorem corrections are O(E_gamma * d log|M0|^2 / dE):
            # measured deviations 5e-5 and 5e-7; bounds carry a 10x margin
            # and still verify the linear-in-E approach to factorization.
            (1e-2, 5e-4),
            (1e-4, 5e-6),
        ],
    )
    def test_soft_photon_factorization(self, egam: float, rtol: float) -> None:
        """|M_rad|^2 factorizes onto the eikonal factor as E_gamma -> 0.

        Checked for all three corpus processes (fermionic vector,
        fermionic scalar, and scalar QED).
        """
        sp = Q * (Q - 2.0 * egam)
        cme_rf = np.sqrt(sp)
        eg_rf = egam * Q / cme_rf
        photon = np.zeros((4, 1, 4))
        photon[0], photon[3] = eg_rf, eg_rf

        pair, _ = Rambo(cme_rf, [mmu, mmu]).generate(4, seed=SEED)
        momenta = np.concatenate((pair, photon), axis=1)
        eik = corpus.eikonal_factor(pair[:, 0], pair[:, 1], momenta[:, 2])

        rad = corpus.msqrd_xx_to_v_to_ffg(momenta, **V_MU)
        nonrad = corpus.msqrd_xx_to_v_to_ff(pair, **V_MU)
        assert_allclose(rad, eik * nonrad, rtol=rtol)

        rad = corpus.msqrd_xx_to_s_to_ffg(momenta, **S_MU)
        nonrad = corpus.msqrd_xx_to_s_to_ff(pair, **S_MU)
        assert_allclose(rad, eik * nonrad, rtol=rtol)

        pair, _ = Rambo(cme_rf, [mpi, mpi]).generate(4, seed=SEED)
        momenta = np.concatenate((pair, photon), axis=1)
        eik = corpus.eikonal_factor(pair[:, 0], pair[:, 1], momenta[:, 2])
        rad = corpus.msqrd_xx_to_v_to_pipig(momenta, **PI)
        nonrad = corpus.msqrd_xx_to_v_to_pipi(pair, **PI)
        assert_allclose(rad, eik * nonrad, rtol=rtol)

    def test_corpus_normalization_matches_model_cross_sections(
        self, vector_model: VectorMediator, scalar_model: ScalarMediator
    ) -> None:
        """Corpus normalization reproduces the models' cross sections.

        sigma = flux x dPhi_2 x |M0|^2 built from the corpus matrix
        elements equals the models' analytic annihilation cross sections
        (independently pinned in test/{vector,scalar}_mediator): an
        exact algebraic identity, rtol 1e-11.
        """
        s = Q**2
        e1 = Q / 2.0
        p = np.sqrt(e1**2 - mmu**2)
        momenta = np.array([[e1, e1], [0.0, 0.0], [0.0, 0.0], [p, -p]])
        flux = cross_section_prefactor(MX, MX, Q)
        dphi2 = np.sqrt(kallen_lambda(s, mmu**2, mmu**2)) / (8.0 * np.pi * s)

        # Beam-averaged |M0|^2 is angle-independent: one point suffices.
        sigma = (
            flux
            * dphi2
            * corpus.msqrd_xx_to_v_to_ff(
                momenta, **{**V_MU, "widthv": vector_model.width_v}
            )
        )
        assert_allclose(sigma, vector_model.sigma_xx_to_v_to_ff(Q, "mu"), rtol=1e-11)

        sigma = (
            flux
            * dphi2
            * corpus.msqrd_xx_to_s_to_ff(
                momenta, **{**S_MU, "widths": scalar_model.width_s}
            )
        )
        assert_allclose(sigma, scalar_model.sigma_xx_to_s_to_ff(Q, "mu"), rtol=1e-11)


# ===================================================================
# ---- 2./3. Oracles: closed forms and Altarelli-Parisi -------------
# ===================================================================


class TestQuadBackendOracles:
    # Quadrature converges to ~4e-7 of the closed forms (epsrel 1.49e-8
    # per angular integral, plus the oracles' own complex-log float
    # noise); rtol 1e-5 has ~25x margin yet fails on any physics-level
    # discrepancy, which for a wrong diagram/normalization is O(1).
    ORACLE_RTOL = 1e-5

    def test_v_to_mumug(
        self, vector_model: VectorMediator, dnde_quad_mumu: FSRSpectrum
    ) -> None:
        oracle = vector_model.dnde_xx_to_v_to_ffg(ES_MU, Q, "mu")
        assert_allclose(dnde_quad_mumu.dnde, oracle, rtol=self.ORACLE_RTOL)
        # The quadrature-error estimate must reflect that convergence.
        assert np.all(dnde_quad_mumu.error < 1e-4 * dnde_quad_mumu.dnde)

    def test_v_to_eeg(self, vector_model: VectorMediator) -> None:
        """Electron final state: the hardest quadrature case.

        The collinear peaks have width ~ 2 me^2/s ~ 5e-7 in cos(theta).
        """
        result = dnde_photon_fsr(
            ES_E,
            Q,
            [me, me],
            ft.partial(corpus.msqrd_xx_to_v_to_ffg, **V_E),
            ft.partial(corpus.msqrd_xx_to_v_to_ff, **V_E),
            method="quad",
        )
        oracle = vector_model.dnde_xx_to_v_to_ffg(ES_E, Q, "e")
        assert_allclose(result.dnde, oracle, rtol=self.ORACLE_RTOL)

    def test_s_to_mumug(self, scalar_model: ScalarMediator) -> None:
        result = dnde_photon_fsr(
            ES_MU,
            Q,
            [mmu, mmu],
            ft.partial(corpus.msqrd_xx_to_s_to_ffg, **S_MU),
            ft.partial(corpus.msqrd_xx_to_s_to_ff, **S_MU),
            method="quad",
        )
        oracle = scalar_model.dnde_xx_to_s_to_ffg(ES_MU, Q, mmu)
        assert_allclose(result.dnde, oracle, rtol=self.ORACLE_RTOL)

    def test_v_to_pipig(self, vector_model: VectorMediator) -> None:
        result = dnde_photon_fsr(
            ES_PI,
            Q,
            [mpi, mpi],
            ft.partial(corpus.msqrd_xx_to_v_to_pipig, **PI),
            ft.partial(corpus.msqrd_xx_to_v_to_pipi, **PI),
            method="quad",
        )
        oracle = vector_model.dnde_xx_to_v_to_pipig(ES_PI, Q)
        assert_allclose(result.dnde, oracle, rtol=self.ORACLE_RTOL)

    def test_altarelli_parisi_collinear_limit(
        self, vector_model: VectorMediator
    ) -> None:
        """Against 2x the single-particle AP spectrum (both legs radiate).

        For electrons the AP log expression is the complete massless
        limit of the vector-current spectrum — measured agreement 1e-6,
        asserted at 1e-4. For muons the residual is a genuine
        O(m/sqrt(s)) mass correction, measured up to ~8% at the hard
        end of the grid; the 10% bound documents where AP stops being
        trustworthy rather than testing the generator.
        """
        es = np.array([50.0, 150.0, 250.0, 350.0, 450.0])
        result = dnde_photon_fsr(
            es,
            Q,
            [me, me],
            ft.partial(corpus.msqrd_xx_to_v_to_ffg, **V_E),
            ft.partial(corpus.msqrd_xx_to_v_to_ff, **V_E),
            method="quad",
        )
        ap = 2.0 * dnde_photon_ap_fermion(es, Q**2, me)
        assert_allclose(result.dnde, ap, rtol=1e-4)

        result = dnde_photon_fsr(
            es,
            Q,
            [mmu, mmu],
            ft.partial(corpus.msqrd_xx_to_v_to_ffg, **V_MU),
            ft.partial(corpus.msqrd_xx_to_v_to_ff, **V_MU),
            method="quad",
        )
        ap = 2.0 * dnde_photon_ap_fermion(es, Q**2, mmu)
        assert np.all(np.abs(result.dnde / ap - 1.0) < AP_MUON_BOUND)


# ===================================================================
# ---- 4. Monte-Carlo backend ---------------------------------------
# ===================================================================


class TestRamboBackend:
    # A |pull| < 4 bound on a fixed seed fails with probability ~1e-4
    # per point if the returned one-sigma error estimate is honest, and
    # catches both bias (wrong prefactor/frame) and a broken estimate.
    MAX_PULL = 4.0

    def test_pion_mc_matches_closed_form(self, vector_model: VectorMediator) -> None:
        result = dnde_photon_fsr(
            ES_PI,
            Q,
            [mpi, mpi],
            ft.partial(corpus.msqrd_xx_to_v_to_pipig, **PI),
            ft.partial(corpus.msqrd_xx_to_v_to_pipi, **PI),
            method="rambo",
            npts=1 << 14,
            seed=SEED,
        )
        oracle = vector_model.dnde_xx_to_v_to_pipig(ES_PI, Q)
        pulls = (result.dnde - oracle) / result.error
        assert np.all(np.abs(pulls) < self.MAX_PULL)
        # Noise-floor sanity: ~1% relative error at 2^14 points.
        assert np.all(result.error < 0.02 * result.dnde)

    def test_fermionic_mc_matches_quad(self, dnde_quad_mumu: FSRSpectrum) -> None:
        es = ES_MU[[1, 3, 5]]
        result = dnde_photon_fsr(
            es,
            Q,
            [mmu, mmu],
            ft.partial(corpus.msqrd_xx_to_v_to_ffg, **V_MU),
            ft.partial(corpus.msqrd_xx_to_v_to_ff, **V_MU),
            method="rambo",
            npts=1 << 12,
            seed=SEED,
        )
        quad = dnde_quad_mumu.dnde[[1, 3, 5]]
        pulls = (result.dnde - quad) / result.error
        assert np.all(np.abs(pulls) < self.MAX_PULL)

    def test_four_body_flat_massless_phase_space(self) -> None:
        """Four-body flat massless phase space has a closed form.

        Three massless particles + photon with flat matrix elements:
        dN/dE = E s'/(4 pi^2 s) exactly, from Phi_3(s) = s/(256 pi^3).

        Exercises the >= 3-body Monte-Carlo numerator and the ThreeBody
        quadrature denominator in one pin. Massless RAMBO weights are
        constant, so the Monte-Carlo mean is deterministic up to float
        noise: rtol 1e-6.
        """
        es = np.array([1.0, 100.0, 300.0, 499.0])
        result = dnde_photon_fsr(
            es, Q, [0.0, 0.0, 0.0], _flat, _flat, npts=1 << 12, seed=SEED
        )
        sp = Q * (Q - 2.0 * es)
        assert_allclose(result.dnde, es * sp / (4.0 * np.pi**2 * Q**2), rtol=1e-6)

    def test_massive_three_body_flat_vs_threebody_quad(self) -> None:
        """The MC numerator agrees with independent ThreeBody quadrature.

        Massive 3-body + photon, flat matrix elements: the numerator must
        reproduce E/(4 pi^2) x Phi_3(s')/Phi_3(s) with Phi_3 computed by
        ThreeBody quadrature.
        """
        masses = (200.0, 150.0, 100.0)
        es = np.array([20.0, 120.0, 240.0])
        result = dnde_photon_fsr(es, Q, masses, _flat, _flat, npts=1 << 14, seed=SEED)
        phi3_s = ThreeBody(Q, masses).integrate(epsabs=0.0)[0]
        expected = np.array(
            [
                e
                / (4.0 * np.pi**2)
                * ThreeBody(np.sqrt(Q * (Q - 2.0 * e)), masses).integrate(epsabs=0.0)[0]
                / phi3_s
                for e in es
            ]
        )
        pulls = (result.dnde - expected) / result.error
        assert np.all(np.abs(pulls) < self.MAX_PULL)

    def test_seed_determinism(self) -> None:
        args = (ES_PI[:3], Q, [mpi, mpi], _flat, _flat)
        kwargs = dict(method="rambo", npts=2000)
        first = dnde_photon_fsr(*args, seed=SEED, **kwargs)
        second = dnde_photon_fsr(*args, seed=SEED, **kwargs)
        third = dnde_photon_fsr(*args, seed=SEED + 1, **kwargs)
        assert np.array_equal(first.dnde, second.dnde)
        assert np.array_equal(first.error, second.error)
        assert not np.array_equal(first.dnde, third.dnde)


# ===================================================================
# ---- 5. Kinematic edges and the public contract -------------------
# ===================================================================


class TestContract:
    def test_flat_massless_two_body_is_exact(self) -> None:
        """Massless flat two-body spectra are exactly E/(4 pi^2).

        dPhi_2 is independent of s', so dN/dE = E/(4 pi^2) — every
        prefactor in the quadrature backend, with zero physics input.
        rtol 1e-10 (quad of a constant is exact to roundoff).
        """
        es = np.array([1.0, 100.0, 300.0, 499.0])
        result = dnde_photon_fsr(es, Q, [0.0, 0.0], _flat, _flat, method="quad")
        assert_allclose(result.dnde, es / (4.0 * np.pi**2), rtol=1e-10)

    @pytest.mark.parametrize("method", ["quad", "rambo"])
    def test_unequal_masses_flat(self, method: str) -> None:
        """Unequal masses with flat matrix elements match the Kallen form.

        dN/dE = E/(4 pi^2) x dPhi_2(s')/dPhi_2(s) analytically (Kallen
        square roots). Deterministic for quad (rtol 1e-8); the rambo
        backend is exact here too because two-body RAMBO weights are
        constant (rtol 1e-8).
        """
        m1, m2 = 300.0, 50.0
        es = np.array([1.0, 100.0, 300.0])

        def dphi2(s: RealOrRealArray) -> RealOrRealArray:
            return np.sqrt(kallen_lambda(s, m1**2, m2**2)) / (8.0 * np.pi * s)

        sp = Q * (Q - 2.0 * es)
        expected = es / (4.0 * np.pi**2) * dphi2(sp) / dphi2(Q**2)
        result = dnde_photon_fsr(
            es, Q, [m1, m2], _flat, _flat, method=method, npts=1 << 12, seed=SEED
        )
        assert_allclose(result.dnde, expected, rtol=1e-8)

    def test_out_of_range_energies_are_zero(self) -> None:
        """Out-of-range energies give (0, 0).

        Below zero, at zero, at/beyond the endpoint, and NaN — rather
        than NaN or negative phase space.
        """
        e_max = (Q**2 - (2.0 * mmu) ** 2) / (2.0 * Q)
        es = np.array([-5.0, 0.0, e_max, e_max + 10.0, np.nan])
        result = dnde_photon_fsr(es, Q, [mmu, mmu], _flat, _flat, method="quad")
        assert np.array_equal(result.dnde, np.zeros(5))
        assert np.array_equal(result.error, np.zeros(5))
        # ... while just inside the endpoint the spectrum is positive.
        inside = dnde_photon_fsr(e_max - 1.0, Q, [mmu, mmu], _flat, _flat)
        assert inside.dnde > 0.0

    def test_scalar_and_array_contract(self, dnde_quad_mumu: FSRSpectrum) -> None:
        result = dnde_photon_fsr(100.0, Q, [mmu, mmu], _flat, _flat)
        assert isinstance(result, FSRSpectrum)
        assert isinstance(result.dnde, float)
        assert isinstance(result.error, float)

        assert isinstance(dnde_quad_mumu, FSRSpectrum)
        assert dnde_quad_mumu.dnde.shape == ES_MU.shape
        assert dnde_quad_mumu.error.shape == ES_MU.shape
        # NamedTuple unpacking is part of the public contract.
        dnde, error = dnde_quad_mumu
        assert dnde is dnde_quad_mumu.dnde and error is dnde_quad_mumu.error
        # In range, the error field is a positive, small fraction.
        inside = slice(0, len(ES_MU))
        assert np.all(dnde_quad_mumu.error[inside] > 0.0)

    def test_raises(self) -> None:
        with pytest.raises(RamboCMETooSmall):
            dnde_photon_fsr(10.0, 100.0, [mmu, mmu], _flat, _flat)
        with pytest.raises(ValueError, match="at least two"):
            dnde_photon_fsr(10.0, Q, [mmu], _flat, _flat)
        with pytest.raises(ValueError, match="method='quad'"):
            dnde_photon_fsr(10.0, Q, [1.0, 1.0, 1.0], _flat, _flat, method="quad")
        with pytest.raises(ValueError, match="Invalid method"):
            dnde_photon_fsr(10.0, Q, [mmu, mmu], _flat, _flat, method="bogus")
        with pytest.raises(ValueError, match="not positive"):
            dnde_photon_fsr(10.0, Q, [mmu, mmu], _flat, lambda m: 0.0)
