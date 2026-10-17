"""LO currents: field coordinates, Ward identity, measures and independent oracles."""

import itertools
from functools import partial

import numpy as np
import pytest

from hazma.form_factors.axial._sources import AxialCurrent
from hazma.form_factors.axial._tensors import current_projections
from hazma.form_factors.axial._three_pi import (
    AxialFormFactorPi0Pi0Pi0,
    AxialFormFactorPiPiPi0,
)
from hazma.parameters import neutral_pion_mass
from hazma.phase_space import ThreeBody

SOURCE = AxialCurrent(
    1.0,
    -1.0,
    0.0,
    qcd_scheme="MSbar",
    scale_mev=2000.0,
    provenance="test pure isovector",
)
CLASSES = (AxialFormFactorPiPiPi0, AxialFormFactorPi0Pi0Pi0)


def dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[0] * b[0] - np.sum(a[1:] * b[1:], axis=0)


def point(
    q: float = 500.0, mass: float = 135.0, fraction: float = 0.4, cosine: float = 0.3
) -> np.ndarray:
    """Sequential two-body decay, with a boost from the charged-pair rest frame."""
    s = 4 * mass**2 + fraction * ((q - mass) ** 2 - 4 * mass**2)
    e0 = (q * q + mass * mass - s) / (2 * q)
    k0 = np.sqrt(max(e0 * e0 - mass * mass, 0))
    ep = np.sqrt(s) / 2
    kp = np.sqrt(max(s / 4 - mass * mass, 0))
    gamma = (q - e0) / np.sqrt(s)
    beta = k0 / (q - e0)
    pp = np.array(
        [
            gamma * (ep - beta * kp * cosine),
            kp * np.sqrt(1 - cosine * cosine),
            0,
            gamma * (kp * cosine - beta * ep),
        ]
    )
    pm = np.array(
        [
            gamma * (ep + beta * kp * cosine),
            -kp * np.sqrt(1 - cosine * cosine),
            0,
            gamma * (-kp * cosine - beta * ep),
        ]
    )
    p0 = np.array([e0, 0, 0, k0])
    return np.stack((pp, pm, p0), axis=1)


def exponential_current(
    ps: np.ndarray, mass: float, decay_constant: float
) -> np.ndarray:
    """Independent expansion in exponential rather than sigma pion coordinates.

    Expanding F² Tr(dU dU†)/4+F²m²Tr(U+U†)/4 with U=exp(i tau.pi/F)
    gives contact (4/3F)(q-3p0), and the off-shell 4pi vertex
    (3s+- - q² - 2m²)/(3F²). Attach the source-pion vertex 2Fq.
    """
    q = np.sum(ps, axis=1)
    q2 = dot(q, q)
    pair = ps[:, 0] + ps[:, 1]
    four_pion = (3 * dot(pair, pair) - q2 - 2 * mass**2) / (3 * decay_constant**2)
    contact = 4 / (3 * decay_constant) * (q - 3 * ps[:, 2])
    pole = -2 * decay_constant * q * four_pion / (q2 - mass**2)
    return contact + pole


def boost_z(ps: np.ndarray, beta: float = 0.47) -> np.ndarray:
    result = np.array(ps, copy=True)
    gamma = 1 / np.sqrt(1 - beta**2)
    result[0] = gamma * (ps[0] + beta * ps[3])
    result[3] = gamma * (ps[3] + beta * ps[0])
    return result


def test_current_from_independent_field_coordinates_and_ward_identity() -> None:
    ps = np.stack([point(fraction=f) for f in (0.1, 0.4, 0.9)], axis=-1)
    q = np.sum(ps, axis=1)
    q2 = dot(q, q)
    pair = ps[:, 0] + ps[:, 1]
    charged = AxialFormFactorPiPiPi0(pion_mass=135)
    neutral = AxialFormFactorPi0Pi0Pi0(pion_mass=135)
    # Only floating-point arithmetic and field-coordinate cancellation: 2e-13.
    np.testing.assert_allclose(
        charged.current(ps, SOURCE),
        exponential_current(ps, 135, 92.2),
        rtol=2e-13,
        atol=2e-14,
    )
    divergence = -2 * 135**2 / 92.2 * (dot(pair, pair) - 135**2) / (q2 - 135**2)
    np.testing.assert_allclose(
        dot(q, charged.current(ps, SOURCE)), divergence, rtol=2e-13
    )
    # PCAC in sigma coordinates: the density is linear in pi3; attach the
    # off-shell 4pi vertex. The massless divergence must vanish, below.
    np.testing.assert_allclose(
        dot(q, neutral.current(ps, SOURCE)),
        -2 * 135**2 * q2 / (92.2 * (q2 - 135**2)),
        rtol=2e-13,
    )


@pytest.mark.parametrize("cls", CLASSES)
def test_lorentz_covariance_shapes_and_invariant_weight(
    cls: type[AxialFormFactorPiPiPi0] | type[AxialFormFactorPi0Pi0Pi0],
) -> None:
    ff = cls(pion_mass=135)
    ps = np.stack([point(fraction=f) for f in (0.2, 0.8)], axis=-1)
    h = ff.current(ps, SOURCE)
    assert h.shape == (4, 2)
    assert ff.current(ps[:, :, 0], SOURCE).shape == (4,)
    np.testing.assert_allclose(
        ff.current(boost_z(ps), SOURCE), boost_z(h), rtol=5e-13, atol=1e-14
    )
    np.testing.assert_allclose(
        ff.differential_weight(boost_z(ps), SOURCE),
        ff.differential_weight(ps, SOURCE),
        rtol=5e-13,
        atol=1e-14,
    )
    t, _ = current_projections(h, np.sum(ps, axis=1))
    np.testing.assert_allclose(
        ff.differential_weight(ps, SOURCE), t, rtol=5e-13, atol=1e-14
    )
    assert isinstance(ff.differential_weight(ps[:, :, 0], SOURCE), float)


def test_charged_permutation_and_neutral_bose_symmetry() -> None:
    ps = point()
    charged, neutral = (cls(pion_mass=135) for cls in CLASSES)
    np.testing.assert_array_equal(
        charged.current(ps[:, [1, 0, 2]], SOURCE), charged.current(ps, SOURCE)
    )
    for order in itertools.permutations(range(3)):
        # Addition order can change the last few ulps of total momentum.
        np.testing.assert_allclose(
            neutral.current(ps[:, order], SOURCE),
            neutral.current(ps, SOURCE),
            rtol=2e-14,
            atol=2e-16,
        )


def test_chiral_ward_adler_limit_and_analytic_massless_integral() -> None:
    charged, neutral = (cls(pion_mass=0) for cls in CLASSES)
    ps = point(mass=0)
    q = np.sum(ps, axis=1)
    np.testing.assert_allclose(dot(q, charged.current(ps, SOURCE)), 0, atol=2e-12)
    # A soft neutral pion is an Adler zero for the full charged current.
    soft = np.array([[250, 250, 0], [0, 0, 0], [0, 0, 0], [250, -250, 0]], dtype=float)
    np.testing.assert_array_equal(charged.current(soft, SOURCE), 0)
    np.testing.assert_array_equal(neutral.current(ps, SOURCE), 0)
    energies = np.array([0, 20, 450, 600])
    result = charged.spectral_functions(energies, SOURCE)
    # dPhi3/ds=(q²-s)/(128pi³q²), so ∫T dPhi=g3²q⁴/(384pi³F²).
    np.testing.assert_allclose(
        result.transverse,
        energies.astype(float) ** 4 / (384 * np.pi**3 * 92.2**2),
        rtol=3e-14,
        atol=0,
    )
    np.testing.assert_array_equal(result.source_longitudinal, 0)
    np.testing.assert_array_equal(
        neutral.spectral_functions(energies, SOURCE).transverse, 0
    )
    np.testing.assert_array_equal(neutral.width(energies, SOURCE), 0)


@pytest.mark.parametrize("q", [450.0, 600.0])
def test_independent_massive_threebody_integration(q: float) -> None:
    charged, neutral = (cls(pion_mass=135) for cls in CLASSES)

    def weight(ps: np.ndarray, longitudinal: bool = False) -> np.ndarray:
        h = exponential_current(ps, 135, 92.2)
        return h[0] ** 2 if longitudinal else np.sum(h[1:] ** 2, axis=0) / 3

    result = charged.spectral_functions(q, SOURCE)
    for longitudinal, expected in (
        (False, result.transverse),
        (True, result.source_longitudinal),
    ):
        # Adaptive two-dimensional Dalitz integration versus one-dimensional
        # endpoint-smoothed Gauss integration; tolerance exceeds their 1e-9 target.
        actual, _ = ThreeBody(
            q,
            (135,) * 3,
            msqrd=partial(weight, longitudinal=longitudinal),
            msqrd_signature="momenta",
        ).integrate(epsabs=1e-10, epsrel=1e-9)
        np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=1e-11)
    phase_volume, _ = ThreeBody(q, (135,) * 3, msqrd=lambda s, t: 1.0).integrate(
        epsabs=1e-11, epsrel=1e-10
    )
    neutral_l = neutral.spectral_functions(q, SOURCE).source_longitudinal
    neutral_density = 4 * 135**4 * q * q / (92.2**2 * (q * q - 135**2) ** 2)
    np.testing.assert_allclose(neutral_l, phase_volume * neutral_density / 6, rtol=2e-9)


def test_v3_independent_symbolic_tree_oracle() -> None:
    # Read-only version3/integrated/ax_xsec.py Model.spectral('Zp', final, q)
    # CHPT, gau=1,gad=-1,gas=0, all pion MASS=135, f=92.2, WIDTH['pi0']=0.
    # n=32 and 48 agreed <6e-15 relative; fixture uses n=48. This independent
    # symbolic tree oracle is supplemental to PCAC and analytic measure tests.
    q = np.array([450, 500, 600])
    expected_t = [1.5906418165653142, 15.231055501000954, 136.87334329168735]
    expected_l = [0.06123401653855351, 0.2071752550990431, 0.5610413513667469]
    expected_neutral_l = [0.09098608979168443, 0.30090781143351225, 0.7726880053721584]
    charged, neutral = (cls(pion_mass=135) for cls in CLASSES)
    result = charged.spectral_functions(q, SOURCE)
    # 1e-11 leaves ample platform-roundoff allowance, no model approximation.
    np.testing.assert_allclose(result.transverse, expected_t, rtol=1e-11)
    np.testing.assert_allclose(result.source_longitudinal, expected_l, rtol=1e-11)
    np.testing.assert_allclose(
        neutral.spectral_functions(q, SOURCE).source_longitudinal,
        expected_neutral_l,
        rtol=1e-11,
    )


@pytest.mark.parametrize("cls", CLASSES)
def test_threshold_width_coupling_scaling_and_quadrature(
    cls: type[AxialFormFactorPiPiPi0] | type[AxialFormFactorPi0Pi0Pi0],
) -> None:
    ff = cls()
    assert ff.fsp_masses == (neutral_pion_mass,) * 3
    q = np.array([[0, ff.threshold], [np.nextafter(ff.threshold, np.inf), 600]])
    result = ff.spectral_functions(q, SOURCE)
    assert np.shape(result.transverse) == q.shape
    np.testing.assert_array_equal(np.asarray(result.transverse)[0], 0)
    np.testing.assert_array_equal(np.asarray(result.source_longitudinal)[0], 0)
    scaled = AxialCurrent(
        -0.5, 0.5, 0, qcd_scheme="MSbar", scale_mev=2000, provenance="test scale"
    )
    scaled_result = ff.spectral_functions(q, scaled)
    np.testing.assert_allclose(
        scaled_result.transverse, np.asarray(result.transverse) / 4, rtol=1e-14, atol=0
    )
    np.testing.assert_allclose(
        scaled_result.source_longitudinal,
        np.asarray(result.source_longitudinal) / 4,
        rtol=1e-14,
        atol=0,
    )
    expected_width = np.divide(
        result.transverse, 2 * q, out=np.zeros_like(q), where=q > 0
    )
    np.testing.assert_allclose(ff.width(q, SOURCE), expected_width, rtol=1e-14, atol=0)
    assert isinstance(ff.width(500, SOURCE), float)
    refined = ff.spectral_functions([450, 500, 600], SOURCE, quadrature_order=96)
    coarse = ff.spectral_functions([450, 500, 600], SOURCE, quadrature_order=32)
    # Smooth endpoint substitution converges to roundoff here.
    np.testing.assert_allclose(
        refined.transverse, coarse.transverse, rtol=2e-12, atol=0
    )
    np.testing.assert_allclose(
        refined.source_longitudinal, coarse.source_longitudinal, rtol=2e-12, atol=0
    )


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize(
    "kwargs",
    [
        {"pion_mass": -1},
        {"pion_mass": np.nan},
        {"decay_constant": 0},
        {"decay_constant": np.inf},
        {"q_max": 400},
        {"q_max": 601},
        {"q_max": 10000},
        {"q_max": np.nan},
    ],
)
def test_bad_parameters(
    cls: type[AxialFormFactorPiPiPi0] | type[AxialFormFactorPi0Pi0Pi0],
    kwargs: dict[str, float],
) -> None:
    with pytest.raises(ValueError):
        cls(**kwargs)


@pytest.mark.parametrize(
    "method", ["current", "differential_weight", "spectral_functions", "width"]
)
@pytest.mark.parametrize("source", [(1, 1, 1), (1, 1, -2), (1, -1, 1)])
def test_reject_unsupported_flavor_directions(
    method: str, source: tuple[float, float, float]
) -> None:
    ff = AxialFormFactorPiPiPi0(pion_mass=135)
    coupling = AxialCurrent(
        *source, qcd_scheme="MSbar", scale_mev=2000, provenance="unsupported"
    )
    with pytest.raises(ValueError, match="only G_A"):
        getattr(ff, method)(
            point() if method in ("current", "differential_weight") else 500, coupling
        )


@pytest.mark.parametrize("q", [-1, 601, np.nan, np.inf, 500 + 1j, [450, np.nan]])
def test_invalid_energy(q: float) -> None:
    with pytest.raises(ValueError):
        AxialFormFactorPiPiPi0().spectral_functions(q, SOURCE)


@pytest.mark.parametrize("order", [0, 7, 513, True, 32.5])
def test_invalid_quadrature(order: int) -> None:
    with pytest.raises(ValueError, match="quadrature_order"):
        AxialFormFactorPiPiPi0().spectral_functions(500, SOURCE, quadrature_order=order)


def test_invalid_momenta_and_wrong_mass_are_rejected() -> None:
    ff = AxialFormFactorPiPiPi0(pion_mass=135)
    invalid = [
        np.zeros((3, 4)),
        point(mass=139.57),
        point(q=650),
        point().astype(complex) + 1j,
        np.full((4, 3), np.nan),
        -point(),
    ]
    for ps in invalid:
        with pytest.raises(ValueError):
            ff.current(ps, SOURCE)
    ps = point()
    ps[0, 0] += 0.01
    with pytest.raises(ValueError, match="on shell"):
        ff.current(ps, SOURCE)
    with pytest.raises(TypeError, match="AxialCurrent"):
        ff.current(point(), (1, -1, 0))


def test_finite_inputs_cannot_escape_as_nan_currents() -> None:
    ff = AxialFormFactorPiPiPi0(pion_mass=135)
    momenta = np.full((4, 3), np.finfo(float).max / 2)
    for method in (ff.current, ff.differential_weight):
        with pytest.raises(ValueError, match="finite"):
            method(momenta, SOURCE)
    huge = AxialCurrent(
        1e308, -1e308, 0, qcd_scheme="MSbar", scale_mev=2000, provenance="overflow test"
    )
    for method in (ff.current, ff.differential_weight):
        with pytest.raises(ValueError, match="finite"):
            method(point(), huge)
    with pytest.raises(ValueError):
        ff.spectral_functions(500, huge)
    # Tiny positive energies have a vanishing representable chiral integral.
    result = AxialFormFactorPiPiPi0(pion_mass=0).spectral_functions(1e-200, SOURCE)
    assert result.transverse == 0.0


def test_massive_integral_under_uniform_unit_rescaling() -> None:
    # H is invariant when q,m,F all scale together; dPhi3 and both spectra
    # scale as energy². This independently tests units and tiny-scale safety.
    for cls in CLASSES:
        reference = cls(pion_mass=135).spectral_functions(500, SOURCE)
        factor = 1e-100
        scaled = cls(
            pion_mass=135 * factor, decay_constant=92.2 * factor
        ).spectral_functions(500 * factor, SOURCE)
        np.testing.assert_allclose(
            scaled.transverse / factor**2, reference.transverse, rtol=2e-13, atol=0
        )
        np.testing.assert_allclose(
            scaled.source_longitudinal / factor**2,
            reference.source_longitudinal,
            rtol=2e-13,
            atol=0,
        )
        tiny = cls(pion_mass=1e-200, decay_constant=1e-200).spectral_functions(
            4e-200, SOURCE
        )
        assert tiny.transverse == 0
        assert tiny.source_longitudinal == 0


def test_independent_analytic_massless_phase_volume() -> None:
    # The full Lorentz invariant measure is q²/(256pi³); a spurious Gram
    # factor of one half or an extra angular average fails this assertion.
    q = 500.0
    actual, _ = ThreeBody(q, (0, 0, 0), msqrd=lambda s, t: 1.0).integrate(
        epsrel=1e-11, epsabs=1e-11
    )
    np.testing.assert_allclose(actual, q * q / (256 * np.pi**3), rtol=2e-14)
