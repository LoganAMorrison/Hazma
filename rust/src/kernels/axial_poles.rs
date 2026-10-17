//! Coherent longitudinal `π⁰`, `η` and `η′` poles of the neutral axial
//! source, and the `γγ` density they produce.
//!
//! The kernels behind `hazma/form_factors/axial/_poles.py`. The source
//! `Z′_μ q̄ γ^μ γ₅ G_A q` mixes with each neutral pseudoscalar through
//! `−c_P Z′·∂P`, `c_P = f Tr(G_A λ_P)`. That mixing is proportional to the
//! source momentum `k`, so it reaches only the source-longitudinal response,
//! and the Proca propagator's `k^μ k^ν/(s M²)` term turns it into a
//! non-resonant contact rather than a spin-zero mediator (project ADR-0003).
//! The meson block is kept whole: the invisible dark-matter cut couples all
//! three mesons, and because it is a rank-one update along the source
//! direction itself, the full inverse propagator has a closed form
//! (Sherman–Morrison) rather than a fresh width on one diagonal entry.
//!
//! The physical basis `(π⁰, η, η′)` uses the single-angle LO mixing
//! `λ_η = cos θ_P T₈ − sin θ_P T₀` and `λ_η′ = sin θ_P T₈ + cos θ_P T₀`, with
//! the project's generators `T₃ = diag(1, −1, 0)`, `T₈ = diag(1, 1, −2)/√3`
//! and `T₀ = √(2/3) 1`, so that `Tr(λ_P λ_Q) = 2 δ_PQ`. Masses, widths and
//! energies are in MeV. The caller validates every input.

use std::f64::consts::PI;

use num_complex::Complex64;

const N_COLORS: f64 = 3.0;
/// Squared quark charges in flavor order `(u, d, s)`.
const QUARK_CHARGES_SQUARED: [f64; 3] = [4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0];
/// `N_c Q²` of the triangle fermions, in the order `(e, μ, τ, c)`.
const TRIANGLE_WEIGHTS: [f64; 4] = [1.0, 1.0, 1.0, N_COLORS * 4.0 / 9.0];

/// The LO U(3) neutral pseudoscalar block, in the order `(π⁰, η, η′)`.
#[derive(Clone, Copy, Debug)]
pub struct MesonBlock {
    /// Pole masses in MeV; zero is the chiral limit.
    pub masses: [f64; 3],
    /// Visible widths in MeV, entering as `i m_P Γ_P`.
    pub widths: [f64; 3],
    /// The `η`–`η′` mixing angle `θ_P` in radians.
    pub mixing_angle: f64,
    /// Common octet and singlet decay constant `f` in MeV.
    pub decay_constant: f64,
}

/// One insertion of the dark-matter pair cut in the meson self-energy.
///
/// The longitudinal contact gives each meson the pseudoscalar coupling
/// `y_P = κ c_P` to `χ̄ γ₅ χ`, with `κ = 2 m_χ g_A^χ / M²`. Its absorptive
/// part adds the rank-one matrix `i ρ(s) κ² c cᵀ`, `ρ = s β_χ/(8π)`, to the
/// inverse propagator `diag(s − m_P² + i m_P Γ_P)`; the dispersive part
/// renormalizes the pole inputs and is omitted.
#[derive(Clone, Copy, Debug)]
pub struct InvisibleCut {
    /// Dark-matter mass `m_χ` in MeV.
    pub dm_mass: f64,
    /// `κ = 2 m_χ g_A^χ / M²` in 1/MeV.
    pub strength: f64,
}

impl InvisibleCut {
    /// Return `s β_χ/(8π)` in MeV², zero at and below the pair threshold.
    fn absorptive_weight(&self, s: f64) -> f64 {
        let threshold = 4.0 * self.dm_mass * self.dm_mass;
        if s <= threshold {
            return 0.0;
        }
        s * (1.0 - threshold / s).sqrt() / (8.0 * PI)
    }
}

/// The heavy fermions whose triangles reach `γγ` without a meson pole, in
/// the order `(e, μ, τ, c)`.
#[derive(Clone, Copy, Debug)]
pub struct Triangles {
    /// Fermion masses in MeV, all positive.
    pub masses: [f64; 4],
    /// Dimensionless axial couplings `g_A^f`.
    pub couplings: [f64; 4],
}

fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

impl MesonBlock {
    /// Return the diagonal of each `λ_P` in flavor order `(u, d, s)`, one
    /// row per meson `(π⁰, η, η′)`.
    pub fn generators(&self) -> [[f64; 3]; 3] {
        let (sin, cos) = self.mixing_angle.sin_cos();
        let t8 = [1.0, 1.0, -2.0].map(|x: f64| x / 3f64.sqrt());
        let t0 = (2.0f64 / 3.0).sqrt();
        [
            [1.0, -1.0, 0.0],
            t8.map(|x| cos * x - sin * t0),
            t8.map(|x| sin * x + cos * t0),
        ]
    }

    /// Return `Tr(G_A λ_P)`, dimensionless, for `G_A = diag(g_u, g_d, g_s)`.
    pub fn source_traces(&self, couplings: &[f64; 3]) -> [f64; 3] {
        self.generators().map(|lambda| dot3(&lambda, couplings))
    }

    /// Return the mixing strengths `c_P = f Tr(G_A λ_P)` in MeV.
    pub fn source_couplings(&self, couplings: &[f64; 3]) -> [f64; 3] {
        self.source_traces(couplings)
            .map(|trace| self.decay_constant * trace)
    }

    /// Return `N_c Tr(λ_P Q²)`, dimensionless: each meson's WZW `γγ`
    /// coupling in units of `α/(π f)`.
    pub fn photon_traces(&self) -> [f64; 3] {
        self.generators()
            .map(|lambda| N_COLORS * dot3(&lambda, &QUARK_CHARGES_SQUARED))
    }

    /// Return `Δ_P = s − m_P² + i m_P Γ_P` in MeV², the diagonal inverse
    /// propagator without a cut.
    fn denominators(&self, s: f64) -> [Complex64; 3] {
        std::array::from_fn(|p| {
            let mass = self.masses[p];
            Complex64::new(s - mass * mass, mass * self.widths[p])
        })
    }

    /// Return `s aᵀ D(s)⁻¹ b`, `a_P = Tr(G_A λ_P)`, the coherent sum over the
    /// three poles.
    ///
    /// The cut adds `i γ a aᵀ`, `γ = ρ κ² f²`, to `D = diag(Δ_P)`, so
    /// Sherman–Morrison gives the exact `aᵀ D⁻¹ b = aᵀΔ⁻¹b / (1 + i γ aᵀΔ⁻¹a)`.
    /// A meson with `a_P = 0` drops out of both sums exactly, even at its own
    /// zero-width pole; a coupled zero-width pole is singular, with or
    /// without a cut.
    fn coherent_sum(
        &self,
        q: f64,
        b: &[f64; 3],
        couplings: &[f64; 3],
        cut: Option<&InvisibleCut>,
    ) -> Complex64 {
        let s = q * q;
        let a = self.source_traces(couplings);
        let delta = self.denominators(s);
        let project = |v: &[f64; 3]| -> Complex64 {
            (0..3)
                .filter(|&p| a[p] != 0.0)
                .map(|p| a[p] * v[p] / delta[p])
                .sum()
        };
        let mut sum = project(b);
        if let Some(cut) = cut {
            let gamma = cut.absorptive_weight(s) * (cut.strength * self.decay_constant).powi(2);
            sum /= Complex64::new(1.0, 0.0) + Complex64::i() * gamma * project(&a);
        }
        s * sum
    }

    /// Return the meson-pole part `s cᵀ D(s)⁻¹ c` of the longitudinal
    /// source self-energy, in MeV².
    ///
    /// It is the Schur correction to the longitudinal kernel `M²`. In the
    /// chiral limit it equals the contact `2 f² Tr(G_A²)` at every `s`,
    /// which is the axial Ward identity.
    pub fn source_self_energy(
        &self,
        q: f64,
        couplings: &[f64; 3],
        cut: Option<&InvisibleCut>,
    ) -> Complex64 {
        let c = self.source_couplings(couplings);
        self.decay_constant * self.coherent_sum(q, &c, couplings, cut)
    }

    /// Return `ε_P = |c_P² s / (M² (s − m_P² + i m_P Γ_P))|` per meson,
    /// dimensionless, with the visible widths only.
    ///
    /// `ε_P` is the note's per-pole measure of the neglected second mixing
    /// insertion; the exact total is `|source_self_energy|/M²`, at most the
    /// sum of the three. It is infinite at an exact pole of zero width.
    pub fn mixing_expansion(&self, q: f64, couplings: &[f64; 3], mediator_mass: f64) -> [f64; 3] {
        let s = q * q;
        let c = self.source_couplings(couplings);
        std::array::from_fn(|p| {
            let numerator = c[p] * c[p] * s;
            if numerator == 0.0 {
                return 0.0;
            }
            let mass = self.masses[p];
            let denominator = Complex64::new(s - mass * mass, mass * self.widths[p]).norm();
            numerator / (mediator_mass * mediator_mass * denominator)
        })
    }

    /// Return `Γ(P → χχ̄) = m_P β_χ κ² c_P² / (8π)` in MeV, the width the
    /// cut adds at each pole; zero for a closed channel.
    pub fn invisible_widths(&self, couplings: &[f64; 3], cut: &InvisibleCut) -> [f64; 3] {
        let c = self.source_couplings(couplings);
        std::array::from_fn(|p| {
            let mass = self.masses[p];
            if mass == 0.0 {
                return 0.0;
            }
            let y = cut.strength * c[p];
            cut.absorptive_weight(mass * mass) * y * y / mass
        })
    }
}

/// Largest `x = s/(4m²)` at which [`triangle_factor`] sums its heavy-fermion
/// series. At `x = 1/4` the closed form `1 − τ arcsin²(1/√τ)` keeps all but
/// a few ulps; below it the subtraction loses digits like `ε/x`.
const TRIANGLE_SERIES_MAX: f64 = 0.25;

/// Return `τ arcsin²(1/√τ) − 1 = Σ_{n≥2} 2^{2n} xⁿ⁻¹ / (2 n² C(2n, n))` for
/// `x = 1/τ ≤ 1/4`, whose successive terms fall by
/// `2x n²/((n + 1)(2n + 1)) < x`.
fn heavy_triangle_series(x: f64) -> f64 {
    let mut term = x / 3.0;
    let mut sum = 0.0;
    let mut n = 2.0;
    while term > 0.5 * f64::EPSILON * sum {
        sum += term;
        term *= 2.0 * x * n * n / ((n + 1.0) * (2.0 * n + 1.0));
        n += 1.0;
    }
    sum
}

/// Return the triangle form factor `1 − τ f(τ)`, `τ = 4m²/s`, at `s = q²`.
///
/// `f(τ) = arcsin²(1/√τ)` for `τ ≥ 1` and
/// `−¼ [ln((1 + r)/(1 − r)) − iπ]²`, `r = √(1 − τ)`, below. The factor is
/// one for a massless fermion and vanishes like `−s/(12 m²)` as `s → 0`,
/// where it is summed as a series to keep full relative precision.
pub fn triangle_factor(q: f64, mass: f64) -> Complex64 {
    let s = q * q;
    if s == 0.0 {
        return Complex64::new(0.0, 0.0);
    }
    let x = s / (4.0 * mass * mass);
    if x <= TRIANGLE_SERIES_MAX {
        return Complex64::new(-heavy_triangle_series(x), 0.0);
    }
    let tau = 4.0 * mass * mass / s;
    if tau == 0.0 {
        return Complex64::new(1.0, 0.0);
    }
    if tau >= 1.0 {
        let angle = tau.sqrt().recip().asin();
        return Complex64::new(1.0 - tau * angle * angle, 0.0);
    }
    let r = (1.0 - tau).sqrt();
    // (1 + r)/(1 − r) = (1 + r)²/τ avoids the cancellation in 1 − r.
    let log = Complex64::new(2.0 * r.ln_1p() - tau.ln(), -PI);
    Complex64::new(1.0, 0.0) + 0.25 * tau * log * log
}

/// Return the dimensionless longitudinal `γγ` amplitude
/// `𝒜(s) = ½ s aᵀ D⁻¹ b + Σ_f N_c Q_f² g_A^f (1 − τ_f f(τ_f))`, with
/// `a_P = Tr(G_A λ_P)` and `b_P = N_c Tr(λ_P Q²)`.
///
/// `k·H_γγ = −i (2α/π) 𝒜 ε(k₁, ε₁, k₂, ε₂)`; for `s ≫ m_P²` and no
/// triangles, `𝒜` tends to the ABJ coefficient `N_c Σ_q g_A^q Q_q²`.
pub fn gamma_gamma_amplitude(
    block: &MesonBlock,
    q: f64,
    couplings: &[f64; 3],
    triangles: &Triangles,
    cut: Option<&InvisibleCut>,
) -> Complex64 {
    let b = block.photon_traces();
    let poles = 0.5 * block.coherent_sum(q, &b, couplings, cut);
    let fermions: Complex64 = TRIANGLE_WEIGHTS
        .iter()
        .zip(triangles.couplings)
        .zip(triangles.masses)
        .map(|((weight, coupling), mass)| weight * coupling * triangle_factor(q, mass))
        .sum();
    poles + fermions
}

/// Return the source-longitudinal density `A_L = s α² |𝒜|² / (8π³)` of
/// `γγ` in MeV², including the identical-photon factor ½.
///
/// Its transverse density is identically zero (Landau–Yang).
pub fn gamma_gamma_density(
    block: &MesonBlock,
    q: f64,
    couplings: &[f64; 3],
    triangles: &Triangles,
    alpha: f64,
    cut: Option<&InvisibleCut>,
) -> f64 {
    let amplitude = gamma_gamma_amplitude(block, q, couplings, triangles, cut);
    q * q * alpha * alpha * amplitude.norm_sqr() / (8.0 * PI.powi(3))
}

#[cfg(test)]
mod tests {
    use super::*;

    const MASSES: [f64; 3] = [134.9768, 547.862, 957.78];
    const WIDTHS: [f64; 3] = [7.8e-6, 1.31e-3, 0.188];
    /// The hypercharge-aligned benchmark of the mediator note, `sec:ax-inv`.
    const HYPERCHARGE: [f64; 3] = [1e-3, -1e-3, -1e-3];
    const NO_TRIANGLES: Triangles = Triangles {
        masses: [0.5109989461, 105.6583745, 1776.86, 1.27e3],
        couplings: [0.0; 4],
    };

    fn block(masses: [f64; 3], widths: [f64; 3]) -> MesonBlock {
        MesonBlock {
            masses,
            widths,
            mixing_angle: (-15.4f64).to_radians(),
            decay_constant: 92.2,
        }
    }

    fn assert_close(actual: f64, expected: f64, rtol: f64) {
        let error = (actual - expected).abs() / expected.abs();
        assert!(
            error <= rtol,
            "{actual} vs {expected}: {error:e} > {rtol:e}"
        );
    }

    #[test]
    fn generators_are_orthonormal_under_the_trace() {
        let lambda = block(MASSES, WIDTHS).generators();
        for p in 0..3 {
            for r in 0..3 {
                let expected = if p == r { 2.0 } else { 0.0 };
                assert!((dot3(&lambda[p], &lambda[r]) - expected).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn benchmark_mixing_strengths_match_the_note() {
        // The note quotes c = (0.184, 0.083, −0.101) MeV to three figures.
        let c = block(MASSES, WIDTHS).source_couplings(&HYPERCHARGE);
        for (actual, expected) in c.into_iter().zip([0.184, 0.083, -0.101]) {
            assert!((actual - expected).abs() < 5e-4, "{actual} vs {expected}");
        }
    }

    #[test]
    fn chiral_pole_sum_cancels_the_contact_at_every_energy() {
        // With massless, stable mesons s/(s − 0) = 1 exactly, so only the
        // rounding of the trace sum separates the poles from the contact.
        let chiral = block([0.0; 3], [0.0; 3]);
        for couplings in [HYPERCHARGE, [0.3, 0.7, -1.1], [1.0, 1.0, 1.0]] {
            let contact = 2.0 * 92.2f64.powi(2) * dot3(&couplings, &couplings);
            for q in [1e-3, 1.0, 300.0, 1e5] {
                let pole = chiral.source_self_energy(q, &couplings, None);
                assert_close(pole.re, contact, 1e-14);
                assert_eq!(pole.im, 0.0);
            }
        }
    }

    #[test]
    fn amplitude_approaches_the_abj_coefficient_far_from_the_poles() {
        let stable = block(MASSES, [0.0; 3]);
        let abj: f64 = (0..3)
            .map(|f| N_COLORS * HYPERCHARGE[f] * QUARK_CHARGES_SQUARED[f])
            .sum();
        let a = stable.source_traces(&HYPERCHARGE);
        let b = stable.photon_traces();
        for q in [2e3, 2e4, 2e5] {
            let s = q * q;
            let amplitude = gamma_gamma_amplitude(&stable, q, &HYPERCHARGE, &NO_TRIANGLES, None);
            // 𝒜 − ABJ = ½ Σ_P a_P b_P m_P²/(s − m_P²) for stable poles.
            let remainder: f64 = (0..3)
                .map(|p| 0.5 * a[p] * b[p] * MASSES[p].powi(2) / (s - MASSES[p].powi(2)))
                .sum();
            assert_close(amplitude.re - abj, remainder, 1e-9);
            assert!((amplitude.re / abj - 1.0).abs() < 2.0 * (MASSES[2] / q).powi(2));
        }
    }

    #[test]
    fn residue_is_recovered_at_each_pole() {
        let stable = block(MASSES, [0.0; 3]);
        let a = stable.source_traces(&HYPERCHARGE);
        let b = stable.photon_traces();
        for p in 0..3 {
            let s = MASSES[p].powi(2);
            let delta = 1e-6;
            // Averaging (s′ − s) 𝒜(s′) at s′ = s(1 ± δ) cancels the O(δ)
            // regular term, leaving O(δ²) plus rounding of order ε/δ.
            let residue = [1.0 + delta, 1.0 - delta]
                .map(|scale: f64| {
                    let amplitude = gamma_gamma_amplitude(
                        &stable,
                        (s * scale).sqrt(),
                        &HYPERCHARGE,
                        &NO_TRIANGLES,
                        None,
                    );
                    (scale - 1.0) * s * amplitude.re
                })
                .iter()
                .sum::<f64>()
                / 2.0;
            assert_close(residue, 0.5 * s * a[p] * b[p], 1e-8);
        }
    }

    #[test]
    fn invisible_widths_follow_the_pseudoscalar_coupling() {
        let mesons = block(MASSES, WIDTHS);
        let cut = InvisibleCut {
            dm_mass: 60.0,
            strength: 2.0 * 60.0 / 2500.0f64.powi(2),
        };
        let gamma = mesons.invisible_widths(&HYPERCHARGE, &cut);
        let c = mesons.source_couplings(&HYPERCHARGE);
        for p in 0..3 {
            let mass = MASSES[p];
            let beta = (1.0 - 4.0 * 60.0f64.powi(2) / mass.powi(2)).sqrt();
            let y = cut.strength * c[p];
            assert_close(gamma[p], mass * beta * y * y / (8.0 * PI), 1e-15);
        }
        let closed = InvisibleCut {
            dm_mass: MASSES[2],
            ..cut
        };
        assert_eq!(mesons.invisible_widths(&HYPERCHARGE, &closed), [0.0; 3]);
    }

    #[test]
    fn invisible_cut_broadens_an_isolated_pole_by_its_width() {
        // G_A = g λ_π couples only the π⁰, so the coherent sum is one
        // Breit–Wigner whose width the cut raises by Γ(π⁰ → χχ̄), however
        // large the cut is against the visible width.
        let aligned = [1e-3, -1e-3, 0.0];
        let mesons = block(MASSES, WIDTHS);
        let a = mesons.source_traces(&aligned);
        assert_eq!([a[1], a[2]], [0.0; 2]);
        let b = mesons.photon_traces();
        for strength in [1.92e-5, 1.92e-2] {
            let cut = InvisibleCut {
                dm_mass: 20.0,
                strength,
            };
            let width = mesons.widths[0] + mesons.invisible_widths(&aligned, &cut)[0];
            let mass = MASSES[0];
            let amplitude =
                gamma_gamma_amplitude(&mesons, mass, &aligned, &NO_TRIANGLES, Some(&cut));
            let expected = 0.5 * mass * mass * a[0] * b[0] / Complex64::new(0.0, mass * width);
            assert!((amplitude - expected).norm() <= 1e-14 * expected.norm());
        }
    }

    #[test]
    fn decoupled_zero_width_pole_drops_out() {
        // An isoscalar source has c_π = 0, so the stable π⁰ pole contributes
        // nothing, at its own mass as on either side of it.
        let isoscalar = [1e-3, 1e-3, 0.0];
        let stable = block(MASSES, [0.0; 3]);
        let mass = MASSES[0];
        let at = gamma_gamma_amplitude(&stable, mass, &isoscalar, &NO_TRIANGLES, None);
        let near = gamma_gamma_amplitude(
            &stable,
            mass * (1.0 + 1e-9),
            &isoscalar,
            &NO_TRIANGLES,
            None,
        );
        assert!(at.is_finite());
        assert!((at - near).norm() <= 1e-8 * at.norm());
        let self_energy = stable.source_self_energy(mass, &isoscalar, None);
        assert!(self_energy.is_finite());
        assert_eq!(stable.mixing_expansion(mass, &isoscalar, 500.0)[0], 0.0);
        // A coupled stable pole is singular.
        let coupled = gamma_gamma_amplitude(&stable, mass, &HYPERCHARGE, &NO_TRIANGLES, None);
        assert!(!coupled.is_finite());
    }

    #[test]
    fn triangle_factor_has_its_limits() {
        // Massless fermions carry the full anomaly.
        assert_eq!(triangle_factor(100.0, 0.0), Complex64::new(1.0, 0.0));
        // A heavy fermion decouples like −(x/3 + 8x²/45), x = s/(4m²); the
        // next term, 4x³/35, is 3e-14 of the sum here.
        let x = 1.0 / 4e6;
        let heavy = triangle_factor(1.0, 1e3);
        assert_close(heavy.re, -(x / 3.0 + 8.0 * x * x / 45.0), 1e-13);
        assert_eq!(heavy.im, 0.0);
        // The series and the closed form meet at x = 1/4 to a few ulps.
        let series = triangle_factor(1.0, 1.0);
        let closed_form = triangle_factor(1.0 + 1e-15, 1.0);
        assert_close(series.re, closed_form.re, 1e-14);
        // 1 − 4 arcsin²(1/2), rounded from mpmath; evaluating it in f64
        // would itself cancel to 2.5e-15.
        assert_close(series.re, -0.096_622_711_232_150_95, 1e-15);
        // Continuity across the pair threshold τ = 1, where f = π²/4.
        let open = triangle_factor(2.0 * (1.0 + 1e-12), 1.0);
        let closed = triangle_factor(2.0 * (1.0 - 1e-12), 1.0);
        assert!((open - closed).norm() < 1e-5);
        assert_close(closed.re, 1.0 - PI * PI / 4.0, 1e-5);
        assert_eq!(triangle_factor(0.0, 1.0), Complex64::new(0.0, 0.0));
    }
}
