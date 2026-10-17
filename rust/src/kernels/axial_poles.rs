//! Coherent longitudinal `π⁰`, `η` and `η′` poles of the neutral axial
//! source, and the `γγ` density they produce.
//!
//! The kernels behind `hazma/form_factors/axial/_poles.py`. The source
//! `Z′_μ q̄ γ^μ γ₅ G_A q` mixes with each neutral pseudoscalar through
//! `−c_P Z′·∂P`, `c_P = f Tr(G_A λ_P)`. That mixing is proportional to the
//! source momentum `k`, so it reaches only the source-longitudinal response,
//! and the Proca propagator's `k^μ k^ν/(s M²)` term turns it into a
//! non-resonant contact rather than a spin-zero mediator (project ADR-0003).
//! The meson block is kept whole: its inverse propagator `D(s)` is a full
//! complex matrix, so an absorptive insertion that couples the mesons, such
//! as the invisible dark-matter cut, is resummed with them rather than
//! added to one diagonal width.
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
/// part adds the rank-one matrix `i ρ(s) κ² c cᵀ`, `ρ = s β_χ/(8π)`, to
/// `D(s)`; the dispersive part renormalizes the pole inputs and is omitted.
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

/// Solve `a x = b` by Gaussian elimination with partial pivoting.
///
/// A zero pivot, which only an exact pole of zero width produces, yields a
/// non-finite solution for the caller to reject.
fn solve(mut a: [[Complex64; 3]; 3], mut b: [Complex64; 3]) -> [Complex64; 3] {
    for k in 0..3 {
        let pivot = (k..3)
            .max_by(|&i, &j| a[i][k].norm().total_cmp(&a[j][k].norm()))
            .unwrap_or(k);
        a.swap(k, pivot);
        b.swap(k, pivot);
        for r in k + 1..3 {
            // Skipping exact zeros keeps a zero pivot from turning the
            // already-decoupled rows of a diagonal block into 0/0.
            if a[r][k] == Complex64::new(0.0, 0.0) {
                continue;
            }
            let factor = a[r][k] / a[k][k];
            let pivot_row = a[k];
            for (entry, pivot_entry) in a[r].iter_mut().zip(pivot_row).skip(k) {
                *entry -= factor * pivot_entry;
            }
            let update = factor * b[k];
            b[r] -= update;
        }
    }
    let mut x = [Complex64::new(0.0, 0.0); 3];
    for k in (0..3).rev() {
        let mut sum = b[k];
        for c in k + 1..3 {
            sum -= a[k][c] * x[c];
        }
        x[k] = sum / a[k][k];
    }
    x
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

    /// Return the inverse meson propagator `D(s)` in MeV² at `s = q²`.
    ///
    /// Without a cut it is `diag(s − m_P² + i m_P Γ_P)`; a cut adds its
    /// rank-one absorptive matrix, which couples all three mesons.
    pub fn inverse_propagator(
        &self,
        q: f64,
        couplings: &[f64; 3],
        cut: Option<&InvisibleCut>,
    ) -> [[Complex64; 3]; 3] {
        let s = q * q;
        let mut d = [[Complex64::new(0.0, 0.0); 3]; 3];
        for (p, row) in d.iter_mut().enumerate() {
            let mass = self.masses[p];
            row[p] = Complex64::new(s - mass * mass, mass * self.widths[p]);
        }
        if let Some(cut) = cut {
            let weight = cut.absorptive_weight(s);
            let y = self.source_couplings(couplings).map(|c| cut.strength * c);
            for (p, row) in d.iter_mut().enumerate() {
                for (r, entry) in row.iter_mut().enumerate() {
                    entry.im += weight * y[p] * y[r];
                }
            }
        }
        d
    }

    /// Return `s aᵀ D(s)⁻¹ b`, the coherent sum over the three poles.
    fn coherent_sum(
        &self,
        q: f64,
        a: &[f64; 3],
        b: &[f64; 3],
        couplings: &[f64; 3],
        cut: Option<&InvisibleCut>,
    ) -> Complex64 {
        let d = self.inverse_propagator(q, couplings, cut);
        let x = solve(d, b.map(|value| Complex64::new(value, 0.0)));
        q * q * (a[0] * x[0] + a[1] * x[1] + a[2] * x[2])
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
        self.coherent_sum(q, &c, &c, couplings, cut)
    }

    /// Return `ε_P = |c_P² s / (M² (s − m_P² + i m_P Γ_P))|` per meson,
    /// dimensionless, with the visible widths only.
    ///
    /// `ε_P` measures the neglected second mixing insertion relative to the
    /// first; it is infinite at an exact pole of zero width.
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

    /// Return `Γ(P → χχ̄) = m_P β_χ κ² c_P² / (8π)` in MeV, the cut's
    /// diagonal at each pole divided by `m_P`; zero for a closed channel.
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

/// Return the triangle form factor `1 − τ f(τ)`, `τ = 4m²/s`, at `s = q²`.
///
/// `f(τ) = arcsin²(1/√τ)` for `τ ≥ 1` and
/// `−¼ [ln((1 + r)/(1 − r)) − iπ]²`, `r = √(1 − τ)`, below. The factor is
/// one for a massless fermion and vanishes like `−s/(12 m²)` as `s → 0`.
pub fn triangle_factor(q: f64, mass: f64) -> Complex64 {
    let s = q * q;
    if s == 0.0 {
        return Complex64::new(0.0, 0.0);
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
    let a = block.source_traces(couplings);
    let b = block.photon_traces();
    let poles = 0.5 * block.coherent_sum(q, &a, &b, couplings, cut);
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
    fn invisible_cut_adds_its_width_to_each_pole_once() {
        let mesons = block(MASSES, WIDTHS);
        let cut = InvisibleCut {
            dm_mass: 60.0,
            strength: 2.0 * 60.0 / 2500.0f64.powi(2),
        };
        let gamma = mesons.invisible_widths(&HYPERCHARGE, &cut);
        let c = mesons.source_couplings(&HYPERCHARGE);
        for p in 0..3 {
            let mass = MASSES[p];
            let d = mesons.inverse_propagator(mass, &HYPERCHARGE, Some(&cut));
            assert_close(d[p][p].im, mass * (WIDTHS[p] + gamma[p]), 1e-15);
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
    fn solve_handles_a_coupled_block() {
        let mesons = block(MASSES, WIDTHS);
        let cut = InvisibleCut {
            dm_mass: 1.0,
            strength: 1e-2,
        };
        let d = mesons.inverse_propagator(MASSES[0], &HYPERCHARGE, Some(&cut));
        let b = [1.0, -2.0, 0.5].map(|v| Complex64::new(v, 0.0));
        let x = solve(d, b);
        for (row, expected) in d.iter().zip(b) {
            let residual = row[0] * x[0] + row[1] * x[1] + row[2] * x[2] - expected;
            assert!(residual.norm() < 1e-12, "{residual}");
        }
    }

    #[test]
    fn triangle_factor_has_its_limits() {
        // Massless fermions carry the full anomaly.
        assert_eq!(triangle_factor(100.0, 0.0), Complex64::new(1.0, 0.0));
        // A heavy fermion decouples like −s/(12 m²); the next term is
        // −s²/(90 m⁴), so the ratio is 1 + O(s/m²).
        let heavy = triangle_factor(1.0, 1e3);
        assert_close(heavy.re, -1.0 / 12e6, 1e-6);
        assert_eq!(heavy.im, 0.0);
        // Continuity across the pair threshold τ = 1, where f = π²/4.
        let open = triangle_factor(2.0 * (1.0 + 1e-12), 1.0);
        let closed = triangle_factor(2.0 * (1.0 - 1e-12), 1.0);
        assert!((open - closed).norm() < 1e-5);
        assert_close(closed.re, 1.0 - PI * PI / 4.0, 1e-5);
        assert_eq!(triangle_factor(0.0, 1.0), Complex64::new(0.0, 0.0));
    }
}
