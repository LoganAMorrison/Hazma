//! Annihilation rates and fermion-pair densities of the axial-vector
//! mediator.
//!
//! The kernels behind `hazma/axial_vector_mediator/`. Every final state
//! enters through its transverse and source-longitudinal densities `A_T`
//! and `A_L`, in MeV², with the normalization of the visible-axial-spectra
//! project's ADR-0001, and the mediator note's master formula
//! (`eq:ax-master`) turns them into `σ v_Møller` per `χχ̄` pair:
//!
//! `σ v = [g_V²(s + 2m²) + g_A²(s − 4m²)] A_T/(s |s − M² + iMΓ|²)
//!        + 2 m² g_A² A_L/(s M⁴)`.
//!
//! The Proca propagator splits into the transverse projector over
//! `s − M²` and the contact `k^μ k^ν/(s M²)`, and only the first has a
//! pole, so the width dresses the transverse term alone (project ADR-0003).
//! The physical width into a final state is `A_T(M²)/(2M)`. Masses, energies
//! and widths are in MeV, rates in MeV⁻². The caller validates every input.

use std::f64::consts::PI;

/// The Dirac dark matter and the mediator that carries its annihilation.
#[derive(Clone, Copy, Debug)]
pub struct DarkSector {
    /// Dark-matter mass `m` in MeV, positive.
    pub dm_mass: f64,
    /// Dark-matter vector coupling `g_V^χ`.
    pub dm_vector: f64,
    /// Dark-matter axial coupling `g_A^χ`.
    pub dm_axial: f64,
    /// Mediator mass `M` in MeV, positive.
    pub mediator_mass: f64,
    /// Total mediator width `Γ` in MeV, entering as `i M Γ`.
    pub mediator_width: f64,
}

impl DarkSector {
    /// Return `σ v_Møller` in MeV⁻² for a final state with densities
    /// `(A_T, A_L)` at invariant mass `q`.
    ///
    /// Zero below the pair threshold `q < 2m`. A vanishing resonant
    /// numerator, such as a final state with no transverse density, gives
    /// no resonant term, so only a nonzero one is infinite on a pole of zero
    /// width.
    pub fn sigma_v(&self, q: f64, transverse: f64, longitudinal: f64) -> f64 {
        let m = self.dm_mass;
        if q < 2.0 * m {
            return 0.0;
        }
        let s = q * q;
        let big = self.mediator_mass;
        // Both differences are factored so that neither cancels near its
        // zero.
        let above_threshold = (q - 2.0 * m) * (q + 2.0 * m);
        let dm_factor =
            self.dm_vector.powi(2) * (s + 2.0 * m * m) + self.dm_axial.powi(2) * above_threshold;
        let numerator = dm_factor * transverse;
        let resonant = if numerator == 0.0 {
            0.0
        } else {
            let off_shell = (q - big) * (q + big);
            let propagator = off_shell * off_shell + (big * self.mediator_width).powi(2);
            numerator / (s * propagator)
        };
        let contact = 2.0 * (m * self.dm_axial).powi(2) * longitudinal / (s * big.powi(4));
        resonant + contact
    }
}

/// Return the centre-of-mass Møller velocity `2 sqrt(1 − 4m²/q²)` of a
/// dark-matter pair, dimensionless; zero at and below threshold.
pub fn moller_velocity(q: f64, dm_mass: f64) -> f64 {
    if q <= 2.0 * dm_mass {
        return 0.0;
    }
    2.0 * ((q - 2.0 * dm_mass) * (q + 2.0 * dm_mass)).sqrt() / q
}

/// Return the cross section `σ = σ v_Møller / v_Møller` in MeV⁻².
///
/// A zero rate is a zero cross section at any velocity; a nonzero rate at
/// exact threshold, an s-wave, is an infinite one.
pub fn cross_section(q: f64, sigma_v: f64, dm_mass: f64) -> f64 {
    if sigma_v == 0.0 {
        return 0.0;
    }
    sigma_v / moller_velocity(q, dm_mass)
}

/// A fermion pair `f f̄` coupled as `Z′_μ f̄ γ^μ (g_V + g_A γ₅) f`.
#[derive(Clone, Copy, Debug)]
pub struct FermionPair {
    /// Fermion mass `m_f` in MeV, nonnegative.
    pub mass: f64,
    /// Vector coupling `g_V^f`.
    pub vector: f64,
    /// Axial coupling `g_A^f`.
    pub axial: f64,
    /// Number of colors `N`.
    pub colors: f64,
}

impl FermionPair {
    /// Return the tree-level `(A_T, A_L)` in MeV² at invariant mass `q`.
    ///
    /// `A_T = N β [g_V²(s + 2m_f²) + g_A² β² s]/(6π)` and
    /// `A_L = N g_A² m_f² β/π`, with `β = sqrt(1 − 4m_f²/s)`; the vector
    /// current is conserved, so only the axial mass term reaches `A_L`.
    /// Both vanish at and below threshold.
    pub fn densities(&self, q: f64) -> (f64, f64) {
        let m = self.mass;
        if q <= 2.0 * m {
            return (0.0, 0.0);
        }
        let s = q * q;
        let beta_squared = (q - 2.0 * m) * (q + 2.0 * m) / s;
        let beta = beta_squared.sqrt();
        let transverse = self.colors * beta / (6.0 * PI)
            * (self.vector.powi(2) * (s + 2.0 * m * m) + self.axial.powi(2) * beta_squared * s);
        let longitudinal = self.colors * (self.axial * m).powi(2) * beta / PI;
        (transverse, longitudinal)
    }

    /// Return the transverse density weighted by the fermion's helicity,
    /// `N β² s g_V g_A/(3π)` in MeV², at invariant mass `q`.
    ///
    /// The opposite-helicity amplitudes are `g_V ± β g_A`, with the upper
    /// sign for a right-handed fermion, and the equal-helicity ones carry
    /// no net helicity, so over [`Self::densities`]' `A_T` this is the
    /// fermion's mean helicity. The antifermion's is its negative. The
    /// source-longitudinal pair is a pseudoscalar of equal helicities and
    /// adds nothing. It vanishes at and below threshold.
    pub fn helicity_density(&self, q: f64) -> f64 {
        let m = self.mass;
        if q <= 2.0 * m {
            return 0.0;
        }
        let beta_squared = (q - 2.0 * m) * (q + 2.0 * m) / (q * q);
        self.colors * beta_squared * q * q * self.vector * self.axial / (3.0 * PI)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MUON: f64 = 105.6583745;

    fn assert_close(actual: f64, expected: f64, rtol: f64) {
        let error = (actual - expected).abs() / expected.abs();
        assert!(
            error <= rtol,
            "actual {actual:e}, expected {expected:e}, relative error {error:e} > {rtol:e}"
        );
    }

    fn pair(mass: f64, vector: f64, axial: f64) -> FermionPair {
        FermionPair {
            mass,
            vector,
            axial,
            colors: 1.0,
        }
    }

    /// `−g_{αβ} ∫ dΦ H^α H^β* = 3A_T − A_L` from the Dirac traces alone:
    /// `g_V² β (s + 2m²)/(2π) + g_A² β (s − 6m²)/(2π)`. Roundoff only, so
    /// 1e-14.
    #[test]
    fn densities_reproduce_the_metric_trace() {
        let (q, gv, ga) = (700.0, 0.3, -0.7);
        let (t, l) = pair(MUON, gv, ga).densities(q);
        let s = q * q;
        let m2 = MUON * MUON;
        let beta = (1.0 - 4.0 * m2 / s).sqrt();
        let trace = beta * (gv * gv * (s + 2.0 * m2) + ga * ga * (s - 6.0 * m2)) / (2.0 * PI);
        assert_close(3.0 * t - l, trace, 1e-14);
    }

    /// A massless vector pair has width `g² M/(12π)` and no `A_L`.
    #[test]
    fn massless_width_is_the_textbook_value() {
        let (t, l) = pair(0.0, 0.4, 0.0).densities(300.0);
        assert_close(t / (2.0 * 300.0), 0.16 * 300.0 / (12.0 * PI), 1e-15);
        assert_eq!(l, 0.0);
    }

    #[test]
    fn densities_vanish_at_and_below_threshold() {
        let fermion = pair(MUON, 1.0, 1.0);
        assert_eq!(fermion.densities(2.0 * MUON), (0.0, 0.0));
        assert_eq!(fermion.densities(MUON), (0.0, 0.0));
        assert_eq!(fermion.helicity_density(2.0 * MUON), 0.0);
    }

    /// Massless, the mean helicity is `(g_R² − g_L²)/(g_R² + g_L²)` with
    /// `g_R, g_L = g_V ± g_A`, and the neutrino's `g_A = −g_V` is purely
    /// left-handed. Roundoff only.
    #[test]
    fn massless_helicity_is_the_chiral_asymmetry() {
        let (gv, ga) = (0.3, -0.7);
        let fermion = pair(0.0, gv, ga);
        let (right, left) = ((gv + ga).powi(2), (gv - ga).powi(2));
        let helicity = fermion.helicity_density(500.0) / fermion.densities(500.0).0;
        assert_close(helicity, (right - left) / (right + left), 1e-14);
        let neutrino = pair(0.0, 0.25, -0.25);
        let helicity = neutrino.helicity_density(500.0) / neutrino.densities(500.0).0;
        assert_close(helicity, -1.0, 1e-14);
    }

    fn sector(width: f64) -> DarkSector {
        DarkSector {
            dm_mass: 150.0,
            dm_vector: 0.2,
            dm_axial: 0.9,
            mediator_mass: 500.0,
            mediator_width: width,
        }
    }

    /// At zero width the split agrees with the unsplit Proca propagator
    /// `(g − kk/M²)/(s − M²)`, whose longitudinal numerator carries
    /// `(1 − s/M²)²`. Roundoff in a few products, so 1e-13.
    #[test]
    fn zero_width_split_matches_the_unsplit_propagator() {
        let dark = sector(0.0);
        let (q, t, l) = (420.0, 3.0e3, 7.0e2);
        let (s, m, big) = (q * q, dark.dm_mass, dark.mediator_mass);
        let (gv, ga) = (dark.dm_vector, dark.dm_axial);
        let trace = gv * gv * (s + 2.0 * m * m) + ga * ga * (s - 6.0 * m * m);
        let k_l_k = 2.0 * m * m * ga * ga;
        let unsplit = (trace * t + k_l_k * (t + l * (1.0 - s / (big * big)).powi(2)))
            / (s * (s - big * big).powi(2));
        assert_close(dark.sigma_v(q, t, l), unsplit, 1e-13);
    }

    /// Pure axial dark matter into a fermion pair at threshold is the
    /// s-wave `g_A^χ² g_A^f² m_f² β_f/(2π M⁴)` at any width.
    #[test]
    fn axial_s_wave_at_threshold() {
        let mut dark = sector(40.0);
        dark.dm_vector = 0.0;
        let q = 2.0 * dark.dm_mass;
        let (t, l) = pair(MUON, 0.5, 0.8).densities(q);
        let beta = (1.0 - 4.0 * MUON * MUON / (q * q)).sqrt();
        let expected = 0.81 * 0.64 * MUON * MUON * beta / (2.0 * PI * 500f64.powi(4));
        assert_close(dark.sigma_v(q, t, l), expected, 1e-14);
    }

    /// The width enters the transverse term only, so a purely longitudinal
    /// final state is unchanged by it and finite on the pole.
    #[test]
    fn width_leaves_the_contact_alone() {
        let q = 500.0;
        let narrow = sector(0.0).sigma_v(q, 0.0, 5.0e2);
        let broad = sector(80.0).sigma_v(q, 0.0, 5.0e2);
        assert!(narrow.is_finite() && narrow > 0.0);
        assert_eq!(narrow, broad);
    }

    /// On a pole of zero width a vanishing numerator stays zero, here pure
    /// axial dark matter at threshold `M = 2m`, and a nonzero one is infinite.
    #[test]
    fn zero_width_pole() {
        let mut dark = sector(0.0);
        dark.dm_vector = 0.0;
        dark.mediator_mass = 2.0 * dark.dm_mass;
        assert_eq!(dark.sigma_v(2.0 * dark.dm_mass, 1.0e3, 0.0), 0.0);
        dark.dm_vector = 0.2;
        assert_eq!(dark.sigma_v(2.0 * dark.dm_mass, 1.0e3, 0.0), f64::INFINITY);
    }

    #[test]
    fn rate_vanishes_below_threshold() {
        assert_eq!(sector(1.0).sigma_v(299.0, 1.0, 1.0), 0.0);
    }

    #[test]
    fn cross_section_divides_by_the_moller_velocity() {
        let (q, m, sv) = (400.0, 150.0, 2.5e-12);
        let v = moller_velocity(q, m);
        assert_close(v, 2.0 * (1.0 - 4.0 * m * m / (q * q)).sqrt(), 1e-15);
        assert_close(cross_section(q, sv, m) * v, sv, 1e-15);
        assert_eq!(cross_section(2.0 * m, sv, m), f64::INFINITY);
        assert_eq!(cross_section(2.0 * m, 0.0, m), 0.0);
    }
}
