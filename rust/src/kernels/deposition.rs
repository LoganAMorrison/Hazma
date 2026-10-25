//! The CMB energy-deposition fold of a lab-frame injection spectrum.
//!
//! The kernel behind `AxialVectorMediator.mediator_pair_f_eff`: it
//! integrates `E dN/dE f(E − m)` over the lab energy `E` of one injected
//! species of mass `m`, with `f` a deposition-efficiency table in kinetic
//! energy (arXiv:1506.03811). It folds the way
//! `hazma.theory._theory_cmb.TheoryCMB.f_eff` folds the annihilation
//! spectra, so the two efficiencies add: the table is linear in energy
//! between its nodes and constant beyond them, as `hazma.cmb.f_eff_ep` is
//! read with its floor, and the caller sets the lower limit, which is the
//! table's first node for photons and the rest mass for electrons.

use crate::interp::interp;

use super::polarized_muon::integrate;

/// Relative tolerance of the fold, whose integrand is itself a decay
/// kernel's integral at a tighter tolerance.
const DEPOSITION_EPSREL: f64 = 1e-8;

/// A deposition-efficiency table: `fractions[i]` of an injected
/// particle's energy reaches the plasma at kinetic energy
/// `kinetic_energies[i]` in MeV, which ascend.
#[derive(Clone, Copy, Debug)]
pub struct Efficiency<'a> {
    /// Kinetic energies in MeV, strictly ascending.
    pub kinetic_energies: &'a [f64],
    /// Deposited fractions, one per energy.
    pub fractions: &'a [f64],
}

impl Efficiency<'_> {
    /// Return the fraction at `kinetic_energy` in MeV.
    fn at(&self, kinetic_energy: f64) -> f64 {
        interp(kinetic_energy, self.kinetic_energies, self.fractions)
    }
}

/// Return `∫ E dN/dE f(E − rest_mass) dE` in MeV over `[lower, upper]`.
///
/// `dnde` is the injected density in MeV⁻¹ at a lab energy in MeV, and
/// `edges` the energies where it jumps, such as a boosted box's ends. The
/// integral also breaks at every table node, where `f` kinks.
pub fn deposited_energy<F>(
    mut dnde: F,
    lower: f64,
    upper: f64,
    rest_mass: f64,
    efficiency: &Efficiency<'_>,
    edges: &[f64],
) -> f64
where
    F: FnMut(f64) -> f64,
{
    if upper <= lower {
        return 0.0;
    }
    let mut points: Vec<f64> = efficiency
        .kinetic_energies
        .iter()
        .map(|kinetic| kinetic + rest_mass)
        .chain(edges.iter().copied())
        .collect();
    points.retain(|&point| lower < point && point < upper);
    let mut integrand = |energy: f64| energy * dnde(energy) * efficiency.at(energy - rest_mass);
    integrate(&mut integrand, lower, upper, &points, DEPOSITION_EPSREL)
}

#[cfg(test)]
mod tests {
    use super::*;

    const ENERGIES: [f64; 4] = [0.01, 1.0, 10.0, 100.0];
    const FRACTIONS: [f64; 4] = [0.9, 0.7, 0.4, 0.2];
    const TABLE: Efficiency<'static> = Efficiency {
        kinetic_energies: &ENERGIES,
        fractions: &FRACTIONS,
    };

    /// `∫ E f(E − m) dE` over `[a, b]` inside one linear table cell, the
    /// closed form of a flat unit density.
    fn cell_moment(a: f64, b: f64, rest_mass: f64, cell: usize) -> f64 {
        let (k0, k1) = (ENERGIES[cell], ENERGIES[cell + 1]);
        let (f0, f1) = (FRACTIONS[cell], FRACTIONS[cell + 1]);
        let slope = (f1 - f0) / (k1 - k0);
        let intercept = f0 - slope * (k0 + rest_mass);
        let primitive = |e: f64| intercept * e * e / 2.0 + slope * e * e * e / 3.0;
        primitive(b) - primitive(a)
    }

    #[test]
    fn a_box_inside_one_cell_folds_to_the_closed_form() {
        let (rest_mass, a, b) = (0.5, 2.0, 7.0);
        let height = 1.0 / (b - a);
        let dnde = |e: f64| if (a..b).contains(&e) { height } else { 0.0 };
        let folded = deposited_energy(dnde, rest_mass, 50.0, rest_mass, &TABLE, &[a, b]);
        let exact = height * cell_moment(a, b, rest_mass, 1);
        assert!((folded / exact - 1.0).abs() < 1e-12, "{folded} vs {exact}");
    }

    #[test]
    fn a_narrow_box_tends_to_its_line() {
        // A box of width w about E folds to E f(E − m) up to O(w²) of the
        // linear cell, whose curvature is only the E weight's.
        let (rest_mass, center) = (0.0, 30.0);
        for width in [1e-2, 1e-3] {
            let (a, b) = (center - width / 2.0, center + width / 2.0);
            let dnde = |e: f64| {
                if (a..b).contains(&e) {
                    1.0 / width
                } else {
                    0.0
                }
            };
            let folded = deposited_energy(dnde, 0.01, 200.0, rest_mass, &TABLE, &[a, b]);
            let line = center * TABLE.at(center);
            assert!(
                (folded / line - 1.0).abs() < width * width,
                "{width}: {folded}"
            );
        }
    }

    #[test]
    fn kinetic_energies_below_the_table_take_its_first_fraction() {
        let rest_mass = 0.511;
        let (a, b) = (rest_mass, rest_mass + 0.005);
        let dnde = |e: f64| if (a..b).contains(&e) { 1.0 } else { 0.0 };
        let folded = deposited_energy(dnde, rest_mass, 1.0, rest_mass, &TABLE, &[b]);
        let exact = FRACTIONS[0] * (b * b - a * a) / 2.0;
        assert!((folded / exact - 1.0).abs() < 1e-12, "{folded} vs {exact}");
    }

    #[test]
    fn an_empty_support_deposits_nothing() {
        let folded = deposited_energy(|_| 1.0, 5.0, 5.0, 0.0, &TABLE, &[]);
        assert_eq!(folded, 0.0);
    }
}
