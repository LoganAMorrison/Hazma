//! The seven tabulated photon spectra, ported from
//! `hazma/spectra/_photon/{_kaon,_eta,_eta_prime,_omega,_phi}.pyx`.
//!
//! PyO3-free (`projects/cython-to-rust/rules.md`, Rust conventions
//! rule 3); [`crate::photon`] is the Python-visible half.
//!
//! # One implementation, seven parameterisations
//!
//! The five `.pyx` files were near-copies of each other — the charged,
//! long and short kaons already shared a helper, and the other four
//! repeated it with the names changed. Everything that actually differs
//! between them is data, so this module has one [`dnde`] and seven
//! [`Spectrum`] values:
//!
//! | Entry point | Table | Parent mass | Lines |
//! | --- | --- | --- | --- |
//! | `dnde_photon_charged_kaon` | `charged_kaon_photon.csv` | [`pdg::MASS_K`] | — |
//! | `dnde_photon_long_kaon` | `long_kaon_photon.csv` | [`pdg::MASS_K0`] | `K_L → γγ` |
//! | `dnde_photon_short_kaon` | `short_kaon_photon.csv` | [`pdg::MASS_K0`] | `K_S → γγ` |
//! | `dnde_photon_eta` | `eta_photon.csv` | [`pdg::MASS_ETA`] | `η → γγ` |
//! | `dnde_photon_eta_prime` | `eta_prime_photon.csv` | [`pdg::MASS_ETAP`] | `η′ → γγ` |
//! | `dnde_photon_omega` | `omega_photon.csv` | [`pdg::MASS_OMEGA`] | `ω → π⁰γ`, `ω → ηγ` |
//! | `dnde_photon_phi` | `phi_photon.csv` | [`pdg::MASS_PHI`] | `φ → ηγ`, `φ → η′γ` |
//!
//! Both kaon flavours take their threshold and boost from
//! [`pdg::MASS_K0`], not from `MASS_KL` / `MASS_KS` — the Cython does,
//! and all three constants hold the same number anyway.
//!
//! # The tables
//!
//! Each CSV is one energy column followed by one column per decay mode;
//! the rest-frame spectrum is their sum. The Cython loaded them with
//! `np.loadtxt(...).T` at **import time**, so importing
//! `hazma.spectra._photon` cost seven file reads and seven NumPy parses
//! whether or not a spectrum was ever evaluated. Here they are
//! [`include_str!`]-ed into the shared object and parsed once, lazily, on
//! first use. The CSVs stay in the repository as the source of truth —
//! `test/parity/generate.py` hashes them into the corpus's kernel digest,
//! and `test/test_core_photon_tables.py` re-parses them with NumPy and
//! compares.
//!
//! Two details of that parse are load-bearing rather than incidental:
//!
//! * the per-row sum reproduces `numpy.sum(axis=0)`, which is **pairwise**
//!   above eight terms. Only `phi_photon.csv` has enough modes (ten) to
//!   reach that path, and a sequential sum differs from NumPy there — so
//!   [`crate::boost::pairwise_sum`] is reused rather than a fold;
//! * `f64::from_str` and NumPy's CSV reader are both correctly rounded, so
//!   the two agree bit-for-bit. That is asserted, not assumed.
//!
//! # Why `mul_add`, and where it is *not* used
//!
//! One multiply-add per line term is written `delta.mul_add(weight, res)`
//! rather than `res + weight * delta`, for the reason
//! [`crate::boost`] documents at length: clang contracts `a * b + c` into
//! a fused multiply-add by default and the parity corpus was captured
//! from a macOS/arm64 build that does. Disassembling the five shipped
//! `.so` files finds exactly eight FMA instructions between them — one
//! per `res += <branching ratio> * boost_delta_function(...)`, all of the
//! form `fmadd d0, d8, d0, d9` — and no others. In particular the
//! rest-frame tail `dnde[0] * emin / photon_energy` is a multiply
//! followed by a divide, which cannot contract, and every other
//! multiply-add in the live path belongs to [`crate::boost`] or
//! [`crate::interp`], which are separate extensions the Cython reaches
//! through `__pyx_capi__` function pointers and so does not inline.
//!
//! # Constant folding
//!
//! `hazma/_utils/constants.pxd` states its masses and branching ratios as
//! Cython `DEF`s, so every combination the `.pyx` writes inline — `2·BR`,
//! `M/2`, `(M² − m²)/(2M)` — is folded into a single immediate in the
//! generated code. They are `const` here for the same reason, and
//! [`tests::every_folded_constant_is_the_shipped_immediate_or_its_declared_repair`]
//! pins each one against the immediate the disassembly loads — bar the
//! three the repairs move ([`ETAP_TO_A_A_WEIGHT`],
//! [`PHI_TO_ETA_A_ENERGY`] and [`PHI_TO_ETAP_A_ENERGY`]), which are
//! pinned against the repaired value with the shipped immediate named
//! beside it.

use std::sync::LazyLock;

use crate::boost::{self, BoostError};
use crate::constants::pdg;
use crate::interp;

// ===========================================================================
// ---- Folded constants -----------------------------------------------------
// ===========================================================================

/// `2·BR(η → γγ)`, the weight of the η's two-photon line.
///
/// The factor of two is the Cython's: two photons per decay, so the line
/// carries twice the branching ratio (`_eta.pyx:99`).
const ETA_TO_A_A_WEIGHT: f64 = 2.0 * pdg::BR_ETA_TO_A_A;
/// `2·BR(η′ → γγ)`, the weight of the η′'s two-photon line.
///
/// The factor of two is this port's, not the Cython's: `_eta_prime.pyx:107`
/// omitted it where its three `X → γγ` siblings above carry it, so 2.1.0
/// shipped 0.02307 photons per decay from this mode instead of 0.04614.
/// The parity corpus still pins the low value, and
/// `test/parity/deltas.py` declares the resulting shift as repair `B1`
/// rather than re-pinning the arrays. See
/// [`tests::every_two_photon_line_carries_twice_its_branching_ratio`].
const ETAP_TO_A_A_WEIGHT: f64 = 2.0 * pdg::BR_ETAP_TO_A_A;
/// `2·BR(K_L → γγ)` (`_kaon.pyx:300`).
const KL_TO_A_A_WEIGHT: f64 = 2.0 * pdg::BR_KL_TO_A_A;
/// `2·BR(K_S → γγ)` (`_kaon.pyx:407`).
const KS_TO_A_A_WEIGHT: f64 = 2.0 * pdg::BR_KS_TO_A_A;

/// Photon energy from `η → γγ` in the η rest frame, MeV.
const ETA_TO_A_A_ENERGY: f64 = pdg::MASS_ETA / 2.0;
/// Photon energy from `η′ → γγ` in the η′ rest frame, MeV.
const ETAP_TO_A_A_ENERGY: f64 = pdg::MASS_ETAP / 2.0;
/// Photon energy from `K⁰ → γγ` in the kaon rest frame, MeV. Shared by
/// the long and short kaons, which the Cython gives the same mass.
const K0_TO_A_A_ENERGY: f64 = pdg::MASS_K0 / 2.0;
/// Photon energy in the two-body decay `X → Y γ`, in X's rest frame, MeV.
///
/// `(M² − m²) / (2 M)`. Its counterpart `(M² + m²) / (2 M)` is the
/// *daughter meson's* energy, and the two sum to `M`; writing the four
/// line energies below through one function is what keeps the pair from
/// being confusable, which is how the two φ lines came to be placed at
/// the meson's energy in 2.1.0 (repair `B2`).
///
/// The operation order is the `.pyx` expressions' own, so the folded
/// constants stay bit-identical to the shipped object code where the
/// energy is unrepaired — see
/// [`tests::every_folded_constant_is_the_shipped_immediate_or_its_declared_repair`].
const fn photon_line_energy(parent: f64, daughter: f64) -> f64 {
    (parent * parent - daughter * daughter) / (2.0 * parent)
}

/// Photon energy from `ω → π⁰γ` in the ω rest frame, MeV.
const OMEGA_TO_PI0_A_ENERGY: f64 = photon_line_energy(pdg::MASS_OMEGA, pdg::MASS_PI0);
/// Photon energy from `ω → ηγ` in the ω rest frame, MeV.
const OMEGA_TO_ETA_A_ENERGY: f64 = photon_line_energy(pdg::MASS_OMEGA, pdg::MASS_ETA);
/// Photon energy from `φ → ηγ` in the φ rest frame, MeV.
///
/// The `−` is this port's, not the Cython's: `_phi.pyx:111` wrote
/// `(M_φ² + M_η²) / (2 M_φ)`, the *η's* energy in the two-body decay
/// rather than the photon's, and fed it to the boost as a photon energy.
/// So 2.1.0 shipped this line at **656.94 MeV where 362.52 belongs**, a
/// factor of 1.81 and above the spectrum's own endpoint. The parity
/// corpus still pins the misplaced line, and `test/parity/deltas.py`
/// declares the relocation as repair `B2` rather than re-pinning the
/// arrays. See [`tests::every_line_energy_is_the_photons_not_the_daughters`].
const PHI_TO_ETA_A_ENERGY: f64 = photon_line_energy(pdg::MASS_PHI, pdg::MASS_ETA);
/// Photon energy from `φ → η′γ` in the φ rest frame, MeV.
///
/// The same repair as [`PHI_TO_ETA_A_ENERGY`] and a far larger move,
/// because the η′ takes almost all of the φ's mass: 2.1.0 shipped this
/// line at **959.65 MeV where 59.82 belongs**, a factor of 16.0 and 94%
/// of the φ's own rest mass carried off by one photon.
const PHI_TO_ETAP_A_ENERGY: f64 = photon_line_energy(pdg::MASS_PHI, pdg::MASS_ETAP);

// ===========================================================================
// ---- Embedded tables ------------------------------------------------------
// ===========================================================================

/// `hazma/spectra/_photon/data/charged_kaon_photon.csv`.
const CHARGED_KAON_CSV: &str =
    include_str!("../../../hazma/spectra/_photon/data/charged_kaon_photon.csv");
/// `hazma/spectra/_photon/data/long_kaon_photon.csv`.
const LONG_KAON_CSV: &str =
    include_str!("../../../hazma/spectra/_photon/data/long_kaon_photon.csv");
/// `hazma/spectra/_photon/data/short_kaon_photon.csv`.
const SHORT_KAON_CSV: &str =
    include_str!("../../../hazma/spectra/_photon/data/short_kaon_photon.csv");
/// `hazma/spectra/_photon/data/eta_photon.csv`.
const ETA_CSV: &str = include_str!("../../../hazma/spectra/_photon/data/eta_photon.csv");
/// `hazma/spectra/_photon/data/eta_prime_photon.csv`.
const ETA_PRIME_CSV: &str =
    include_str!("../../../hazma/spectra/_photon/data/eta_prime_photon.csv");
/// `hazma/spectra/_photon/data/omega_photon.csv`.
const OMEGA_CSV: &str = include_str!("../../../hazma/spectra/_photon/data/omega_photon.csv");
/// `hazma/spectra/_photon/data/phi_photon.csv`.
const PHI_CSV: &str = include_str!("../../../hazma/spectra/_photon/data/phi_photon.csv");

// ===========================================================================
// ---- Types ----------------------------------------------------------------
// ===========================================================================

/// A rest-frame photon spectrum, tabulated on an ascending energy grid.
pub struct Table {
    /// Rest-frame photon energies, MeV, ascending.
    energies: Vec<f64>,
    /// `dN/dE` in MeV⁻¹ at each energy — the row's decay-mode columns
    /// summed.
    dnde: Vec<f64>,
}

impl Table {
    /// Parse one of the shipped CSVs.
    ///
    /// Lines beginning with `#` are comments, as `numpy.loadtxt` treats
    /// them. Field 0 is the energy and the rest are per-mode spectra,
    /// summed with [`boost::pairwise_sum`] so the result is bit-equal to
    /// the `numpy.sum(data[1:], axis=0)` the Cython computed.
    ///
    /// # Panics
    ///
    /// Panics on a malformed table — a field that is not a float, a row
    /// with no columns after the energy, a ragged file, or no data rows
    /// at all. The input is compiled into the shared object, so any of
    /// these is a build-time defect rather than a reachable input, and
    /// [`tests::every_table_parses_to_an_ascending_grid`] is what keeps
    /// it that way.
    fn parse(csv: &str) -> Self {
        let mut energies = Vec::new();
        let mut dnde = Vec::new();
        let mut components = Vec::new();

        for line in csv.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            components.clear();
            let mut fields = line.split(',').map(|field| {
                field
                    .trim()
                    .parse::<f64>()
                    .unwrap_or_else(|_| panic!("spectrum table holds a non-numeric field: {field}"))
            });
            let energy = fields.next().expect("split always yields one field");
            components.extend(fields);
            assert!(
                !components.is_empty(),
                "spectrum table row {energy} has no decay-mode columns"
            );
            energies.push(energy);
            dnde.push(boost::pairwise_sum(&components));
        }

        assert!(!energies.is_empty(), "spectrum table has no data rows");
        Self { energies, dnde }
    }

    /// The table's lowest energy, MeV.
    fn emin(&self) -> f64 {
        self.energies[0]
    }

    /// The table's highest energy, MeV.
    fn emax(&self) -> f64 {
        self.energies[self.energies.len() - 1]
    }

    /// The rest-frame spectrum `dN/dE` in MeV⁻¹ at `photon_energy`.
    ///
    /// Three regimes, all the Cython's (`_eta.pyx:35-44`): zero above the
    /// table, a `1/E` extrapolation below it — anchored so the tail
    /// matches the first tabulated value — and linear interpolation
    /// inside. A `NaN` argument fails both comparisons and reaches
    /// [`interp::interp`], which propagates it.
    fn rest_frame(&self, photon_energy: f64) -> f64 {
        if photon_energy > self.emax() {
            return 0.0;
        }
        if photon_energy < self.emin() {
            return self.dnde[0] * self.emin() / photon_energy;
        }
        interp::interp(photon_energy, &self.energies, &self.dnde)
    }
}

/// A monochromatic photon line the parent emits on top of its continuum.
struct Line {
    /// The line's rest-frame photon energy, MeV.
    energy: f64,
    /// What the line contributes per decay — a branching ratio, doubled
    /// where the mode yields two photons.
    weight: f64,
}

/// Everything one tabulated photon spectrum needs: its table, its parent's
/// mass, and its lines.
pub struct Spectrum {
    table: Table,
    /// The parent's mass in MeV — the decay threshold and the boost's
    /// second argument.
    pub mass: f64,
    lines: Vec<Line>,
}

impl Spectrum {
    fn new(csv: &str, mass: f64, lines: Vec<Line>) -> Self {
        Self {
            table: Table::parse(csv),
            mass,
            lines,
        }
    }
}

/// The photon spectrum from charged-kaon decay. No line term: the Cython
/// gives `K± → γγ` nothing, which is right — the mode does not exist.
pub static CHARGED_KAON: LazyLock<Spectrum> =
    LazyLock::new(|| Spectrum::new(CHARGED_KAON_CSV, pdg::MASS_K, Vec::new()));

/// The photon spectrum from long-kaon decay.
pub static LONG_KAON: LazyLock<Spectrum> = LazyLock::new(|| {
    Spectrum::new(
        LONG_KAON_CSV,
        pdg::MASS_K0,
        vec![Line {
            energy: K0_TO_A_A_ENERGY,
            weight: KL_TO_A_A_WEIGHT,
        }],
    )
});

/// The photon spectrum from short-kaon decay.
pub static SHORT_KAON: LazyLock<Spectrum> = LazyLock::new(|| {
    Spectrum::new(
        SHORT_KAON_CSV,
        pdg::MASS_K0,
        vec![Line {
            energy: K0_TO_A_A_ENERGY,
            weight: KS_TO_A_A_WEIGHT,
        }],
    )
});

/// The photon spectrum from η decay.
pub static ETA: LazyLock<Spectrum> = LazyLock::new(|| {
    Spectrum::new(
        ETA_CSV,
        pdg::MASS_ETA,
        vec![Line {
            energy: ETA_TO_A_A_ENERGY,
            weight: ETA_TO_A_A_WEIGHT,
        }],
    )
});

/// The photon spectrum from η′ decay.
pub static ETA_PRIME: LazyLock<Spectrum> = LazyLock::new(|| {
    Spectrum::new(
        ETA_PRIME_CSV,
        pdg::MASS_ETAP,
        vec![Line {
            energy: ETAP_TO_A_A_ENERGY,
            weight: ETAP_TO_A_A_WEIGHT,
        }],
    )
});

/// The photon spectrum from ω decay.
pub static OMEGA: LazyLock<Spectrum> = LazyLock::new(|| {
    Spectrum::new(
        OMEGA_CSV,
        pdg::MASS_OMEGA,
        vec![
            Line {
                energy: OMEGA_TO_PI0_A_ENERGY,
                weight: pdg::BR_OMEGA_TO_PI0_A,
            },
            Line {
                energy: OMEGA_TO_ETA_A_ENERGY,
                weight: pdg::BR_OMEGA_TO_ETA_A,
            },
        ],
    )
});

/// The photon spectrum from φ decay.
pub static PHI: LazyLock<Spectrum> = LazyLock::new(|| {
    Spectrum::new(
        PHI_CSV,
        pdg::MASS_PHI,
        vec![
            Line {
                energy: PHI_TO_ETA_A_ENERGY,
                weight: pdg::BR_PHI_TO_ETA_A,
            },
            Line {
                energy: PHI_TO_ETAP_A_ENERGY,
                weight: pdg::BR_PHI_TO_ETAP_A,
            },
        ],
    )
});

// ===========================================================================
// ---- The kernel -----------------------------------------------------------
// ===========================================================================

/// Which formula the parent's energy selects.
///
/// Split out of [`dnde`] because it depends on the parent energy alone:
/// resolving it once per call rather than once per grid point is the same
/// arithmetic on the same inputs, and it gives the guard somewhere to
/// fail before any element is evaluated.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Branch {
    /// The parent is below its own mass: the spectrum is identically zero.
    BelowThreshold,
    /// The parent is within one `f64::EPSILON` **MeV** of rest, so the
    /// rest-frame table is returned untouched. That is the Cython's guard
    /// in the Cython's units — an absolute comparison on `E − m` in MeV,
    /// not a dimensionless ratio — and it exists because the boost
    /// integral's `1/(2γβ)` prefactor diverges as `β → 0`.
    RestFrame,
    /// The parent is in flight, with the boost velocity it implies.
    InFlight {
        /// `β`, in units of `c`, strictly inside `(0, 1)`.
        beta: f64,
    },
}

/// Resolve a parent energy to its [`Branch`].
///
/// # Parameters
///
/// * `parent_energy` — the decaying meson's total energy, MeV.
/// * `mass` — that meson's mass, MeV; [`Spectrum::mass`].
///
/// # Errors
///
/// [`BoostError::BetaOutOfRange`] when the parent's energy implies a `β`
/// outside `(0, 1)`, which the Cython states as
/// `assert 0.0 < beta < 1.0` inside the boost integral
/// (`hazma/_utils/boost.pyx:173`). `projects/cython-to-rust/rules.md`
/// rule 9 turns that assert into an unconditional error.
///
/// The only reachable way in is a `NaN` parent energy: both branch
/// comparisons are false for a `NaN`, so it falls through to `β = NaN`.
/// A finite parent energy cannot get here — the smallest representable
/// step above any of these masses already gives `β ≈ 1.5e-8`, and
/// anything smaller took the [`Branch::RestFrame`] arm.
// Not a disguised equality test, which is what `float_equality_without_abs`
// is looking for: `parent_energy >= mass` is already established above, so
// this is the one-sided "within one epsilon MeV of rest" threshold the
// Cython writes, and `.abs()` would change nothing. Same guard, same lint,
// same answer as `crate::kernels::positron_muon::dnde_positron_muon`.
#[allow(clippy::float_equality_without_abs)]
pub fn branch(parent_energy: f64, mass: f64) -> Result<Branch, BoostError> {
    if parent_energy < mass {
        return Ok(Branch::BelowThreshold);
    }
    if parent_energy - mass < f64::EPSILON {
        return Ok(Branch::RestFrame);
    }
    let beta = boost::boost_beta(parent_energy, mass);
    if !(0.0 < beta && beta < 1.0) {
        return Err(BoostError::BetaOutOfRange { beta });
    }
    Ok(Branch::InFlight { beta })
}

/// The photon spectrum `dN/dE` in MeV⁻¹ at one energy.
///
/// # Parameters
///
/// * `photon_energy` — the photon's lab-frame energy, MeV.
/// * `branch` — from [`branch`], for the parent energy in question.
/// * `spectrum` — one of the seven statics in this module.
///
/// # Returns
///
/// `dN/dE` in MeV⁻¹: exactly `0.0` below threshold, the rest-frame table
/// at rest, and otherwise the boosted continuum plus each line's boosted
/// contribution. A `NaN` photon energy gives `NaN`.
#[must_use]
pub fn dnde(photon_energy: f64, branch: Branch, spectrum: &Spectrum) -> f64 {
    match branch {
        Branch::BelowThreshold => 0.0,
        Branch::RestFrame => spectrum.table.rest_frame(photon_energy),
        Branch::InFlight { beta } => {
            let table = &spectrum.table;
            let mut result = boost::boost_integrate_linear_interp(
                photon_energy,
                beta,
                &table.energies,
                &table.dnde,
            )
            .expect("beta was checked by `branch` and the table is non-empty and paired");
            for line in &spectrum.lines {
                // One `fmadd d0, d8, d0, d9` per line term in the shipped
                // objects — eight across the five, and the only FMA sites
                // any of them has: the boosted line, times its folded
                // weight, plus the running total.
                result = boost::boost_delta_function(line.energy, photon_energy, 0.0, beta)
                    .mul_add(line.weight, result);
            }
            result
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The seven spectra, with the name each is registered under.
    fn all() -> Vec<(&'static str, &'static Spectrum)> {
        vec![
            ("charged_kaon", &CHARGED_KAON),
            ("long_kaon", &LONG_KAON),
            ("short_kaon", &SHORT_KAON),
            ("eta", &ETA),
            ("eta_prime", &ETA_PRIME),
            ("omega", &OMEGA),
            ("phi", &PHI),
        ]
    }

    /// Every folded constant, against the immediate the shipped
    /// `.cpython-312-darwin.so` loads at that site — or, for the three a
    /// repair has moved, against the value that replaces it.
    ///
    /// The shipped immediates were read out of `objdump -d` as the `movk`
    /// sequences that build each one (little-endian halfwords, high
    /// halfword last). Rust's const evaluator and clang's constant folder
    /// have to agree here to the last bit: the parity budget for these
    /// entry points is 1e-12 and a one-ulp shift in a line's energy moves
    /// the whole line. Each repaired constant names the immediate it
    /// replaces beside it, so the object code stays recoverable from this
    /// test and a silent revert to it still fails.
    #[test]
    fn every_folded_constant_is_the_shipped_immediate_or_its_declared_repair() {
        assert_eq!(ETA_TO_A_A_WEIGHT.to_bits(), 0x3fe9_38ef_34d6_a162);
        assert_eq!(ETA_TO_A_A_ENERGY.to_bits(), 0x4071_1ee5_6041_8937);
        // B1: the shipped object code loads `0x3f97_9fa9_7e13_2b56` here
        // and the two-photon line needs twice it. Doubling only increments
        // the exponent field, so the mantissa below is the shipped
        // immediate's, digit for digit.
        assert_eq!(ETAP_TO_A_A_WEIGHT.to_bits(), 0x3fa7_9fa9_7e13_2b56);
        assert_eq!(ETAP_TO_A_A_ENERGY.to_bits(), 0x407d_ee3d_70a3_d70a);
        assert_eq!(KL_TO_A_A_WEIGHT.to_bits(), 0x3f51_ec91_8e32_5d4a);
        assert_eq!(KS_TO_A_A_WEIGHT.to_bits(), 0x3ed6_0fe1_ca5f_e00f);
        assert_eq!(K0_TO_A_A_ENERGY.to_bits(), 0x406f_19c6_a7ef_9db2);
        assert_eq!(pdg::BR_OMEGA_TO_PI0_A.to_bits(), 0x3fb5_59b3_d07c_84b6);
        assert_eq!(pdg::BR_OMEGA_TO_ETA_A.to_bits(), 0x3f3d_7dbf_487f_cb92);
        assert_eq!(OMEGA_TO_PI0_A_ENERGY.to_bits(), 0x4077_bb0e_6562_5c1c);
        assert_eq!(OMEGA_TO_ETA_A_ENERGY.to_bits(), 0x4068_f281_6f00_68dc);
        assert_eq!(pdg::BR_PHI_TO_ETA_A.to_bits(), 0x3f8a_af78_feef_5ec8);
        assert_eq!(pdg::BR_PHI_TO_ETAP_A.to_bits(), 0x3f10_4e2b_dcfd_9c78);
        // B2: the shipped object code loads the *daughter meson's* energy
        // at both sites — `0x4084_8789_3897_9d28` (656.94 MeV) and
        // `0x408d_fd2a_ecc8_ec19` (959.65 MeV). The photon's energy shares
        // no bits with either, because the repair changes one sign in the
        // numerator rather than scaling the result.
        assert_eq!(PHI_TO_ETA_A_ENERGY.to_bits(), 0x4076_a84d_d059_fcfd);
        assert_eq!(PHI_TO_ETAP_A_ENERGY.to_bits(), 0x404d_e853_3fba_f8c5);
    }

    /// All four `X → γγ` lines carry `2·BR`, the η′ included.
    ///
    /// Two photons leave the decay, so the line's weight is twice the
    /// branching ratio. The η′ was the one sibling that shipped without
    /// the factor (`_eta_prime.pyx:107`, and the disassembled immediate
    /// confirms it was the code rather than a reading of it), which left
    /// its spectrum short one photon per `η′ → γγ` decay; this test is
    /// what makes a silent revert to the shipped weight fail. The ω and φ
    /// weights are *correctly* un-doubled and are deliberately absent:
    /// their lines are `X → Yγ`, one photon each.
    #[test]
    fn every_two_photon_line_carries_twice_its_branching_ratio() {
        assert_eq!(ETA_TO_A_A_WEIGHT, 2.0 * pdg::BR_ETA_TO_A_A);
        assert_eq!(KL_TO_A_A_WEIGHT, 2.0 * pdg::BR_KL_TO_A_A);
        assert_eq!(KS_TO_A_A_WEIGHT, 2.0 * pdg::BR_KS_TO_A_A);
        assert_eq!(ETAP_TO_A_A_WEIGHT, 2.0 * pdg::BR_ETAP_TO_A_A);
        assert_ne!(ETAP_TO_A_A_WEIGHT, pdg::BR_ETAP_TO_A_A);
    }

    /// All four `X → Y γ` lines sit at the photon's energy, below the
    /// daughter meson's.
    ///
    /// `(M² − m²)/(2M)` is `E_γ` and `(M² + m²)/(2M)` is `E_Y`; the two
    /// sum to `M`, which is what makes the distinction checkable rather
    /// than a reading — a line energy that pairs with the *photon's* to
    /// give the parent mass is the daughter's. The φ's two lines shipped
    /// that way in 2.1.0 (`_phi.pyx:111,113`): 656.94 MeV instead of
    /// 362.52 for `φ → ηγ`, a factor of 1.81, and 959.65 instead of 59.82
    /// for `φ → η′γ`, a factor of 16.0 that put 94% of the φ's whole rest
    /// mass into one photon. Repaired as roster entry B2 of
    /// `projects/parity-pinned-defect-repair`; the corpus still pins the
    /// misplaced lines and `test/parity/deltas.py` declares the shift.
    ///
    /// The ω's two lines were always the photon's, and are held here with
    /// the φ's rather than as a control, because the two mesons run
    /// through the same [`photon_line_energy`] now.
    #[test]
    fn every_line_energy_is_the_photons_not_the_daughters() {
        for (line, parent, mass) in [
            (PHI_TO_ETA_A_ENERGY, pdg::MASS_PHI, pdg::MASS_ETA),
            (PHI_TO_ETAP_A_ENERGY, pdg::MASS_PHI, pdg::MASS_ETAP),
            (OMEGA_TO_PI0_A_ENERGY, pdg::MASS_OMEGA, pdg::MASS_PI0),
            (OMEGA_TO_ETA_A_ENERGY, pdg::MASS_OMEGA, pdg::MASS_ETA),
        ] {
            let daughter = (parent * parent + mass * mass) / (2.0 * parent);
            assert!((line + daughter - parent).abs() < 1e-12 * parent);
            assert!(line < daughter);
            // No photon from a two-body decay carries more than half the
            // parent's mass; both shipped φ energies did.
            assert!(line < 0.5 * parent);
        }
        // Stated as the energies as well as the form, so a revert to the
        // shipped `+` fails on the number and not only on the identity.
        assert!((PHI_TO_ETA_A_ENERGY - 362.518_997_527_615_1).abs() < 1e-9);
        assert!((PHI_TO_ETAP_A_ENERGY - 59.815_040_556_235_125).abs() < 1e-9);
        assert_ne!(PHI_TO_ETA_A_ENERGY, 656.942_002_472_385);
        assert_ne!(PHI_TO_ETAP_A_ENERGY, 959.645_959_443_764_8);
    }

    /// Every table parses to a paired, non-empty, strictly ascending grid.
    ///
    /// Ascending is what [`interp::interp`] and
    /// [`boost::boost_integrate_linear_interp`] both assume without
    /// checking, so a table that violated it would give meaningless
    /// numbers rather than an error.
    #[test]
    fn every_table_parses_to_an_ascending_grid() {
        for (name, spectrum) in all() {
            let table = &spectrum.table;
            assert_eq!(table.energies.len(), table.dnde.len(), "{name}");
            assert!(!table.energies.is_empty(), "{name}");
            assert!(
                table.energies.windows(2).all(|pair| pair[0] < pair[1]),
                "{name} energies are not strictly ascending"
            );
            assert!(table.dnde.iter().all(|value| *value >= 0.0), "{name}");
        }
    }

    /// The shipped row counts, so a truncated or duplicated table is
    /// caught here rather than as a moved number three layers up.
    ///
    /// The η is the 100-row outlier; the other six are 500. The header
    /// comment line is not a row.
    #[test]
    fn the_tables_have_the_row_counts_the_csvs_ship() {
        assert_eq!(ETA.table.energies.len(), 100);
        for (name, spectrum) in all() {
            if name == "eta" {
                continue;
            }
            assert_eq!(spectrum.table.energies.len(), 500, "{name}");
        }
    }

    /// Every table runs up to its own parent's mass, bit-for-bit.
    ///
    /// That is the structural tie between a CSV and the [`Spectrum`] it
    /// is paired with: the tables were generated over
    /// `[M/10^k, M]`, so a swapped pair of CSVs — which would otherwise
    /// surface only as a wrong number four layers up — fails here.
    #[test]
    fn each_table_runs_up_to_its_parent_mass() {
        for (name, spectrum) in all() {
            assert_eq!(
                spectrum.table.emax().to_bits(),
                spectrum.mass.to_bits(),
                "{name}: the table's top energy is not the parent mass"
            );
            assert!(spectrum.table.emin() > 0.0, "{name}");
            assert!(
                spectrum.table.emin() < spectrum.mass * 1e-3,
                "{name}: the table starts too close to the parent mass"
            );
        }
    }

    /// The parsed tables are bit-equal to `numpy.loadtxt(...).T` followed
    /// by `numpy.sum(axis=0)`, which is what the Cython computed.
    ///
    /// Two independent things have to hold for this to pass, and both are
    /// assumptions this module would otherwise be making silently:
    /// `f64::from_str` and NumPy's CSV reader must round the same decimal
    /// literal to the same double, and the per-row sum must reproduce
    /// NumPy's reduction *order*. The second is not free — `phi` has ten
    /// mode columns, which is past the eight where `numpy.sum` stops
    /// being a sequential fold, and a sequential sum differs from NumPy
    /// on that table (measured; the other six agree either way).
    ///
    /// The reference bit patterns were read out of NumPy on the parity
    /// corpus's capturing environment (CPython 3.12.12, NumPy 2.5.1,
    /// macOS/arm64). `test/test_core_photon_tables.py` re-derives the
    /// same comparison against the live NumPy rather than against these
    /// literals, so a NumPy that ever changed would be caught there.
    #[test]
    fn the_parsed_tables_are_bit_equal_to_numpys() {
        // (name, emin, first dN/dE, last dN/dE)
        for (name, emin, first, last) in [
            (
                "charged_kaon",
                0x3f40_2d43_48ee_cb13,
                0x4049_0f29_3810_ab86,
                0u64,
            ),
            ("long_kaon", 0x3f40_4e43_7c4d_fbae, 0x4051_1168_5498_b7f9, 0),
            (
                "short_kaon",
                0x3f40_4e43_7c4d_fbae,
                0x404e_70d6_8ed6_5230,
                0,
            ),
            ("eta", 0x3fac_0cef_d28b_52af, 0x3fc9_32a2_bc2a_c21e, 0),
            ("eta_prime", 0x3fb8_84e8_31ad_2136, 0x3fd8_c4fe_2a3d_3684, 0),
            ("omega", 0x3f49_a56d_8d4c_2e56, 0x4048_f0fd_4cb7_f87a, 0),
            ("phi", 0x3fba_1923_bd74_6a35, 0x3fe1_00e2_562a_df4a, 0),
        ] {
            let table = &all()
                .into_iter()
                .find(|(known, _)| *known == name)
                .expect("the roster covers every name above")
                .1
                .table;
            assert_eq!(table.emin().to_bits(), emin, "{name} emin");
            assert_eq!(table.dnde[0].to_bits(), first, "{name} first dN/dE");
            assert_eq!(
                table.dnde[table.dnde.len() - 1].to_bits(),
                last,
                "{name} last dN/dE"
            );
        }
    }

    /// The three branch arms, at and around each boundary.
    #[test]
    fn the_branch_boundaries_are_the_cython_s() {
        let mass = pdg::MASS_ETA;
        assert_eq!(branch(mass * 0.999, mass).unwrap(), Branch::BelowThreshold);
        // `E - m < EPSILON` is absolute in MeV, so `E = m` and one
        // epsilon above both take the rest frame.
        assert_eq!(branch(mass, mass).unwrap(), Branch::RestFrame);
        assert_eq!(
            branch(mass + 0.5 * f64::EPSILON, mass).unwrap(),
            Branch::RestFrame
        );
        // The next representable double above the mass is 1.1e-13 away,
        // well past one epsilon, so it is already in flight.
        let stepped = f64::from_bits(mass.to_bits() + 1);
        assert!(matches!(
            branch(stepped, mass).unwrap(),
            Branch::InFlight { .. }
        ));
    }

    /// A `NaN` parent energy is the one reachable route to the guard.
    ///
    /// The Cython raises a bare `AssertionError` here (measured on the
    /// shipped build); rule 9 makes it an error return that
    /// [`crate::photon`] raises as `ValueError`.
    #[test]
    fn a_nan_parent_energy_is_rejected_rather_than_evaluated() {
        assert!(matches!(
            branch(f64::NAN, pdg::MASS_ETA),
            Err(BoostError::BetaOutOfRange { .. })
        ));
        // Every finite parent energy that reaches the in-flight arm has a
        // usable beta, for every one of the seven masses.
        for (name, spectrum) in all() {
            let stepped = f64::from_bits(spectrum.mass.to_bits() + 1);
            match branch(stepped, spectrum.mass) {
                Ok(Branch::InFlight { beta }) => {
                    assert!(beta > 1e-9, "{name}: beta {beta} is too small to be usable");
                }
                other => panic!("{name}: expected an in-flight branch, got {other:?}"),
            }
        }
    }

    /// Below threshold the spectrum is exactly zero, at every energy.
    #[test]
    fn below_threshold_the_spectrum_vanishes() {
        for (name, spectrum) in all() {
            let below = branch(spectrum.mass * 0.5, spectrum.mass).unwrap();
            for energy in [1e-6, 1.0, 100.0, 1e6] {
                assert_eq!(dnde(energy, below, spectrum), 0.0, "{name} at {energy}");
            }
        }
    }

    /// The rest-frame branch is the table itself: it reproduces every
    /// tabulated value exactly at that value's own energy.
    ///
    /// Exact rather than approximate because [`interp::interp`] returns
    /// the node's value at a node instead of interpolating, so this is a
    /// statement about the wiring rather than about interpolation error.
    #[test]
    fn at_rest_the_spectrum_is_the_table() {
        for (name, spectrum) in all() {
            let rest = branch(spectrum.mass, spectrum.mass).unwrap();
            let table = &spectrum.table;
            for index in [0, 1, table.energies.len() / 2, table.energies.len() - 1] {
                assert_eq!(
                    dnde(table.energies[index], rest, spectrum).to_bits(),
                    table.dnde[index].to_bits(),
                    "{name} at node {index}"
                );
            }
        }
    }

    /// Above the table the rest-frame spectrum is zero and below it the
    /// tail is exactly `1/E`.
    #[test]
    fn the_rest_frame_tails_are_the_cython_s() {
        for (name, spectrum) in all() {
            let rest = branch(spectrum.mass, spectrum.mass).unwrap();
            let table = &spectrum.table;
            assert_eq!(
                dnde(table.emax() * 1.000_001, rest, spectrum),
                0.0,
                "{name} above the table"
            );
            // y = y0 * emin / E below the table, so halving E doubles it.
            let low = table.emin() * 0.5;
            assert_eq!(
                dnde(low, rest, spectrum),
                table.dnde[0] * table.emin() / low,
                "{name} below the table"
            );
            assert_eq!(
                dnde(low * 0.5, rest, spectrum) / dnde(low, rest, spectrum),
                2.0,
                "{name}: the tail is not 1/E"
            );
        }
    }

    /// The line terms are what separates the seven spectra from a bare
    /// boost integral, so each one is measured on its own.
    ///
    /// At the parent's rest-frame line energy, mildly boosted, the line
    /// contributes `weight / (2 γ β E_line)` — a flat plateau. Comparing
    /// the full kernel against the continuum alone isolates exactly that
    /// term, and the difference must be the sum over the spectrum's
    /// lines. The charged kaon has none, and must therefore show no
    /// difference at all.
    #[test]
    fn each_line_contributes_its_own_boosted_plateau() {
        for (name, spectrum) in all() {
            let parent_energy = spectrum.mass * 2.0;
            let Branch::InFlight { beta } = branch(parent_energy, spectrum.mass).unwrap() else {
                unreachable!("2 m is in flight")
            };
            let table = &spectrum.table;
            // A photon energy every line's boosted window covers: the
            // window around a line at e0 is [γ e0 (1−β), γ e0 (1+β)], and
            // the lines here are all within a factor of two of m/2.
            let probe = spectrum.mass / 2.0;
            let continuum =
                boost::boost_integrate_linear_interp(probe, beta, &table.energies, &table.dnde)
                    .unwrap();
            let expected: f64 = spectrum
                .lines
                .iter()
                .map(|line| {
                    line.weight * boost::boost_delta_function(line.energy, probe, 0.0, beta)
                })
                .sum();
            let got = dnde(probe, Branch::InFlight { beta }, spectrum) - continuum;
            assert!(
                (got - expected).abs() <= 1e-12 * expected.abs().max(continuum.abs()),
                "{name}: line contribution {got} vs {expected}"
            );
            if name == "charged_kaon" {
                assert_eq!(expected, 0.0, "the charged kaon has no line");
            } else {
                assert!(expected > 0.0, "{name}: the line contributes nothing");
            }
        }
    }

    /// A `NaN` photon energy propagates rather than raising or panicking.
    ///
    /// Both branches agree on this, which they did not in the Cython: the
    /// rest-frame arm returned `nan` (`np.interp` propagates) while the
    /// in-flight arm raised `IndexError` out of
    /// `np.flatnonzero(lb <= x)[0]` on an empty match. See
    /// [`crate::boost::boost_integrate_linear_interp`] for why the port
    /// answers `NaN` in both.
    #[test]
    fn a_nan_photon_energy_propagates_through_both_branches() {
        for (name, spectrum) in all() {
            let rest = branch(spectrum.mass, spectrum.mass).unwrap();
            let flight = branch(spectrum.mass * 2.0, spectrum.mass).unwrap();
            assert!(dnde(f64::NAN, rest, spectrum).is_nan(), "{name} at rest");
            assert!(
                dnde(f64::NAN, flight, spectrum).is_nan(),
                "{name} in flight"
            );
        }
    }

    /// Zero and negative photon energies take the below-the-table tail,
    /// which is where the Cython's `1/E` extrapolation sends them.
    #[test]
    fn the_unphysical_low_end_follows_the_tail_rather_than_erroring() {
        let spectrum = &*ETA;
        let flight = branch(spectrum.mass * 2.0, spectrum.mass).unwrap();
        assert_eq!(dnde(0.0, flight, spectrum), f64::INFINITY);
        assert!(dnde(-5.0, flight, spectrum) < 0.0);
        // And a photon far above the boosted window gets nothing.
        assert_eq!(dnde(1e12, flight, spectrum), 0.0);
    }
}
