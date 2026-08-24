//! The shared foundation of the four mediator-spectrum `.pyx`:
//! rest-frame interpolation tables, their memo cache, and the mode
//! selectors.
//!
//! PyO3-free (`projects/cython-to-rust/rules.md`, Rust conventions
//! rule 3). Nothing here is a hazma entry point; Tasks 6.2 and 6.3 build
//! `scalar_mediator_decay_spectrum`, `dnde_decay_v`/`dnde_decay_v_pt`
//! and `dnde_decay_s`/`dnde_decay_s_pt` on top of it.
//!
//! # Which `.pyx` this serves
//!
//! Like [`crate::kernels::soft_complex`] and unlike the rest of
//! [`crate::kernels`], this module is not the port of one `.pyx`. It is
//! the part all four mediator-spectrum modules repeat verbatim:
//!
//! | `.pyx` | Tables | Grid start | Below-grid |
//! | --- | --- | --- | --- |
//! | `scalar_mediator/scalar_mediator_decay_spectrum.pyx` | charged pion photon | `10⁻¹` MeV | `1/E` tail |
//! | `vector_mediator/vector_mediator_decay_spectrum.pyx` | charged pion photon, muon photon | `10⁻¹` MeV | `1/E` tail |
//! | `scalar_mediator/scalar_mediator_positron_spec.pyx` | charged pion positron, muon positron | `m_e` | clamp |
//! | `vector_mediator/vector_mediator_positron_spec.pyx` | charged pion positron, muon positron | `m_e` | clamp |
//!
//! The two decay modules are one clone-pair and the two positron
//! modules another, so there are two table *sets* here, each
//! parameterised by the mediator mass: [`PhotonTables`] and
//! [`PositronTables`].
//!
//! # The grid is `numpy.logspace`, bit for bit
//!
//! All four `.pyx` build their abscissae with
//! `np.logspace(start, log10(m/2), num=500)`, and the values a
//! bit-equality-class comparison sees depend on reproducing that
//! exactly rather than on any equivalent spelling. [`logspace`] is
//! NumPy's own arithmetic: `step = (stop - start) / (num - 1)`, then
//! `10**(i * step + start)` with the multiply and the add rounded
//! separately — **not** fused — and the final point taken as `10**stop`
//! rather than as `10**(499 * step + start)`. That last substitution is
//! worth one ulp at about 9% of mediator masses — 732 of 8,008
//! (mass, start) pairs over `numpy.linspace(1, 2000, 4001)` MeV crossed
//! with the two grid starts — and none of the corpus's three masses is
//! among them, so it is a correctness detail rather than something a
//! Phase 06 measurement would have caught.
//!
//! Note also that the grid's last point is not `m/2` exactly: at
//! `m = 550` MeV, `10**log10(275)` is `275.0000000000001`.
//!
//! `cargo` gates the *algorithm* only — the tests below assert the
//! unfused step, the substituted endpoint and the endpoints themselves.
//! Agreement with NumPy is a claim about libm, which
//! `test/test_core_mediator_tables.py` re-derives live through
//! [`crate::mediator_tables_probe`] on whatever platform the suite runs;
//! hard-coding one platform's bits here would turn a Linux CI job red
//! for a libm difference rather than a defect
//! (`projects/cython-to-rust/learnings/phase-04-spectra-kernels.md` §4).
//!
//! One spelling difference between the four sources is *not* a
//! numerical one: `scalar_mediator_decay_spectrum.pyx:45` takes its
//! upper endpoint from `np.log10`, the other three from libc `log10`.
//! Measured on the capturing platform, the two agree bit-for-bit at
//! every mass the corpus samples, so this module takes one `log10`.
//!
//! # Why the cache is real here and dead in the Cython
//!
//! Neither decay module caches at all — `__set_spectra` is called
//! unconditionally at the top of every entry point
//! (`scalar_mediator_decay_spectrum.pyx:245`,
//! `vector_mediator_decay_spectrum.pyx:250` and `:274`), so every call rebuilds a
//! 500-point quadrature-backed table. Both positron modules *look*
//! cached and are not: `__recompute_rf_spectra` compares against
//! `cache_ms` / `cache_pws` and **no line anywhere assigns to either**,
//! so the sentinel `-1.0` they are initialised with never changes and
//! the predicate is always true
//! (`scalar_mediator_positron_spec.pyx:49-55`, and the identical
//! `vector_mediator_positron_spec.pyx:50-56`).
//!
//! [`TableCache`] is the fix, and it is a performance change only
//! (`rules.md` rules 3 and 12): a cache hit returns the table a miss
//! would have rebuilt from the same inputs, so no value moves.
//!
//! The key is the mediator mass alone. The Cython's dead predicate also
//! named three partial widths, but `__set_spectra` takes only the mass
//! and reads no width — the tables are a pure function of `m/2`, the
//! daughter energy the Phase 04 kernels are evaluated at. Keying on the
//! widths as well would be equally correct and strictly slower: it
//! would rebuild both tables whenever a caller varies a coupling at
//! fixed mass, which is the sweep the cache exists to make cheap.
//!
//! # Modes are parsed once, and an unknown mode is not an error
//!
//! The Cython re-dispatches on `str` **inside the integrand**, so a
//! mode string is compared once per quadrature node
//! (`vector_mediator_decay_spectrum.pyx:166-178`,
//! `scalar_mediator_positron_spec.pyx:150-161`). [`PhotonMode`],
//! [`PositronMode`] and [`ScalarPhotonModes`] move that to the call
//! boundary.
//!
//! An unrecognised selector raises nothing today. Every `cdef double`
//! integrand ends in a chain of `if mode == ...: return ...` with no
//! `else`, and a C function that falls off its end returns zero — so a
//! typo'd mode integrates a zero integrand and the entry point returns
//! `0.0`. The parsers below therefore return [`Option`] and leave that
//! `0.0` to their caller rather than tightening it into a raise; the
//! behaviour is reproduced under `rules.md` rule 1 and filed as
//! `docs/followups/todo/mediator-spectra-accept-unknown-mode-strings.md`.
//! Rule 9's "Cython `assert`s become raises" does not reach it: there
//! is no `assert` here, only a fall-through.

use std::sync::{LazyLock, Mutex};

use std::sync::Arc;

use crate::constants::legacy;
use crate::interp;
use crate::kernels::{photon_muon, photon_pion, positron_muon, positron_pion};

// ===========================================================================
// ---- The grid -------------------------------------------------------------
// ===========================================================================

/// Points in every mediator-spectrum interpolation table.
///
/// `n_interp_pts` in all four `.pyx`.
pub const N_INTERP_PTS: usize = 500;

/// The decay modules' lower grid endpoint, `10⁻¹` MeV.
///
/// Written as the literal exponent `-1.0` in both
/// (`scalar_mediator_decay_spectrum.pyx:45`,
/// `vector_mediator_decay_spectrum.pyx:33`), and reused as the
/// threshold of the `1/E` tail below the grid — the same number in both
/// roles, which is why it is one constant.
pub const PHOTON_GRID_LOG10_START: f64 = -1.0;

/// `numpy.logspace(start, stop, num)`, reproduced exactly.
///
/// NumPy evaluates `10 ** linspace(start, stop, num)`, and its
/// `linspace` computes `step = (stop - start) / (num - 1)` once, forms
/// `i * step` and adds `start` as two separately rounded operations, then
/// **overwrites the last element with `stop`**. That last assignment is
/// not cosmetic: `10**stop` and `10**(499 * step + start)` differ by one
/// ulp at about 9% of mediator masses (see this module's docs), and
/// `m = 2` MeV is one of them.
///
/// `i * step + start` is deliberately *not* written `mul_add`. NumPy's
/// loop is two IEEE operations and a fused one would round differently.
///
/// # Panics
///
/// Panics if `num < 2`. Every call site passes [`N_INTERP_PTS`]; NumPy
/// defines `num == 1` as `[10**start]` and `num == 0` as empty, and
/// reproducing either would be dead code.
#[must_use]
pub fn logspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    assert!(num >= 2, "logspace: num must be at least 2");
    let step = (stop - start) / ((num - 1) as f64);
    let mut grid: Vec<f64> = (0..num).map(|i| (i as f64) * step + start).collect();
    grid[num - 1] = stop;
    for exponent in &mut grid {
        *exponent = 10.0_f64.powf(*exponent);
    }
    grid
}

// ===========================================================================
// ---- The table ------------------------------------------------------------
// ===========================================================================

/// What a table returns below its first abscissa.
///
/// The two clone-pairs disagree, and both behaviours are shipped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BelowGrid {
    /// `dnde[0] * energies[0] / energy` — the decay modules' `1/E` tail
    /// (`scalar_mediator_decay_spectrum.pyx:55-56`,
    /// `vector_mediator_decay_spectrum.pyx:49-56`), anchored so it meets
    /// the first tabulated value at the threshold.
    ///
    /// The threshold is the grid's own lower endpoint written a second
    /// time as `10**-1`; the Cython compares against that literal rather
    /// than against `e_gams[0]`, and the two are the same double.
    InverseEnergy,
    /// `numpy.interp`'s own clamp to `dnde[0]` — the positron modules,
    /// which interpolate with no guard at all
    /// (`scalar_mediator_positron_spec.pyx:96-101`).
    Clamp,
}

/// A rest-frame spectrum tabulated on a log-spaced energy grid.
///
/// `dnde[i]` is the Phase 04 kernel evaluated at `energies[i]` — called
/// natively, with no Python round trip, where the Cython went through
/// `dnde_*_array` and a NumPy allocation.
pub struct RestFrameTable {
    /// Rest-frame product energies, MeV, ascending.
    energies: Vec<f64>,
    /// `dN/dE` in MeV⁻¹ at each energy.
    dnde: Vec<f64>,
    below: BelowGrid,
}

impl RestFrameTable {
    /// Tabulate `kernel` on `logspace(log10_start, log10_stop, num)`.
    ///
    /// `log10_start` and `log10_stop` are base-10 logarithms of energies
    /// in MeV; `kernel` maps an energy in MeV to `dN/dE` in MeV⁻¹.
    #[must_use]
    pub fn build<F>(log10_start: f64, log10_stop: f64, below: BelowGrid, kernel: F) -> Self
    where
        F: Fn(f64) -> f64,
    {
        let energies = logspace(log10_start, log10_stop, N_INTERP_PTS);
        let dnde = energies.iter().map(|&energy| kernel(energy)).collect();
        Self {
            energies,
            dnde,
            below,
        }
    }

    /// Wrap columns that were tabulated elsewhere.
    ///
    /// Only [`crate::mediator_tables_probe`] calls this: it is how a
    /// Python test puts abscissae through [`Self::lookup`] without
    /// rebuilding a quadrature-backed table first. Every table hazma
    /// itself evaluates comes from [`Self::build`].
    ///
    /// # Errors
    ///
    /// A message naming the violation if the columns are empty or of
    /// different lengths — the two cases [`interp::interp`] would
    /// otherwise panic on.
    pub fn from_columns(
        energies: Vec<f64>,
        dnde: Vec<f64>,
        below: BelowGrid,
    ) -> Result<Self, String> {
        if energies.is_empty() {
            return Err("array of sample points is empty".to_owned());
        }
        if energies.len() != dnde.len() {
            return Err("fp and xp are not of the same length.".to_owned());
        }
        Ok(Self {
            energies,
            dnde,
            below,
        })
    }

    /// The tabulated `dN/dE` in MeV⁻¹ at `energy` MeV.
    ///
    /// Inside the grid this is `numpy.interp`; below it, whichever of
    /// [`BelowGrid`]'s two behaviours the source `.pyx` had. Above the
    /// grid both clone-pairs take NumPy's clamp to the last value, which
    /// is what [`interp::interp`] already does — neither `.pyx` guards
    /// that side.
    ///
    /// A `NaN` energy fails the `<` comparison, so it reaches
    /// [`interp::interp`] and propagates, exactly as the Cython's
    /// `np.interp` did.
    #[must_use]
    pub fn lookup(&self, energy: f64) -> f64 {
        if self.below == BelowGrid::InverseEnergy && energy < PHOTON_GRID_LOG10_TAIL_THRESHOLD {
            return self.dnde[0] * self.energies[0] / energy;
        }
        interp::interp(energy, &self.energies, &self.dnde)
    }

    /// The grid's abscissae, MeV — for tests and for
    /// [`crate::kernels`]'s own assertions.
    #[must_use]
    pub fn energies(&self) -> &[f64] {
        &self.energies
    }

    /// The tabulated values, MeV⁻¹, aligned with [`Self::energies`].
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.dnde
    }
}

/// The `10**-1` the decay modules compare against before extrapolating.
///
/// Named separately from [`PHOTON_GRID_LOG10_START`] because it is an
/// *energy* in MeV where that is a base-10 exponent, even though the
/// Cython spells both `-1` and the tail therefore begins exactly at the
/// grid's first point.
const PHOTON_GRID_LOG10_TAIL_THRESHOLD: f64 = 0.1;

// ===========================================================================
// ---- The cache ------------------------------------------------------------
// ===========================================================================

/// A one-slot memo keyed on the mediator mass.
///
/// One slot, because the Cython had one set of module globals and
/// because every consumer sweeps a whole energy grid at one mass before
/// moving to the next. The key is compared on the mass's **bit
/// pattern**, so it is reflexive at every value a caller can pass —
/// including `NaN`, which `==` would make a permanent miss and an
/// unbounded rebuild loop.
///
/// The stored value is an [`Arc`] so a caller can drop the lock before
/// integrating: a quadrature over 500-point tables must not hold a mutex
/// across thousands of integrand evaluations.
pub struct TableCache<T> {
    slot: Mutex<Option<(u64, Arc<T>)>>,
}

impl<T> Default for TableCache<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> TableCache<T> {
    /// An empty cache.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            slot: Mutex::new(None),
        }
    }

    /// The tables for `mass` MeV, building them with `build` on a miss.
    ///
    /// # Panics
    ///
    /// Panics if the mutex was poisoned by a previous panic inside
    /// `build`. `build` calls kernel `fn`s that return `NaN` rather than
    /// panicking on bad input, so this is a crate bug rather than a
    /// reachable path.
    pub fn get_or_build<F>(&self, mass: f64, build: F) -> Arc<T>
    where
        F: FnOnce(f64) -> T,
    {
        let key = mass.to_bits();
        let mut slot = self.slot.lock().expect("mediator table cache poisoned");
        if let Some((cached_key, tables)) = slot.as_ref()
            && *cached_key == key
        {
            return Arc::clone(tables);
        }
        let tables = Arc::new(build(mass));
        *slot = Some((key, Arc::clone(&tables)));
        tables
    }
}

// ===========================================================================
// ---- The two table sets ---------------------------------------------------
// ===========================================================================

/// The rest-frame photon tables both decay modules interpolate.
///
/// Built at the daughter energy `m/2`, which is what a two-body decay of
/// a mediator at rest gives each daughter.
pub struct PhotonTables {
    /// `π± → …γ` at `E_π = m/2`.
    pub charged_pion: RestFrameTable,
    /// `μ± → e ν ν γ` at `E_μ = m/2`.
    ///
    /// Only `vector_mediator_decay_spectrum.pyx` builds this one; the
    /// scalar module calls `dnde_photon_muon_point` per node instead of
    /// tabulating it. Both table sets are built together anyway — see
    /// the note on [`photon_tables_for`].
    pub muon: RestFrameTable,
}

/// The rest-frame positron tables both positron modules interpolate.
pub struct PositronTables {
    /// `π± → μ ν → e ν ν ν` at `E_π = m/2`.
    pub charged_pion: RestFrameTable,
    /// `μ± → e ν ν` at `E_μ = m/2`.
    pub muon: RestFrameTable,
}

/// Photon tables for a mediator of mass `mass` MeV.
///
/// The scalar decay module tabulates only the charged pion and the
/// vector module tabulates both, so a shared set builds one table the
/// scalar half will not read. That is deliberate: the alternative is two
/// caches whose only difference is a field, and the wasted table is one
/// 500-point evaluation per *distinct mass*, where the dead cache it
/// replaces cost two per *call*. Task 6.2 measures the result either
/// way.
fn photon_tables_for(mass: f64) -> PhotonTables {
    let daughter_energy = mass / 2.0;
    let log10_stop = daughter_energy.log10();
    PhotonTables {
        charged_pion: RestFrameTable::build(
            PHOTON_GRID_LOG10_START,
            log10_stop,
            BelowGrid::InverseEnergy,
            |energy| photon_pion::dnde_photon_charged_pion(energy, daughter_energy),
        ),
        muon: RestFrameTable::build(
            PHOTON_GRID_LOG10_START,
            log10_stop,
            BelowGrid::InverseEnergy,
            |energy| photon_muon::dnde_photon_muon(energy, daughter_energy),
        ),
    }
}

/// Positron tables for a mediator of mass `mass` MeV.
///
/// The grid starts at the **legacy** electron mass — all four `.pyx`
/// `include "../_utils/legacy_parameters.pxd"`, so `me` here is
/// `0.510998928` and not [`crate::constants::pdg::MASS_E`]
/// (`rules.md` rule 4).
fn positron_tables_for(mass: f64) -> PositronTables {
    let daughter_energy = mass / 2.0;
    let log10_start = legacy::MASS_E.log10();
    let log10_stop = daughter_energy.log10();
    PositronTables {
        charged_pion: RestFrameTable::build(log10_start, log10_stop, BelowGrid::Clamp, |energy| {
            positron_pion::dnde_positron_charged_pion(energy, daughter_energy)
        }),
        muon: RestFrameTable::build(log10_start, log10_stop, BelowGrid::Clamp, |energy| {
            positron_muon::dnde_positron_muon(energy, daughter_energy)
        }),
    }
}

/// The process-wide photon-table cache.
static PHOTON_TABLES: LazyLock<TableCache<PhotonTables>> = LazyLock::new(TableCache::new);

/// The process-wide positron-table cache.
static POSITRON_TABLES: LazyLock<TableCache<PositronTables>> = LazyLock::new(TableCache::new);

/// Memoized [`PhotonTables`] for a mediator of mass `mass` MeV.
#[must_use]
pub fn photon_tables(mass: f64) -> Arc<PhotonTables> {
    PHOTON_TABLES.get_or_build(mass, photon_tables_for)
}

/// Memoized [`PositronTables`] for a mediator of mass `mass` MeV.
#[must_use]
pub fn positron_tables(mass: f64) -> Arc<PositronTables> {
    POSITRON_TABLES.get_or_build(mass, positron_tables_for)
}

// ===========================================================================
// ---- Mode selectors -------------------------------------------------------
// ===========================================================================

/// The vector decay module's `mode` argument.
///
/// The seven strings `vector_mediator_decay_spectrum.pyx:166-178`
/// compares against, in that order. Parsed once per call by
/// [`PhotonMode::parse`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PhotonMode {
    /// `"total"` — every channel, plus the `π⁰γ` line.
    Total,
    /// `"e e g"` — FSR off the electron pair.
    ElectronFsr,
    /// `"pi pi g"` — FSR off the charged pions.
    ChargedPionFsr,
    /// `"pi pi"` — the charged pions' decay continuum.
    ChargedPionDecay,
    /// `"pi0 g"` — the `π⁰` continuum plus the monochromatic line.
    NeutralPionLine,
    /// `"mu mu g"` — FSR off the muon pair.
    MuonFsr,
    /// `"mu mu"` — the muons' decay continuum.
    MuonDecay,
}

impl PhotonMode {
    /// The mode `mode` names, or `None` if it names nothing.
    ///
    /// `None` is not an error: see this module's docs — the Cython
    /// returns `0.0` for an unrecognised mode and the caller reproduces
    /// that.
    #[must_use]
    pub fn parse(mode: &str) -> Option<Self> {
        match mode {
            "total" => Some(Self::Total),
            "e e g" => Some(Self::ElectronFsr),
            "pi pi g" => Some(Self::ChargedPionFsr),
            "pi pi" => Some(Self::ChargedPionDecay),
            "pi0 g" => Some(Self::NeutralPionLine),
            "mu mu g" => Some(Self::MuonFsr),
            "mu mu" => Some(Self::MuonDecay),
            _ => None,
        }
    }

    /// Whether this mode carries the `π⁰ → γ` line term.
    ///
    /// `vector_mediator_decay_spectrum.pyx:223` adds it for `"pi0 g"`
    /// and `"total"` only.
    #[must_use]
    pub fn has_line(self) -> bool {
        matches!(self, Self::Total | Self::NeutralPionLine)
    }
}

/// The positron modules' `fs` argument.
///
/// Both `scalar_mediator_positron_spec.pyx:150-161` and its vector clone
/// compare against these four strings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PositronMode {
    /// `"total"` — the charged-pion and muon continua, plus the `e⁺e⁻`
    /// line.
    Total,
    /// `"e e"` — the `e⁺e⁻` line alone; no integral is performed.
    ElectronLine,
    /// `"mu mu"` — the muon continuum, plus the line.
    MuonDecay,
    /// `"pi pi"` — the charged-pion continuum, plus the line.
    ChargedPionDecay,
}

impl PositronMode {
    /// The final state `fs` names, or `None` if it names nothing.
    #[must_use]
    pub fn parse(fs: &str) -> Option<Self> {
        match fs {
            "total" => Some(Self::Total),
            "e e" => Some(Self::ElectronLine),
            "mu mu" => Some(Self::MuonDecay),
            "pi pi" => Some(Self::ChargedPionDecay),
            _ => None,
        }
    }
}

/// The scalar decay module's `modes` argument, reduced to a bit set.
///
/// Alone among the four, this entry point takes a *list* of modes and
/// folds it into an `int` bitflag
/// (`scalar_mediator_decay_spectrum.pyx:253-266`). The bit values are
/// that file's, so a set built here and one built there compare equal.
///
/// Two properties of the Cython's fold are reproduced deliberately: an
/// unrecognised entry sets no bit and raises nothing, and a repeated
/// entry sets its bit once — the `.pyx` writes `bitflag += BITFLAG_PP`
/// under `if "pi pi" in modes`, and `in` is tested once per mode name
/// rather than once per list element, so a duplicate cannot double a
/// flag into its neighbour's bit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ScalarPhotonModes(u32);

impl ScalarPhotonModes {
    /// `"pi pi"` — the charged pions' decay continuum.
    pub const CHARGED_PION_DECAY: u32 = 1;
    /// `"mu mu"` — the muons' decay continuum.
    pub const MUON_DECAY: u32 = 2;
    /// `"pi0 pi0"` — the neutral pions' decay continuum.
    pub const NEUTRAL_PION_DECAY: u32 = 4;
    /// `"g g"` — the monochromatic two-photon line.
    pub const TWO_PHOTON_LINE: u32 = 8;
    /// `"e e g"` — FSR off the electron pair.
    pub const ELECTRON_FSR: u32 = 16;
    /// `"pi pi g"` — FSR off the charged pions.
    pub const CHARGED_PION_FSR: u32 = 32;
    /// `"mu mu g"` — FSR off the muon pair.
    pub const MUON_FSR: u32 = 64;

    /// The bit a mode name sets, or `None` if it names nothing.
    #[must_use]
    pub fn bit_for(mode: &str) -> Option<u32> {
        match mode {
            "pi pi" => Some(Self::CHARGED_PION_DECAY),
            "mu mu" => Some(Self::MUON_DECAY),
            "pi0 pi0" => Some(Self::NEUTRAL_PION_DECAY),
            "g g" => Some(Self::TWO_PHOTON_LINE),
            "e e g" => Some(Self::ELECTRON_FSR),
            "pi pi g" => Some(Self::CHARGED_PION_FSR),
            "mu mu g" => Some(Self::MUON_FSR),
            _ => None,
        }
    }

    /// Fold an iterator of mode names into a bit set.
    #[must_use]
    pub fn from_names<I, S>(modes: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut bits = 0;
        for mode in modes {
            if let Some(bit) = Self::bit_for(mode.as_ref()) {
                bits |= bit;
            }
        }
        Self(bits)
    }

    /// The raw bitflag, for comparison against the Cython's `int`.
    ///
    /// The only reader the bit set needs today. A `contains(bit)`
    /// helper belongs with the integrand that branches on it, so Task
    /// 6.2 adds one when it has a caller rather than leaving a
    /// `dead_code` allowance here.
    #[must_use]
    pub fn bits(self) -> u32 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BelowGrid, N_INTERP_PTS, PHOTON_GRID_LOG10_START, PhotonMode, PositronMode, RestFrameTable,
        ScalarPhotonModes, TableCache, logspace, photon_tables, positron_tables,
    };
    use crate::constants::legacy;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// The *fused* spelling of the exponent loop — the wrong one.
    ///
    /// The foil [`logspace`]'s unfused arithmetic is compared against.
    /// It is written out here rather than derived from `logspace` so an
    /// edit to the implementation cannot silently make the comparison
    /// vacuous.
    fn fused_exponents(start: f64, stop: f64, num: usize) -> Vec<f64> {
        let step = (stop - start) / ((num - 1) as f64);
        let mut exponents: Vec<f64> = (0..num).map(|i| (i as f64).mul_add(step, start)).collect();
        exponents[num - 1] = stop;
        exponents
    }

    #[test]
    fn logspace_does_not_fuse_the_step_arithmetic() {
        // `i * step + start` is two roundings in NumPy and one in a
        // `mul_add`. `fused_exponents` above is the fused variant; this
        // asserts both halves of the contract -- that the two really do
        // disagree on this grid, so the comparison is not vacuous, and
        // that `logspace` is the unfused one.
        let (start, stop) = (PHOTON_GRID_LOG10_START, 275.0_f64.log10());
        let grid = logspace(start, stop, N_INTERP_PTS);
        let step = (stop - start) / ((N_INTERP_PTS - 1) as f64);
        let fused = fused_exponents(start, stop, N_INTERP_PTS);

        // The last position is excluded: it is substituted from `stop`
        // and belongs to the test below, not to the step arithmetic.
        let mut differing = 0;
        for (index, &fused_exponent) in fused.iter().enumerate().take(N_INTERP_PTS - 1) {
            let unfused_exponent = (index as f64) * step + start;
            if unfused_exponent.to_bits() != fused_exponent.to_bits() {
                differing += 1;
            }
            assert_eq!(
                grid[index].to_bits(),
                10.0_f64.powf(unfused_exponent).to_bits(),
                "grid position {index} is not the unfused exponent"
            );
        }
        assert!(
            differing > 0,
            "fused and unfused step arithmetic agree everywhere on this \
             grid; the test no longer gates the spelling"
        );
    }

    #[test]
    fn logspace_takes_its_last_point_from_stop_not_from_the_step() {
        // The property the final assignment exists for, stated without
        // the captured constants: the last point is `10**stop` exactly,
        // and it is *not* what continuing the step arithmetic gives.
        //
        // `m = 2` MeV rather than one of the corpus's three masses,
        // because at 250/550/900 MeV the two spellings happen to agree —
        // they part company at about 9% of masses and a test built on one
        // of the other 91% would assert nothing.
        let (start, stop) = (PHOTON_GRID_LOG10_START, 1.0_f64.log10());
        let grid = logspace(start, stop, N_INTERP_PTS);
        let step = (stop - start) / ((N_INTERP_PTS - 1) as f64);
        let stepped = 10.0_f64.powf(((N_INTERP_PTS - 1) as f64) * step + start);
        assert_eq!(
            grid[N_INTERP_PTS - 1].to_bits(),
            10.0_f64.powf(stop).to_bits()
        );
        assert_ne!(grid[N_INTERP_PTS - 1].to_bits(), stepped.to_bits());
    }

    #[test]
    fn logspace_is_ascending_and_spans_its_endpoints() {
        let grid = logspace(-1.0, 3.0, N_INTERP_PTS);
        assert!(grid.windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(grid[0].to_bits(), 0.1_f64.to_bits());
        assert_eq!(grid[N_INTERP_PTS - 1].to_bits(), 1000.0_f64.to_bits());
    }

    /// A table over `dN/dE = 1/E`, which makes every regime checkable in
    /// closed form.
    fn reciprocal_table(below: BelowGrid) -> RestFrameTable {
        RestFrameTable::build(-1.0, 3.0, below, |energy| 1.0 / energy)
    }

    #[test]
    fn table_values_are_the_kernel_at_the_grid_points() {
        let table = reciprocal_table(BelowGrid::Clamp);
        assert_eq!(table.energies().len(), N_INTERP_PTS);
        assert_eq!(table.values().len(), N_INTERP_PTS);
        for (&energy, &value) in table.energies().iter().zip(table.values()) {
            assert_eq!(value.to_bits(), (1.0 / energy).to_bits());
        }
    }

    #[test]
    fn table_lookup_hits_a_node_exactly() {
        let table = reciprocal_table(BelowGrid::Clamp);
        let energy = table.energies()[123];
        assert_eq!(
            table.lookup(energy).to_bits(),
            table.values()[123].to_bits()
        );
    }

    #[test]
    fn inverse_energy_tail_is_continuous_at_the_threshold() {
        // The tail's whole point: `dnde[0] * e[0] / e` meets `dnde[0]` at
        // `e = e[0]`, so the branch introduces no jump.
        let table = reciprocal_table(BelowGrid::InverseEnergy);
        let first = table.energies()[0];
        assert_eq!(first.to_bits(), 0.1_f64.to_bits());
        let just_below = f64::from_bits(first.to_bits() - 1);
        let ratio = table.lookup(just_below) / table.values()[0];
        assert!((ratio - 1.0).abs() < 1e-15, "tail jumps by {ratio}");
        // Below the threshold it really is a `1/E` tail, not a clamp.
        assert_eq!(
            table.lookup(0.01).to_bits(),
            (table.values()[0] * first / 0.01).to_bits()
        );
    }

    #[test]
    fn the_tail_branch_opens_exactly_at_the_threshold_and_not_above_it() {
        // The threshold is a boundary, not a region: at and above `10**-1`
        // the table interpolates. Without this, moving the constant up
        // silently converts the first decade of the grid into
        // extrapolation and every other test still passes.
        let table = reciprocal_table(BelowGrid::InverseEnergy);
        for energy in [0.1, 0.11, 0.15, 0.2, 0.5, 1.0] {
            assert_eq!(
                table.lookup(energy).to_bits(),
                crate::interp::interp(energy, table.energies(), table.values()).to_bits(),
                "lookup at {energy} extrapolated where it should interpolate"
            );
        }
    }

    #[test]
    fn clamp_below_grid_returns_the_first_value() {
        let table = reciprocal_table(BelowGrid::Clamp);
        assert_eq!(table.lookup(1e-6).to_bits(), table.values()[0].to_bits());
        assert_eq!(table.lookup(-5.0).to_bits(), table.values()[0].to_bits());
    }

    #[test]
    fn both_policies_clamp_above_the_grid() {
        // Neither `.pyx` guards the upper side, so both inherit
        // `numpy.interp`'s clamp.
        let last = N_INTERP_PTS - 1;
        for below in [BelowGrid::Clamp, BelowGrid::InverseEnergy] {
            let table = reciprocal_table(below);
            assert_eq!(table.lookup(1e6).to_bits(), table.values()[last].to_bits());
        }
    }

    #[test]
    fn nan_propagates_through_both_policies() {
        for below in [BelowGrid::Clamp, BelowGrid::InverseEnergy] {
            assert!(reciprocal_table(below).lookup(f64::NAN).is_nan());
        }
    }

    #[test]
    fn cache_builds_once_per_mass_and_rebuilds_when_it_changes() {
        let cache: TableCache<f64> = TableCache::new();
        let builds = AtomicUsize::new(0);
        let mut build = |mass: f64| {
            builds.fetch_add(1, Ordering::SeqCst);
            mass * 2.0
        };

        assert_eq!(*cache.get_or_build(550.0, &mut build), 1100.0);
        assert_eq!(*cache.get_or_build(550.0, &mut build), 1100.0);
        assert_eq!(builds.load(Ordering::SeqCst), 1, "a hit rebuilt the tables");

        assert_eq!(*cache.get_or_build(900.0, &mut build), 1800.0);
        assert_eq!(builds.load(Ordering::SeqCst), 2);
        // The slot holds one entry, so returning to the first mass is a
        // miss — the Cython's single set of module globals, made real.
        assert_eq!(*cache.get_or_build(550.0, &mut build), 1100.0);
        assert_eq!(builds.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn cache_hits_on_a_nan_mass() {
        // Keyed on bits rather than on `==`, so `NaN` is a hit and not an
        // unbounded rebuild. The tables it caches are all-`NaN`, which is
        // what the Cython produced too.
        let cache: TableCache<f64> = TableCache::new();
        let builds = AtomicUsize::new(0);
        let mut build = |_mass: f64| {
            builds.fetch_add(1, Ordering::SeqCst);
            0.0
        };
        cache.get_or_build(f64::NAN, &mut build);
        cache.get_or_build(f64::NAN, &mut build);
        assert_eq!(builds.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn cache_distinguishes_the_two_zeros() {
        // `0.0 == -0.0` but their bits differ, so this is the one place
        // bit-keying is *stricter* than `==`. It costs one extra build at
        // a mass that is unphysical anyway; asserted so the choice is
        // deliberate rather than discovered later.
        let cache: TableCache<f64> = TableCache::new();
        let builds = AtomicUsize::new(0);
        let mut build = |_mass: f64| {
            builds.fetch_add(1, Ordering::SeqCst);
            0.0
        };
        cache.get_or_build(0.0, &mut build);
        cache.get_or_build(-0.0, &mut build);
        assert_eq!(builds.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn photon_tables_are_memoized_and_hold_the_phase_04_kernels() {
        let tables = photon_tables(550.0);
        let again = photon_tables(550.0);
        assert!(
            std::sync::Arc::ptr_eq(&tables, &again),
            "a second call rebuilt the photon tables"
        );

        // Grid endpoints: `10**-1` MeV to the daughter energy `m/2`.
        assert_eq!(
            tables.charged_pion.energies()[0].to_bits(),
            0.1_f64.to_bits()
        );
        assert_eq!(
            tables.charged_pion.energies()[N_INTERP_PTS - 1].to_bits(),
            10.0_f64.powf(275.0_f64.log10()).to_bits()
        );

        // The tabulated values are the native kernels, not a re-derivation.
        for index in [0, 137, N_INTERP_PTS - 1] {
            let energy = tables.charged_pion.energies()[index];
            assert_eq!(
                tables.charged_pion.values()[index].to_bits(),
                crate::kernels::photon_pion::dnde_photon_charged_pion(energy, 275.0).to_bits()
            );
            let energy = tables.muon.energies()[index];
            assert_eq!(
                tables.muon.values()[index].to_bits(),
                crate::kernels::photon_muon::dnde_photon_muon(energy, 275.0).to_bits()
            );
        }
    }

    #[test]
    fn positron_tables_start_at_the_legacy_electron_mass() {
        let tables = positron_tables(550.0);
        // `rules.md` rule 4: the mediator `.pyx` include the legacy
        // header, so this is 0.510998928 and not the PDG value.
        assert_eq!(
            tables.charged_pion.energies()[0].to_bits(),
            legacy::MASS_E.to_bits()
        );
        assert_ne!(legacy::MASS_E, crate::constants::pdg::MASS_E);

        for index in [0, 137, N_INTERP_PTS - 1] {
            let energy = tables.muon.energies()[index];
            assert_eq!(
                tables.muon.values()[index].to_bits(),
                crate::kernels::positron_muon::dnde_positron_muon(energy, 275.0).to_bits()
            );
            let energy = tables.charged_pion.energies()[index];
            assert_eq!(
                tables.charged_pion.values()[index].to_bits(),
                crate::kernels::positron_pion::dnde_positron_charged_pion(energy, 275.0).to_bits()
            );
        }
    }

    #[test]
    fn photon_mode_accepts_exactly_the_cython_strings() {
        let accepted = [
            ("total", PhotonMode::Total),
            ("e e g", PhotonMode::ElectronFsr),
            ("pi pi g", PhotonMode::ChargedPionFsr),
            ("pi pi", PhotonMode::ChargedPionDecay),
            ("pi0 g", PhotonMode::NeutralPionLine),
            ("mu mu g", PhotonMode::MuonFsr),
            ("mu mu", PhotonMode::MuonDecay),
        ];
        for (name, mode) in accepted {
            assert_eq!(PhotonMode::parse(name), Some(mode));
        }
        // Near-misses a caller will actually make, all of which the
        // Cython answers with 0.0 rather than an exception.
        for rejected in ["", "Total", "total ", "pi0g", "e e", "pi0 pi0", "g g"] {
            assert_eq!(PhotonMode::parse(rejected), None, "{rejected:?}");
        }
    }

    #[test]
    fn only_the_two_line_carrying_photon_modes_have_a_line() {
        assert!(PhotonMode::Total.has_line());
        assert!(PhotonMode::NeutralPionLine.has_line());
        for mode in [
            PhotonMode::ElectronFsr,
            PhotonMode::ChargedPionFsr,
            PhotonMode::ChargedPionDecay,
            PhotonMode::MuonFsr,
            PhotonMode::MuonDecay,
        ] {
            assert!(!mode.has_line(), "{mode:?} claims a line term");
        }
    }

    #[test]
    fn positron_mode_accepts_exactly_the_cython_strings() {
        assert_eq!(PositronMode::parse("total"), Some(PositronMode::Total));
        assert_eq!(PositronMode::parse("e e"), Some(PositronMode::ElectronLine));
        assert_eq!(PositronMode::parse("mu mu"), Some(PositronMode::MuonDecay));
        assert_eq!(
            PositronMode::parse("pi pi"),
            Some(PositronMode::ChargedPionDecay)
        );
        // "e e g" is a *photon* mode; the positron modules never accept it.
        for rejected in ["", "e e g", "pi0 g", "ee", "TOTAL"] {
            assert_eq!(PositronMode::parse(rejected), None, "{rejected:?}");
        }
    }

    #[test]
    fn scalar_photon_bits_match_the_cython_bitflags() {
        // The literal values at
        // `scalar_mediator_decay_spectrum.pyx:16-22`.
        assert_eq!(ScalarPhotonModes::CHARGED_PION_DECAY, 1);
        assert_eq!(ScalarPhotonModes::MUON_DECAY, 2);
        assert_eq!(ScalarPhotonModes::NEUTRAL_PION_DECAY, 4);
        assert_eq!(ScalarPhotonModes::TWO_PHOTON_LINE, 8);
        assert_eq!(ScalarPhotonModes::ELECTRON_FSR, 16);
        assert_eq!(ScalarPhotonModes::CHARGED_PION_FSR, 32);
        assert_eq!(ScalarPhotonModes::MUON_FSR, 64);
    }

    #[test]
    fn scalar_photon_default_mode_list_sets_every_bit() {
        // The list `_scalar_mediator_spectra.py` passes when the caller
        // gives none.
        let default = [
            "pi pi", "mu mu", "pi0 pi0", "g g", "e e g", "pi pi g", "mu mu g",
        ];
        let modes = ScalarPhotonModes::from_names(default);
        assert_eq!(modes.bits(), 127);
        for bit in [
            ScalarPhotonModes::CHARGED_PION_DECAY,
            ScalarPhotonModes::MUON_DECAY,
            ScalarPhotonModes::NEUTRAL_PION_DECAY,
            ScalarPhotonModes::TWO_PHOTON_LINE,
            ScalarPhotonModes::ELECTRON_FSR,
            ScalarPhotonModes::CHARGED_PION_FSR,
            ScalarPhotonModes::MUON_FSR,
        ] {
            assert!(modes.bits() & bit != 0);
        }
    }

    #[test]
    fn scalar_photon_modes_ignore_unknown_and_repeated_names() {
        // Both are the Cython's `if "x" in modes: bitflag += BIT`
        // reduction: a repeat cannot carry into the next bit, and an
        // unknown name contributes nothing and raises nothing.
        let modes = ScalarPhotonModes::from_names(["mu mu", "mu mu", "not a mode", ""]);
        assert_eq!(modes.bits(), ScalarPhotonModes::MUON_DECAY);
        assert_eq!(modes.bits() & ScalarPhotonModes::NEUTRAL_PION_DECAY, 0);
        assert_eq!(
            ScalarPhotonModes::from_names::<[&str; 0], &str>([]).bits(),
            0
        );
        assert_eq!(ScalarPhotonModes::default().bits(), 0);
    }
}
