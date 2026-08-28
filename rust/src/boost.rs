//! The boost kernels of `hazma/_utils/boost.pyx`, ported.
//!
//! PyO3-free (`projects/cython-to-rust/rules.md`, Rust conventions
//! rule 3); [`crate::boost_probe`] is the Python-visible half.
//!
//! Four routines, in the order the spectra call them: [`boost_gamma`]
//! and [`boost_beta`] turn a parent's energy and mass into boost
//! parameters, [`boost_delta_function`] boosts a two-body line, and
//! [`boost_integrate_linear_interp`] boosts a tabulated continuum. The
//! `.pyx` also carried `boost_jac` and `boost_eng`; both were exported
//! through `__pyx_capi__` and called by nothing in the tree (checked with
//! `rg` at execution time, 2026-08-10), so they were left out of this
//! port's scope and went with the file when Phase 06 Task 6.4 deleted it.
//!
//! # What calls these, and with what
//!
//! The `.pyx` column is the pre-port record and every file in it has
//! since been deleted — the last four by Phase 06 Task 6.4. Task 4.2
//! replaced the five tabulated photon files with
//! [`crate::kernels::photon_tables`], which calls all four of these
//! natively and is the only kernel that reaches
//! [`boost_integrate_linear_interp`] at all. [`crate::boost_probe`]
//! exposes these to Python for `test/test_core_boost.py` only.
//!
//! | Function | Cython call sites |
//! | --- | --- |
//! | [`boost_beta`] / [`boost_gamma`] | every `_photon`, `_positron` and `_neutrino` kernel that boosts out of a rest frame — `_photon/{_eta,_eta_prime,_kaon,_omega,_phi,_pion,_rho}.pyx`, `_positron/{_muon,_pion}.pyx`, `_neutrino/_pion.pyx` |
//! | [`boost_delta_function`] | `_photon/{_eta,_eta_prime,_kaon,_omega,_phi}.pyx` (the line terms), `_positron/_pion.pyx`, `_neutrino/_pion.pyx` |
//! | [`boost_integrate_linear_interp`] | `_photon/{_eta,_eta_prime,_kaon,_omega,_phi}.pyx` — seven tabulated spectra over 100- or 500-row CSVs |
//!
//! # Why `mul_add`, and where it is *not* used
//!
//! Twelve multiply-adds in this module are written `a.mul_add(b, c)`
//! instead of `a * b + c` — eight distinct expressions in the `.pyx`,
//! three of which appear once per partial cell. That is not a rewrite:
//! it is what the shipped Cython computes. C compilers contract
//! `a * b + c` into a fused multiply-add by default
//! (`-ffp-contract=on`), and the parity corpus was captured from a
//! macOS/arm64 build that does. Each site below was
//! confirmed by disassembling the shipped
//! `hazma/_utils/boost.cpython-312-darwin.so` (`fmsub` / `fmadd`
//! instructions) *and*, while the extension still built, by bisection
//! against the live kernel through `__pyx_capi__`; both records are in
//! `projects/cython-to-rust/task-notes/phase-03/task-3.4-interp-boost.md`.
//!
//! The measurement that makes this load-bearing rather than pedantic:
//! evaluated on the parity corpus's own grids for all seven tabulated
//! photon spectra, the unfused arithmetic misses the corpus by up to
//! **3.6e-12** relative, against the 1e-12 `TABULATED` budget in
//! `test/parity/tolerances.py`. The fused form is bit-equal at every one
//! of those points. A Phase 04 swap written the obvious way would have
//! failed its own gate, and widening the budget instead would have hidden
//! a three-decade-wider class of real errors.
//!
//! Where the Cython does *not* contract, neither does this module.
//! [`boost_beta`] is the case that matters: `(mass / energy) ** 2` is a
//! rounded product before the subtraction, and none of the ten inlining
//! call sites contract `1.0 - t` (checked by disassembly in `_eta`,
//! `_kaon` and `_positron/_pion`). Writing it fused would move every
//! boosted spectrum for no reason.

/// Errors [`boost_integrate_linear_interp`] returns instead of panicking.
///
/// The Cython twin states two of these as `assert`s and leaves the third
/// undefined; `projects/cython-to-rust/rules.md` rule 9 (Rust conventions
/// 4) makes every such guard an unconditional error return.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BoostError {
    /// `beta` outside `(0, 1)` — the Cython's `assert 0.0 < beta < 1.0`.
    ///
    /// The integral divides by `beta` and by `sqrt(1 - beta^2)`, so both
    /// endpoints are singular. Callers short-circuit to the rest-frame
    /// value when the parent is within one epsilon of rest, which is why
    /// the live path never reaches this.
    BetaOutOfRange {
        /// The `beta` that was passed.
        beta: f64,
    },
    /// The table's two columns have different lengths — the Cython's
    /// `assert npts == len(y)`.
    LengthMismatch {
        /// Length of the abscissa column.
        x: usize,
        /// Length of the ordinate column.
        y: usize,
    },
    /// The table is empty.
    ///
    /// New in the port. The Cython reads `x[npts - 1]` with
    /// `wraparound(False)` and no bounds check, so an empty table is
    /// undefined behavior there rather than an error.
    EmptyTable,
}

impl std::fmt::Display for BoostError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BoostError::BetaOutOfRange { beta } => {
                write!(f, "boost velocity must satisfy 0 < beta < 1; got {beta}")
            }
            BoostError::LengthMismatch { x, y } => write!(
                f,
                "spectrum table columns must have equal length; got {x} energies and {y} values"
            ),
            BoostError::EmptyTable => write!(f, "spectrum table must not be empty"),
        }
    }
}

impl std::error::Error for BoostError {}

/// The Lorentz factor of a particle — `γ = E / m`.
///
/// # Parameters
///
/// * `energy` — the particle's energy, MeV.
/// * `mass` — the particle's mass, MeV.
///
/// # Returns
///
/// The dimensionless `γ`. No guard: `mass = 0` gives `±∞` here exactly as
/// in the Cython, and every call site passes a hadron or lepton mass.
pub fn boost_gamma(energy: f64, mass: f64) -> f64 {
    energy / mass
}

/// The velocity of a particle — `β = sqrt(1 − (m/E)²)`, in units of `c`.
///
/// # Parameters
///
/// * `energy` — the particle's energy, MeV.
/// * `mass` — the particle's mass, MeV.
///
/// # Returns
///
/// The dimensionless `β`. `energy < mass` gives `NaN`, as in the Cython;
/// callers guard the near-rest region themselves with an
/// `E − m < DBL_EPSILON` short circuit.
///
/// Deliberately **not** fused — see the module docs. The Cython spells
/// this `(mass / energy) ** 2`, whose rounded product is complete before
/// the subtraction, and no call site's generated code contracts it.
pub fn boost_beta(energy: f64, mass: f64) -> f64 {
    let ratio = mass / energy;
    (1.0 - ratio * ratio).sqrt()
}

/// Boost a rest-frame line `δ(E − e0)` into the lab frame.
///
/// The boosted line is flat in energy across the window the boost opens,
/// `[γ(e − βk), γ(e + βk)]`, with height `1 / (2 γ β k₀)` where
/// `k₀ = sqrt(e0² − m²)` is the product's rest-frame momentum.
///
/// # Parameters
///
/// * `e0` — the line's rest-frame energy, MeV.
/// * `e` — the product's lab-frame energy, MeV.
/// * `m` — the product's mass, MeV (`0` for a photon or neutrino).
/// * `beta` — the parent's boost velocity, in units of `c`.
///
/// # Returns
///
/// `dN/dE` in MeV⁻¹ inside the window, and `0` outside it. `0` is also
/// what an unphysical argument gives — `beta > 1`, `beta <= 0`, or
/// `e < m` — reproducing the Cython's guard rather than raising, because
/// callers sum this term into a spectrum and rely on it vanishing.
pub fn boost_delta_function(e0: f64, e: f64, m: f64, beta: f64) -> f64 {
    if beta > 1.0 || beta <= 0.0 || e < m {
        return 0.0;
    }

    let gamma = 1.0 / (-beta).mul_add(beta, 1.0).sqrt();
    let k = e.mul_add(e, -(m * m)).sqrt();
    let eminus = gamma * (-beta).mul_add(k, e);
    let eplus = gamma * beta.mul_add(k, e);

    if eminus < e0 && e0 < eplus {
        let k0 = e0.mul_add(e0, -(m * m)).sqrt();
        return 1.0 / (2.0 * gamma * beta * k0);
    }

    0.0
}

/// Boost a tabulated rest-frame spectrum into the lab frame.
///
/// Evaluates
///
/// ```text
///   dN            1     ⌠ ub  y(x)
///   ──(E, β)  =  ────   ⎮     ──── dx ,
///   dE           2 γ β  ⌡ lb    x
/// ```
///
/// with `lb = E γ (1 − β)` and `ub = E γ (1 + β)`, over the rest-frame
/// spectrum tabulated as `(x, y)`. The integral is the trapezoidal rule
/// across whole interior cells, plus a closed-form correction on each
/// partial cell at the two ends, plus an analytic `1/E` tail when `lb`
/// falls below the table.
///
/// # Parameters
///
/// * `photon_energy` — the lab-frame energy to evaluate at, MeV. (The
///   Cython's name; the routine is energy-agnostic and the positron and
///   neutrino families would use it the same way.)
/// * `beta` — the parent's boost velocity, in units of `c`, in `(0, 1)`.
/// * `x` — the table's rest-frame energies, MeV, ascending.
/// * `y` — the table's `dN/dE` values, MeV⁻¹.
///
/// # Returns
///
/// The boosted `dN/dE` in MeV⁻¹, or a [`BoostError`] for the guards the
/// Cython asserts.
///
/// # Faithfulness notes
///
/// Four details are reproduced rather than repaired, per
/// `projects/cython-to-rust/rules.md` rule 1 — the corpus pins what the
/// Cython returns, and a repair is a separate declared change. The first
/// two are the same off-by-one read from opposite sides and are a real
/// defect, tracked in
/// `docs/followups/todo/boost-integral-drops-last-interior-cell.md`:
///
/// * the trapezoid runs over `x[ilow..ihigh]`, an **exclusive** upper
///   bound, while the upper partial-cell term starts at `x[ihigh]` — so
///   `[x[ihigh - 1], x[ihigh]]` is covered by *nothing*. When `ub` is
///   clamped, `ihigh` is the last index and the upper term is skipped
///   too, so the table's final row contributes to no term at all;
/// * when both bounds fall inside one cell, `ihigh = ilow - 1` and the
///   two partial-cell terms **overlap** instead, covering about two whole
///   cells rather than the sliver between the bounds. The over-count is
///   `cell width / window width`, which diverges as `β → 0` — which is
///   why all seven tabulated photon spectra blow up near threshold
///   rather than converging to their rest-frame values;
/// * cell membership is decided with a **1e-6 absolute** tolerance on
///   energies that span six decades, so "the bound sits on a node" means
///   something different at 0.005 MeV than at 1000 MeV;
/// * the below-table tail assumes `y ∝ 1/E` and is added whole, even when
///   `ub` also falls below the table (that case returns earlier).
///
/// One behavior is deliberately **not** reproduced. A `NaN`
/// `photon_energy` makes both bounds `NaN`, every comparison below false,
/// and the Cython's `np.flatnonzero(lb <= x)[0]` an index into an empty
/// array — so `dnde_photon_eta(float('nan'), 1000.0)` raises `IndexError`
/// on the shipped build (measured, cython-to-rust Task 4.2). There is no
/// faithful way to carry that across: the port evaluates a grid element
/// by element behind `dispatch::map_unary`, which has no per-element
/// error channel, so reproducing it would mean panicking — a
/// `PanicException` where Python could catch an `IndexError` today. This
/// function answers `NaN` instead, which is what the same kernels' own
/// rest-frame branch already does (`np.interp` propagates) and what the
/// rest of the port does with a `NaN` energy. The parity corpus samples
/// no `NaN` abscissa, so no pinned value moves.
pub fn boost_integrate_linear_interp(
    photon_energy: f64,
    beta: f64,
    x: &[f64],
    y: &[f64],
) -> Result<f64, BoostError> {
    if !(0.0 < beta && beta < 1.0) {
        return Err(BoostError::BetaOutOfRange { beta });
    }
    if x.len() != y.len() {
        return Err(BoostError::LengthMismatch {
            x: x.len(),
            y: y.len(),
        });
    }
    let npts = x.len();
    if npts == 0 {
        return Err(BoostError::EmptyTable);
    }

    let xmax = x[npts - 1];
    let x0 = x[0];
    let y0 = y[0];

    // Fused: the Cython's `1.0 - beta * beta` contracts to `fmsub`.
    let gamma = 1.0 / (-beta).mul_add(beta, 1.0).sqrt();
    let mut lb = photon_energy * gamma * (1.0 - beta);
    let mut ub = photon_energy * gamma * (1.0 + beta);

    // A `NaN` window: see the "Faithfulness notes" above for why this
    // answers `NaN` rather than reproducing the Cython's `IndexError`.
    // Checked on the bounds rather than on `photon_energy` because they
    // are what the dead end below is reached through; with `beta` inside
    // `(0, 1)` the two conditions coincide, since `1 - beta*beta` cannot
    // round to zero for any representable `beta < 1` and so `gamma` is
    // always finite.
    if lb.is_nan() || ub.is_nan() {
        return Ok(f64::NAN);
    }

    // The whole boosted window sits above the table: nothing to integrate.
    if lb > xmax {
        return Ok(0.0);
    }
    // The whole window sits below it: only the analytic 1/E tail survives,
    // and it integrates to the closed form below.
    if ub < x0 {
        return Ok(y0 * x0 / photon_energy);
    }

    let mut integral = 0.0;
    // `-1` is the Cython's "not yet decided" sentinel for both indices;
    // `Option` carries it here without a magic value.
    let mut ilow: Option<usize> = None;
    let mut ihigh: Option<usize> = None;

    if ub > xmax {
        ub = xmax;
        ihigh = Some(npts - 1);
    }

    if lb < x0 {
        let rat = (1.0 - beta) * photon_energy * gamma / x0;
        integral += y0 * (1.0 - rat) / rat;
        lb = x0;
        ilow = Some(0);
    }

    // `yy[i] = y[i] / x[i]` — the Cython materialises the whole column;
    // reading it pointwise gives the same values without the allocation.
    let yy = |i: usize| y[i] / x[i];

    // `np.flatnonzero(bound <= x)[0]`: the first index at or above the
    // bound. Both bounds are inside the table by the early returns above,
    // so both scans find one.
    let ilow = ilow.unwrap_or_else(|| {
        (0..npts)
            .find(|&i| lb <= x[i])
            .expect("lb <= xmax, so some node is at or above it")
    });
    let ihigh = ihigh.unwrap_or_else(|| {
        let first = (0..npts)
            .find(|&i| ub <= x[i])
            .expect("ub <= xmax, so some node is at or above it");
        // Step back unless that node *is* the bound, so the interior sum
        // never reaches past `ub`. The 1e-6 is absolute and the Cython's.
        if (x[first] - ub).abs() > 1e-6 {
            first - 1
        } else {
            first
        }
    });

    if ilow < ihigh {
        integral += trapezoid(&x[ilow..ihigh], ilow, &yy);
    }

    // Partial cell at the lower bound: integrate the linear interpolant
    // of `y/x` across `[lb, x[ilow]]`.
    if ilow > 0 && (x[ilow] - lb).abs() > 1e-6 {
        let x2 = x[ilow];
        let x1 = x[ilow - 1];
        let m = (yy(ilow) - yy(ilow - 1)) / (x2 - x1);
        // Fused: `y1 - m * x1`, then `0.5 * m * (x2 + lb) + b`, then the
        // accumulation itself — three `fmsub`/`fmadd` in the Cython.
        let b = (-m).mul_add(x1, yy(ilow - 1));
        let inner = (0.5 * m).mul_add(x2 + lb, b);
        integral = (x2 - lb).mul_add(inner, integral);
    }

    // Partial cell at the upper bound, same shape anchored on `x[ihigh]`.
    if ihigh < npts - 1 && (ub - x[ihigh]).abs() > 1e-6 {
        let x2 = x[ihigh + 1];
        let x1 = x[ihigh];
        let m = (yy(ihigh + 1) - yy(ihigh)) / (x2 - x1);
        let b = (-m).mul_add(x1, yy(ihigh));
        let inner = (0.5 * m).mul_add(ub + x1, b);
        integral = (ub - x1).mul_add(inner, integral);
    }

    Ok(integral / (2.0 * gamma * beta))
}

/// `np.trapezoid(yy[offset..offset + xs.len()], x=xs)`.
///
/// `xs` is the abscissa slice and `yy` reads the ordinate by *absolute*
/// index, so the caller does not have to materialise the `y / x` column
/// the Cython builds. `offset` is the absolute index of `xs[0]`.
///
/// NumPy forms the per-cell terms as one array and reduces it with
/// `ndarray.sum`, which is pairwise rather than sequential — see
/// [`pairwise_sum`]. Reproducing that is what makes this function
/// bit-equal to the call it replaces.
fn trapezoid<F>(xs: &[f64], offset: usize, yy: &F) -> f64
where
    F: Fn(usize) -> f64,
{
    if xs.len() < 2 {
        return 0.0;
    }
    let terms: Vec<f64> = (0..xs.len() - 1)
        .map(|i| ((xs[i + 1] - xs[i]) * (yy(offset + i + 1) + yy(offset + i))) / 2.0)
        .collect();
    pairwise_sum(&terms)
}

/// NumPy's pairwise summation for `float64`, reproduced.
///
/// `ndarray.sum` is not a sequential accumulation: below 8 elements it is
/// sequential, up to a 128-element block it runs eight independent
/// accumulators and combines them as a balanced tree, and above that it
/// splits in half on a multiple of 8 and recurses. The order is what the
/// result depends on, so a sequential sum here would be a different
/// number — up to 1.8e-15 relative on the 500-row tables, measured.
///
/// This mirrors an implementation detail rather than a documented
/// contract, which is a real cost: a future NumPy could reduce
/// differently. `test/test_core_boost.py::TestTrapezoid` compares against
/// the live `np.trapezoid`, so the day that happens the test says so
/// instead of the number drifting quietly.
///
/// Reused outside this module by
/// [`crate::kernels::photon_tables`], whose CSV tables are summed across
/// decay-mode columns with `numpy.sum(axis=0)` — the same reduction, and
/// on the ten-column φ table the same non-sequential answer.
pub(crate) fn pairwise_sum(values: &[f64]) -> f64 {
    /// NumPy's `PW_BLOCKSIZE`.
    const BLOCK: usize = 128;

    let n = values.len();
    if n < 8 {
        return values.iter().fold(0.0, |acc, &v| acc + v);
    }
    if n <= BLOCK {
        let mut r = [
            values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7],
        ];
        let unrolled = n - (n % 8);
        let mut i = 8;
        while i < unrolled {
            for (k, slot) in r.iter_mut().enumerate() {
                *slot += values[i + k];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += values[i];
            i += 1;
        }
        return res;
    }
    let half = (n / 2) - (n / 2) % 8;
    pairwise_sum(&values[..half]) + pairwise_sum(&values[half..])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `γ = E/m` and `β = sqrt(1 − 1/γ²)` are the same boost, so the two
    /// helpers must satisfy `γ = 1/sqrt(1 − β²)`.
    ///
    /// The tolerance is derived rather than chosen. Recovering `γ` from
    /// `β` evaluates `1 − β²`, whose true value is `1/γ²`; the rounding
    /// of `β²` is absolute at the `eps` level, so the subtraction's
    /// relative error is `~eps·γ²` and the square root halves it. `4 eps
    /// γ²` therefore covers every case with margin — and it is the
    /// cancellation that sets it, not the implementation, which is why
    /// the ultrarelativistic electron below is allowed nearly a part in
    /// 1e5 while the near-rest eta is held to a part in 1e15.
    #[test]
    fn beta_and_gamma_describe_the_same_boost() {
        for &(energy, mass) in &[(1000.0, 139.57039), (547.9, 547.862), (5e4, 0.5109989461)] {
            let beta = boost_beta(energy, mass);
            let gamma = boost_gamma(energy, mass);
            let from_beta = 1.0 / (1.0 - beta * beta).sqrt();
            let tol = 4.0 * f64::EPSILON * gamma * gamma;
            assert!(
                (gamma - from_beta).abs() <= tol * gamma,
                "gamma {gamma} vs {from_beta} at E = {energy}, m = {mass} (rtol {tol:.3e})"
            );
        }
    }

    #[test]
    fn a_particle_at_rest_has_zero_velocity_and_unit_gamma() {
        assert_eq!(boost_beta(139.57039, 139.57039), 0.0);
        assert_eq!(boost_gamma(139.57039, 139.57039), 1.0);
    }

    #[test]
    fn below_rest_energy_beta_is_nan() {
        assert!(boost_beta(100.0, 139.57039).is_nan());
    }

    /// The boosted line integrates back to 1 over its own window: it is a
    /// normalised δ that the boost only spreads out.
    #[test]
    fn the_boosted_line_carries_unit_normalisation() {
        let (e0, m, beta): (f64, f64, f64) = (200.0, 0.0, 0.6);
        let gamma = 1.0 / (1.0 - beta * beta).sqrt();
        let k0 = e0;
        let lo = gamma * (e0 - beta * k0);
        let hi = gamma * (e0 + beta * k0);
        let height = boost_delta_function(e0, 0.5 * (lo + hi), m, beta);
        let integral = height * (hi - lo);
        assert!(
            (integral - 1.0).abs() < 1e-12,
            "normalisation {integral} != 1"
        );
    }

    #[test]
    fn the_boosted_line_vanishes_outside_its_window() {
        let (e0, m, beta): (f64, f64, f64) = (200.0, 0.0, 0.6);
        let gamma = 1.0 / (1.0 - beta * beta).sqrt();
        // Just outside each edge of [γ(e0 − βe0), γ(e0 + βe0)].
        assert_eq!(
            boost_delta_function(e0, gamma * e0 * (1.0 - beta) * 0.99, m, beta),
            0.0
        );
        assert_eq!(
            boost_delta_function(e0, gamma * e0 * (1.0 + beta) * 1.01, m, beta),
            0.0
        );
    }

    #[test]
    fn the_boosted_line_rejects_unphysical_arguments() {
        assert_eq!(boost_delta_function(200.0, 200.0, 0.0, 1.5), 0.0);
        assert_eq!(boost_delta_function(200.0, 200.0, 0.0, 0.0), 0.0);
        assert_eq!(boost_delta_function(200.0, 200.0, 0.0, -0.3), 0.0);
        // Product below its own mass.
        assert_eq!(boost_delta_function(200.0, 0.1, 0.5109989461, 0.6), 0.0);
    }

    /// A flat rest-frame spectrum `y = c` boosts to `c ln(ub/lb)/(2γβ)`
    /// when the window sits inside the table.
    ///
    /// The tolerance is set by the dropped interior cell, not by the
    /// quadrature: at cell width `h` the missing cell is `h·c/ub` out of
    /// `c·ln(ub/lb)`, which at `h = 1e-3` here is 2.6e-5 of the answer,
    /// while the trapezoidal error on `c/x` over the same cells is
    /// `~1e-10`. See [`the_last_interior_cell_is_dropped`] for the pin on
    /// the drop itself; this test only has to stay loose enough to let it
    /// through.
    #[test]
    fn a_flat_table_boosts_to_the_log_of_the_window() {
        let x: Vec<f64> = (0..40_001).map(|i| 1.0 + f64::from(i) * 1e-3).collect();
        let y = vec![2.0; x.len()];
        let beta: f64 = 0.5;
        let gamma = 1.0 / (1.0 - beta * beta).sqrt();
        let energy = 20.0;
        let (lb, ub) = (energy * gamma * (1.0 - beta), energy * gamma * (1.0 + beta));
        let want = 2.0 * (ub / lb).ln() / (2.0 * gamma * beta);
        let got = boost_integrate_linear_interp(energy, beta, &x, &y).unwrap();
        assert!(
            (got - want).abs() < 1e-4 * want,
            "boosted flat spectrum {got} vs {want}"
        );
    }

    /// The interior sum stops one cell short of `ihigh`, so the cell
    /// `[x[ihigh - 1], x[ihigh]]` is covered by nothing.
    ///
    /// Reproduced, not repaired — `projects/cython-to-rust/rules.md`
    /// rule 1, and
    /// `docs/followups/todo/boost-integral-drops-last-interior-cell.md`.
    /// The numbers are hand-computable: with `y = x` the integrand `y/x`
    /// is 1, `beta = 0.6` gives `gamma = 1.25`, and `E = 2.2` gives
    /// `lb = 1.1` and `ub = 4.4`, clamped to `xmax = 4`. The Cython then
    /// covers `[1.1, 2]` (lower partial cell) and `[2, 3]` (the interior
    /// sum) and nothing else, for `1.9 / (2γβ) = 1.9 / 1.5`. The true
    /// integral over `[1.1, 4]` would be `2.9 / 1.5`.
    ///
    /// Checked against the live Cython through `__pyx_capi__` while that
    /// extension still existed: it returned this same 1.2666666666666666
    /// (Task 3.4 task note). The literal is what carries that measurement
    /// now.
    #[test]
    fn the_last_interior_cell_is_dropped() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = x;
        let got = boost_integrate_linear_interp(2.2, 0.6, &x, &y).unwrap();
        assert_eq!(got, 1.9 / 1.5);
        assert_ne!(got, 2.9 / 1.5);
    }

    /// A `NaN` energy propagates instead of panicking.
    ///
    /// The Cython raises `IndexError` here; reproducing that from inside
    /// an element-wise map would mean a panic, so the port answers `NaN`
    /// — see the function's "Faithfulness notes".
    ///
    /// The infinities are checked alongside because they look like the
    /// same case and are not: they reach the window comparisons with a
    /// definite answer, so they return a number rather than `NaN`, at
    /// every `beta` the guard admits. `beta` one ulp below 1 is the
    /// extreme, and shows `gamma` staying finite — `1 - beta*beta` is
    /// 2.2e-16 there, not 0.
    #[test]
    fn a_nan_window_is_nan_rather_than_a_panic() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = x;
        let extreme = f64::from_bits(1.0_f64.to_bits() - 1);
        for beta in [0.6, extreme] {
            assert!(
                boost_integrate_linear_interp(f64::NAN, beta, &x, &y)
                    .unwrap()
                    .is_nan(),
                "beta = {beta}"
            );
            assert_eq!(
                boost_integrate_linear_interp(f64::INFINITY, beta, &x, &y).unwrap(),
                0.0
            );
            assert_eq!(
                boost_integrate_linear_interp(f64::NEG_INFINITY, beta, &x, &y).unwrap(),
                -0.0
            );
        }
    }

    #[test]
    fn a_window_entirely_above_the_table_is_zero() {
        let x = [1.0, 2.0, 3.0];
        let y = [1.0, 1.0, 1.0];
        // lb = E γ (1 − β) > 3 for a large enough E.
        assert_eq!(
            boost_integrate_linear_interp(1e6, 0.5, &x, &y).unwrap(),
            0.0
        );
    }

    #[test]
    fn a_window_entirely_below_the_table_is_the_analytic_tail() {
        let x = [1.0, 2.0, 3.0];
        let y = [7.0, 1.0, 1.0];
        let energy = 1e-6;
        let got = boost_integrate_linear_interp(energy, 0.5, &x, &y).unwrap();
        assert_eq!(got, 7.0 * 1.0 / energy);
    }

    #[test]
    fn the_guards_return_errors_rather_than_panicking() {
        let x = [1.0, 2.0];
        let y = [1.0, 1.0];
        assert_eq!(
            boost_integrate_linear_interp(1.0, 0.0, &x, &y),
            Err(BoostError::BetaOutOfRange { beta: 0.0 })
        );
        assert_eq!(
            boost_integrate_linear_interp(1.0, 1.0, &x, &y),
            Err(BoostError::BetaOutOfRange { beta: 1.0 })
        );
        assert!(matches!(
            boost_integrate_linear_interp(1.0, f64::NAN, &x, &y),
            Err(BoostError::BetaOutOfRange { .. })
        ));
        assert_eq!(
            boost_integrate_linear_interp(1.0, 0.5, &x, &[1.0]),
            Err(BoostError::LengthMismatch { x: 2, y: 1 })
        );
        assert_eq!(
            boost_integrate_linear_interp(1.0, 0.5, &[], &[]),
            Err(BoostError::EmptyTable)
        );
    }

    /// Every branch of [`pairwise_sum`]'s length dispatch, against a
    /// sequential sum on values that make the two agree exactly (powers
    /// of two lose nothing to reassociation).
    #[test]
    fn pairwise_sum_totals_exactly_representable_values() {
        for n in [0usize, 1, 7, 8, 9, 128, 129, 300, 1000] {
            let values: Vec<f64> = (0..n).map(|i| f64::from(i as u32 % 8) * 0.25).collect();
            let want: f64 = values.iter().sum();
            assert_eq!(pairwise_sum(&values), want, "n = {n}");
        }
    }

    /// The trapezoidal rule is exact on a straight line, whatever the
    /// cell widths.
    #[test]
    fn trapezoid_is_exact_on_a_linear_integrand() {
        let xs: Vec<f64> = vec![0.5, 1.0, 2.5, 3.0, 7.0, 11.5];
        // yy(i) must return 3 * xs[i] + 1, and `trapezoid` indexes it
        // absolutely from `offset`.
        let yy = |i: usize| 3.0f64.mul_add(xs[i], 1.0);
        let got = trapezoid(&xs, 0, &yy);
        let (a, b) = (xs[0], xs[xs.len() - 1]);
        let want = 1.5 * (b * b - a * a) + (b - a);
        assert!((got - want).abs() < 1e-12 * want, "{got} vs {want}");
    }
}
