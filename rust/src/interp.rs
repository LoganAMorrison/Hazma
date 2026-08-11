//! `np.interp`, reproduced exactly.
//!
//! PyO3-free (`projects/cython-to-rust/rules.md`, Rust conventions
//! rule 3); [`crate::interp_probe`] is the Python-visible half.
//!
//! # Sources and licensing
//!
//! This is an independent Rust implementation of the behavior of
//! `numpy.interp`, written against NumPy's documented contract and the
//! observable behavior of the shipped binary, then pinned against it by
//! exhaustive comparison in `test/test_core_interp.py`. No NumPy source
//! is transcribed. NumPy is BSD-3-Clause, which
//! `projects/cython-to-rust/adrs/ADR-0002-license-clean-numerics.md`
//! permits in any case — the rule it enforces is that nothing
//! GSL-derived (GPL-3) enters the tree, and nothing here is.
//!
//! # What calls this, and with what
//!
//! Five rest-frame photon spectra reach `np.interp` inside `cdef`
//! functions, always as `np.interp(energy, table_energies, table_dnde)`
//! on an ascending 100- or 500-row table read from a shipped CSV:
//! `hazma/spectra/_photon/_eta.pyx:44`, `_eta_prime.pyx:48`,
//! `_kaon.pyx:127`, `_omega.pyx:48`, `_phi.pyx:47`. The four mediator
//! spectrum modules call it the same way on their own tables
//! (`hazma/{scalar,vector}_mediator/*_decay_spectrum.pyx`,
//! `*_positron_spec.pyx`). Every call site passes a scalar `x`, so the
//! signature here is scalar-in / scalar-out and the array form lives in
//! the probe.
//!
//! # Why `mul_add`
//!
//! The interpolation step is written `slope.mul_add(x - xp[j], fp[j])`
//! rather than `slope * (x - xp[j]) + fp[j]`, because that is what the
//! shipped NumPy computes. C compilers contract `a * b + c` into a fused
//! multiply-add by default (`-ffp-contract=on`), and on the corpus's
//! capturing platform — macOS/arm64 — NumPy's `arr_interp` does exactly
//! that. Measured on the eta and charged-kaon tables over 20,000 random
//! abscissae plus every node and every node nudged by one part in 1e13:
//! the fused form is **bit-equal to `np.interp` at every point**, and the
//! unfused form misses it at 1,549 of 20,204 points, by up to 1.1e-13
//! relative (`test/test_core_interp.py::TestAgainstNumpy`).
//!
//! That trade is worth stating, because it is not free. `mul_add` is a
//! single instruction only where the target has hardware FMA; elsewhere
//! it lowers to a libm `fma()` call. And a NumPy built for a target
//! *without* FMA does not contract, so on such a platform this function
//! would differ from its `np.interp` by that same ≤1.1e-13 — an order
//! inside the 1e-12 budget `test/parity/tolerances.py` sets for the
//! tabulated spectra, where the unfused form's error against the corpus
//! is not (see [`crate::boost`]).

/// Linear interpolation on an ascending grid — `numpy.interp(x, xp, fp)`.
///
/// Reproduces NumPy's full contract for the three-argument form:
///
/// * outside the grid the result is **clamped**, not extrapolated:
///   `fp[0]` below `xp[0]`, `fp[len - 1]` above `xp[len - 1]`;
/// * an exact node hit returns that node's value, which is also NumPy's
///   guard against a non-finite interpolation at a node;
/// * `NaN` propagates — except on a one-point grid, where NumPy has no
///   `NaN` check and every argument returns `fp[0]`. That asymmetry is
///   NumPy's, reproduced deliberately and pinned by test.
///
/// # Parameters
///
/// * `x` — the abscissa to evaluate at, in the grid's units.
/// * `xp` — grid abscissae, **ascending**. NumPy does not check this and
///   neither does this function; an unsorted grid gives NumPy's own
///   (meaningless) answer.
/// * `fp` — grid ordinates, same length as `xp`.
///
/// # Panics
///
/// Panics if `xp` is empty or `xp.len() != fp.len()`. NumPy raises
/// `ValueError` for both; [`crate::interp_probe`] raises it with NumPy's
/// wording before this function is reached, so the panic is a
/// caller-side bug rather than a reachable path from Python.
pub fn interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    assert!(!xp.is_empty(), "interp: the grid must not be empty");
    assert!(
        xp.len() == fp.len(),
        "interp: xp and fp must have the same length"
    );

    let n = xp.len();

    // NumPy's one-point branch, which runs *before* its NaN check and so
    // returns `fp[0]` for every argument including NaN.
    if n == 1 {
        return fp[0];
    }

    if x.is_nan() {
        return x;
    }

    match search(x, xp) {
        Slot::Below => fp[0],
        Slot::Above => fp[n - 1],
        Slot::Cell(j) if j == n - 1 => fp[j],
        // A node hit takes the node's value rather than interpolating,
        // so a zero-width or non-finite cell cannot turn an exact
        // abscissa into a NaN.
        Slot::Cell(j) if xp[j] == x => fp[j],
        Slot::Cell(j) => {
            let slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
            // Fused, to match the shipped NumPy — see the module docs.
            let value = slope.mul_add(x - xp[j], fp[j]);
            if !value.is_nan() {
                return value;
            }
            // NumPy's two-step NaN rescue: try the cell's other end, and
            // if that fails too fall back to the node value when the cell
            // is flat. Reachable when a cell edge is infinite.
            let from_right = slope.mul_add(x - xp[j + 1], fp[j + 1]);
            if !from_right.is_nan() {
                from_right
            } else if fp[j] == fp[j + 1] {
                fp[j]
            } else {
                from_right
            }
        }
    }
}

/// Where `x` sits relative to an ascending grid.
enum Slot {
    /// Strictly below `xp[0]`.
    Below,
    /// Strictly above `xp[len - 1]`.
    Above,
    /// In the grid: the largest `j` with `xp[j] <= x`.
    Cell(usize),
}

/// NumPy's `binary_search_with_guess`, without the guess.
///
/// Returns the largest index `j` with `xp[j] <= x`, or [`Slot::Below`] /
/// [`Slot::Above`] outside the grid. NumPy's version starts from the
/// previous result and expands outwards before bisecting, which is a
/// speedup for a sorted query array and cannot change the index it
/// returns; this one bisects immediately.
fn search(x: f64, xp: &[f64]) -> Slot {
    let n = xp.len();
    if x > xp[n - 1] {
        return Slot::Above;
    }
    if x < xp[0] {
        return Slot::Below;
    }

    // `x >= xp[0]` here, so `imin` ends at 1 or more and the subtraction
    // cannot underflow. Ties resolve to the *last* matching index,
    // matching NumPy.
    let mut imin = 0usize;
    let mut imax = n;
    while imin < imax {
        let imid = imin + ((imax - imin) >> 1);
        if x >= xp[imid] {
            imin = imid + 1;
        } else {
            imax = imid;
        }
    }
    Slot::Cell(imin - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A grid with unequal cells, so a wrong cell index cannot pass by
    /// symmetry.
    const XP: [f64; 5] = [0.0, 1.0, 3.0, 4.0, 8.0];
    const FP: [f64; 5] = [10.0, 20.0, 0.0, -5.0, 15.0];

    #[test]
    fn clamps_outside_the_grid() {
        assert_eq!(interp(-1.0, &XP, &FP), 10.0);
        assert_eq!(interp(-1e300, &XP, &FP), 10.0);
        assert_eq!(interp(9.0, &XP, &FP), 15.0);
        assert_eq!(interp(f64::INFINITY, &XP, &FP), 15.0);
        assert_eq!(interp(f64::NEG_INFINITY, &XP, &FP), 10.0);
    }

    #[test]
    fn returns_node_values_at_nodes() {
        for (&x, &f) in XP.iter().zip(FP.iter()) {
            assert_eq!(interp(x, &XP, &FP), f);
        }
    }

    #[test]
    fn interpolates_linearly_inside_a_cell() {
        // Midpoint of [1, 3] with values 20 -> 0.
        assert_eq!(interp(2.0, &XP, &FP), 10.0);
        // Quarter point of [4, 8] with values -5 -> 15.
        assert_eq!(interp(5.0, &XP, &FP), 0.0);
    }

    #[test]
    fn propagates_nan_on_a_multi_point_grid() {
        assert!(interp(f64::NAN, &XP, &FP).is_nan());
    }

    #[test]
    fn a_one_point_grid_never_propagates_nan() {
        // NumPy checks for NaN only on the multi-point path, so its
        // one-point branch answers `fp[0]` to everything.
        assert_eq!(interp(f64::NAN, &[2.0], &[7.0]), 7.0);
        assert_eq!(interp(-1.0, &[2.0], &[7.0]), 7.0);
        assert_eq!(interp(5.0, &[2.0], &[7.0]), 7.0);
    }

    #[test]
    fn duplicate_nodes_resolve_to_the_last_match() {
        // NumPy's bisection walks past equal keys, so `x = 1` lands on
        // the *second* copy and returns its value.
        let xp = [0.0, 1.0, 1.0, 2.0];
        let fp = [0.0, 5.0, 9.0, 9.0];
        assert_eq!(interp(1.0, &xp, &fp), 9.0);
    }

    #[test]
    fn the_last_cell_index_short_circuits_to_the_node() {
        // `x == xp[n - 1]` reaches `Slot::Cell(n - 1)`, where there is no
        // cell to the right; NumPy returns the node rather than reading
        // past the end.
        assert_eq!(interp(8.0, &XP, &FP), 15.0);
    }

    #[test]
    fn an_infinite_ordinate_falls_back_to_the_other_end() {
        // The slope is -inf, so the left-anchored form is
        // `fma(-inf, 0.5, inf)` = NaN; NumPy retries anchored on the
        // right node, where `fma(-inf, -0.5, 0)` = +inf.
        let xp = [0.0, 1.0];
        let fp = [f64::INFINITY, 0.0];
        assert_eq!(interp(0.0, &xp, &fp), f64::INFINITY);
        assert_eq!(interp(0.5, &xp, &fp), f64::INFINITY);
    }

    #[test]
    fn a_nan_from_both_ends_survives_unless_the_cell_is_flat() {
        // Both anchors give NaN when the cell is infinitely wide. NumPy
        // rescues the flat case and leaves the rest NaN.
        let flat = [f64::NEG_INFINITY, f64::INFINITY];
        assert_eq!(interp(0.0, &flat, &[3.0, 3.0]), 3.0);
        assert!(interp(0.0, &flat, &[3.0, 4.0]).is_nan());
    }

    #[test]
    #[should_panic(expected = "must not be empty")]
    fn rejects_an_empty_grid() {
        interp(0.0, &[], &[]);
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn rejects_mismatched_lengths() {
        interp(0.0, &[0.0, 1.0], &[0.0]);
    }
}
