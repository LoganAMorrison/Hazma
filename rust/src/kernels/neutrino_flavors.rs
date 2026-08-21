//! The per-flavor spectrum record, ported from
//! `hazma/spectra/_neutrino/_neutrino.pyx`.
//!
//! PyO3-free (`projects/cython-to-rust/rules.md`, Rust conventions
//! rule 3). The whole `.pyx` is a three-`double` `cdef struct` plus a
//! zeroing constructor, and it exists because Cython has no way to return
//! three values from a `cdef` without one.
//!
//! # Why the module is not named for its `.pyx`
//!
//! [`super`]'s convention is one submodule per ported `.pyx`, named for
//! it — [`super::positron_muon`] is `_positron/_muon.pyx`. Applied
//! literally, `_neutrino/_neutrino.pyx` would give `neutrino_neutrino`,
//! and shortening it to `neutrino` would collide in the reader's eye with
//! [`crate::neutrino`], which is the PyO3 registration module and a
//! different layer entirely. So this one is named for what it holds. It
//! is the second documented exception, after [`super::photon_tables`].
//!
//! # What changes in the port
//!
//! [`NeutrinoSpectrumPoint`] keeps the Cython's name and field order —
//! electron, muon, tau — because that order is the row order of the
//! `(3, N)` array the public entry points return, and a silent permutation
//! there would be invisible to any tolerance. What it gains is that the
//! rows are *named* at every site that builds one, where the Cython's
//! `spec_view[0][i]` was positional.
//!
//! `new_neutrino_spectrum_point()` becomes [`NeutrinoSpectrumPoint::ZERO`]:
//! a `const`, so the "start from zero and fill what applies" idiom every
//! neutrino kernel uses costs nothing and cannot be forgotten. The tau row
//! is never written by any kernel in the tree — no hadron in hazma's
//! spectra decays to a tau neutrino at these energies — so it is zero in
//! every returned triple, which [`super::neutrino_muon`] and
//! [`super::neutrino_pion`] both assert.

/// One point of a neutrino spectrum, decomposed into flavors.
///
/// Every field is `dN/dE` in MeV⁻¹ at one energy, for that flavor and its
/// antiparticle summed as the Cython sums them.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct NeutrinoSpectrumPoint {
    /// The electron-neutrino spectrum, MeV⁻¹.
    pub electron: f64,
    /// The muon-neutrino spectrum, MeV⁻¹.
    pub muon: f64,
    /// The tau-neutrino spectrum, MeV⁻¹.
    pub tau: f64,
}

impl NeutrinoSpectrumPoint {
    /// All three flavors zero — `new_neutrino_spectrum_point()`.
    pub const ZERO: Self = Self {
        electron: 0.0,
        muon: 0.0,
        tau: 0.0,
    };

    /// The triple in the row order [`crate::dispatch::map_flavors`] writes:
    /// electron, muon, tau.
    #[must_use]
    pub const fn to_array(self) -> [f64; 3] {
        [self.electron, self.muon, self.tau]
    }
}

#[cfg(test)]
mod tests {
    use super::NeutrinoSpectrumPoint;

    /// `ZERO` is the derived `Default`, and both are the Cython
    /// constructor's three explicit `0.0` assignments.
    ///
    /// Compared on the bit pattern: `-0.0 == 0.0` is true, and a
    /// constructor that produced negative zeros would pass an `==`
    /// comparison while changing what a below-threshold spectrum stores.
    /// `test/parity/tolerances.py` treats a stored zero as exact, so that
    /// distinction is load-bearing.
    #[test]
    fn the_zero_point_is_three_positive_zeros() {
        for value in NeutrinoSpectrumPoint::ZERO.to_array() {
            assert_eq!(value.to_bits(), 0.0_f64.to_bits());
        }
        assert_eq!(
            NeutrinoSpectrumPoint::default(),
            NeutrinoSpectrumPoint::ZERO
        );
    }

    /// The row order is electron, muon, tau — the Cython's, and the one
    /// the `(3, N)` return shape publishes.
    ///
    /// Three distinct values, so a permutation cannot pass: with equal
    /// rows, a swapped pair or a row written twice would satisfy any
    /// value-by-value assertion.
    #[test]
    fn to_array_preserves_the_published_row_order() {
        let point = NeutrinoSpectrumPoint {
            electron: 1.0,
            muon: 2.0,
            tau: 3.0,
        };
        assert_eq!(point.to_array(), [1.0, 2.0, 3.0]);
    }
}
