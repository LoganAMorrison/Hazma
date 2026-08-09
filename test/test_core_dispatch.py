"""Cross-language plumbing: the ``hazma._core`` entry-point dispatch contract.

This module is the **template** every later kernel swap copies (cython-to-rust
Phase 02, Task 2.3). It pins the argument-conversion, return-type and error
behavior that :func:`hazma._core.dispatch.map_unary` implements once for all of
Phases 03-06, exercised through the scaffold's plumbing probe
``hazma._core.roundtrip`` -- the identity, so every assertion here is about the
*plumbing* and none of it is about physics.

The contract, from
``projects/cython-to-rust/references/numerics-replacements.md`` ("Entry-point
dispatch contract"):

* a Python ``float``, a NumPy scalar, or a 0-d ``float64`` array returns a
  Python ``float``;
* a 1-D ``float64`` array returns a **fresh** 1-D ``float64`` array of the same
  length;
* anything else raises ``ValueError``, naming the quantity the caller passed.

Copying this module for a real kernel means: keep every test below, swap
``roundtrip`` for the kernel and ``QUANTITY`` for the wording that kernel passes
to ``map_unary`` (``"Photon energies"``, ``"Positron energies"``, ...), and add
the kernel's *numerical* tests beside them. Do not merge the two halves --
plumbing failures and physics failures should not need the same debugging.

Two things this module deliberately does **not** do:

* It does not pin values against the Cython twin. That is the parity corpus's
  job (``test/parity/``), which holds all 41 consumed entry points to
  bit-equality on its capturing platform. A second, looser numerical gate here
  would only be one more thing to keep in sync.
* It does not assume the live Cython dispatch and this contract agree. They do
  not, in four measured ways recorded in the reference above; the two that
  surface at this layer -- a 0-d array (Cython raises, ``map_unary`` returns a
  float) and a Python list (Cython accepts, ``map_unary`` raises) -- are called
  out at their assertions below. **Task 3.5 decides each divergence**; until
  then these tests pin the target contract, which is what the scaffold
  implements, and are expected to be revisited by that task rather than by a
  kernel swap.

Every assertion below was measured against the built extension before it was
written, not derived from the contract prose.
"""

from __future__ import annotations

import inspect
import struct
from typing import TYPE_CHECKING

import numpy as np
import pytest

from hazma._core import roundtrip

if TYPE_CHECKING:
    from collections.abc import Callable

# The wording `roundtrip` passes to `map_unary` as its `quantity`. Every error
# message the contract produces is prefixed with it, which is how a ported
# kernel keeps its Cython twin's user-visible exception text (rules.md rule 1).
QUANTITY = "Input values"

DIMENSION_ERROR = f"{QUANTITY} must be 0 or 1-dimensional."
TYPE_ERROR = f"{QUANTITY} must be a float or a NumPy array."


def dtype_error(dtype: str) -> str:
    """The dtype-rejection message for a non-``float64`` array."""
    return f"{QUANTITY} must be a float64 array; got dtype {dtype}."


def bits(x: float) -> bytes:
    """The IEEE-754 bit pattern of ``x``.

    Compared instead of ``==`` so that ``-0.0`` is distinguished from ``0.0``
    and a NaN compares equal to itself. The kernel under test is the identity,
    so *every* input bit pattern must survive the round trip, including the
    ones ``==`` cannot see.
    """
    return struct.pack("<d", x)


# Values chosen to cover the float64 range and its special cases: signed zeros
# (invisible to `==`), a subnormal, both finite extremes, both infinities, and
# a NaN (whose payload is also preserved -- measured, not assumed).
SCALARS = [
    0.0,
    -0.0,
    1.0,
    -1.5,
    5e-324,  # smallest positive subnormal
    2.2250738585072014e-308,  # smallest positive normal
    1.7976931348623157e308,  # largest finite
    float("inf"),
    float("-inf"),
    float("nan"),
]


class TestScalarPath:
    """float in -> float out."""

    @pytest.mark.parametrize("value", SCALARS)
    def test_python_float_round_trips_bit_for_bit(self, value: float) -> None:
        result = roundtrip(value)
        assert type(result) is float
        assert bits(result) == bits(value)

    def test_numpy_float64_returns_a_python_float(self) -> None:
        # np.float64 subclasses float, so it takes the PyFloat fast path that
        # exists specifically to keep a scalar call from touching NumPy at all
        # (the `numpy` crate panics rather than raises when NumPy is absent --
        # Task 2.1). The return type is the plain builtin either way.
        result = roundtrip(np.float64(2.5))
        assert type(result) is float
        assert bits(result) == bits(2.5)

    @pytest.mark.parametrize(
        "value",
        [np.float32(2.5), np.int64(7), np.uint8(3), np.bool_(True)],
        ids=["float32", "int64", "uint8", "bool_"],
    )
    def test_other_numpy_scalars_are_accepted(self, value: np.generic) -> None:
        # These are neither `float` subclasses nor ndarrays, so they fall past
        # both fast paths to the `extract::<f64>` arm. They are still scalars
        # and the contract says a scalar returns a float.
        result = roundtrip(value)
        assert type(result) is float
        assert bits(result) == bits(float(value))

    @pytest.mark.parametrize("value", [3, True], ids=["int", "bool"])
    def test_python_int_and_bool_are_accepted(self, value: int) -> None:
        # `float(x)` semantics, matching what the Cython entry points accept
        # for an energy argument today.
        result = roundtrip(value)
        assert type(result) is float
        assert bits(result) == bits(float(value))


class TestZeroDimensionalArray:
    """0-d array in -> float out.

    This is the first of the two contract/Cython divergences this layer sees:
    the live Cython raises ``AssertionError`` on a 0-d array because
    ``ndarray`` defines ``__len__`` on the *type*, so its ``hasattr(x,
    '__len__')`` dispatch sends a 0-d array down the array path and the
    ``len(shape) == 1`` guard then rejects it. ``map_unary`` treats it as the
    scalar it is. Task 3.5 decides whether the ported entry points keep the
    Cython behavior or take this (widening, user-visible) change.
    """

    @pytest.mark.parametrize("value", SCALARS)
    def test_zero_dim_float64_array_returns_a_python_float(self, value: float) -> None:
        result = roundtrip(np.array(value))
        assert type(result) is float
        assert bits(result) == bits(value)

    def test_zero_dim_array_still_enforces_dtype(self) -> None:
        # The scalar path an ndarray takes is inside the array branch, so the
        # dtype check applies to it too -- a 0-d int64 array is rejected even
        # though a Python int is accepted. That asymmetry is deliberate: the
        # int comes through `extract::<f64>`, the array through a typed view.
        with pytest.raises(ValueError, match=r"got dtype int64\."):
            roundtrip(np.array(4, dtype=np.int64))


class TestArrayPath:
    """1-D float64 array in -> fresh 1-D float64 array out."""

    def test_dtype_shape_and_values(self) -> None:
        values = np.array([1.0, 2.0, 3.5, -7.25])
        result = roundtrip(values)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        assert result.shape == values.shape
        assert result.tobytes() == values.tobytes()

    def test_special_values_survive_bit_for_bit(self) -> None:
        values = np.array(SCALARS, dtype=np.float64)
        assert roundtrip(values).tobytes() == values.tobytes()

    def test_empty_array_returns_an_empty_array(self) -> None:
        # A length-0 array is a legitimate energy grid, and the mapping loop
        # must not special-case it into a scalar or an error.
        result = roundtrip(np.array([], dtype=np.float64))
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        assert result.shape == (0,)

    def test_non_contiguous_input_is_read_in_order(self) -> None:
        # `map_unary` iterates the ndarray *view*, not the underlying buffer,
        # so a strided slice must yield its own elements rather than the
        # buffer's first n. This test fails loudly if the array path is ever
        # rewritten to read a raw pointer.
        values = np.arange(6.0)[::2]
        assert not values.flags.c_contiguous
        result = roundtrip(values)
        assert result.tobytes() == np.array([0.0, 2.0, 4.0]).tobytes()
        assert result.flags.c_contiguous

    def test_readonly_input_is_accepted(self) -> None:
        # The view taken on the caller's array is read-only, so a read-only
        # input must not be rejected -- callers pass sliced or frozen grids.
        values = np.array([1.0, 2.0])
        values.flags.writeable = False
        assert roundtrip(values).tobytes() == values.tobytes()

    def test_result_is_a_fresh_array_that_does_not_alias_the_input(self) -> None:
        # The point of the whole probe. The identity kernel means an equal
        # result proves nothing on its own: a dispatch layer that returned its
        # argument untouched would satisfy every value assertion above. A
        # *fresh* allocation is what proves Rust ran and produced the values,
        # and it is also the contract -- callers must be able to mutate a
        # returned spectrum without corrupting the grid they passed in.
        values = np.array([1.0, 2.0, 3.0])
        result = roundtrip(values)

        assert result is not values
        assert result.base is not values
        assert not np.shares_memory(result, values)

        result[0] = 99.0
        assert values[0] == 1.0


class TestErrorPaths:
    """Everything the contract rejects, and the message it rejects it with."""

    @pytest.mark.parametrize(
        "values",
        [
            np.ones((2, 2)),
            np.ones((2, 2, 2)),
            np.asfortranarray(np.ones((2, 2))),
            np.ones((1, 1)),
        ],
        ids=["2d", "3d", "2d-fortran-order", "2d-single-element"],
    )
    def test_multidimensional_arrays_raise_value_error(
        self, values: np.ndarray
    ) -> None:
        # Note the last case: a (1, 1) array holds exactly one value but is
        # still 2-D, and the contract is about rank, not size.
        with pytest.raises(ValueError, match=r"must be 0 or 1-dimensional\.$"):
            roundtrip(values)

    @pytest.mark.parametrize(
        ("dtype", "name"),
        [
            (np.int64, "int64"),
            (np.int32, "int32"),
            (np.float32, "float32"),
            (np.complex128, "complex128"),
            (np.bool_, "bool"),
        ],
    )
    def test_wrong_dtype_arrays_raise_value_error_naming_the_dtype(
        self, dtype: type, name: str
    ) -> None:
        # PyO3 would raise TypeError when the typed view fails to extract;
        # `map_unary` maps it to ValueError so the whole contract raises one
        # exception type, and names the offending dtype so the message is
        # actionable rather than merely correct.
        with pytest.raises(ValueError) as excinfo:
            roundtrip(np.array([1, 2], dtype=dtype))
        assert str(excinfo.value) == dtype_error(name)

    def test_rank_is_checked_before_dtype(self) -> None:
        # An array that is both 2-D and wrong-dtype reports the dimension. The
        # ordering is worth pinning because it is the order the checks appear
        # in `map_unary`, and a reordering would silently change a
        # user-visible message that this suite otherwise matches exactly.
        with pytest.raises(ValueError) as excinfo:
            roundtrip(np.ones((2, 2), dtype=np.int64))
        assert str(excinfo.value) == DIMENSION_ERROR

    @pytest.mark.parametrize(
        "value",
        ["1.0", None, 1 + 2j, {"a": 1.0}, object()],
        ids=["str", "none", "complex", "dict", "object"],
    )
    def test_non_numeric_input_raises_value_error(self, value: object) -> None:
        with pytest.raises(ValueError) as excinfo:
            roundtrip(value)
        assert str(excinfo.value) == TYPE_ERROR

    def test_a_python_list_is_rejected(self) -> None:
        # The second contract/Cython divergence at this layer, and the one that
        # is a *narrowing*: the Cython entry points call `np.array(...)` before
        # the memoryview cast, so `dnde_photon([10.0, 20.0], 200.0)` works
        # today. A typed PyO3 view will not take a list. Task 3.5 must either
        # call `np.asarray` at the public boundary or declare the narrowing;
        # this assertion pins what the scaffold does so that decision is made
        # deliberately rather than discovered by a user.
        with pytest.raises(ValueError) as excinfo:
            roundtrip([1.0, 2.0])
        assert str(excinfo.value) == TYPE_ERROR

    @pytest.mark.parametrize(
        "call",
        [
            lambda: roundtrip(np.ones((2, 2))),
            lambda: roundtrip(np.array([1, 2])),
            lambda: roundtrip(None),
        ],
        ids=["dimension", "dtype", "type"],
    )
    def test_every_message_names_the_quantity(self, call: Callable[[], object]) -> None:
        # `quantity` is how a ported kernel keeps its twin's wording. If it
        # ever stops reaching the message, every Phase 04-06 kernel silently
        # starts reporting the wrong argument name.
        with pytest.raises(ValueError) as excinfo:
            call()
        assert str(excinfo.value).startswith(QUANTITY)


class TestSignature:
    """The advertised signature matches the one that actually works."""

    def test_signature_is_introspectable_and_accepts_a_keyword(self) -> None:
        # PyO3's `text_signature` is a *claim*; it does not constrain the call.
        # The Cython entry points are `def` functions and accept their
        # arguments by keyword (`dnde_photon(egam=..., emu=...)` works today),
        # so a positional-only claim here would both misdescribe this function
        # and, copied into a Phase 04 wrapper, narrow the public API.
        assert str(inspect.signature(roundtrip)) == "(x)"
        assert bits(roundtrip(x=1.5)) == bits(1.5)
