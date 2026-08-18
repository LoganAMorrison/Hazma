"""Cross-language plumbing: the ``hazma._core`` entry-point dispatch contract.

This module is the **template** every later kernel swap copies (cython-to-rust
Phase 02 Task 2.3; contract settled by Phase 03 Task 3.5). It pins the
argument-conversion, return-type and error behavior that ``rust/src/dispatch.rs``
implements once for all of Phases 04-06, exercised through probes that compute
nothing -- the identity and three distinguishable functions of it -- so every
assertion here is about the *plumbing* and none of it is about physics.

The contract, settled in Task 3.5 against all 43 surviving top-level ``def``s:

* a Python ``float``, a NumPy scalar, or a 0-d array of any numeric dtype
  returns a Python ``float``;
* a 1-D ``float64`` array, or any sequence that converts to one, returns a
  **fresh** 1-D ``float64`` array of the same length;
* a higher-rank array, or an array whose dtype is not ``float64``, raises
  ``ValueError`` naming the quantity;
* anything that is neither a real number nor a sequence raises ``TypeError``
  naming the quantity.

Two variants share that classification:
:func:`hazma._core.dispatch.roundtrip_flavors` returns the neutrino shape (a
3-tuple for a scalar, a ``(3, N)`` array for a grid), and
:func:`hazma._core.dispatch.roundtrip_vector` is the ``partial_widths`` shape,
which must be 1-D and is never a scalar.

Copying this module for a real kernel means: keep every test below, swap the
probe for the kernel and the quantity wording for the one that kernel passes
(``"Photon energies"``, ``"Positron energies"``, ...), and add the kernel's
*numerical* tests beside them. Do not merge the two halves -- plumbing failures
and physics failures should not need the same debugging.

Two things this module deliberately does **not** do:

* It does not pin values against the Cython twin. That is the parity corpus's
  job (``test/parity/``), which holds all 41 consumed entry points to
  bit-equality on its capturing platform. A second, looser numerical gate here
  would only be one more thing to keep in sync.
* It does not claim the port and the live Cython agree everywhere. They do not,
  and :class:`TestDeclaredDivergencesFromCython` is where each difference is
  written down as an assertion rather than as prose.

Every assertion below was measured against the built extension before it was
written, not derived from the contract prose.
"""

from __future__ import annotations

import inspect
import re
import struct
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from hazma._core import dispatch as core_dispatch
from hazma._core import roundtrip
from hazma.scalar_mediator import _c_scalar_mediator_cross_sections as cython_xs
from hazma.spectra._positron import _pion as cython_positron_pion

# The `hazma._core.dispatch` probes, bound here so the tests below read like
# ordinary calls. Each takes the quantity wording as an argument, which the
# top-level `roundtrip` (Phase 02's, wording fixed to `QUANTITY`) cannot.
roundtrip_as = core_dispatch.roundtrip
roundtrip_flavors = core_dispatch.roundtrip_flavors
roundtrip_vector = core_dispatch.roundtrip_vector

if TYPE_CHECKING:
    from collections.abc import Callable

# The wording the top-level `roundtrip` passes to `map_unary` as its
# `quantity`. Every error message the contract produces is prefixed with it,
# which is how a ported kernel keeps its Cython twin's user-visible exception
# text. The `hazma._core.dispatch` probes take the wording as an argument
# instead, which is what lets `TestCythonMessageParity` render the tree's real
# quantities.
QUANTITY = "Input values"

#: Neutrino flavors a spectrum kernel returns: electron, muon, tau.
N_FLAVORS = 3

DIMENSION_ERROR = f"{QUANTITY} must be 0 or 1-dimensional."
TYPE_ERROR = f"{QUANTITY} must be a float or a NumPy array."

REPO_ROOT = Path(__file__).resolve().parents[1]


def dtype_error(dtype: str, quantity: str = QUANTITY) -> str:
    """The dtype-rejection message for a non-``float64`` array."""
    return f"{quantity} must be a float64 array; got dtype {dtype}."


def bits(x: float) -> bytes:
    """The IEEE-754 bit pattern of ``x``.

    Compared instead of ``==`` so that ``-0.0`` is distinguished from ``0.0``
    and a NaN compares equal to itself. The probe kernels are exactly
    reproducible, so *every* input bit pattern must survive the round trip,
    including the ones ``==`` cannot see.
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


def expected_flavors(x: float) -> np.ndarray:
    """``kernels::roundtrip_flavors`` recomputed in Python.

    ``[x, -x, 1/x]``. Negation and division are both correctly rounded by
    IEEE-754, so this reproduces the Rust bit for bit and the comparison
    argues about no tolerance. Computed through NumPy rather than with the
    ``/`` operator because Python raises ``ZeroDivisionError`` where the
    hardware returns an infinity.
    """
    value = np.float64(x)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return np.array([value, -value, np.float64(1.0) / value])


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
        # These are neither `float` subclasses nor ndarrays, and none of them
        # defines `__len__`, so they fall past every earlier arm to the
        # `extract::<f64>` one. They are still scalars and the contract says a
        # scalar returns a float.
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
    """0-d array in -> float out, for any numeric dtype.

    Task 3.5's first decision. The live Cython disagrees with itself here: the
    17 entry points that dispatch on ``hasattr(x, '__len__')`` -- the 16 under
    ``hazma/spectra/`` plus ``scalar_mediator_decay_spectrum`` -- raise
    ``AssertionError`` (``ndarray`` defines ``__len__`` on the *type*, so the
    dispatch sends a 0-d array down the array path and the ``len(shape) == 1``
    guard rejects it), while the 18 cross-section entry points call ``.item()``
    and return a float. The
    port takes the cross sections' answer for all of them -- it is a widening
    no working call can notice, and it is what the spectra's own message
    ("must be **0** or 1-dimensional") already promises.
    """

    @pytest.mark.parametrize("value", SCALARS)
    def test_zero_dim_float64_array_returns_a_python_float(self, value: float) -> None:
        result = roundtrip(np.array(value))
        assert type(result) is float
        assert bits(result) == bits(value)

    @pytest.mark.parametrize(
        "dtype", [np.int64, np.int32, np.float32, np.uint8, np.bool_]
    )
    def test_zero_dim_arrays_of_any_numeric_dtype_are_the_scalar_they_hold(
        self, dtype: type
    ) -> None:
        # A 0-d array *is* a scalar, and `np.int64(4)` is accepted, so
        # rejecting `np.array(4)` would be arbitrary. The float64 rule binds
        # 1-D arrays only, where it is about the grid's storage.
        result = roundtrip(np.array(1, dtype=dtype))
        assert type(result) is float
        assert bits(result) == bits(1.0)

    @pytest.mark.parametrize(
        ("value", "dtype"),
        [("15.0", "<U4"), (None, "object")],
        ids=["str", "object"],
    )
    def test_zero_dim_non_numeric_arrays_are_rejected(
        self, value: object, dtype: str
    ) -> None:
        # The reason the 0-d path asks the *dtype* rather than trying
        # `float(array)`: NumPy's 0-d `__float__` forwards to the element and
        # `np.str_` subclasses `str`, so `float(np.array("15.0"))` is 15.0.
        # Extraction alone would let a string through the front door.
        with pytest.raises(ValueError) as excinfo:
            roundtrip(np.array(value))
        assert str(excinfo.value) == dtype_error(dtype)


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


class TestSequenceInput:
    """A list or tuple of floats is an energy grid.

    Task 3.5's second decision, and the one that would have been a
    **narrowing**: those same 17 entry points call ``np.array(...)`` before
    their memoryview cast, so ``dnde_photon([10.0, 20.0], 200.0)`` works today
    and a typed PyO3 view would not have taken it. The port converts with
    ``numpy.asarray`` at the boundary and then applies the same array rules, so
    a list behaves exactly like the array it converts to -- including being
    rejected when that array is not float64.
    """

    @pytest.mark.parametrize(
        "values",
        [[1.0, 2.0, 3.0], (1.0, 2.0, 3.0)],
        ids=["list", "tuple"],
    )
    def test_a_sequence_of_floats_is_accepted(self, values: object) -> None:
        result = roundtrip(values)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        assert result.tobytes() == np.array([1.0, 2.0, 3.0]).tobytes()

    def test_an_empty_list_is_an_empty_grid(self) -> None:
        result = roundtrip([])
        assert isinstance(result, np.ndarray)
        assert result.shape == (0,)

    def test_a_list_of_ints_is_rejected_like_an_int_array(self) -> None:
        # `np.asarray([1, 2])` is int64, and the Cython rejects exactly this
        # for exactly this reason. The whole value of routing lists through
        # `asarray` is that there is one dtype rule, not two.
        with pytest.raises(ValueError) as excinfo:
            roundtrip([1, 2])
        assert str(excinfo.value) == dtype_error("int64")

    def test_a_nested_list_is_a_rank_error(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            roundtrip([[1.0, 2.0], [3.0, 4.0]])
        assert str(excinfo.value) == DIMENSION_ERROR


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
        # `map_unary` maps it to ValueError -- which is also what the Cython
        # raises for a dtype mismatch -- and names the offending dtype so the
        # message is actionable rather than merely correct.
        with pytest.raises(ValueError) as excinfo:
            roundtrip(np.array([1, 2], dtype=dtype))
        assert str(excinfo.value) == dtype_error(name)

    def test_rank_is_checked_before_dtype(self) -> None:
        # An array that is both 2-D and wrong-dtype reports the dimension. The
        # ordering is worth pinning because it is the order the checks appear
        # in `classify_array`, and a reordering would silently change a
        # user-visible message that this suite otherwise matches exactly.
        with pytest.raises(ValueError) as excinfo:
            roundtrip(np.ones((2, 2), dtype=np.int64))
        assert str(excinfo.value) == DIMENSION_ERROR

    @pytest.mark.parametrize(
        "value",
        [None, 1 + 2j, object()],
        ids=["none", "complex", "object"],
    )
    def test_non_numeric_input_raises_type_error(self, value: object) -> None:
        # `TypeError`, not `ValueError`: it is the type CPython raises today
        # when such a value reaches the Cython entry point's `cdef double`
        # parameter ("must be real number, not NoneType"), and the port keeps
        # every exception type the Cython raises explicitly. Only its
        # `assert`s change type (rules.md rule 9).
        with pytest.raises(TypeError) as excinfo:
            roundtrip(value)
        assert str(excinfo.value) == TYPE_ERROR

    def test_a_string_is_not_a_number(self) -> None:
        # Guarding this is not paranoia: an earlier draft returned 15.0 for
        # `roundtrip("15.0")`. A string has `__len__`, so it reaches
        # `numpy.asarray` and becomes a 0-d `<U4` array -- and NumPy's 0-d
        # `__float__` forwards to the element, which is an `np.str_` and so a
        # `str`, which `float()` parses. `has_numeric_dtype` is what stops it,
        # which is why the 0-d path asks the dtype rather than trying the
        # conversion.
        with pytest.raises(ValueError) as excinfo:
            roundtrip("15.0")
        assert str(excinfo.value) == dtype_error("<U4")

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
        with pytest.raises((ValueError, TypeError)) as excinfo:
            call()
        assert str(excinfo.value).startswith(QUANTITY)


class TestQuantityIsParameterised:
    """The probe with caller-chosen wording behaves like the fixed one."""

    @pytest.mark.parametrize(
        ("value", "exception"),
        [
            (np.ones((2, 2)), ValueError),
            (np.array([1, 2]), ValueError),
            (None, TypeError),
        ],
        ids=["dimension", "dtype", "type"],
    )
    def test_the_wording_is_whatever_the_caller_passed(
        self, value: object, exception: type[BaseException]
    ) -> None:
        with pytest.raises(exception) as excinfo:
            roundtrip_as(value, "Photon energies")
        assert str(excinfo.value).startswith("Photon energies ")

    def test_the_value_paths_are_the_fixed_probe_s(self) -> None:
        grid = np.linspace(-3.0, 3.0, 41)
        assert roundtrip_as(grid, "Photon energies").tobytes() == grid.tobytes()
        assert bits(roundtrip_as(1.5, "Photon energies")) == bits(1.5)


class TestFlavorPath:
    """The neutrino shape: 3-tuple for a scalar, ``(3, N)`` for a grid.

    The one non-uniform return shape in hazma's public surface
    (``hazma/spectra/_neutrino/*.pyx``). The probe's three rows are ``x``,
    ``-x`` and ``1/x`` rather than three copies of ``x`` precisely so that a
    transposed result, a reversed row order or a row written twice cannot pass
    a value assertion.
    """

    QUANTITY = "Neutrino energies"

    @pytest.mark.parametrize("value", SCALARS)
    def test_scalar_in_three_tuple_out(self, value: float) -> None:
        result = roundtrip_flavors(value, self.QUANTITY)
        assert type(result) is tuple
        assert len(result) == N_FLAVORS
        assert all(type(item) is float for item in result)
        # Bit equality, NaN included: negation and reciprocal are both
        # correctly rounded, so Python and Rust agree exactly and there is
        # nothing here to give a tolerance to.
        expected = expected_flavors(value)
        for got, want in zip(result, expected, strict=True):
            assert bits(got) == bits(float(want))

    def test_array_in_three_by_n_array_out(self) -> None:
        grid = np.array([1.0, 2.0, 4.0, -0.5])
        result = roundtrip_flavors(grid, self.QUANTITY)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        assert result.shape == (3, grid.size)
        assert result.flags.c_contiguous

        expected = np.stack([expected_flavors(float(x)) for x in grid], axis=1)
        assert result.tobytes() == expected.tobytes()

    def test_rows_are_flavors_not_energies(self) -> None:
        # The assertion a transposed implementation fails. Length 4 != 3, so
        # the shape alone catches a transpose here -- which is why the grid is
        # not three points long.
        grid = np.array([1.0, 2.0, 4.0, 8.0])
        electron, muon, tau = roundtrip_flavors(grid, self.QUANTITY)
        assert electron.tobytes() == grid.tobytes()
        assert muon.tobytes() == (-grid).tobytes()
        assert tau.tobytes() == (1.0 / grid).tobytes()

    def test_a_three_point_grid_is_not_confused_with_the_flavor_axis(self) -> None:
        # The one length where a transpose is shape-invisible, so it gets its
        # own test with values that make the two orientations differ.
        grid = np.array([1.0, 2.0, 4.0])
        result = roundtrip_flavors(grid, self.QUANTITY)
        expected = np.stack([expected_flavors(float(x)) for x in grid], axis=1)
        assert result.tobytes() == expected.tobytes()
        assert result.tobytes() != expected.T.tobytes()

    def test_empty_grid_returns_three_empty_rows(self) -> None:
        result = roundtrip_flavors(np.array([], dtype=np.float64), self.QUANTITY)
        assert result.shape == (3, 0)
        assert result.dtype == np.float64

    def test_result_does_not_alias_the_input(self) -> None:
        grid = np.array([1.0, 2.0, 3.0])
        result = roundtrip_flavors(grid, self.QUANTITY)
        assert not np.shares_memory(result, grid)
        result[0, 0] = 99.0
        assert grid[0] == 1.0

    def test_non_contiguous_and_readonly_inputs_are_accepted(self) -> None:
        strided = np.arange(6.0)[1::2]
        assert not strided.flags.c_contiguous
        assert roundtrip_flavors(strided, self.QUANTITY).shape == (3, 3)

        frozen = np.array([1.0, 2.0])
        frozen.flags.writeable = False
        assert roundtrip_flavors(frozen, self.QUANTITY).shape == (3, 2)

    def test_a_zero_dimensional_array_takes_the_tuple_path(self) -> None:
        assert type(roundtrip_flavors(np.array(2.0), self.QUANTITY)) is tuple

    def test_a_sequence_is_accepted(self) -> None:
        assert roundtrip_flavors([1.0, 2.0], self.QUANTITY).shape == (3, 2)

    @pytest.mark.parametrize(
        ("value", "exception", "message"),
        [
            (np.ones((2, 2)), ValueError, f"{QUANTITY} must be 0 or 1-dimensional."),
            (np.array([1, 2]), ValueError, dtype_error("int64", QUANTITY)),
            (None, TypeError, f"{QUANTITY} must be a float or a NumPy array."),
        ],
        ids=["dimension", "dtype", "type"],
    )
    def test_it_shares_the_unary_error_contract(
        self, value: object, exception: type[BaseException], message: str
    ) -> None:
        # Same `classify`, so a Phase 04 neutrino wrapper inherits the same
        # messages as its photon sibling without restating them.
        with pytest.raises(exception) as excinfo:
            roundtrip_flavors(value, self.QUANTITY)
        assert str(excinfo.value) == message


class TestRequireVector:
    """The ``partial_widths`` shape: 1-D array required, scalar refused.

    ``hazma/scalar_mediator/scalar_mediator_decay_spectrum.py``'s ``pws``
    argument is the only public argument the Cython refuses a scalar for with
    an explicit ``raise`` rather than by falling out of a dispatch branch, so
    it is the one place where both message strings are the call site's own and
    are reproduced verbatim.
    """

    QUANTITY = "Partial widths"

    def test_a_float64_array_round_trips(self) -> None:
        values = np.array([1.0, 2.0, 3.0])
        result = roundtrip_vector(values, self.QUANTITY)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        assert result.tobytes() == values.tobytes()
        assert not np.shares_memory(result, values)

    @pytest.mark.parametrize("values", [[1.0, 2.0], (1.0, 2.0)], ids=["list", "tuple"])
    def test_a_sequence_is_converted(self, values: object) -> None:
        assert (
            roundtrip_vector(values, self.QUANTITY).tobytes()
            == np.array([1.0, 2.0]).tobytes()
        )

    def test_a_non_contiguous_view_is_read_in_order(self) -> None:
        strided = np.arange(6.0)[::2]
        assert (
            roundtrip_vector(strided, self.QUANTITY).tobytes()
            == np.array([0.0, 2.0, 4.0]).tobytes()
        )

    def test_an_empty_array_is_accepted(self) -> None:
        # Length is not this layer's business: the Cython's own `pws` handling
        # indexes seven entries and raises IndexError from the *kernel* when
        # they are not there. Phase 06 owns that check; `require_vector` owns
        # rank and dtype only.
        assert roundtrip_vector(np.array([]), self.QUANTITY).shape == (0,)

    @pytest.mark.parametrize(
        "value",
        [1.0, 3, True, np.float64(1.0), np.int64(1), None, 1 + 2j],
        ids=["float", "int", "bool", "np.float64", "np.int64", "none", "complex"],
    )
    def test_anything_without_a_length_is_refused(self, value: object) -> None:
        with pytest.raises(ValueError) as excinfo:
            roundtrip_vector(value, self.QUANTITY)
        assert str(excinfo.value) == f"{self.QUANTITY} must be a list or array."

    @pytest.mark.parametrize(
        "value",
        [np.array(1.0), np.ones((2, 2)), np.ones((1, 1, 1)), "1.0", {"a": 1.0}],
        ids=["0d", "2d", "3d", "str", "dict"],
    )
    def test_anything_that_is_not_one_dimensional_is_refused(
        self, value: object
    ) -> None:
        # Note `str` and `dict`: both have `__len__`, so they pass the first
        # guard and are caught by the rank of what they convert to -- exactly
        # what the Cython does with them.
        with pytest.raises(ValueError) as excinfo:
            roundtrip_vector(value, self.QUANTITY)
        assert str(excinfo.value) == f"{self.QUANTITY} must be 1-dimensional."

    @pytest.mark.parametrize(
        ("dtype", "name"), [(np.int64, "int64"), (np.float32, "float32")]
    )
    def test_a_wrong_dtype_array_is_refused(self, dtype: type, name: str) -> None:
        with pytest.raises(ValueError) as excinfo:
            roundtrip_vector(np.array([1, 2], dtype=dtype), self.QUANTITY)
        assert str(excinfo.value) == dtype_error(name, self.QUANTITY)


def cython_dispatch_messages() -> dict[str, set[str]]:
    """Every dispatch error string in the surviving ``.pyx``, from source.

    Returns a mapping from message *kind* to the exact strings the tree
    contains. Derived by scanning the files rather than transcribed, so a
    ``.pyx`` edit that changes a message -- or a Phase 04-06 deletion that
    removes the last site carrying one -- shows up here instead of leaving a
    hand-typed roster quietly wrong.
    """
    patterns = {
        "assert": re.compile(r"assert\s+len\([\w.]+\.shape\)\s*==\s*1,\s*\"([^\"]+)\""),
        "raise": re.compile(r"raise\s+ValueError\(\s*\"([^\"]+)\"\s*\)"),
    }
    found: dict[str, set[str]] = {kind: set() for kind in patterns}
    for path in sorted((REPO_ROOT / "hazma").rglob("*.pyx")):
        text = path.read_text()
        for kind, pattern in patterns.items():
            found[kind].update(pattern.findall(text))
    return found


class TestCythonMessageParity:
    """The port's messages are the Cython's, byte for byte.

    The phase file's second exit criterion. The oracle is the ``.pyx`` sources
    themselves -- read here, not copied into this file -- because the messages
    are a user-visible part of the public API and a reworded one is a silent
    break that no numerical gate can see.
    """

    def test_the_tree_carries_exactly_the_expected_message_roster(self) -> None:
        # Enumerated from source so the four templates below are known to
        # cover everything, and so a new wording cannot appear unnoticed.
        # `"Photon energies ..."` in `hazma/spectra/_neutrino/_muon.pyx` is a
        # copy-paste defect its `_pion.pyx` sibling does not share; Task 3.5
        # decided the port says "Neutrino energies" there, which is why the
        # roster is pinned rather than merely counted.
        found = cython_dispatch_messages()
        assert found["assert"] == {
            "Photon energies must be 0 or 1-dimensional.",
            "Positron energies must be 0 or 1-dimensional.",
            "Neutrino energies must be 0 or 1-dimensional.",
            "Partial widths must be 1-dimensional.",
        }
        assert found["raise"] == {"Partial widths must be a list or array."}

    def test_every_rank_message_is_reproduced_exactly(self) -> None:
        suffix = " must be 0 or 1-dimensional."
        messages = [
            message
            for message in cython_dispatch_messages()["assert"]
            if message.endswith(suffix)
        ]
        assert messages, "no rank messages found in the .pyx sources"
        for message in messages:
            quantity = message[: -len(suffix)]
            with pytest.raises(ValueError) as excinfo:
                roundtrip_as(np.ones((2, 2)), quantity)
            assert str(excinfo.value) == message

    def test_the_partial_width_messages_are_reproduced_exactly(self) -> None:
        found = cython_dispatch_messages()
        rank = "Partial widths must be 1-dimensional."
        missing = "Partial widths must be a list or array."
        assert rank in found["assert"]
        assert missing in found["raise"]

        with pytest.raises(ValueError) as excinfo:
            roundtrip_vector(np.ones((2, 2)), "Partial widths")
        assert str(excinfo.value) == rank

        with pytest.raises(ValueError) as excinfo:
            roundtrip_vector(1.0, "Partial widths")
        assert str(excinfo.value) == missing


class TestDeclaredDivergencesFromCython:
    """Each place the port deliberately does not match the live Cython.

    Written as assertions against **both** implementations so that the claim
    "this is a widening, not a break" is checked rather than asserted, and so
    that a Phase 04-06 swap changing one of these fails here first.

    The spectra half of the oracle is whichever ``.pyx`` still exports a
    ``def`` with the unary "scalar or 1-D array in, same out" shape. That
    was ``_photon/_muon`` until cython-to-rust Task 4.3 swapped it, then
    ``_photon/_pion`` until Task 4.4, then ``_photon/_rho`` until Task 4.5
    exhausted the photon candidates. It is ``_positron/_pion`` now: the
    same ``hasattr(__len__)`` / ``assert`` / array-return shape as all
    three predecessors, differing only in the quantity its message names
    (``"Positron energies"``), which these tests take from the source
    rather than hard-code.

    **Task 4.6 swaps that one too, and it is the last unary spectra
    ``def`` in the tree** — the neutrino entry points survive it but
    return a 3-tuple or a ``(3, N)`` array, so they do not fit these
    assertions without rewriting them. At that point either rewrite them
    around the neutrino shape or retire them; by the end of Phase 04 the
    widening they describe stops having a live Cython side outside
    ``cython_xs``.
    """

    def test_a_zero_dimensional_array_raises_in_cython_and_returns_in_rust(
        self,
    ) -> None:
        with pytest.raises(AssertionError):
            cython_positron_pion.dnde_positron_charged_pion(np.array(15.0), 200.0)
        assert type(roundtrip(np.array(15.0))) is float

    def test_the_cross_sections_already_take_the_ported_zero_dimensional_path(
        self,
    ) -> None:
        # The reason the widening above is the cross sections' behavior rather
        # than a new invention.
        args = (5.0, 200.0, 1.0, 1.0, 0.1, 0.1, 1e5, 1.0, 0.0, 0.511)
        assert cython_xs.sigma_xx_to_s_to_ff(
            np.array(15.0), *args
        ) == cython_xs.sigma_xx_to_s_to_ff(15.0, *args)

    def test_a_rank_error_is_an_assertion_in_cython_and_a_value_error_in_rust(
        self,
    ) -> None:
        # rules.md rule 9: the Cython's `assert`s become unconditional raises,
        # because today they vanish under `python -O` and leave the user a
        # downstream failure instead. The *message* is unchanged, which is
        # what `TestCythonMessageParity` checks.
        with pytest.raises(AssertionError) as cython_error:
            cython_positron_pion.dnde_positron_charged_pion(np.ones((2, 2)), 200.0)
        with pytest.raises(ValueError) as rust_error:
            roundtrip_as(np.ones((2, 2)), "Positron energies")
        assert str(cython_error.value) == str(rust_error.value)

    def test_a_sequence_is_accepted_by_the_spectra_and_by_the_port(self) -> None:
        assert cython_positron_pion.dnde_positron_charged_pion(
            [15.0, 25.0], 200.0
        ).shape == (2,)
        assert roundtrip([15.0, 25.0]).shape == (2,)

    def test_a_sequence_is_refused_by_the_cross_sections_and_accepted_by_the_port(
        self,
    ) -> None:
        # The one place the port is *wider* than the Cython twin it replaces
        # rather than equal to it: the cross-section dispatch reads `.ndim` on
        # whatever it is given, so a list raises AttributeError today. Phase 05
        # inherits the widening, which no working call can notice.
        with pytest.raises(AttributeError):
            cython_xs.sigma_xx_to_s_to_ff(
                [15.0, 25.0], 5.0, 200.0, 1.0, 1.0, 0.1, 0.1, 1e5, 1.0, 0.0, 0.511
            )
        assert roundtrip_as([15.0, 25.0], "Center-of-mass energies").shape == (2,)


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

    @pytest.mark.parametrize(
        ("probe", "signature"),
        [
            (roundtrip_as, "(x, quantity)"),
            (roundtrip_flavors, "(x, quantity)"),
            (roundtrip_vector, "(values, quantity)"),
        ],
        ids=["roundtrip", "roundtrip_flavors", "roundtrip_vector"],
    )
    def test_the_dispatch_probes_advertise_what_they_accept(
        self, probe: Callable[..., object], signature: str
    ) -> None:
        assert str(inspect.signature(probe)) == signature
