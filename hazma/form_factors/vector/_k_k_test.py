"""Tests on the k + k form factor."""

# pylint: disable=invalid-name,too-few-public-methods

import pytest

from hazma.form_factors.vector import VectorFormFactorK0K0, VectorFormFactorKK

from .test_utils import FormFactorTestDataItem, load_test_data


@pytest.fixture(name="vector_form_factor_k_k")
def fixture_vector_form_factor_k_k():
    """Fixture for K^+ + K^- form-factor."""
    return VectorFormFactorKK()


@pytest.fixture(name="vector_form_factor_k0_k0")
def fixture_vector_form_factor_k0_k0():
    """Fixture for K^0 + K^0 form-factor."""
    return VectorFormFactorK0K0()


class TestFormFactorKK:
    """Tests for the `K^+ + K^-` form factor."""

    testdata = load_test_data("k_k.json")

    @pytest.mark.parametrize("entry", testdata)
    def test_form_factor(
        self,
        vector_form_factor_k_k: VectorFormFactorKK,
        entry: FormFactorTestDataItem,
    ):
        """
        Test that the `hazma` and `herwig4DM` implementations of the V-K-K
        form factor agree for the input model.
        """
        form_factor = vector_form_factor_k_k.form_factor(
            q=entry.mv, couplings=entry.couplings
        )

        assert form_factor.real == pytest.approx(entry.form_factor_re)
        assert form_factor.imag == pytest.approx(entry.form_factor_im)

    @pytest.mark.parametrize("entry", testdata)
    def test_width(
        self,
        vector_form_factor_k_k: VectorFormFactorKK,
        entry: FormFactorTestDataItem,
    ):
        """
        Test that the `hazma` and `herwig4DM` implementations of the V-K-K
        form factor agree for the input model.
        """
        width = vector_form_factor_k_k.width(mv=entry.mv, couplings=entry.couplings)

        assert width == pytest.approx(entry.width)


class TestFormFactorK0K0:
    """Tests for the `K^0 + K^0` form factor."""

    testdata = load_test_data("k0_k0.json")

    @pytest.mark.parametrize("entry", testdata)
    def test_form_factor(
        self,
        vector_form_factor_k0_k0: VectorFormFactorK0K0,
        entry: FormFactorTestDataItem,
    ):
        """
        Test that the `hazma` and `herwig4DM` implementations of the V-K-K
        form factor agree for the input model.
        """
        form_factor = vector_form_factor_k0_k0.form_factor(
            q=entry.mv, couplings=entry.couplings
        )

        assert form_factor.real == pytest.approx(entry.form_factor_re, rel=1e-2)
        assert form_factor.imag == pytest.approx(entry.form_factor_im, rel=1e-2)

    @pytest.mark.parametrize("entry", testdata)
    def test_width(
        self,
        vector_form_factor_k0_k0: VectorFormFactorK0K0,
        entry: FormFactorTestDataItem,
    ):
        """
        Test that the `hazma` and `herwig4DM` implementations of the V-K-K
        width agree for the input model.
        """
        width = vector_form_factor_k0_k0.width(mv=entry.mv, couplings=entry.couplings)
        assert width == pytest.approx(entry.width, rel=1e-2)
