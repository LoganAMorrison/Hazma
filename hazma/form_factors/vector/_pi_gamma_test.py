"""Tests on the pi + gamma form factor."""

# pylint: disable=invalid-name,too-few-public-methods

import pytest

from hazma.form_factors.vector import VectorFormFactorPi0Gamma

from .test_utils import FormFactorTestDataItem, load_test_data


@pytest.fixture(name="vector_form_factor_pi0_gamma")
def fixture_vector_form_factor_pi0_gamma():
    """Fixture for pi^+ + pi^- form-factor."""
    return VectorFormFactorPi0Gamma()


class TestFormFactorPi0Gamma:
    """Tests for the `pi^0 + gamma` form factor."""

    testdata = load_test_data("pi0_gamma.json")

    @pytest.mark.skip("Known to be broken")
    @pytest.mark.parametrize("entry", testdata)
    def test_form_factor(
        self,
        vector_form_factor_pi0_gamma: VectorFormFactorPi0Gamma,
        entry: FormFactorTestDataItem,
    ):
        """
        Test that the `hazma` and `herwig4DM` implementations of the V-pi-pi
        form factor agree for the input model.
        """
        form_factor = vector_form_factor_pi0_gamma.form_factor(
            q=entry.mv, couplings=entry.couplings
        )

        assert form_factor.real == pytest.approx(entry.form_factor_re, rel=5e-2)
        assert form_factor.imag == pytest.approx(entry.form_factor_im, rel=5e-2)

    @pytest.mark.parametrize("entry", testdata)
    def test_width(
        self,
        vector_form_factor_pi0_gamma: VectorFormFactorPi0Gamma,
        entry: FormFactorTestDataItem,
    ):
        """
        Test that the `hazma` and `herwig4DM` implementations of the V-pi-pi
        form factor agree for the input model.
        """
        width = vector_form_factor_pi0_gamma.width(
            mv=entry.mv, couplings=entry.couplings
        )

        assert width == pytest.approx(entry.width, rel=5e-2)
