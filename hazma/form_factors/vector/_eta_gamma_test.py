"""Tests on the eta + gamma form factor."""

# pylint: disable=invalid-name,too-few-public-methods

import pytest

from hazma.form_factors.vector import VectorFormFactorEtaGamma

from .test_utils import FormFactorTestDataItem, load_test_data


@pytest.fixture(name="vector_form_factor_eta_gamma")
def fixture_vector_form_factor_eta_gamma():
    """Fixture for pi^+ + pi^- form-factor."""
    return VectorFormFactorEtaGamma()


class TestFormFactorEtaGamma:
    """Tests for the `eta + gamma` form factor."""

    testdata = load_test_data("eta_gamma.json")

    @pytest.mark.skip("Known to be broken")
    @pytest.mark.parametrize("entry", testdata)
    def test_form_factor(
        self,
        vector_form_factor_eta_gamma: VectorFormFactorEtaGamma,
        entry: FormFactorTestDataItem,
    ):
        """
        Test that the `hazma` and `herwig4DM` implementations of the V-eta-gamma
        form factor agree for the input model.
        """
        form_factor = vector_form_factor_eta_gamma.form_factor(
            q=entry.mv, couplings=entry.couplings
        )

        assert form_factor.real == pytest.approx(entry.form_factor_re)
        assert form_factor.imag == pytest.approx(entry.form_factor_im)

    @pytest.mark.parametrize("entry", testdata)
    def test_width(
        self,
        vector_form_factor_eta_gamma: VectorFormFactorEtaGamma,
        entry: FormFactorTestDataItem,
    ):
        """
        Test that the `hazma` and `herwig4DM` implementations of the V-eta-gamma
        form factor agree for the input model.
        """
        width = vector_form_factor_eta_gamma.width(
            mv=entry.mv, couplings=entry.couplings
        )

        assert width == pytest.approx(entry.width)
