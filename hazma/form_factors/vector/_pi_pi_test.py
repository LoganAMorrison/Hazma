"""Tests on the pi + pi form factor."""

# pylint: disable=invalid-name,too-few-public-methods

import pytest

from hazma.form_factors.vector import VectorFormFactorPi0Pi0, VectorFormFactorPiPi

from .test_utils import FormFactorTestDataItem, load_test_data


@pytest.fixture(name="vector_form_factor_pi_pi")
def fixture_vector_form_factor_pi_pi():
    """Fixture for pi^+ + pi^- form-factor."""
    return VectorFormFactorPiPi()


@pytest.fixture(name="vector_form_factor_pi0_pi0")
def fixture_vector_form_factor_pi0_pi0():
    """Fixture for pi^0 + pi^0 form-factor."""
    return VectorFormFactorPi0Pi0()


class TestFormFactorPiPi:
    """Tests for the `pi^+ + pi^-` form factor."""

    testdata = load_test_data("pi_pi.json")

    @pytest.mark.parametrize("entry", testdata)
    def test_form_factor(
        self,
        vector_form_factor_pi_pi: VectorFormFactorPiPi,
        entry: FormFactorTestDataItem,
    ):
        """
        Test that the `hazma` and `herwig4DM` implementations of the V-pi-pi
        form factor agree for the input model.
        """
        form_factor = vector_form_factor_pi_pi.form_factor(
            q=entry.mv, couplings=entry.couplings
        )

        assert form_factor.real == pytest.approx(entry.form_factor_re)
        assert form_factor.imag == pytest.approx(entry.form_factor_im)

    @pytest.mark.parametrize("entry", testdata)
    def test_width(
        self,
        vector_form_factor_pi_pi: VectorFormFactorPiPi,
        entry: FormFactorTestDataItem,
    ):
        """
        Test that the `hazma` and `herwig4DM` implementations of the V-pi-pi
        form factor agree for the input model.
        """
        width = vector_form_factor_pi_pi.width(mv=entry.mv, couplings=entry.couplings)

        assert width == pytest.approx(entry.width)


class TestFormFactorPi0Pi0:
    """Tests for the `pi^0 + pi^0` form factor."""

    testdata = load_test_data("pi0_pi0.json")

    @pytest.mark.parametrize("entry", testdata)
    def test_form_factor(
        self,
        vector_form_factor_pi0_pi0: VectorFormFactorPi0Pi0,
        entry: FormFactorTestDataItem,
    ):
        """
        Test that the `hazma` and `herwig4DM` implementations of the V-pi-pi
        form factor agree for the input model.
        """
        form_factor = vector_form_factor_pi0_pi0.form_factor(
            q=entry.mv, couplings=entry.couplings
        )

        assert form_factor.real == pytest.approx(entry.form_factor_re)
        assert form_factor.imag == pytest.approx(entry.form_factor_im)

    @pytest.mark.parametrize("entry", testdata)
    def test_width(
        self,
        vector_form_factor_pi0_pi0: VectorFormFactorPi0Pi0,
        entry: FormFactorTestDataItem,
    ):
        """
        Test that the `hazma` and `herwig4DM` implementations of the V-pi-pi
        form factor agree for the input model.
        """
        width = vector_form_factor_pi0_pi0.width(mv=entry.mv, couplings=entry.couplings)
        assert width == pytest.approx(entry.width)
