"""
Tests for the vector form factors.
"""

import pytest

from hazma.vector_mediator.form_factors.pi_gamma import form_factor_pi_gamma
from hazma.vector_mediator.form_factors.pipi import (
    compute_pipi_form_factor_parameters, form_factor_pipi)


@pytest.fixture
def pipi_ff_parameters():
    return compute_pipi_form_factor_parameters(2000)


def test_ff_pipi_hazma_vs_herwig4dm(pipi_ff_parameters):
    """
    Test that the `hazma` and `herwig4DM` implementations of the V-pi-pi
    form factor agree at 1.0 GeV.
    """
    from herwig4dm import F2pi  # type:ignore

    # Initialize the F2pi functions
    F2pi.initialize()

    hazma = form_factor_pipi(1.0, pipi_ff_parameters, F2pi.cI1_, imode=1)
    herwig = F2pi.Fpi(1.0, 1)
    re_diff = abs(hazma.real - herwig.real) / herwig.real * 100
    im_diff = abs(hazma.imag - herwig.imag) / herwig.imag * 100

    assert re_diff <= 1.0
    assert im_diff <= 1.0


def test_ff_pigamma_hazma_vs_herwig4dm():
    """
    Test that the `hazma` and `herwig4DM` implementations of the V-pi-pi
    form factor agree at 1.0 GeV.
    """
    from herwig4dm import FPiGamma  # type:ignore
    gvuu, gvdd, gvss = 1.0, -1.0, 1.0
    FPiGamma.resetParameters(1.0, 1.0, 2.0, 1e-3, gvuu, gvdd, gvss)

    hazma = form_factor_pi_gamma(1.0, gvuu, gvdd, gvss)
    herwig = FPiGamma.FPiGamma(1.0)
    re_diff = abs(hazma.real - herwig.real) / herwig.real * 100
    im_diff = abs(hazma.imag - herwig.imag) / herwig.imag * 100

    assert re_diff <= 1.0
    assert im_diff <= 1.0
