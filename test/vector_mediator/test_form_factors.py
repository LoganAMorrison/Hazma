"""
Tests for the vector form factors.
"""
# from typing import
from typing import NamedTuple, Tuple

import pytest
from pytest import approx
import numpy as np

from hazma.vector_mediator import VectorMediatorGeV
from hazma.parameters import Qd, Qe, Qu
from hazma.parameters import qe


class HazmaFunction(NamedTuple):
    model: VectorMediatorGeV
    function: str
    extra_args: Tuple

    def __call__(self, val):
        func = getattr(self.model, self.function)
        return func(val, *self.extra_args)


class ModuleFunction(NamedTuple):
    module: object
    function: str
    extra_args: Tuple

    def __call__(self, val):
        func = getattr(self.module, self.function)
        return func(val, *self.extra_args)


@pytest.fixture
def model1() -> VectorMediatorGeV:
    return VectorMediatorGeV(
        mx=5e3,
        mv=2e3,
        gvxx=1.0,
        gvuu=3.0,
        gvdd=1.0,
        gvss=-1.0,
        gvee=0.0,
        gvmumu=0.0,
    )


@pytest.fixture
def model2() -> VectorMediatorGeV:
    return VectorMediatorGeV(
        mx=5e3,
        mv=4e3,
        gvxx=1.0,
        gvuu=1.3,
        gvdd=1.2,
        gvss=1.1,
        gvee=0.0,
        gvmumu=0.0,
    )


@pytest.fixture
def kinetic_mixing_model() -> VectorMediatorGeV:
    eps = 1.0
    return VectorMediatorGeV(
        mx=5e3,
        mv=2e3,
        gvxx=1.0,
        gvuu=Qu * eps * qe,
        gvdd=Qd * eps * qe,
        gvss=Qd * eps * qe,
        gvee=Qe * eps * qe,
        gvmumu=Qe * eps * qe,
    )


def synchronize_herwig(model: VectorMediatorGeV, module) -> None:
    module.resetParameters(
        model.gvxx, model.mx, model.mv, np.nan, model.gvuu, model.gvdd, model.gvss
    )


def compare_widths(
    model: VectorMediatorGeV, func: str, herwig_module, reltol=1e-4, abstol=0.0
):
    mvgev = model.mv * 1e-3
    synchronize_herwig(model, herwig_module)
    hazma = getattr(model, func)() / model.mv
    herwig = herwig_module.GammaDM(mvgev) / mvgev
    assert hazma == approx(herwig, rel=reltol, abs=abstol)


def compare_form_factors(
    hazma: HazmaFunction, herwig: ModuleFunction, herwig_units: int = -1
):

    synchronize_herwig(hazma.model, herwig.module)
    mvmev = hazma.model.mv
    mvgev = mvmev * 1e-3
    hazma_val = hazma(mvmev)
    herwig_val = herwig(mvgev ** 2) / (mvgev ** herwig_units)

    assert np.real(hazma_val) == approx(np.real(herwig_val), rel=1e-4, abs=0.0)
    assert np.imag(hazma_val) == approx(np.imag(herwig_val), rel=1e-4, abs=0.0)


def test_form_factor_pipi(model1: VectorMediatorGeV):
    """
    Test that the `hazma` and `herwig4DM` implementations of the V-pi-pi
    form factor agree for the input model.
    """
    from herwig4dm import F2pi

    # model = model1
    # synchronize_herwig(model, F2pi)
    # hazma = model.form_factor_pipi(model.mv ** 2)
    # herwig = F2pi.Fpi(1e-6 * model.mv ** 2, 1)
    # assert np.real(hazma) == approx(np.real(herwig), rel=1e-4, abs=0.0)
    # assert np.imag(hazma) == approx(np.imag(herwig), rel=1e-4, abs=0.0)

    hazma = HazmaFunction(model1, "form_factor_pipi", (1,))
    herwig = ModuleFunction(F2pi, "Fpi", (1,))

    compare_form_factors(hazma, herwig, 0)


def test_form_factor_pi_gamma(
    model1: VectorMediatorGeV,
    model2: VectorMediatorGeV,
    kinetic_mixing_model: VectorMediatorGeV,
):
    """
    Test that the `hazma` and `herwig4DM` implementations of the V-pi-gamma
    form factor agree for the input model.
    """
    from herwig4dm import FPiGamma

    hazma = HazmaFunction(model1, "_form_factor_pi_gamma", tuple())
    herwig = ModuleFunction(FPiGamma, "FPiGamma", tuple())

    for model in [model1, model2, kinetic_mixing_model]:
        hazma = HazmaFunction(model, "_form_factor_pi_gamma", tuple())
        compare_form_factors(hazma, herwig)


def test_form_factor_pi_gamma2(model2: VectorMediatorGeV):
    """
    Test that the `hazma` and `herwig4DM` implementations of the V-pi-gamma
    form factor agree for the input model.
    """
    from herwig4dm import FPiGamma

    hazma = HazmaFunction(model2, "_form_factor_pi_gamma", tuple())
    herwig = ModuleFunction(FPiGamma, "FPiGamma", tuple())

    compare_form_factors(hazma, herwig)


def test_form_factor_eta_gamma(model1: VectorMediatorGeV):
    """
    Test that the `hazma` and `herwig4DM` implementations of the V-eta-gamma
    form factor agree for the input model.
    """
    from herwig4dm import FEtaGamma

    hazma = HazmaFunction(model1, "_form_factor_eta_gamma", tuple())
    herwig = ModuleFunction(FEtaGamma, "FEtaGamma", tuple())

    compare_form_factors(hazma, herwig)


def test_form_factor_k0k0(model1: VectorMediatorGeV):
    """
    Test that the `hazma` and `herwig4DM` implementations of the K0 K0
    form factor agree for the input model.
    """
    from herwig4dm import FK

    hazma = HazmaFunction(model1, "form_factor_kk", (0,))
    herwig = ModuleFunction(FK, "Fkaon", (0,))

    compare_form_factors(hazma, herwig, 0)


def test_form_factor_kk(model1: VectorMediatorGeV):
    """
    Test that the `hazma` and `herwig4DM` implementations of the K K
    form factor agree for the input model.
    """
    from herwig4dm import FK

    hazma = HazmaFunction(model1, "form_factor_kk", (1,))
    herwig = ModuleFunction(FK, "Fkaon", (1,))

    compare_form_factors(hazma, herwig, 0)


def test_form_factor_omega_pi(model1: VectorMediatorGeV):
    """
    Test that the `hazma` and `herwig4DM` implementations of the V-omega-pi
    form factor agree for the input model.
    """
    from herwig4dm import FOmegaPion

    hazma = HazmaFunction(model1, "_form_factor_omega_pi", tuple())
    herwig = ModuleFunction(FOmegaPion, "FOmPiGamma", tuple())

    compare_form_factors(hazma, herwig)


def test_width_omega_pi(model1: VectorMediatorGeV):
    """
    Test that the `hazma` and `herwig4DM` implementations of the V-omega-pi
    widths agree for the input model.
    """
    from herwig4dm import FOmegaPion

    compare_widths(model1, "width_v_to_omega_pi", FOmegaPion)


def test_width_phi_pi(model1: VectorMediatorGeV):
    """
    Test that the `hazma` and `herwig4DM` implementations of the V-phi-pi
    widths agree for the input model.
    """
    from herwig4dm import FPhiPi

    compare_widths(model1, "width_v_to_phi_pi", FPhiPi)


def test_width_eta_phi(model1: VectorMediatorGeV):
    """
    Test that the `hazma` and `herwig4DM` implementations of the V-eta-phi
    widths agree for the input model.
    """
    from herwig4dm import FEtaPhi

    compare_widths(model1, "width_v_to_eta_phi", FEtaPhi)


def test_width_eta_omega(model1: VectorMediatorGeV):
    """
    Test that the `hazma` and `herwig4DM` implementations of the V-eta-omega
    widths agree for the input model.
    """
    from herwig4dm import FEtaOmega

    compare_widths(model1, "width_v_to_eta_omega", FEtaOmega)


def test_width_pi_pi_pi0(model1: VectorMediatorGeV):
    from herwig4dm import F3pi

    compare_widths(model1, "width_v_to_pi_pi_pi0", F3pi, reltol=1e-2)


def test_width_pi_pi_eta(model1: VectorMediatorGeV):
    from herwig4dm import FEtaPiPi

    compare_widths(model1, "width_v_to_pi_pi_eta", FEtaPiPi)


def test_width_pi_pi_etap(model1: VectorMediatorGeV):
    from herwig4dm import FEtaPrimePiPi

    compare_widths(model1, "width_v_to_pi_pi_etap", FEtaPrimePiPi)


def test_width_pi_pi_omega(model1: VectorMediatorGeV):
    from herwig4dm import FOmegaPiPi

    compare_widths(model1, "width_v_to_pi_pi_omega", FOmegaPiPi)
