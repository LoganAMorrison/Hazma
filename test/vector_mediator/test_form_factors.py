"""
Tests for the vector form factors.
"""

# from typing import
from typing import List, NamedTuple, Tuple, Union

import numpy as np
import pytest
from pytest import approx

from hazma.vector_mediator import BLGeV, KineticMixingGeV, VectorMediatorGeV

VectorModel = Union[VectorMediatorGeV, KineticMixingGeV, BLGeV]


class HazmaFunction(NamedTuple):
    model: VectorModel
    function: str
    extra_args: Tuple = tuple()

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
def models() -> List[VectorModel]:
    model1 = VectorMediatorGeV(
        mx=5e3,
        mv=2e3,
        gvxx=1.0,
        gvuu=3.0,
        gvdd=1.0,
        gvss=-1.0,
        gvee=0.0,
        gvmumu=0.0,
        gvveve=0.0,
        gvvmvm=0.0,
        gvvtvt=0.0,
    )

    model2 = VectorMediatorGeV(
        mx=5e3,
        mv=1.75e3,
        gvxx=1.0,
        gvuu=1.3,
        gvdd=1.2,
        gvss=1.1,
        gvee=0.0,
        gvmumu=0.0,
        gvveve=0.0,
        gvvmvm=0.0,
        gvvtvt=0.0,
    )

    eps = 1.0
    km = KineticMixingGeV(mx=5e3, mv=2e3, gvxx=1.0, eps=eps)
    bl = BLGeV(mx=5e3, mv=2e3, gvxx=1.0, g=1.0)

    return [model1, model2, km, bl]


def error_msg(hazma, herwig, model):
    return f"model:{model}, hazma, herwig = {hazma:.3e}, {herwig:.3e}"


def synchronize_herwig(model: VectorModel, module) -> None:
    module.resetParameters(
        model.gvxx,
        model.mx * 1e-3,
        model.mv * 1e-3,
        model.width_v() * 1e-3,
        model.gvuu,
        model.gvdd,
        model.gvss,
    )


def compare_widths(
    model: VectorModel,
    func: str,
    herwig_module,
    reltol=1e-4,
    abstol=0.0,
    herwig_args=tuple(),
):
    mvgev = model.mv * 1e-3
    synchronize_herwig(model, herwig_module)
    hazma = getattr(model, func)() / model.mv
    herwig = herwig_module.GammaDM(mvgev, *herwig_args) / mvgev
    assert hazma == approx(herwig, rel=reltol, abs=abstol)


def compare_form_factors(
    hazma: HazmaFunction, herwig: ModuleFunction, herwig_units: int = -1, reltol=1e-4
):

    synchronize_herwig(hazma.model, herwig.module)
    mvmev = hazma.model.mv
    mvgev = mvmev * 1e-3
    hazma_val = np.abs(hazma(mvmev))
    herwig_val = np.abs(herwig(mvgev**2)) / (mvgev**herwig_units)

    assert np.real(hazma_val) == approx(np.real(herwig_val), rel=reltol, abs=0.0)
    assert np.imag(hazma_val) == approx(np.imag(herwig_val), rel=reltol, abs=0.0)


# ===============================================================================
# ---- Test Form-Factors --------------------------------------------------------
# ===============================================================================


def test_form_factor_pi_pi(models: List[VectorModel]):
    """
    Test that the `hazma` and `herwig4DM` implementations of the V-pi-pi
    form factor agree for the input model.
    """
    from herwig4dm import F2pi

    for model in models:
        hazma = HazmaFunction(model, "form_factor_pi_pi")
        herwig = ModuleFunction(F2pi, "Fpi", (1,))
        compare_form_factors(hazma, herwig, 0)


def test_form_factor_pi0_gamma(models: List[VectorModel]):
    """
    Test that the `hazma` and `herwig4DM` implementations of the V-pi-gamma
    form factor agree for the input model.
    """
    from herwig4dm import FPiGamma

    for model in models:
        hazma = HazmaFunction(model, "form_factor_pi0_gamma", tuple())
        herwig = ModuleFunction(FPiGamma, "FPiGamma", tuple())
        compare_form_factors(hazma, herwig)


def test_form_factor_pi0_omega(models: List[VectorModel]):
    """
    Test that the `hazma` and `herwig4DM` implementations of the V-omega-pi
    form factor agree for the input model.
    """
    from herwig4dm import FOmegaPion

    for model in models:
        hazma = HazmaFunction(model, "form_factor_pi0_omega", tuple())
        herwig = ModuleFunction(FOmegaPion, "FOmPiGamma", tuple())
        compare_form_factors(hazma, herwig, reltol=1)


def test_form_factor_k0_k0(models: List[VectorModel]):
    """
    Test that the `hazma` and `herwig4DM` implementations of the K0 K0
    form factor agree for the input model.
    """
    from herwig4dm import FK

    for model in models:
        hazma = HazmaFunction(model, "form_factor_k0_k0")
        herwig = ModuleFunction(FK, "Fkaon", (0,))
        compare_form_factors(hazma, herwig, 0, reltol=5e-1)


def test_form_factor_kk(models: List[VectorModel]):
    """
    Test that the `hazma` and `herwig4DM` implementations of the K K
    form factor agree for the input model.
    """
    from herwig4dm import FK

    for model in models:
        hazma = HazmaFunction(model, "form_factor_k_k")
        herwig = ModuleFunction(FK, "Fkaon", (1,))
        compare_form_factors(hazma, herwig, 0)


@pytest.mark.skip(reason="Need to check why this form-factor seems to be so wrong.")
def test_form_factor_eta_gamma(models: List[VectorModel]):
    """
    Test that the `hazma` and `herwig4DM` implementations of the V-eta-gamma
    form factor agree for the input model.
    """
    from herwig4dm import FEtaGamma

    for model in models:
        hazma = HazmaFunction(model, "form_factor_eta_gamma", tuple())
        herwig = ModuleFunction(FEtaGamma, "FEtaGamma", tuple())

        compare_form_factors(hazma, herwig)


# ===============================================================================
# ---- Test Widths --------------------------------------------------------------
# ===============================================================================


def test_width_pi_pi(models: List[VectorModel]):
    from herwig4dm import FOmegaPion as herwig_module

    model_names = ["model1", "model2", "km", "bl"]

    for i, model in enumerate(models):
        mvgev = model.mv * 1e-3
        synchronize_herwig(model, herwig_module)
        hazma = model.width_v_to_pi_pi() / model.mv
        herwig = herwig_module.GammaDM(mvgev) / mvgev
        assert hazma == approx(herwig, rel=0.5, abs=0.0), error_msg(
            hazma, herwig, model_names[i]
        )


@pytest.mark.skip(
    reason="There is something strange going on for lower masses with this channel."
)
def test_width_k0_k0(models: List[VectorModel]):
    from herwig4dm import FK as herwig_module

    model_names = ["model1", "model2", "km", "bl"]

    for i, model in enumerate(models):
        mvgev = model.mv * 1e-3
        synchronize_herwig(model, herwig_module)
        hazma = model.width_v_to_k0_k0() / model.mv
        herwig = herwig_module.GammaDM(mvgev, imode=0) / mvgev
        assert hazma == approx(herwig, rel=1, abs=0.0), error_msg(
            hazma, herwig, model_names[i]
        )


def test_width_pi0_omega(models: List[VectorModel]):
    """
    Test that the `hazma` and `herwig4DM` implementations of the V-omega-pi
    widths agree for the input model.
    """
    from herwig4dm import FOmegaPion as herwig_module

    model_names = ["model1", "model2", "km", "bl"]

    for i, model in enumerate(models):
        mvgev = model.mv * 1e-3
        synchronize_herwig(model, herwig_module)
        hazma = model.width_v_to_pi0_pi0_gamma() / model.mv / 8.34e-2
        herwig = herwig_module.GammaDM(mvgev) / mvgev
        assert hazma == approx(herwig, rel=1e-4, abs=0.0), error_msg(
            hazma, herwig, model_names[i]
        )


def test_width_phi_pi(models: List[VectorModel]):
    """
    Test that the `hazma` and `herwig4DM` implementations of the V-phi-pi
    widths agree for the input model.
    """
    from herwig4dm import FPhiPi as herwig_module

    model_names = ["model1", "model2", "km", "bl"]

    for i, model in enumerate(models):
        # compare_widths(model, "width_v_to_pi0_phi", FPhiPi)
        mvgev = model.mv * 1e-3
        synchronize_herwig(model, herwig_module)
        hazma = model.width_v_to_pi0_phi() / model.mv
        herwig = herwig_module.GammaDM(mvgev) / mvgev
        assert hazma == approx(herwig, rel=1e-4, abs=0.0), error_msg(
            hazma, herwig, model_names[i]
        )


def test_width_eta_phi(models: List[VectorModel]):
    """
    Test that the `hazma` and `herwig4DM` implementations of the V-eta-phi
    widths agree for the input model.
    """
    from herwig4dm import FEtaPhi

    for model in models:
        compare_widths(model, "width_v_to_eta_phi", FEtaPhi)


def test_width_eta_omega(models: List[VectorModel]):
    """
    Test that the `hazma` and `herwig4DM` implementations of the V-eta-omega
    widths agree for the input model.
    """
    from herwig4dm import FEtaOmega

    for model in models:
        compare_widths(model, "width_v_to_eta_omega", FEtaOmega)


def test_width_pi_pi_pi0(models: List[VectorModel]):
    from herwig4dm import F3pi

    for model in models:
        compare_widths(model, "width_v_to_pi_pi_pi0", F3pi, reltol=1e-1)


def test_width_pi_pi_eta(models: List[VectorModel]):
    from herwig4dm import FEtaPiPi

    for model in models:
        compare_widths(model, "width_v_to_pi_pi_eta", FEtaPiPi, reltol=1e-3)


def test_width_pi_pi_etap(models: List[VectorModel]):
    from herwig4dm import FEtaPrimePiPi

    for model in models:
        compare_widths(model, "width_v_to_pi_pi_etap", FEtaPrimePiPi, reltol=1e-2)


def test_width_pi_pi_omega(models: List[VectorModel]):
    from herwig4dm import FOmegaPiPi

    for model in models:
        mvgev = model.mv * 1e-3
        synchronize_herwig(model, FOmegaPiPi)
        hazma = model.width_v_to_pi_pi_omega() / model.mv
        herwig = FOmegaPiPi.GammaDM(mvgev, mode=1) / mvgev
        assert hazma == approx(herwig, rel=1e-1, abs=1e-10)


def test_width_pi0_pi0_omega(models: List[VectorModel]):
    from herwig4dm import FOmegaPiPi

    for model in models:
        mvgev = model.mv * 1e-3
        synchronize_herwig(model, FOmegaPiPi)
        hazma = model.width_v_to_pi0_pi0_omega() / model.mv
        herwig = FOmegaPiPi.GammaDM(mvgev, mode=0) / mvgev
        assert hazma == approx(herwig, rel=1e-1, abs=1e-10)


def test_width_pi0_k0_k0(models: List[VectorModel]):
    from herwig4dm import FKKpi

    # compare_widths(model1, "width_v_to_pi0_k0_k0", FKKpi, herwig_args=(0,))

    for model in models:
        mvgev = model.mv * 1e-3
        synchronize_herwig(model, FKKpi)
        hazma = model.width_v_to_pi0_k0_k0() / model.mv
        herwig = FKKpi.GammaDM(mvgev, imode=0) / mvgev
        assert hazma == approx(herwig, rel=1e-1, abs=1e-10)


def test_width_pi0_k_k(models: List[VectorModel]):
    from herwig4dm import FKKpi

    # compare_widths(model1, "width_v_to_pi0_k_k", FKKpi, herwig_args=(1,))

    for model in models:
        mvgev = model.mv * 1e-3
        synchronize_herwig(model, FKKpi)
        hazma = model.width_v_to_pi0_k_k() / model.mv
        herwig = FKKpi.GammaDM(mvgev, imode=1) / mvgev
        assert hazma == approx(herwig, rel=1e-1, abs=1e-10)


def test_width_pi_k_k0(models: List[VectorModel]):
    from herwig4dm import FKKpi

    # compare_widths(model1, "width_v_to_pi0_k_k", FKKpi, herwig_args=(1,))

    for model in models:
        mvgev = model.mv * 1e-3
        synchronize_herwig(model, FKKpi)
        hazma = model.width_v_to_pi_k_k0() / model.mv
        # herwig4dm includes both charged modes so divide by 2
        herwig = FKKpi.GammaDM(mvgev, imode=2) / mvgev / 2
        assert hazma == approx(herwig, rel=1e-1, abs=1e-10)


def test_width_v_to_pi_pi_pi_pi(models: List[VectorModel]):
    from herwig4dm import F4pi

    for model in models:
        F4pi.cI1_ = 1.0
        synchronize_herwig(model, F4pi)
        F4pi.readHadronic_Current()

        mvmev = model.mv
        mvgev = mvmev * 1e-3
        hazma = model.width_v_to_pi_pi_pi_pi(npts=1 << 15) / mvmev
        herwig = F4pi.GammaDM(mvgev, "charged") / mvgev
        assert hazma == approx(herwig, rel=1e-1, abs=0.0)


def test_width_v_to_pi_pi_pi0_pi0(models: List[VectorModel]):
    from herwig4dm import F4pi

    for model in models:
        F4pi.cI1_ = 1.0
        synchronize_herwig(model, F4pi)
        F4pi.readHadronic_Current()
        mvmev = model.mv
        mvgev = mvmev * 1e-3
        hazma = model.width_v_to_pi_pi_pi0_pi0(npts=1 << 15) / mvmev
        herwig = F4pi.GammaDM(mvgev, "neutral") / mvgev
        assert hazma == approx(herwig, rel=1e-1, abs=0.0)


# ===============================================================================
# ---- Test Cross Sections ------------------------------------------------------
# ===============================================================================


def test_cross_section_pi_pi(models: List[VectorModel]):
    """
    Test that the `hazma` and `herwig4DM` implementations of the V-phi-pi
    widths agree for the input model.
    """
    from herwig4dm import F2pi as herwig_module

    model_names = ["model1", "model2", "km", "bl"]

    for i, model in enumerate(models):
        cme = 4 * model.mx
        cmegev = cme * 1e-3
        synchronize_herwig(model, herwig_module)
        hazma = model.sigma_xx_to_pi_pi(cme) * cme**2
        herwig = herwig_module.sigmaDM(cmegev**2) * cmegev**2
        assert hazma == approx(herwig, rel=1, abs=0.0), error_msg(
            hazma, herwig, model_names[i]
        )


def test_cross_section_k0_k0(models: List[VectorModel]):
    """
    Test that the `hazma` and `herwig4DM` implementations of the V-phi-pi
    widths agree for the input model.
    """
    from herwig4dm import FK as herwig_module

    model_names = ["model1", "model2", "km", "bl"]

    for i, model in enumerate(models):
        cme = 4 * model.mx
        cmegev = cme * 1e-3
        synchronize_herwig(model, herwig_module)
        hazma = model.sigma_xx_to_k0_k0(cme) * cme**2
        herwig = herwig_module.sigmaDM0(cmegev**2) * cmegev**2
        assert hazma == approx(herwig, rel=1, abs=0.0), error_msg(
            hazma, herwig, model_names[i]
        )


def test_cross_section_k_k(models: List[VectorModel]):
    """
    Test that the `hazma` and `herwig4DM` implementations of the V-phi-pi
    widths agree for the input model.
    """
    from herwig4dm import FK as herwig_module

    model_names = ["model1", "model2", "km", "bl"]

    for i, model in enumerate(models):
        # compare_widths(model, "width_v_to_pi0_phi", FPhiPi)
        cme = 4 * model.mx
        cmegev = cme * 1e-3
        synchronize_herwig(model, herwig_module)
        hazma = model.sigma_xx_to_k_k(cme) * cme**2
        herwig = herwig_module.sigmaDMP(cmegev**2) * cmegev**2
        assert hazma == approx(herwig, rel=1, abs=0.0), error_msg(
            hazma, herwig, model_names[i]
        )


def test_cross_section_pi_phi(models: List[VectorModel]):
    from herwig4dm import FPhiPi as herwig_module

    model_names = ["model1", "model2", "km", "bl"]

    for i, model in enumerate(models):
        # compare_widths(model, "width_v_to_pi0_phi", FPhiPi)
        cme = 4 * model.mx
        cmegev = cme * 1e-3
        synchronize_herwig(model, herwig_module)
        hazma = model.sigma_xx_to_pi0_phi(cme) * cme**2
        herwig = herwig_module.sigmaDMPhiPi(cmegev**2) * cmegev**2
        assert hazma == approx(herwig, rel=1, abs=0.0), error_msg(
            hazma, herwig, model_names[i]
        )


def test_cross_section_eta_phi(models: List[VectorModel]):
    from herwig4dm import FEtaPhi as herwig_module

    model_names = ["model1", "model2", "km", "bl"]

    for i, model in enumerate(models):
        # compare_widths(model, "width_v_to_pi0_phi", FPhiPi)
        cme = 4 * model.mx
        cmegev = cme * 1e-3
        synchronize_herwig(model, herwig_module)
        hazma = model.sigma_xx_to_eta_phi(cme) * cme**2
        herwig = herwig_module.sigmaDMEtaPhi(cmegev**2) * cmegev**2
        assert hazma == approx(herwig, rel=1, abs=0.0), error_msg(
            hazma, herwig, model_names[i]
        )
