from hazma.fsr_helper_functions.scalar_mediator_fsr import dnde_xx_to_s_to_ffg
from hazma.fsr_helper_functions.pseudo_scalar_mediator_fsr \
    import dnde_xx_to_p_to_ffg
from hazma.fsr_helper_functions.vector_mediator_fsr import dnde_xx_to_v_to_ffg

from hazma.parameters import muon_mass as mmu

import numpy as np


def test_scalar_mediator_fsr(cme):
    eng_gams = np.logspace(-3., np.log10(cme), num=150)
    return dnde_xx_to_s_to_ffg(eng_gams, cme, mmu)


def test_pseudo_scalar_mediator_fsr(cme):
    eng_gams = np.logspace(-3., np.log10(cme), num=150)
    return dnde_xx_to_p_to_ffg(eng_gams, cme, mmu)


def test_vector_mediator_fsr(cme):
    eng_gams = np.logspace(-3., np.log10(cme), num=150)
    return dnde_xx_to_v_to_ffg(eng_gams, cme, mmu)


test_scalar_mediator_fsr(1000.)
test_pseudo_scalar_mediator_fsr(1000.)
test_vector_mediator_fsr(1000.)
