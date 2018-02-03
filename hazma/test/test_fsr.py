from hazma.fsr_helper_functions.scalar_mediator_fsr import dnde_s
from hazma.fsr_helper_functions.pseudo_scalar_mediator_fsr import dnde_p
from hazma.fsr_helper_functions.vector_mediator_fsr import dnde_v

from hazma.parameters import muon_mass as mmu

import numpy as np


def test_scalar_mediator_fsr(cme):
    eng_gams = np.logspace(-3., np.log10(cme), num=150)
    return dnde_s(eng_gams, cme, mmu)


def test_pseudo_scalar_mediator_fsr(cme):
    eng_gams = np.logspace(-3., np.log10(cme), num=150)
    return dnde_p(eng_gams, cme, mmu)


def test_vector_mediator_fsr(cme):
    eng_gams = np.logspace(-3., np.log10(cme), num=150)
    return dnde_v(eng_gams, cme, mmu)


test_scalar_mediator_fsr(1000.)
test_pseudo_scalar_mediator_fsr(1000.)
test_vector_mediator_fsr(1000.)
