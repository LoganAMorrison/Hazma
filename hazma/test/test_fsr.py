from hazma.fsr_helper_functions import scalar_mediator_fsr
from hazma.fsr_helper_functions import pseudo_scalar_mediator_fsr
from hazma.fsr_helper_functions import vector_mediator_fsr

from hazma.parameters import muon_mass

import numpy as np


def test_scalar_mediator_fsr():
    eng_gams = np.logspace(-3., 3., num=100)
    scalar_mediator_fsr.fermion(eng_gams, 1000., muon_mass)


def test_pseudo_scalar_mediator_fsr():
    eng_gams = np.logspace(-3., 3., num=100)
    pseudo_scalar_mediator_fsr.fermion(eng_gams, 1000., muon_mass)


def test_vector_mediator_fsr():
    eng_gams = np.logspace(-3., 3., num=100)
    vector_mediator_fsr.fermion(eng_gams, 1000., muon_mass)
