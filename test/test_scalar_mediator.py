from hazma.scalar_mediator import ScalarMediator
from hazma.parameters import vh
import numpy as np


def test_list_final_states():
    mx, ms = [100., 200.]
    gsxx, gsff, gsGG, gsFF = [1., 1., 1., 1.]
    lam = vh

    SM = ScalarMediator(mx, ms, gsxx, gsff, gsGG, gsFF, lam)
    SM.list_annihilation_final_states()


def test_cross_sections():
    mx, ms = [100., 200.]
    gsxx, gsff, gsGG, gsFF = [1., 1., 1., 1.]
    cme = 1000.
    lam = vh

    SM = ScalarMediator(mx, ms, gsxx, gsff, gsGG, gsFF, lam)
    SM.annihilation_cross_sections(cme)


def test_branching_fractions():
    mx, ms = [100., 200.]
    gsxx, gsff, gsGG, gsFF = [1., 1., 1., 1.]
    cme = 1000.
    lam = vh

    SM = ScalarMediator(mx, ms, gsxx, gsff, gsGG, gsFF, lam)
    SM.annihilation_branching_fractions(cme)


def test_spectra():
    mx, ms = [100., 200.]
    gsxx, gsff, gsGG, gsFF = [1., 1., 1., 1.]
    cme = 1000.
    egams = np.logspace(0., np.log10(cme), num=10)
    lam = vh

    SM = ScalarMediator(mx, ms, gsxx, gsff, gsGG, gsFF, lam)
    SM.spectra(egams, cme)


def test_spectrum_functions():
    mx, ms = [100., 200.]
    gsxx, gsff, gsGG, gsFF = [1., 1., 1., 1.]
    lam = vh

    SM = ScalarMediator(mx, ms, gsxx, gsff, gsGG, gsFF, lam)
    SM.spectrum_functions()


def test_partial_widths():
    mx, ms = [100., 200.]
    gsxx, gsff, gsGG, gsFF = [1., 1., 1., 1.]
    lam = vh

    SM = ScalarMediator(mx, ms, gsxx, gsff, gsGG, gsFF, lam)
    SM.partial_widths()
