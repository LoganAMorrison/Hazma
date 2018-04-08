from hazma.scalar_mediator import ScalarMediator
import numpy as np


def test_list_final_states():
    mx, ms = [100., 200.]
    gsxx, gsff, gsGG, gsFF = [1., 1., 1., 1.]

    SM = ScalarMediator(mx, ms, gsxx, gsff, gsGG, gsFF)
    SM.list_final_states()


def test_cross_sections():
    mx, ms = [100., 200.]
    gsxx, gsff, gsGG, gsFF = [1., 1., 1., 1.]
    cme = 1000.

    SM = ScalarMediator(mx, ms, gsxx, gsff, gsGG, gsFF)
    SM.cross_sections(cme)


def test_branching_fractions():
    mx, ms = [100., 200.]
    gsxx, gsff, gsGG, gsFF = [1., 1., 1., 1.]
    cme = 1000.

    SM = ScalarMediator(mx, ms, gsxx, gsff, gsGG, gsFF)
    SM.branching_fractions(cme)


def test_spectra():
    mx, ms = [100., 200.]
    gsxx, gsff, gsGG, gsFF = [1., 1., 1., 1.]
    cme = 1000.
    eng_gams = np.logspace(0., np.log10(cme), num=50)

    SM = ScalarMediator(mx, ms, gsxx, gsff, gsGG, gsFF)
    SM.spectra(eng_gams, cme)


def test_spectrum_functions():
    mx, ms = [100., 200.]
    gsxx, gsff, gsGG, gsFF = [1., 1., 1., 1.]

    SM = ScalarMediator(mx, ms, gsxx, gsff, gsGG, gsFF)
    SM.spectrum_functions()


def test_partial_widths():
    mx, ms = [100., 200.]
    gsxx, gsff, gsGG, gsFF = [1., 1., 1., 1.]

    SM = ScalarMediator(mx, ms, gsxx, gsff, gsGG, gsFF)
    SM.partial_widths()
