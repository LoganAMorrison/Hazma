from hazma.pseudo_scalar_mediator import PseudoScalarMediator
import numpy as np


def test_list_final_states():
    mx, mp = [100., 200.]
    gpxx, gpff = [1., 1.]
    PSM = PseudoScalarMediator(mx, mp, gpxx, gpff)

    PSM.list_final_states()


def test_cross_sections():
    mx, mp = [100., 200.]
    gpxx, gpff = [1., 1.]
    PSM = PseudoScalarMediator(mx, mp, gpxx, gpff)

    cme = 1000.

    PSM.cross_sections(cme)


def test_branching_fractions():
    mx, mp = [100., 200.]
    gpxx, gpff = [1., 1.]
    PSM = PseudoScalarMediator(mx, mp, gpxx, gpff)

    cme = 1000.

    PSM.branching_fractions(cme)


def test_spectra():
    mx, mp = [100., 200.]
    gpxx, gpff = [1., 1.]
    PSM = PseudoScalarMediator(mx, mp, gpxx, gpff)

    cme = 1000.
    egams = np.logspace(0., np.log10(cme), num=50)

    PSM.spectra(egams, cme)


def test_spectrum_functions():
    mx, mp = [100., 200.]
    gpxx, gpff = [1., 1.]
    PSM = PseudoScalarMediator(mx, mp, gpxx, gpff)

    PSM.spectrum_functions()


def test_partial_widths():
    pass
