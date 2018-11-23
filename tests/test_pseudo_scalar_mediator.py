from hazma.pseudo_scalar_mediator import PseudoScalarMediator
import numpy as np


def test_list_final_states():
    PSM = PseudoScalarMediator(mx=200., mp=1e3, gpxx=1., gpuu=1., gpdd=1.,
                               gpss=1., gpee=1., gpmumu=1., gpGG=1., gpFF=1.)

    PSM.list_annihilation_final_states()


def test_cross_sections():
    PSM = PseudoScalarMediator(mx=200., mp=1e3, gpxx=1., gpuu=1., gpdd=1.,
                               gpss=1., gpee=1., gpmumu=1., gpGG=1., gpFF=1.)

    cme = 1000.

    PSM.annihilation_cross_sections(cme)


def test_branching_fractions():
    PSM = PseudoScalarMediator(mx=200., mp=1e3, gpxx=1., gpuu=1., gpdd=1.,
                               gpss=1., gpee=1., gpmumu=1., gpGG=1., gpFF=1.)

    cme = 1000.

    PSM.annihilation_branching_fractions(cme)


def test_spectra():
    PSM = PseudoScalarMediator(mx=200., mp=1e3, gpxx=1., gpuu=1., gpdd=1.,
                               gpss=1., gpee=1., gpmumu=1., gpGG=1., gpFF=1.)

    cme = 1000.
    egams = np.logspace(0., np.log10(cme), num=50)

    PSM.spectra(egams, cme)


def test_spectrum_functions():
    PSM = PseudoScalarMediator(mx=200., mp=1e3, gpxx=1., gpuu=1., gpdd=1.,
                               gpss=1., gpee=1., gpmumu=1., gpGG=1., gpFF=1.)

    PSM.spectrum_functions()


def test_partial_widths():
    pass
