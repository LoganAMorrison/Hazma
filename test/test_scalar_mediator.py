from hazma.scalar_mediator import ScalarMediator
from hazma.parameters import vh
from hazma.parameters import electron_mass as me
import pickle
import numpy as np

mx = 250.
ms1 = 550.
ms2 = 200.
vrel = 1e-3
gsxx = 1.
stheta = 1e-3
gsff = stheta
gsGG = 3. * stheta
gsFF = -5. * stheta / 6.
lam = vh

cme = 2. * mx * (1. + 0.5 * vrel**2)

params1 = {"mx": mx, "ms": ms1, "gsxx": gsxx,
           "gsff": gsff, "gsGG": gsGG, "gsFF": gsFF, "lam": lam}
params2 = {"mx": mx, "ms": ms2, "gsxx": gsxx,
           "gsff": gsff, "gsGG": gsGG, "gsFF": gsFF, "lam": lam}

SM1 = ScalarMediator(**params1)
SM2 = ScalarMediator(**params2)


def test_description():
    description = SM1.description()


def test_list_final_states():
    list_fs = SM1.list_annihilation_final_states()

    assert list_fs == ['mu mu', 'e e', 'g g', 'pi0 pi0', 'pi pi', 's s']


def test_cross_sections():
    css1 = SM1.annihilation_cross_sections(cme)
    css2 = SM2.annihilation_cross_sections(cme)

    data1 = pickle.load("test_sm_acs_1.pkl", "rb")
    data2 = pickle.load("test_sm_acs_2.pkl", "rb")

    is_close = True
    for key in data1.keys():
        is_close = is_close * np.isclose(data1[key], css1[key],
                                         rtol=1e-5, atol=0.0)

    assert is_close is True

    is_close = True
    for key in data1.keys():
        is_close = is_close * np.isclose(data2[key], css2[key],
                                         rtol=1e-5, atol=0.0)

    assert is_close is True


def test_branching_fractions():
    cbfs1 = SM1.annihilation_cross_sections(cme)
    cbfs2 = SM2.annihilation_cross_sections(cme)

    data1 = pickle.load("test_sm_abfs_1.pkl", "rb")
    data2 = pickle.load("test_sm_abfs_2.pkl", "rb")

    is_close = True
    for key in data1.keys():
        is_close = is_close * np.isclose(data1[key], cbfs1[key],
                                         rtol=1e-5, atol=0.0)

    assert is_close is True

    is_close = True
    for key in data1.keys():
        is_close = is_close * np.isclose(data2[key], cbfs2[key],
                                         rtol=1e-5, atol=0.0)

    assert is_close is True


def test_compute_vs():
    assert SM1.compute_vs() == 0.
    assert SM2.compute_vs() == 0.


def test_spectra():
    egams = np.logspace(0., np.log10(cme), num=10)

    spectra1 = SM1.spectra(egams, cme)
    spectra2 = SM2.spectra(egams, cme)

    data1 = pickle.load("test_sm_spectra_1.pkl", "rb")
    data2 = pickle.load("test_sm_spectra_2.pkl", "rb")

    isclose = True
    for key in spectra1.keys():
        isclose = np.bool(
            np.prod(np.isclose(spectra1[key], data1[key],
                               atol=0., rtol=100)))

    assert isclose is True

    isclose = True
    for key in spectra1.keys():
        isclose = np.bool(
            np.prod(np.isclose(spectra2[key], data2[key],
                               atol=0., rtol=100)))

    assert isclose is True


def test_spectrum_functions():
    spec_funcs = SM.spectrum_functions()


def test_partial_widths():
    pws = SM.partial_widths()


def test_positron_spectra():
    cme = 1e3
    eng_ps = np.logspace(me, np.log10(cme), num=10)
    pos_spec = SM.positron_spectra(eng_ps, cme)


def test_positron_lines():
    cme = 1e3
    pos_lines = SM.positron_lines(cme)
