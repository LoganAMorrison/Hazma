from hazma.scalar_mediator import ScalarMediator
from hazma.parameters import vh
from hazma.parameters import electron_mass as me
# import pickle
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

# with open('test_sm_acs_1.pkl', 'rb') as f:
#     acs_1_pickle = pickle.load(f)
#
# with open('test_sm_acs_2.pkl', 'rb') as f:
#     acs_2_pickle = pickle.load(f)
#
# with open('test_sm_abfs_1.pkl', 'rb') as f:
#     abfs_1_pickle = pickle.load(f)
#
# with open('test_sm_abfs_2.pkl', 'rb') as f:
#     abfs_2_pickle = pickle.load(f)
#
# with open('test_sm_spectra_1.pkl', 'rb') as f:
#     spec_1_pickle = pickle.load(f)
#
# with open('test_sm_spectra_2.pkl', 'rb') as f:
#     spec_2_pickle = pickle.load(f)


def test_description():
    SM1.description()


def test_list_final_states():
    list_fs = SM1.list_annihilation_final_states()

    assert list_fs == ['mu mu', 'e e', 'g g', 'pi0 pi0', 'pi pi', 's s']


def test_cross_sections():
    SM1.annihilation_cross_sections(cme)
    SM2.annihilation_cross_sections(cme)

    # is_close = True
    # for key in acs_1_pickle.keys():
    #     is_close = is_close * np.isclose(acs_1_pickle[key], css1[key],
    #                                      rtol=1e-5, atol=0.0)
    #
    # assert is_close is True
    #
    # is_close = True
    # for key in acs_2_pickle.keys():
    #     is_close = is_close * np.isclose(acs_2_pickle[key], css2[key],
    #                                      rtol=1e-5, atol=0.0)
    #
    # assert is_close is True


def test_branching_fractions():
    SM1.annihilation_cross_sections(cme)
    SM2.annihilation_cross_sections(cme)

    # is_close = True
    # for key in abfs_1_pickle.keys():
    #     is_close = is_close * np.isclose(abfs_1_pickle[key], cbfs1[key],
    #                                      rtol=1e-5, atol=0.0)
    #
    # assert is_close is True
    #
    # is_close = True
    # for key in abfs_2_pickle.keys():
    #     is_close = is_close * np.isclose(abfs_2_pickle[key], cbfs2[key],
    #                                      rtol=1e-5, atol=0.0)
    #
    # assert is_close is True


def test_compute_vs():
    assert SM1.compute_vs() == 0.
    assert SM2.compute_vs() == 0.


def test_spectra():
    egams = np.logspace(0., np.log10(cme), num=10)

    SM1.spectra(egams, cme)
    SM2.spectra(egams, cme)

    # isclose = True
    # for key in spec_1_pickle.keys():
    #     isclose = np.bool(
    #         np.prod(np.isclose(spectra1[key], spec_1_pickle[key],
    #                            atol=0., rtol=100)))
    #
    # assert isclose is True
    #
    # isclose = True
    # for key in spec_2_pickle.keys():
    #     isclose = np.bool(
    #         np.prod(np.isclose(spectra2[key], spec_2_pickle[key],
    #                            atol=0., rtol=100)))
    #
    # assert isclose is True


def test_spectrum_functions():
    SM1.spectrum_functions()


def test_partial_widths():
    SM1.partial_widths()


def test_positron_spectra():
    cme = 1e3
    eng_ps = np.logspace(me, np.log10(cme), num=10)
    SM1.positron_spectra(eng_ps, cme)


def test_positron_lines():
    cme = 1e3
    SM1.positron_lines(cme)
