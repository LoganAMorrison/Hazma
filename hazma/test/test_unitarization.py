from math import pi, cos

import numpy as np

from hazma.parameters import pion_mass_chiral_limit as mPI
from hazma.parameters import kaon_mass_chiral_limit as mK

from hazma.unitarization.lo_amplitudes import amp_pipi_to_pipi_LO
from hazma.unitarization.lo_amplitudes import amp_pipi_to_pipi_LO_I
from hazma.unitarization.lo_amplitudes import msqrd_pipi_to_pipi_LO_I
from hazma.unitarization.lo_amplitudes import partial_wave_pipi_to_pipi_LO_I


from hazma.unitarization.nlo_amplitudes import amp_pipi_to_pipi_NLO
from hazma.unitarization.nlo_amplitudes import amp_pipi_to_pipi_NLO_I
from hazma.unitarization.nlo_amplitudes import msqrd_pipi_to_pipi_NLO_I
from hazma.unitarization.nlo_amplitudes import partial_wave_pipi_to_pipi_NLO_I
from hazma.unitarization.nlo_amplitudes import \
    high_L_partial_wave_pipi_to_pipi_NLO_I

from hazma.unitarization.loops import bubble_loop
from hazma.unitarization.loops import loop_matrix
from hazma.unitarization.loops import Q_MAX

from hazma.unitarization.bethe_salpeter import amp_kk_to_kk_bse
from hazma.unitarization.bethe_salpeter import amp_pipi_to_kk_bse
from hazma.unitarization.bethe_salpeter import amp_pipi_to_pipi_bse
from hazma.unitarization.bethe_salpeter import phase_shift
from hazma.unitarization.bethe_salpeter import fix_phases

from hazma.unitarization.inverse_amplitude_method import amp_pipi_to_pipi_iam
from hazma.unitarization.inverse_amplitude_method import \
    msqrd_inverse_amplitude_pipi_to_pipi


# ################################
""" lo_amplitudes.py functions """
# ################################


def test_amp_pipi_to_pipi_LO():
    s = 500.**2
    x = cos(pi / 4.)
    t = 0.5 * (4. * mPI**2 - s) * (1. - x)
    u = 4. * mPI**2 - s - t

    amp_pipi_to_pipi_LO(s, t, u)


def test_amp_pipi_to_pipi_LO_I():
    s = 500.**2
    x = cos(pi / 4.)
    t = 0.5 * (4. * mPI**2 - s) * (1. - x)

    amp_pipi_to_pipi_LO_I(s, t, iso=0)
    amp_pipi_to_pipi_LO_I(s, t, iso=1)


def test_msqrd_pipi_to_pipi_LO_I():
    s = 500.**2

    msqrd_pipi_to_pipi_LO_I(s, iso=0)
    msqrd_pipi_to_pipi_LO_I(s, iso=1)


def test_partial_wave_pipi_to_pipi_LO_I():
    s = 500.**2
    ell = 0
    partial_wave_pipi_to_pipi_LO_I(s, ell, iso=0)
    partial_wave_pipi_to_pipi_LO_I(s, ell, iso=1)


# #################################
""" nlo_amplitudes.py functions """
# #################################


def test_amp_pipi_to_pipi_NLO():
    s = 500.**2
    x = cos(pi / 4.)
    t = 0.5 * (4. * mPI**2 - s) * (1. - x)
    u = 4. * mPI**2 - s - t

    amp_pipi_to_pipi_NLO(s, t, u, su=3)
    amp_pipi_to_pipi_NLO(s, t, u, su=2)


def test_amp_pipi_to_pipi_NLO_I():
    s = 500.**2
    x = cos(pi / 4.)
    t = 0.5 * (4. * mPI**2 - s) * (1. - x)

    amp_pipi_to_pipi_NLO_I(s, t, iso=0, su=3)
    amp_pipi_to_pipi_NLO_I(s, t, iso=0, su=2)

    amp_pipi_to_pipi_NLO_I(s, t, iso=1, su=3)
    amp_pipi_to_pipi_NLO_I(s, t, iso=1, su=2)


def test_msqrd_pipi_to_pipi_NLO_I():
    s = 500.**2

    msqrd_pipi_to_pipi_NLO_I(s, iso=0, su=3)


def test_partial_wave_pipi_to_pipi_NLO_I():
    s = 500.**2
    ell = 0

    partial_wave_pipi_to_pipi_NLO_I(s, ell, iso=0, su=3)
    partial_wave_pipi_to_pipi_NLO_I(s, ell, iso=0, su=2)

    partial_wave_pipi_to_pipi_NLO_I(s, ell, iso=1, su=3)
    partial_wave_pipi_to_pipi_NLO_I(s, ell, iso=1, su=2)


def test_high_L_partial_wave_pipi_to_pipi_NLO_I():
    s = 500.**2

    high_L_partial_wave_pipi_to_pipi_NLO_I(s, iso=0, su=3)
    high_L_partial_wave_pipi_to_pipi_NLO_I(s, iso=0, su=2)

    high_L_partial_wave_pipi_to_pipi_NLO_I(s, iso=1, su=3)
    high_L_partial_wave_pipi_to_pipi_NLO_I(s, iso=1, su=2)


# ########################
""" loops.py functions """
# ########################


def test_bubble_loop():
    cme = 1000.

    bubble_loop(cme, mPI, q_max=Q_MAX)
    bubble_loop(cme, mK, q_max=Q_MAX)


def test_loop_matrix():
    cme = 1000.

    loop_matrix(cme, q_max=Q_MAX)


# ###########################################
""" inverse_amplitude_method.py functions """
# ###########################################


def test_amp_pipi_to_pipi_iam(cmes):
    ell = 0
    iso = 0
    su = 3

    cmes = 1500.
    amp_pipi_to_pipi_iam(cmes, ell=ell, iso=iso, su=su)

    cmes = np.linspace(2. * mPI, 2000.)
    amp_pipi_to_pipi_iam(cmes, ell=ell, iso=iso, su=su)


def test_msqrd_inverse_amplitude_pipi_to_pipi():
    ell = 0
    iso = 0
    su = 3

    cmes = 1500.
    msqrd_inverse_amplitude_pipi_to_pipi(cmes, ell=ell, iso=iso, su=su)

    cmes = np.linspace(2. * mPI, 2000.)
    msqrd_inverse_amplitude_pipi_to_pipi(cmes, ell=ell, iso=iso, su=su)


# #################################
""" bethe_salpeter.py functions """
# #################################


def test_amp_kk_to_kk_bse():
    cmes = 1500.
    amp_kk_to_kk_bse(cmes, q_max=Q_MAX)

    cmes = np.linspace(2. * mK, 2000.)
    amp_kk_to_kk_bse(cmes, q_max=Q_MAX)


def test_amp_pipi_to_kk_bse():
    cmes = 1500.
    amp_pipi_to_kk_bse(cmes, q_max=Q_MAX)

    cmes = np.linspace(2. * mK, 2000.)
    amp_pipi_to_kk_bse(cmes, q_max=Q_MAX)


def test_amp_pipi_to_pipi_bse():
    cmes = 1500.
    amp_pipi_to_pipi_bse(cmes, q_max=Q_MAX)

    cmes = np.linspace(2. * mPI, 2000.)
    amp_pipi_to_pipi_bse(cmes, q_max=Q_MAX)


def test_phase_shift():
    cmes = np.linspace(2 * mPI * (1.001), 1300., dtype=complex)
    ss = cmes**2

    t0sLO = partial_wave_pipi_to_pipi_LO_I(ss, 0, 0)
    tsBSE = amp_pipi_to_pipi_bse(cmes, q_max=np.sqrt((1 * 1e3)**2 - mK**2))
    t0sNLOSU2 = partial_wave_pipi_to_pipi_NLO_I(ss, 0, 0, su=2)
    t0sNLOSU3 = partial_wave_pipi_to_pipi_NLO_I(ss, 0, 0)
    t0sIAMSU2 = amp_pipi_to_pipi_iam(cmes, ell=0, iso=0, su=2)
    t0sIAMSU3 = amp_pipi_to_pipi_iam(cmes, ell=0, iso=0, su=3)

    phase_shift(ss, t0sLO, deg=True)
    phase_shift(cmes, t0sNLOSU2, deg=True)
    phase_shift(cmes, t0sNLOSU3, deg=True)
    phase_shift(cmes, t0sIAMSU2, deg=True)
    phase_shift(cmes, t0sIAMSU3, deg=True)
    phase_shift(cmes, tsBSE, deg=True)


def test_fix_phases():
    cmes = np.linspace(2 * mPI * (1.001), 1300., dtype=complex)
    ss = cmes**2

    t0sLO = partial_wave_pipi_to_pipi_LO_I(ss, 0, 0)
    tsBSE = amp_pipi_to_pipi_bse(cmes, q_max=np.sqrt((1 * 1e3)**2 - mK**2))
    t0sNLOSU2 = partial_wave_pipi_to_pipi_NLO_I(ss, 0, 0, su=2)
    t0sNLOSU3 = partial_wave_pipi_to_pipi_NLO_I(ss, 0, 0)
    t0sIAMSU2 = amp_pipi_to_pipi_iam(cmes, ell=0, iso=0, su=2)
    t0sIAMSU3 = amp_pipi_to_pipi_iam(cmes, ell=0, iso=0, su=3)

    fix_phases((phase_shift(ss, t0sLO, deg=True)), trigger=80.)
    fix_phases((phase_shift(cmes, t0sNLOSU2, deg=True)), trigger=80.)
    fix_phases((phase_shift(cmes, t0sNLOSU3, deg=True)), trigger=80.)
    fix_phases((phase_shift(cmes, t0sIAMSU2, deg=True)), trigger=80.)
    fix_phases((phase_shift(cmes, t0sIAMSU3, deg=True)), trigger=80.)
    fix_phases(phase_shift(cmes, tsBSE, deg=True), trigger=80.)
