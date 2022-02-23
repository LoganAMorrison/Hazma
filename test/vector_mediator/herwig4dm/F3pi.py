# Libraries to load
import math

from scipy import integrate

from . import Resonance
from . import alpha


ii = complex(0.0, 1.0)
gev2nb = 389379.3656

###########################################
# parametrization based on hep-ph/0512180 #
###########################################

# set of parameter
# assuming that approximately mpi+ = mpi- = mpi0
mPi_ = 0.140
# I=0 part consisting of omega, phi, omega, omega
mV_ = [0.7824, 1.01924, 1.375, 1.631]
gV_ = [0.00869, 0.00414, 0.250, 0.245]
cV_ = [18.20, -0.87, -0.77, -1.12]
mRho_ = [0.77609, 1.465, 1.7]
gRho_ = [0.14446, 0.31, 0.235]
cRho_ = [0.0, -0.72, -0.59]

# I=1 part, rho resonances
mRhoI1_ = [0.77609, 1.7]
gRhoI1_ = [0.14446, 0.26]
mOmI1_ = 0.78259
gOmI1_ = 0.00849
GW_pre_ = 1.55 / math.sqrt(2.0) * 12.924 * 0.266
g_omega_pi_pi_ = 0.185
sigma_ = -0.1

# Parameter set for DM part
gDM_ = 1.0
mDM_ = 0.41
mMed_ = 5
wMed_ = 10.0

# coupling strength to rho (I1), omega (I0) and phi (S) contributions
cI1_ = 1.0
cI0_ = 1.0
cS_ = 1.0
cVector_ = [cI1_, cI0_, cS_]

coeffs = {}
hadronic_interpolator = None


def resetParameters(gDM, mDM, mMed, wMed, cMedu, cMedd, cMeds):
    global gDM_, mDM_, mMed_, wMed_, cI1_, cI0_, cS_, cVector_
    cI1_ = cMedu - cMedd
    cI0_ = 3 * (cMedu + cMedd)
    cS_ = -3 * cMeds
    cVector_ = [cI1_, cI0_, cS_]
    gDM_ = gDM
    mDM_ = mDM
    mMed_ = mMed
    wMed_ = wMed


###########################
# necessary functions #
###########################

# p-wave width
def Gammarho(Qi2, mRho, gRho, mj, mk):
    return (
        gRho
        * mRho ** 2
        / Qi2
        * ((Qi2 - (mj + mk) ** 2) / (mRho ** 2 - (mj + mk) ** 2)) ** 1.5
    )


# Breit-Wigner for rhos
def BWrho(Qi2, mRho, gRho, mj, mk):
    return mRho ** 2 / (
        Qi2 - mRho ** 2 + ii * math.sqrt(Qi2) * Gammarho(Qi2, mRho, gRho, mj, mk)
    )


def Hrho(Qp2, Qm2, Q02, mRho, gRho):
    return (
        BWrho(Q02, mRho, gRho, mPi_, mPi_)
        + BWrho(Qp2, mRho, gRho, mPi_, mPi_)
        + BWrho(Qm2, mRho, gRho, mPi_, mPi_)
    )


"""
Form Factors
"""


def F0(Q2, Qp2, Qm2, Q02):
    """I=0 component"""
    if cI0_ == 0 and cS_ == 0:
        return 0
    c0 = 0.0
    for i in range(0, len(mV_)):
        if i != 1:
            c0 += cI0_ * cV_[i] * Resonance.BreitWignerFW(Q2, mV_[i], gV_[i])
        else:
            c0 += cS_ * cV_[i] * Resonance.BreitWignerFW(Q2, mV_[i], gV_[i])
    f0 = c0 * Hrho(Qp2, Qm2, Q02, mRho_[0], gRho_[0])

    f0 += (
        cS_
        * cRho_[1]
        * Resonance.BreitWignerFW(Q2, mV_[1], gV_[1])
        * Hrho(Qp2, Qm2, Q02, mRho_[1], gRho_[1])
    )
    f0 += (
        cI0_
        * cRho_[2]
        * Resonance.BreitWignerFW(Q2, mV_[3], gV_[3])
        * Hrho(Qp2, Qm2, Q02, mRho_[2], gRho_[2])
    )
    return f0


# Omega of I=0 only
def F0_Omega(Q2, Qp2, Qm2, Q02):
    if cI0_ == 0:
        return 0
    c0 = 0.0
    for i in range(0, len(mV_)):
        if i != 1:
            c0 += cI0_ * cV_[i] * Resonance.BreitWignerFW(Q2, mV_[i], gV_[i])
    f0 = c0 * Hrho(Qp2, Qm2, Q02, mRho_[0], gRho_[0])
    f0 += (
        cI0_
        * cRho_[2]
        * Resonance.BreitWignerFW(Q2, mV_[3], gV_[3])
        * Hrho(Qp2, Qm2, Q02, mRho_[2], gRho_[2])
    )
    return f0


# Phi of I=0 only
def F0_Phi(Q2, Qp2, Qm2, Q02):
    if cS_ == 0:
        return 0
    c0 = 0.0
    c0 += cS_ * cV_[1] * Resonance.BreitWignerFW(Q2, mV_[1], gV_[1])
    f0 = c0 * Hrho(Qp2, Qm2, Q02, mRho_[0], gRho_[0])
    f0 += (
        cS_
        * cRho_[1]
        * Resonance.BreitWignerFW(Q2, mV_[1], gV_[1])
        * Hrho(Qp2, Qm2, Q02, mRho_[1], gRho_[1])
    )
    return f0


# I=1 component
def F1(Q2, Q02):
    if cI1_ == 0:
        return 0
    f1 = BWrho(Q02, mRhoI1_[0], gRhoI1_[0], mPi_, mPi_) / mRhoI1_[0] ** 2
    f1 += sigma_ * BWrho(Q02, mRhoI1_[1], gRhoI1_[1], mPi_, mPi_) / mRhoI1_[1] ** 2
    GW = GW_pre_ * mRhoI1_[0] ** 2 * g_omega_pi_pi_
    f1 *= cI1_ * GW * Resonance.BreitWignerFW(Q2, mOmI1_, gOmI1_) / mOmI1_ ** 2
    return f1


# Total form factor, addition of isospin I=0 and isospin I=1 contributions
def F3pi(Q2, Qp2, Qm2, Q02):
    return F0(Q2, Qp2, Qm2, Q02) + F1(Q2, Q02)


"""
Contract hadronic current and integrate over phase space....
"""

# Lorentz contraction of Lorentz part in eq.(1) of hep-ph/0512180, assuming that
# mpi+ = mpi- = mpi0


def Lcontracted(Qp2, Qm2, Q02, s):
    return (
        1.0
        / 4.0
        * (
            -(Qp2 ** 2) * Qm2
            + Qp2 * Qm2 * (3 * mPi_ ** 2 - Qm2 + s)
            - (mPi_ ** 3 - mPi_ * s) ** 2
        )
    )


# Q02 expressed in other variables
def QZero2(s, Qp2, Qm2):
    return 3 * mPi_ ** 2 + s - Qm2 - Qp2


def Integrand(Qp2, Qm2, s):
    Q02 = QZero2(s, Qp2, Qm2)
    Lorentzpart = Lcontracted(Qp2, Qm2, Q02, s)
    form = abs(F3pi(s, Qp2, Qm2, Q02)) ** 2
    # output.append(Lorentzpart*form)
    return Lorentzpart * form


# bounds for Qm2
def bounds_Qm(s):
    upp = (math.sqrt(s) - mPi_) ** 2
    low = 4.0 * mPi_ ** 2
    return (low, upp)


# bounds for Qp2
def bounds_Qp(Qm2, s):
    E2s = 0.5 * math.sqrt(Qm2)
    E3s = 0.5 * (s - Qm2 - mPi_ ** 2) / math.sqrt(Qm2)
    low = (E2s + E3s) ** 2 - (
        math.sqrt(E2s ** 2 - mPi_ ** 2) + math.sqrt(E3s ** 2 - mPi_ ** 2)
    ) ** 2
    upp = (E2s + E3s) ** 2 - (
        math.sqrt(E2s ** 2 - mPi_ ** 2) - math.sqrt(E3s ** 2 - mPi_ ** 2)
    ) ** 2
    return (low, upp)


def int_current(s):
    return integrate.nquad(Integrand, [bounds_Qp, bounds_Qm], args=(s,))[0]


"""
Functions for mediator decay rate and cross-sections
"""


def GammaDM(medMass):
    """Decay rate of mediator-> 3pions"""
    if medMass < 3 * mPi_:
        return 0
    pre = medMass / 3.0
    # phase space prefactor
    pre *= 1.0 / (2 * math.pi) ** 3 / 32.0 / medMass ** 4
    s = medMass ** 2
    form = integrate.nquad(Integrand, [bounds_Qp, bounds_Qm], args=(s,))[0]
    return pre * form


# DM annihilation DM DM -> 3 pions
def sigmaDM(s):
    if s < (3 * mPi_) ** 2:
        return 0
    en = math.sqrt(s)
    cDM = gDM_
    DMmed = cDM / (s - mMed_ ** 2 + complex(0.0, 1.0) * mMed_ * wMed_)
    DMmed2 = abs(DMmed) ** 2
    # DM initial current+ mediator propagator
    pre = 1 / 3.0 * DMmed2 * s * (1 + 2 * mDM_ ** 2 / s)
    # phase space prefactor
    pre *= 1.0 / (2 * math.pi) ** 3 / 32.0 / s ** 2
    form = integrate.nquad(Integrand, [bounds_Qp, bounds_Qm], args=(s,))[0]
    return pre * form * Resonance.gev2nb


# SM e+e- -> 3 pions
def sigmaSM(s):
    if s < (3 * mPi_) ** 2:
        return 0
    # leptonic current +1/s photon propagator
    pre = 16.0 * math.pi ** 2 * alpha.alphaEM(s) ** 2 / 3.0 / s
    # phase space prefactor
    pre *= 1.0 / (2 * math.pi) ** 3 / 32.0 / s ** 2
    # form factor integration
    form = integrate.nquad(Integrand, [bounds_Qp, bounds_Qm], args=(s,))[0]
    return pre * form * Resonance.gev2nb
