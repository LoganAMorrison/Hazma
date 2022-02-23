# Libraries to Load
import math
import cmath

import scipy
import scipy.integrate

from . import Resonance
from . import alpha
from .masses import mpi, metap, fpi


ii = complex(0.0, 1.0)
gev2nb = 389379.3656

# parametrization taken from 1306.1985 with own fit values as in 1911.11147
# Fit values
mRho1 = 1.54
mRho2 = 1.76
mRho3 = 2.11
gRho1 = 0.356
gRho2 = 0.113
gRho3 = 0.176
a1 = 0.0
a2 = 0.0
a3 = 0.02
phi1 = 3.14
phi2 = 3.14
phi3 = 3.14
# metap = 0.95778
# mpi = 0.13957018
# fpi = 0.0933
mRho_ = [0.77549, 1.54, 1.76, 2.11]
gRho_ = [0.1494, 0.356, 0.113, 0.176]
amp_ = [1.0, 0.0, 0.0, 0.02]
phase_ = [0.0, 3.14, 3.14, 3.14]
upp_lim = 3.0


# coupling modification depending on mediator quark couplings
# coupling strength to rho (I1),omega (I0) and phi (S) contributions
cI1_ = 1.0
cI0_ = 1.0
cS_ = 1.0

camp_ = []
for i in range(0, len(amp_)):
    camp_.append(amp_[i] * cmath.exp(ii * phase_[i]))

total = sum(camp_)
for i in range(0, len(camp_)):
    camp_[i] /= total


def resetParameters(gDM, mDM, mMed, wMed, cMedu, cMedd, cMeds):
    global gDM_, mDM_, mMed_, wMed_, cI1_, cI0_, cS_
    cI1_ = cMedu - cMedd
    cI0_ = 3 * (cMedu + cMedd)
    cS_ = -3 * cMeds
    gDM_ = gDM
    mDM_ = mDM
    mMed_ = mMed
    wMed_ = wMed


#    camp_ = []
#    if cRho_!=0:
#        for i in range(0,len(amp_)) :
#            camp_.append(cRho_*amp_[i]*cmath.exp(ii*phase_[i]))
#        total = sum(camp_)
#        for i in range(0,len(camp_)) :
#            camp_[i] /=total
#    else:
#        camp_ = [0.,0.,0.,0.]


def gammaRho(i, Q2):
    if i == 0:
        return (
            gRho_[0]
            * mRho_[0] ** 2
            / Q2
            * ((Q2 - 4.0 * mpi ** 2) / (mRho_[0] ** 2 - 4.0 * mpi ** 2)) ** 1.5
        )
    else:
        return gRho_[i] * Q2 / mRho_[i] ** 2


def BW(i, Q2):
    gam = gammaRho(i, Q2)
    mr2 = mRho_[i] ** 2
    return mr2 / (mr2 - Q2 - ii * math.sqrt(Q2) * gam)


def pcm(m02, m12, m22):
    return 0.5 * math.sqrt(
        (
            m02 ** 2
            + m12 ** 2
            + m22 ** 2
            - 2.0 * m02 * m12
            - 2.0 * m02 * m22
            - 2.0 * m12 * m22
        )
        / m02
    )


def FEtaPrimePiPi(s, Q2):
    form = 0.0
    pre = 0.25 * math.sqrt(2) / math.sqrt(3) / math.pi ** 2 / fpi ** 3
    for i in range(0, len(camp_)):
        form += BW(i, s) * camp_[i] * cI1_
    form *= BW(0, Q2)
    return pre * form


def integrand(rho, s):
    output = []
    for val in rho:
        # change of variables for easier integration with scipy
        # Q2 is momentum squared of the rho of rho->pipi
        Q2 = mRho_[0] * gRho_[0] * math.tan(val) + mRho_[0] ** 2
        # momentum dependence after one phase space integration (see eq.139 notes)
        peta = pcm(s, metap ** 2, Q2)
        ppi = pcm(Q2, mpi ** 2, mpi ** 2)
        Q = math.sqrt(Q2)
        # 'Jabobi' due to change of variables in integration
        pre = (
            ((Q2 - mRho_[0] ** 2) ** 2 + (mRho_[0] * gRho_[0]) ** 2)
            / mRho_[0]
            / gRho_[0]
        )
        output.append(pre * (peta * ppi) ** 3 / Q * abs(FEtaPrimePiPi(s, Q2)) ** 2)
    return output


def phase(s):
    # integration limits
    upp = (math.sqrt(s) - metap) ** 2
    low = 4.0 * mpi ** 2
    if upp <= low:
        return 0.0
    # transform to new variables
    upp = math.atan((upp - mRho_[0] ** 2) / gRho_[0] / mRho_[0])
    low = math.atan((low - mRho_[0] ** 2) / gRho_[0] / mRho_[0])
    return scipy.integrate.quadrature(
        integrand, low, upp, args=s, tol=1e-12, maxiter=200
    )[0]


# partial decay width for Eta Pi Pi
def GammaDM(mMed):
    if mMed ** 2 < (2 * mpi + metap) ** 2:
        return 0
    if cI1_ == 0:
        return 0
    # vector spin average
    pre = 1 / 3.0
    # coming from phase space (see eq.139 notes)
    pre *= (1.0 / 12.0 / (2.0 * math.pi) ** 3 / mMed) * mMed
    return pre * phase(mMed ** 2)


# DM annihilation
def sigmaDM(s):
    if s < (2 * mpi + metap) ** 2:
        return 0
    if cI1_ == 0:
        return 0
    cDM = gDM_
    DMmed = cDM / (s - mMed_ ** 2 + complex(0.0, 1.0) * mMed_ * wMed_)
    DMmed2 = abs(DMmed) ** 2
    pre = DMmed2 * s * (1 + 2 * mDM_ ** 2 / s) / 3.0
    # coming from phase space (see eq.139 notes)
    pre *= 1.0 / 12.0 / (2 * math.pi) ** 3 / s ** 0.5
    return pre * phase(s) * Resonance.gev2nb


# cross-section for e+e- -> Eta' Pi Pi
def sigmaSM(s):
    if cI1_ == 0 or s < (metap + 2 * mpi) ** 2:
        return 0
    pre = 16.0 * math.pi ** 2 * alpha.alphaEM(s) ** 2 / 3.0 / s
    # coming from phase space (see eq.139 notes)
    pre *= 1.0 / 12.0 / (2 * math.pi) ** 3 / s ** 0.5
    return pre * phase(s) * Resonance.gev2nb
