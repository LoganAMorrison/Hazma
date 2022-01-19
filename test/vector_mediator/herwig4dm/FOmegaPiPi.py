import math
import scipy.integrate
from . import alpha
import cmath
import os
import matplotlib.pyplot as plt

ii = complex(0.0, 1.0)
gev2nb = 389379.3656
cF0_ = [0.165, 0.695]
# mPi_       = [0.1349770,0.13957018]
# mK_        = [0.493677,0.497611]#K+-, K0

# PDG values
mPi_ = [0.1349770, 0.13957018]
mK_ = [0.493677, 0.497611]

#############################################
# own parametrization, see arXiv:1911.11147 #
#############################################

m_F0_ = 0.990
g_F0_ = 0.1
mF0_ = [0.600, 0.980]
gF0_ = [1.0, 0.10]
aF0_ = [1.0, 0.883]
phasef0_ = 0.0
# mB1_       = 1.2295
# gB1_       = 0.142
mOm_ = 0.78265  # omega mass for decay product
mOmega_ = [0.783, 1.420, 1.6608543573197]  # omega masses for vector meson mixing
gOmega_ = [0.00849, 0.315, 0.3982595005228462]
aOm_ = [0.0, 0.0, 2.728870588760009]
phaseOm_ = [0.0, math.pi, 0.0]

# coupling modification depending on mediator quark couplings
cI1_ = 1.0
cI0_ = 1.0
cS_ = 1.0


# change rho, omega, phi contributions
def resetParameters(gDM, mDM, mMed, wMed, cMedu, cMedd, cMeds):
    global gDM_, mDM_, mMed_, wMed_, cI1_, cI0_, cS_
    cI1_ = cMedu - cMedd
    cI0_ = 3 * (cMedu + cMedd)
    cS_ = -3 * cMeds
    gDM_ = gDM
    mDM_ = mDM
    mMed_ = mMed
    wMed_ = wMed


# Some Breit-Wigner functions, can probably be found in Resonance.py as well


def BWs(a, s, mMed, gMed):
    return a * mMed ** 2 / ((mMed ** 2 - s) - ii * math.sqrt(s) * gMed)


def BW(a, s, mMed, gMed):
    return a * mMed ** 2 / ((mMed ** 2 - s) - ii * mMed * gMed)


# momentum
def pcm(m02, m12, m22):
    if (
        (
            m02 ** 2
            + m12 ** 2
            + m22 ** 2
            - 2.0 * m02 * m12
            - 2.0 * m02 * m22
            - 2.0 * m12 * m22
        )
        / m02
    ) < 0:
        print(
            (
                m02 ** 2
                + m12 ** 2
                + m22 ** 2
                - 2.0 * m02 * m12
                - 2.0 * m02 * m22
                - 2.0 * m12 * m22
            )
            / m02,
            m02,
            m12,
            m22,
        )
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


# def production_vec(s,mMed, gMed,a_s,phaseOm):
#     pre = alpha.alphaEM(s)**2/48./math.pi/s**3
#     med_prop=0.
#     for i in range(0,len(mMed)):
#         med_prop+=cOmega_*BWs(a_s[i],s,mMed[i],gMed[i])*cmath.exp(ii*phaseOm[i])
#     return pre*abs(med_prop)**2


# Integrand for mpipi^2 integration,
# one m_ij (m_{OmPi}) already analytically performed
def Integrand(mpp, s, mode):
    output = []
    sqrts = math.sqrt(s)
    for val in mpp:
        Q2 = val
        Q = math.sqrt(Q2)
        # see eq. (141) in low_energy.pdf notes
        P2 = pcm(Q2, mPi_[mode] ** 2, mPi_[mode] ** 2)
        P3 = pcm(s, Q2, mOm_ ** 2)
        mom_term = P2 * P3 / Q * (1 + P3 ** 2 / 3.0 / mOm_ ** 2)
        output.append(mom_term)
    return output


# Integration over mpipi^2
def phase(s, mode):
    if s < (mOm_ + 2 * mPi_[mode]) ** 2:
        return 0
    upp = (math.sqrt(s) - mOm_) ** 2
    low = 4.0 * mPi_[1] ** 2
    pre = 1.0
    if mode == 0:
        pre /= 2.0
    return (
        pre
        * scipy.integrate.quadrature(
            Integrand, low, upp, args=(s, mode), tol=1e-10, maxiter=200
        )[0]
    )


###############
# Form factor #
###############
def FOmegaPiPi(s):
    med_prop = 0.0
    for i in range(0, len(mOmega_)):
        med_prop += (
            cI0_ * BWs(aOm_[i], s, mOmega_[i], gOmega_[i]) * cmath.exp(ii * phaseOm_[i])
        )
    return med_prop


##################
# Cross sections #
##################
# e+e- -> Omega Pi Pi, mode=0:
def sigmaSM(s, mode):
    if s <= (mOm_ + 2 * mPi_[1]) ** 2:
        return 0
    # initial leptonic current (e+e-)
    pre = 16.0 * math.pi ** 2 * alpha.alphaEM(s) ** 2 / 3.0 / s
    # coming from phase space, see eq. (141) in low_energy.pdf notes
    pre *= 3 / 64.0 / math.pi ** 3 / s ** 1.5
    return pre * phase(s, mode) * gev2nb * abs(FOmegaPiPi(s)) ** 2


# Dark sector


def GammaDM(mMed, mode=1):
    if mMed ** 2 <= (2 * mPi_[1] + mOm_) ** 2:
        return 0
    if cI0_ == 0:
        return 0
    # vector spin average
    pre = 1 / 3.0
    # coming from phase space, see eq. (141) in low_energy.pdf notes
    pre *= 3 / 64.0 / math.pi ** 3 / mMed ** 2
    return pre * phase(mMed ** 2, mode) * abs(FOmegaPiPi(mMed ** 2)) ** 2


def sigmaDM(s, mode):
    if s < (2 * mPi_[mode] + mOm_) ** 2:
        return 0
    if cI0_ == 0:
        return 0
    cDM = gDM_
    # initial DM current
    DMmed = cDM / (s - mMed_ ** 2 + complex(0.0, 1.0) * mMed_ * wMed_)
    DMmed2 = abs(DMmed) ** 2
    pre = DMmed2 * s * (1 + 2 * mDM_ ** 2 / s) / 3.0
    # coming from phase space, see eq. (141) in low_energy.pdf notes
    pre *= 3 / 64.0 / math.pi ** 3 / s ** 1.5
    return pre * phase(s, mode) * gev2nb * abs(FOmegaPiPi(s)) ** 2
