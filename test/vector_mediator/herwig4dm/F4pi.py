# Libraries to load
import os
import math
import random
import glob

import numpy
from scipy.interpolate import interp1d

from . import Resonance
from . import alpha


"""
Parametrization taken from 0804.0359 with new fit values
"""

# masses and widths from PDG
mpip = 0.13957061
mpi0 = 0.1349770
mRho = 0.7755
gRho = 0.1494
mRho1 = 1.459
gRho1 = 0.4
mRho2 = 1.72
gRho2 = 0.25
ma1 = 1.23
ga1 = 0.2
mf0 = 1.35
gf0 = 0.2
mOmega = 0.78265
gOmega = 0.00849

# as given in 0804.0359
g_omega_pi_rho = 42.3
g_rho_pi_pi = 5.997
g_rho_gamma = 0.1212

# fit parameters of own fit
c_f0 = 124.10534971287902
beta1_f0 = 73860.28659732222
beta2_f0 = -26182.725634782986
beta3_f0 = 333.6314358023821

c_omega = -1.5791482789120541
beta1_omega = -0.36687866443745953
beta2_omega = 0.036253295280213906
beta3_omega = -0.004717302695776386

c_a1 = -201.79098091602876
beta1_a1 = -0.051871563361440096
beta2_a1 = -0.041610293030827125
beta3_a1 = -0.0018934309483457441

c_rho = -2.3089567893904537

mBar1 = 1.437
mBar2 = 1.738
mBar3 = 2.12

gBar1 = 0.6784824438511003
gBar2 = 0.8049287553822373
gBar3 = 0.20919646790795576

br_omega_pi_gamma = 0.084

coeffs_neutral = {}
coeffs_charged = {}

omega_interpolator = None

hadronic_interpolator_n = None
hadronic_interpolator_c = None

# Parameter set for DM part
gDM_ = 1.0
mDM_ = 0.41
mMed_ = 5
wMed_ = 10.0
cI1_ = 1.0
cI0_ = 1.0
cS_ = 1.0


def resetParameters(gDM, mDM, mMed, wMed, cMedu, cMedd, cMeds):
    """change rho, omega, phi contributions"""
    global cI1_, cI0_, cS_
    global cRhoOmPhi_
    global gDM_, mDM_, mMed_, wMed_
    gDM_ = gDM
    mDM_ = mDM
    mMed_ = mMed
    wMed_ = wMed
    cI1_ = cMedu - cMedd
    cI0_ = 3 * (cMedu + cMedd)
    cS_ = -3 * cMeds
    # readHadronic_Current()


def sigmaOmega(Q2):
    """
    Cross section, if we consider the omega contribution only
    """
    Q = math.sqrt(Q2)
    res = c_omega * Frho(Q2, beta1_omega, beta2_omega, beta3_omega)
    pre = (
        alpha.alphaEM(Q2) ** 2
        * 2.0
        / 3.0
        / Q2 ** 2
        * (2.0 * math.pi) ** 6
        * Resonance.gev2nb
    )
    return (
        pre
        * omega_interpolator(Q)
        * (res * res.conjugate()).real
        / (1.0 - br_omega_pi_gamma)
    )


def phaseSpace1(m0, m1, m2, m3, m4, M12, G12, M34, G34):
    if random.random() < 0.5:
        # generate the mass of the 12 system
        m12min = m1 + m2
        m12max = m0 - m3 - m4
        rhomin = math.atan2(m12min ** 2 - M12 ** 2, M12 * G12)
        rhomax = math.atan2(m12max ** 2 - M12 ** 2, M12 * G12) - rhomin
        rho = rhomin + rhomax * random.random()
        m122 = max(
            m12min ** 2, min(m12max ** 2, (M12) ** 2 + M12 * G12 * math.tan(rho))
        )
        m12 = math.sqrt(m122)
        # generate the mass of the 34 system
        m34min = m3 + m4
        m34max = m0 - m12
        rhomin = math.atan2(m34min ** 2 - M34 ** 2, M34 * G34)
        rhomax = math.atan2(m34max ** 2 - M34 ** 2, M34 * G34) - rhomin
        rho = rhomin + rhomax * random.random()
        m342 = max(m34min ** 2, min(m34max ** 2, M34 ** 2 + M34 * G34 * math.tan(rho)))
        m34 = math.sqrt(m342)
    else:
        m34min = m3 + m4
        m34max = m0 - m1 - m2
        rhomin = math.atan2(m34min ** 2 - M34 ** 2, M34 * G34)
        rhomax = math.atan2(m34max ** 2 - M34 ** 2, M34 * G34) - rhomin
        rho = rhomin + rhomax * random.random()
        m342 = max(m34min ** 2, min(m34max ** 2, M34 ** 2 + M34 * G34 * math.tan(rho)))
        m34 = math.sqrt(m342)
        m12min = m1 + m2
        m12max = m0 - m34
        rhomin = math.atan2(m12min ** 2 - M12 ** 2, M12 * G12)
        rhomax = math.atan2(m12max ** 2 - M12 ** 2, M12 * G12) - rhomin
        rho = rhomin + rhomax * random.random()
        m122 = max(m12min ** 2, min(m12max ** 2, M12 ** 2 + M12 * G12 * math.tan(rho)))
        m12 = math.sqrt(m122)
    (q1, q2) = twoBodyDecay(numpy.array([m0, 0.0, 0.0, 0.0]), m0, m12, m34)
    (p1, p2) = twoBodyDecay(q1, m12, m1, m2)
    (p3, p4) = twoBodyDecay(q2, m34, m3, m4)
    return (p1, p2, p3, p4)


def weight1(m0, m1, m2, m3, m4, m12, m34, M12, G12, M34, G34):
    # weight for 1st ordering
    m12min = m1 + m2
    m12max = m0 - m3 - m4
    rhomin = math.atan2(m12min ** 2 - M12 ** 2, M12 * G12)
    rhomax = math.atan2(m12max ** 2 - M12 ** 2, M12 * G12) - rhomin
    wgt1 = M12 * G12 / rhomax / ((m12 ** 2 - M12 ** 2) ** 2 + G12 ** 2 * M12 ** 2)
    m34min = m3 + m4
    m34max = m0 - m12
    rhomin = math.atan2(m34min ** 2 - M34 ** 2, M34 * G34)
    rhomax = math.atan2(m34max ** 2 - M34 ** 2, M34 * G34) - rhomin
    wgt1 *= M34 * G34 / rhomax / ((m34 ** 2 - M34 ** 2) ** 2 + +(G34 ** 2) * M34 ** 2)
    # weight for second ordering
    m34min = m3 + m4
    m34max = m0 - m1 - m2
    rhomin = math.atan2(m34min ** 2 - M34 ** 2, M34 * G34)
    rhomax = math.atan2(m34max ** 2 - M34 ** 2, M34 * G34) - rhomin
    wgt2 = M34 * G34 / rhomax / ((m34 ** 2 - M34 ** 2) ** 2 + +(G34 ** 2) * M34 ** 2)
    m12min = m1 + m2
    m12max = m0 - m34
    rhomin = math.atan2(m12min ** 2 - M12 ** 2, M12 * G12)
    rhomax = math.atan2(m12max ** 2 - M12 ** 2, M12 * G12) - rhomin
    wgt2 *= M12 * G12 / rhomax / ((m12 ** 2 - M12 ** 2) ** 2 + +(G12 ** 2) * M12 ** 2)
    # resonance piece of the weight
    wgt = 0.5 * (wgt1 + wgt2)
    # phase-space bits
    wgt *= m0 * 8.0 * math.pi ** 2 / pStar(m0, m12, m34)
    wgt *= m12 * 8.0 * math.pi ** 2 / pStar(m12, m1, m2)
    wgt *= m34 * 8.0 * math.pi ** 2 / pStar(m34, m3, m4)
    return wgt * (2.0 * math.pi) ** 3


def phaseSpace2(m0, m1, m2, m3, m4, M234, G234, M34, G34):
    m234min = m2 + m3 + m4
    m234max = m0 - m1
    rhomin = math.atan2(m234min ** 2 - M234 ** 2, M234 * G234)
    rhomax = math.atan2(m234max ** 2 - M234 ** 2, M234 * G234) - rhomin
    rho = rhomin + rhomax * random.random()
    m2342 = max(
        m234min ** 2, min(m234max ** 2, M234 ** 2 + M234 * G234 * math.tan(rho))
    )
    m234 = math.sqrt(m2342)
    (p1, q234) = twoBodyDecay(numpy.array([m0, 0.0, 0.0, 0.0]), m0, m1, m234)
    m34min = m3 + m4
    m34max = m234 - m2
    rhomin = math.atan2(m34min ** 2 - M34 ** 2, M34 * G34)
    rhomax = math.atan2(m34max ** 2 - M34 ** 2, M34 * G34) - rhomin
    rho = rhomin + rhomax * random.random()
    m342 = max(m34min ** 2, min(m34max ** 2, M34 ** 2 + M34 * G34 * math.tan(rho)))
    m34 = math.sqrt(m342)
    (p2, q34) = twoBodyDecay(q234, m234, m2, m34)
    (p3, p4) = twoBodyDecay(q34, m34, m3, m4)
    return (p1, p2, p3, p4)


def weight2(m0, m1, m2, m3, m4, m234, m34, M234, G234, M34, G34):
    # weight for 0 - > 1 (234)
    m234min = m2 + m3 + m4
    m234max = m0 - m1
    rhomin = math.atan2(m234min ** 2 - M234 ** 2, M234 * G234)
    rhomax = math.atan2(m234max ** 2 - M234 ** 2, M234 * G234) - rhomin
    wgt = M234 * G234 / rhomax / ((m234 ** 2 - M234 ** 2) ** 2 + G234 ** 2 * M234 ** 2)
    wgt *= m0 * 8.0 * math.pi ** 2 / pStar(m0, m1, m234)
    # weight for (234) -> 2 (34)
    m34min = m3 + m4
    m34max = m234 - m2
    rhomin = math.atan2(m34min ** 2 - M34 ** 2, M34 * G34)
    rhomax = math.atan2(m34max ** 2 - M34 ** 2, M34 * G34) - rhomin
    wgt *= M34 * G34 / rhomax / ((m34 ** 2 - M34 ** 2) ** 2 + G34 ** 2 * M34 ** 2)
    wgt *= m234 * 8.0 * math.pi ** 2 / pStar(m234, m2, m34)
    # weight for (34) -> 3 4
    wgt *= m34 * 8.0 * math.pi ** 2 / pStar(m34, m3, m4)
    return wgt * (2.0 * math.pi) ** 3


def pStar(m0, m1, m2):
    return 0.5 / m0 * math.sqrt((m0 ** 2 - (m1 + m2) ** 2) * (m0 ** 2 - (m1 - m2) ** 2))


def twoBodyDecay(p0, m0, m1, m2):
    ctheta = 2.0 * random.random() - 1.0
    stheta = math.sqrt(1.0 - ctheta ** 2)
    phi = math.pi * 2.0 * random.random()
    pcm = pStar(m0, m1, m2)
    p1 = numpy.array(
        [
            math.sqrt(pcm ** 2 + m1 ** 2),
            pcm * math.cos(phi) * stheta,
            pcm * math.sin(phi) * stheta,
            pcm * ctheta,
        ]
    )
    p2 = numpy.array([math.sqrt(pcm ** 2 + m2 ** 2), -p1[1], -p1[2], -p1[3]])
    bv = numpy.array([p0[1], p0[2], p0[3]]) / p0[0]
    b2 = bv[0] ** 2 + bv[1] ** 2 + bv[2] ** 2
    if b2 == 0.0:
        return (p1, p2)
    gamma = p0[0] / m0
    gamma2 = (gamma - 1.0) / b2
    for p in [p1, p2]:
        bp = bv[0] * p[1] + bv[1] * p[2] + bv[2] * p[3]
        for i in range(0, 3):
            p[i + 1] += gamma2 * bp * bv[i] + gamma * bv[i] * p[0]
        p[0] = gamma * (p[0] + bp)
    return (p1, p2)


"""
Function to implement the form factors
"""


def BW3(Q2, M, G):
    return M ** 2 / complex(
        M ** 2 - Q2,
        -G
        * M ** 2
        * math.sqrt(
            max(0.0, ((Q2 - 4.0 * mpip ** 2) / (M ** 2 - 4.0 * mpip ** 2)) ** 3) / Q2
        ),
    )


def Brho(Q2):
    beta = -0.145
    return 1.0 / (1.0 + beta) * (BW3(Q2, mRho, gRho) + beta * BW3(Q2, mRho1, gRho1))


def Trho(Q2):
    beta1 = 0.08
    beta2 = -0.0075
    return (
        1.0
        / (1.0 + beta1 + beta2)
        * (
            BW3(Q2, mRho, gRho)
            + beta1 * BW3(Q2, mRho1, gRho1)
            + beta2 * BW3(Q2, mRho2, gRho2)
        )
    )


def Bf0(Q2):
    return mf0 ** 2 / complex(
        mf0 ** 2 - Q2,
        -gf0
        * mf0 ** 2
        * math.sqrt(
            max(0.0, ((Q2 - 4.0 * mpip ** 2) / (mf0 ** 2 - 4.0 * mpip ** 2))) / Q2
        ),
    )


def H(s1, s2, s3):
    return BW3(s1, mRho, gRho) + BW3(s2, mRho, gRho) + BW3(s3, mRho, gRho)


#############################################
# might be subject to change in Hazma... : ##
#############################################
# own definition of dot product of four vectors, used in Jomega, Jf0, Ja1, Jrho
def dot(p1, p2):
    return p1[0] * p2[0] - p1[1] * p2[1] - p1[2] * p2[2] - p1[3] * p2[3]


# mass squared of four vectors
def m2(p):
    return p[0] ** 2 - p[1] ** 2 - p[2] ** 2 - p[3] ** 2


#############################################


def BWRR(Q2):
    return BW3(Q2, mRho, gRho) / mRho ** 2 - BW3(Q2, mRho1, gRho1) / mRho1 ** 2


def Grho(p1, p2, p3, p4):
    pre = BWRR(m2(p1 + p3)) * (
        BWRR(m2(p2 + p4)) * dot(p1 + p2 + 3.0 * p3 + p4, p2 - p4) + 2.0
    )
    return pre * p1


"""
Currents for each contribution
"""
# p1,p2,p3,p4 are four vectors, that are numpy arrays with
# numpy.array([p1[0],p1[1],p1[2],p1[3]])


def Jomega(p1, p2, p3, p4):
    """Omega contribution"""
    Q = p1 + p2 + p3 + p4
    s1 = m2(p2 + p3)
    s2 = m2(p2 + p4)
    s3 = m2(p3 + p4)
    pre = (
        2.0
        * Resonance.BreitWignerFW(m2(p2 + p3 + p4), mOmega, gOmega)
        * g_omega_pi_rho
        * g_rho_pi_pi
        * H(s1, s2, s3)
    )
    p2Q = dot(p2, Q)
    p3Q = dot(p3, Q)
    p4Q = dot(p4, Q)
    p1p2 = dot(p1, p2)
    p1p3 = dot(p1, p3)
    p1p4 = dot(p1, p4)
    current = (
        p2 * (p1p4 * p3Q - p1p3 * p4Q)
        + p3 * (p1p2 * p4Q - p1p4 * p2Q)
        + p4 * (p1p3 * p2Q - p1p2 * p3Q)
    )
    return -pre * current


# f0 contribution
def Jf0(Q2, p1, p2, p3, p4):
    m342 = m2(p3 + p4)
    m122 = m2(p1 + p2)
    q = p1 + p2 + p3 + p4
    current = p3 - p4 - q * dot(q, p3 - p4) / Q2
    return -Trho(m342) * Bf0(m122) * current


# a1 contribution
def Ja1(Q2, p1, p2, p3, p4):
    Q = p1 + p2 + p3 + p4
    msq = m2(Q - p1)
    pre = Resonance.BreitWignera1(msq, ma1, ga1) * Brho(m2(p3 + p4))
    current = (
        p3
        - p4
        + p1 * dot(p2, p3 - p4) / msq
        - Q * (dot(Q, p3 - p4) / Q2 + dot(Q, p1) * dot(p2, p3 - p4) / Q2 / msq)
    )
    return pre * current


# rho contribution
def Jrho(Q2, p1, p2, p3, p4):
    Q = p1 + p2 + p3 + p4
    c1 = (
        Grho(p1, p2, p3, p4)
        + Grho(p4, p1, p2, p3)
        - Grho(p1, p2, p4, p3)
        - Grho(p3, p1, p2, p4)
        + Grho(p2, p1, p3, p4)
        + Grho(p4, p2, p1, p3)
        - Grho(p2, p1, p4, p3)
        - Grho(p3, p2, p1, p4)
    )
    d1 = dot(Q, c1) / Q2
    return -(g_rho_pi_pi ** 3) * g_rho_gamma * BWRR(Q2) * (c1 - Q * d1)


# resonant rho contribution of the form factor
def Frho(Q2, beta1, beta2, beta3):
    return (
        1.0
        / (1 + beta1 + beta2 + beta3)
        * (
            BW3(Q2, mRho, gRho)
            + beta1 * BW3(Q2, mBar1, gBar1)
            + beta2 * BW3(Q2, mBar2, gBar2)
            + beta3 * BW3(Q2, mBar3, gBar3)
        )
    )


"""
functions for phase space calculation
"""


def generatepm00(rootS):
    rnd = random.random()
    if rnd < 0.25:
        (p01, p02, pp, pm) = phaseSpace1(
            rootS, mpi0, mpi0, mpip, mpip, mf0, gf0, mRho, gRho
        )
    elif rnd < 0.5:
        rnd = 4.0 * rnd - 1.0
        if rnd < 0.25:
            (pm, p02, p01, pp) = phaseSpace2(
                rootS, mpip, mpi0, mpi0, mpip, ma1, ga1, mRho, gRho
            )
        elif rnd < 0.5:
            (pm, p01, p02, pp) = phaseSpace2(
                rootS, mpip, mpi0, mpi0, mpip, ma1, ga1, mRho, gRho
            )
        elif rnd < 0.75:
            (pp, p02, p01, pm) = phaseSpace2(
                rootS, mpip, mpi0, mpi0, mpip, ma1, ga1, mRho, gRho
            )
        else:
            (pp, p01, p02, pm) = phaseSpace2(
                rootS, mpip, mpi0, mpi0, mpip, ma1, ga1, mRho, gRho
            )
    elif rnd < 0.75:
        rnd = 4.0 * rnd - 2.0
        if rnd < 0.5:
            (p01, pm, p02, pp) = phaseSpace1(
                rootS, mpi0, mpip, mpi0, mpip, mRho, gRho, mRho, gRho
            )
        else:
            (p02, pm, p01, pp) = phaseSpace1(
                rootS, mpi0, mpip, mpi0, mpip, mRho, gRho, mRho, gRho
            )
    else:
        rnd = 4.0 * rnd - 3.0
        if rnd < 1.0 / 6.0:
            (p01, p02, pp, pm) = phaseSpace2(
                rootS, mpi0, mpi0, mpip, mpip, mOmega, gOmega, mRho, gRho
            )
        elif rnd < 1.0 / 3.0:
            (p02, p01, pp, pm) = phaseSpace2(
                rootS, mpi0, mpi0, mpip, mpip, mOmega, gOmega, mRho, gRho
            )
        elif rnd < 0.5:
            (p01, pp, p02, pm) = phaseSpace2(
                rootS, mpi0, mpip, mpi0, mpip, mOmega, gOmega, mRho, gRho
            )
        elif rnd < 2.0 / 3.0:
            (p02, pp, p01, pm) = phaseSpace2(
                rootS, mpi0, mpip, mpi0, mpip, mOmega, gOmega, mRho, gRho
            )
        elif rnd < 5.0 / 6.0:
            (p01, pm, p02, pp) = phaseSpace2(
                rootS, mpi0, mpip, mpi0, mpip, mOmega, gOmega, mRho, gRho
            )
        else:
            (p02, pm, p01, pp) = phaseSpace2(
                rootS, mpi0, mpip, mpi0, mpip, mOmega, gOmega, mRho, gRho
            )

    wgt = (
        1.0
        / 4.0
        * weight1(
            rootS,
            mpi0,
            mpi0,
            mpip,
            mpip,
            math.sqrt(m2(p01 + p02)),
            math.sqrt(m2(pp + pm)),
            mf0,
            gf0,
            mRho,
            gRho,
        )
        + 1.0
        / 4.0
        / 6.0
        * (
            weight2(
                rootS,
                mpi0,
                mpi0,
                mpip,
                mpip,
                math.sqrt(m2(p02 + pp + pm)),
                math.sqrt(m2(pp + pm)),
                mOmega,
                gOmega,
                mRho,
                gRho,
            )
            + weight2(
                rootS,
                mpi0,
                mpi0,
                mpip,
                mpip,
                math.sqrt(m2(p01 + pp + pm)),
                math.sqrt(m2(pp + pm)),
                mOmega,
                gOmega,
                mRho,
                gRho,
            )
            + weight2(
                rootS,
                mpi0,
                mpip,
                mpi0,
                mpip,
                math.sqrt(m2(p02 + pp + pm)),
                math.sqrt(m2(p02 + pm)),
                mOmega,
                gOmega,
                mRho,
                gRho,
            )
            + weight2(
                rootS,
                mpi0,
                mpip,
                mpi0,
                mpip,
                math.sqrt(m2(p01 + pp + pm)),
                math.sqrt(m2(p01 + pm)),
                mOmega,
                gOmega,
                mRho,
                gRho,
            )
            + weight2(
                rootS,
                mpi0,
                mpip,
                mpi0,
                mpip,
                math.sqrt(m2(p02 + pp + pm)),
                math.sqrt(m2(p02 + pp)),
                mOmega,
                gOmega,
                mRho,
                gRho,
            )
            + weight2(
                rootS,
                mpi0,
                mpip,
                mpi0,
                mpip,
                math.sqrt(m2(p01 + pp + pm)),
                math.sqrt(m2(p01 + pp)),
                mOmega,
                gOmega,
                mRho,
                gRho,
            )
        )
        + 1.0
        / 4.0
        / 4.0
        * (
            weight2(
                rootS,
                mpip,
                mpi0,
                mpi0,
                mpip,
                math.sqrt(m2(p02 + p01 + pp)),
                math.sqrt(m2(p01 + pp)),
                ma1,
                ga1,
                mRho,
                gRho,
            )
            + weight2(
                rootS,
                mpip,
                mpi0,
                mpi0,
                mpip,
                math.sqrt(m2(p01 + p02 + pp)),
                math.sqrt(m2(p02 + pp)),
                ma1,
                ga1,
                mRho,
                gRho,
            )
            + weight2(
                rootS,
                mpip,
                mpi0,
                mpi0,
                mpip,
                math.sqrt(m2(p02 + p01 + pm)),
                math.sqrt(m2(p01 + pm)),
                ma1,
                ga1,
                mRho,
                gRho,
            )
            + weight2(
                rootS,
                mpip,
                mpi0,
                mpi0,
                mpip,
                math.sqrt(m2(p01 + p02 + pm)),
                math.sqrt(m2(p02 + pm)),
                ma1,
                ga1,
                mRho,
                gRho,
            )
        )
        + 1.0
        / 4.0
        / 2.0
        * (
            weight1(
                rootS,
                mpi0,
                mpip,
                mpi0,
                mpip,
                math.sqrt(m2(p01 + pm)),
                math.sqrt(m2(p02 + pp)),
                mRho,
                gRho,
                mRho,
                gRho,
            )
            + weight1(
                rootS,
                mpi0,
                mpip,
                mpi0,
                mpip,
                math.sqrt(m2(p02 + pm)),
                math.sqrt(m2(p01 + pp)),
                mRho,
                gRho,
                mRho,
                gRho,
            )
        )
    )
    return (1.0 / wgt, p01, p02, pp, pm)


def generatepmpm(rootS):
    rnd = random.random()
    if rnd < 0.5:
        rnd *= 2.0
        if rnd < 0.25:
            (pp1, pm1, pp2, pm2) = phaseSpace1(
                rootS, mpip, mpip, mpip, mpip, mf0, gf0, mRho, gRho
            )
        elif rnd < 0.5:
            (pp1, pm2, pp2, pm1) = phaseSpace1(
                rootS, mpip, mpip, mpip, mpip, mf0, gf0, mRho, gRho
            )
        elif rnd < 0.75:
            (pp2, pm1, pp1, pm2) = phaseSpace1(
                rootS, mpip, mpip, mpip, mpip, mf0, gf0, mRho, gRho
            )
        else:
            (pp2, pm2, pp1, pm1) = phaseSpace1(
                rootS, mpip, mpip, mpip, mpip, mf0, gf0, mRho, gRho
            )
    else:
        rnd = 2.0 * rnd - 1.0
        if rnd < 0.125:
            (pp1, pm1, pp2, pm2) = phaseSpace2(
                rootS, mpip, mpip, mpip, mpip, ma1, ga1, mRho, gRho
            )
        elif rnd < 0.25:
            (pp1, pm2, pp2, pm1) = phaseSpace2(
                rootS, mpip, mpip, mpip, mpip, ma1, ga1, mRho, gRho
            )
        elif rnd < 0.375:
            (pp2, pm1, pp1, pm2) = phaseSpace2(
                rootS, mpip, mpip, mpip, mpip, ma1, ga1, mRho, gRho
            )
        elif rnd < 0.5:
            (pp2, pm2, pp1, pm1) = phaseSpace2(
                rootS, mpip, mpip, mpip, mpip, ma1, ga1, mRho, gRho
            )
        elif rnd < 0.625:
            (pm1, pp1, pm2, pp2) = phaseSpace2(
                rootS, mpip, mpip, mpip, mpip, ma1, ga1, mRho, gRho
            )
        elif rnd < 0.75:
            (pm1, pp2, pm2, pp1) = phaseSpace2(
                rootS, mpip, mpip, mpip, mpip, ma1, ga1, mRho, gRho
            )
        elif rnd < 0.875:
            (pm2, pp1, pm1, pp2) = phaseSpace2(
                rootS, mpip, mpip, mpip, mpip, ma1, ga1, mRho, gRho
            )
        else:
            (pm2, pp2, pm1, pp1) = phaseSpace2(
                rootS, mpip, mpip, mpip, mpip, ma1, ga1, mRho, gRho
            )
    wgt = 0.125 * (
        weight1(
            rootS,
            mpip,
            mpip,
            mpip,
            mpip,
            math.sqrt(m2(pp2 + pm2)),
            math.sqrt(m2(pp1 + pm1)),
            mf0,
            gf0,
            mRho,
            gRho,
        )
        + weight1(
            rootS,
            mpip,
            mpip,
            mpip,
            mpip,
            math.sqrt(m2(pp2 + pm1)),
            math.sqrt(m2(pp1 + pm2)),
            mf0,
            gf0,
            mRho,
            gRho,
        )
        + weight1(
            rootS,
            mpip,
            mpip,
            mpip,
            mpip,
            math.sqrt(m2(pp1 + pm2)),
            math.sqrt(m2(pp2 + pm1)),
            mf0,
            gf0,
            mRho,
            gRho,
        )
        + weight1(
            rootS,
            mpip,
            mpip,
            mpip,
            mpip,
            math.sqrt(m2(pp1 + pm1)),
            math.sqrt(m2(pp2 + pm2)),
            mf0,
            gf0,
            mRho,
            gRho,
        )
    ) + 0.0625 * (
        weight2(
            rootS,
            mpip,
            mpip,
            mpip,
            mpip,
            math.sqrt(m2(pp2 + pp1 + pm1)),
            math.sqrt(m2(pp1 + pm1)),
            ma1,
            ga1,
            mRho,
            gRho,
        )
        + weight2(
            rootS,
            mpip,
            mpip,
            mpip,
            mpip,
            math.sqrt(m2(pp2 + pp1 + pm2)),
            math.sqrt(m2(pp1 + pm2)),
            ma1,
            ga1,
            mRho,
            gRho,
        )
        + weight2(
            rootS,
            mpip,
            mpip,
            mpip,
            mpip,
            math.sqrt(m2(pp2 + pp1 + pm1)),
            math.sqrt(m2(pp2 + pm1)),
            ma1,
            ga1,
            mRho,
            gRho,
        )
        + weight2(
            rootS,
            mpip,
            mpip,
            mpip,
            mpip,
            math.sqrt(m2(pp2 + pp1 + pm2)),
            math.sqrt(m2(pp2 + pm2)),
            ma1,
            ga1,
            mRho,
            gRho,
        )
        + weight2(
            rootS,
            mpip,
            mpip,
            mpip,
            mpip,
            math.sqrt(m2(pm2 + pm1 + pp1)),
            math.sqrt(m2(pm1 + pp1)),
            ma1,
            ga1,
            mRho,
            gRho,
        )
        + weight2(
            rootS,
            mpip,
            mpip,
            mpip,
            mpip,
            math.sqrt(m2(pm2 + pm1 + pp2)),
            math.sqrt(m2(pm1 + pp2)),
            ma1,
            ga1,
            mRho,
            gRho,
        )
        + weight2(
            rootS,
            mpip,
            mpip,
            mpip,
            mpip,
            math.sqrt(m2(pm2 + pm1 + pp1)),
            math.sqrt(m2(pm2 + pp1)),
            ma1,
            ga1,
            mRho,
            gRho,
        )
        + weight2(
            rootS,
            mpip,
            mpip,
            mpip,
            mpip,
            math.sqrt(m2(pm2 + pm1 + pp2)),
            math.sqrt(m2(pm2 + pp2)),
            ma1,
            ga1,
            mRho,
            gRho,
        )
    )
    return (1.0 / wgt, pp1, pp2, pm1, pm2)


def contributions(Q2):
    # coefficients for the cross section
    coeffs = numpy.zeros((4, 4), dtype=complex)
    for i in range(0, 4):
        f1 = 0.0
        if i == 0:
            f1 = c_a1 * Frho(Q2, beta1_a1, beta2_a1, beta3_a1)
        elif i == 1:
            f1 = c_omega * Frho(Q2, beta1_omega, beta2_omega, beta3_omega)
        elif i == 2:
            f1 = c_f0 * Frho(Q2, beta1_f0, beta2_f0, beta3_f0)
        elif i == 3:
            f1 = c_rho
        for j in range(0, 4):
            f2 = 0.0
            if j == 0:
                f2 = c_a1 * Frho(Q2, beta1_a1, beta2_a1, beta3_a1)
            elif j == 1:
                f2 = c_omega * Frho(Q2, beta1_omega, beta2_omega, beta3_omega)
            elif j == 2:
                f2 = c_f0 * Frho(Q2, beta1_f0, beta2_f0, beta3_f0)
            elif j == 3:
                f2 = c_rho
            coeffs[i, j] = f1 * numpy.conj(f2)
    return coeffs


"""
Full Form factor
"""


def F4pi(Q2, p1, p2, p3, p4):
    """
    depends on the c.m. energy and all momenta
    4-body phase space integral still has to be performed
    """
    coeffs = numpy.zeros((4, 4), dtype=complex)
    for i in range(0, 4):
        f1 = 0.0
        if i == 0:
            f1 = c_a1 * Frho(Q2, beta1_a1, beta2_a1, beta3_a1) * Ja1(Q2, p1, p2, p3, p4)
        elif i == 1:
            f1 = (
                c_omega
                * Frho(Q2, beta1_omega, beta2_omega, beta3_omega)
                * Jomega(Q2, p1, p2, p3, p4)
            )
        elif i == 2:
            f1 = c_f0 * Frho(Q2, beta1_f0, beta2_f0, beta3_f0) * Jf0(Q2, p1, p2, p3, p4)
        elif i == 3:
            f1 = c_rho * Jrho(Q2, p1, p2, p3, p4)
        for j in range(0, 4):
            f2 = 0.0
            if j == 0:
                f2 = (
                    c_a1
                    * Frho(Q2, beta1_a1, beta2_a1, beta3_a1)
                    * Ja1(Q2, p1, p2, p3, p4)
                )
            elif j == 1:
                f2 = (
                    c_omega
                    * Frho(Q2, beta1_omega, beta2_omega, beta3_omega)
                    * Jomega(Q2, p1, p2, p3, p4)
                )
            elif j == 2:
                f2 = (
                    c_f0
                    * Frho(Q2, beta1_f0, beta2_f0, beta3_f0)
                    * Jf0(Q2, p1, p2, p3, p4)
                )
            elif j == 3:
                f2 = c_rho * Jrho(Q2, p1, p2, p3, p4)
            coeffs[i, j] = f1 * numpy.conj(f2)
    return coeffs


# phase space + form factors, the integration of the form factor over the phase space has already been performed
# phase space values are read in by wgt, wgt2....
def hadronic_current(Q2, npoints, wgt, wgt2, omegaOnly=False):
    # contributions from several subsequent processes
    coeffs = contributions(Q2)
    # compute the cross section and error
    total = complex(0.0, 0.0)
    toterr = complex(0.0, 0.0)
    if not omegaOnly:
        for i1 in range(0, 4):
            for i2 in range(0, 4):
                total += cI1_ ** 2 * wgt[i1, i2] * coeffs[i1, i2]
                for j1 in range(0, 4):
                    for j2 in range(0, 4):
                        toterr += (
                            cI1_ ** 4
                            * coeffs[i1, i2]
                            * coeffs[j1, j2]
                            * wgt2[i1, i2, j1, j2]
                        )
    else:
        total += cI1_ ** 2 * wgt[1, 1] * coeffs[1, 1]
        toterr += cI1_ ** 4 * coeffs[1, 1] * coeffs[1, 1] * wgt2[1, 1, 1, 1]
    toterr = math.sqrt((toterr.real - total.real ** 2) / npoints)
    return total, toterr


# full integrated hadronic current
# 2pi+2pi- can be called with hadronic_interpolator_c
# pi+pi-2pi) can be called with hadronic_interpolator_n
def readHadronic_Current():
    global hadronic_interpolator_n
    global hadronic_interpolator_c
    readCoefficients()
    # neutral: pi+pi-2pi0
    x = []
    y = []
    for (key, val) in sorted(coeffs_neutral.items()):
        en = key
        s = en ** 2
        x.append(en)
        (npoints, wgt, wgt2) = val
        hadcurr, hadcurr_err = hadronic_current(s, npoints, wgt, wgt2, omegaOnly=False)
        y.append(abs(hadcurr))
    hadronic_interpolator_n = interp1d(x, y, kind="cubic", fill_value=(0.0, 0.0))
    # charged: 2pi+2pi-
    x = []
    y = []
    for (key, val) in sorted(coeffs_charged.items()):
        en = key
        s = en ** 2
        x.append(en)
        (npoints, wgt, wgt2) = val
        hadcurr, hadcurr_err = hadronic_current(s, npoints, wgt, wgt2, omegaOnly=False)
        y.append(abs(hadcurr))
    hadronic_interpolator_c = interp1d(x, y, kind="cubic", fill_value=(0.0, 0.0))


###############################################################
#### Functions for mediator decay rate and cross-sections #####
###############################################################
# neutral: pi+pi-2pi0
# charged: 2pi+2pi-


def GammaDM(mMed, mode):
    if cI1_ == 0:
        return 0
    if mode == "neutral":
        if mMed < (0.85):
            return 0
    if mode == "charged":
        if mMed < (0.6125):
            return 0
    # m4Pi_ = 4*mpip
    pre = 1.0 / 3
    # phase space prefactor, see e.g.
    pre *= (2.0 * math.pi) ** 4 / 2.0 / mMed
    if mode == "neutral":
        hadcurr = hadronic_interpolator_n(mMed)
    if mode == "charged":
        hadcurr = hadronic_interpolator_c(mMed)
    # print (pre, hadcurr)
    return pre * hadcurr


def sigmaDM(s, mode):
    if cI1_ == 0:
        return 0
    if mode == "neutral":
        if s < (0.85) ** 2:
            return 0
        m4Pi_ = 2 * mpip + 2 * mpi0
    if mode == "charged":
        if s < (0.6125) ** 2:
            return 0
        m4Pi_ = 4 * mpip
    if s < (m4Pi_) ** 2:
        return 0
    # leptonic part contracted
    sqrts = math.sqrt(s)
    cDM = gDM_
    DMmed = cDM / (s - mMed_ ** 2 + complex(0.0, 1.0) * mMed_ * wMed_)
    DMmed2 = abs(DMmed) ** 2
    pre = DMmed2 * s * (1 + 2 * mDM_ ** 2 / s) / 3.0
    # prefactor of phase space
    pre *= (2.0 * math.pi) ** 4 / 2.0 / s * Resonance.gev2nb
    if mode == "neutral":
        hadcurr = hadronic_interpolator_n(sqrts)
    if mode == "charged":
        hadcurr = hadronic_interpolator_c(sqrts)
    return pre * hadcurr


def sigmaSM(s, mode):
    if mode == "neutral":
        if s < (0.85) ** 2:
            return 0.0
        m4Pi_ = 2 * mpip + 2 * mpi0
    if mode == "charged":
        if s < (0.6125) ** 2:
            return 0.0
        m4Pi_ = 4 * mpip
    if s < (m4Pi_) ** 2:
        return 0.0
    # leptonic part contracted
    sqrts = math.sqrt(s)
    pre = 16.0 * math.pi ** 2 * alpha.alphaEM(s) ** 2 / 3.0 / s
    # prefactor of phase space
    pre *= (2.0 * math.pi) ** 4 / 2.0 / s * Resonance.gev2nb
    if mode == "neutral":
        hadcurr = hadronic_interpolator_n(sqrts)
    if mode == "charged":
        hadcurr = hadronic_interpolator_c(sqrts)
    return pre * hadcurr


"""
Useful functions for precalculated phase space values
"""


def readCoefficients():
    global coeffs_neutral, coeffs_charged
    if len(coeffs_neutral) != 0:
        return
    omega = {}
    for fname in glob.glob(
        os.path.dirname(os.path.abspath(__file__)) + "/4pi/*neutral*.dat"
    ):
        output = readPoint(fname)
        coeffs_neutral[output[0]] = output[1]
        omega[output[0]] = output[1][1][1, 1]
    for fname in glob.glob(
        os.path.dirname(os.path.abspath(__file__)) + "/4pi/*charged*.dat"
    ):
        output = readPoint(fname)
        coeffs_charged[output[0]] = output[1]
    x = []
    y = []
    for val in sorted(omega.keys()):
        x.append(val)
        y.append(omega[val].real)
    global omega_interpolator
    omega_interpolator = interp1d(x, y, kind="cubic", fill_value="extrapolate")


def readPoint(fname):
    file = open(fname)
    line = file.readline().strip().split()
    energy = float(line[0])
    npoints = int(line[1])
    line = file.readline().strip()
    ix = 0
    iy = 0
    wgtsum = numpy.zeros((4, 4), dtype=complex)
    while len(line) != 0:
        if line[0] == "(":
            index = line.find(")") + 1
            wgtsum[ix][iy] = complex(line[0:index])
            iy += 1
            line = line[index:]
        elif line[0:2] == "0j":
            wgtsum[ix][iy] = 0.0
            iy += 1
            line = line[2:]
        else:
            print("fails", line)
            quit()

        if iy == 4:
            iy = 0
            ix += 1
    line = file.readline().strip()
    ix1 = 0
    iy1 = 0
    ix2 = 0
    iy2 = 0
    wgt2sum = numpy.zeros((4, 4, 4, 4), dtype=complex)
    while len(line) != 0:
        if line[0] == "(":
            index = line.find(")") + 1
            wgt2sum[ix1][iy1][ix2][iy2] = complex(line[0:index])
            iy2 += 1
            line = line[index:]
        elif line[0:2] == "0j":
            wgt2sum[ix1][iy1][ix2][iy2] = 0.0
            iy2 += 1
            line = line[2:]
        else:
            print("fails", line)
            quit()
        if iy2 == 4:
            iy2 = 0
            ix2 += 1
            if ix2 == 4:
                ix2 = 0
                iy1 += 1
                if iy1 == 4:
                    iy1 = 0
                    ix1 += 1
    file.close()
    return (energy, [npoints, wgtsum, wgt2sum])
