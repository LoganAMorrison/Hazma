# Libraries to load
import math

from . import Resonance
from . import alpha


# parametrization losely based on https://arxiv.org/pdf/1711.00820.pdf

# fpi_= 0.092663579634360449
fpi_ = 0.09266
hV_ = 0.007594981126020603
Aom_ = 0.8846540224221084
Aphi_ = -0.06460651106718258
mRho_ = 0.77526
wRho_ = 0.1491
mOmega_ = 0.78265
wOmega_ = 0.00849
mPhi_ = 1.01946
wPhi_ = 0.004247

mPi_ = 0.1349770


ii = complex(0.0, 1.0)

# Parameter set for DM part
gDM_ = 1.0
mDM_ = 0.41
mMed_ = 5
wMed_ = 10.0
cI1_ = 1.0
cI0_ = 1.0
cS_ = 1.0
cD_ = 1.0


def resetParameters(gDM, mDM, mMed, wMed, cMedu, cMedd, cMeds):
    """
    change rho, omega, phi contributions
    """
    global cI1_, cI0_, cS_, cD_
    global cRhoOmPhi_
    global gDM_, mDM_, mMed_, wMed_
    gDM_ = gDM
    mDM_ = mDM
    mMed_ = mMed
    wMed_ = wMed
    cI1_ = cMedu - cMedd
    cI0_ = 3 * (cMedu + cMedd)
    cS_ = -3 * cMeds
    cD_ = 2 * cMedu + cMedd  # direct photon-photon-pion -> boson-photon-pion coupling


# Form factor of PiGamma
def FPiGamma(Q2):
    Q = math.sqrt(Q2)
    form = -1.0 / 4.0 / math.pi ** 2 / fpi_ * cD_
    Drho = cI1_ / (Q2 - mRho_ ** 2 + ii * Q * wRho_)
    Domega = cI0_ / (Q2 - mOmega_ ** 2 + ii * Q * wOmega_)
    Dphi = cS_ / (Q2 - mPhi_ ** 2 + ii * Q * wPhi_)
    form_i = Drho + Aom_ * Domega + Aphi_ * Dphi
    form_i *= 4.0 * math.sqrt(2) * hV_ * Q2 / 3.0 / fpi_
    form += form_i
    return math.sqrt(4.0 * math.pi * alpha.alphaEM(Q2)) * form


# Decay rate of the dark mediator into PiGamma
def GammaDM(mMed):
    Q2 = mMed ** 2
    if mMed > mPi_:
        pcm = 0.5 * (Q2 - mPi_ ** 2) / mMed
    else:
        return 0.0
    return 1.0 / 12.0 / math.pi * pcm ** 3 * abs(FPiGamma(Q2)) ** 2 * Resonance.gev2nb


# cross section for Pi Gamma
def sigmaSMPiGamma(Q2):
    Q = math.sqrt(Q2)
    if Q > mPi_:
        pcm = 0.5 * (Q2 - mPi_ ** 2) / Q
    else:
        return 0.0
    return (
        4.0
        * math.pi
        * alpha.alphaEM(Q2) ** 2
        * pcm ** 3
        / 3.0
        / Q
        / Q2
        * abs(FPiGamma(Q2)) ** 2
        * Resonance.gev2nb
    )


# cross section for Pi Gamma
def sigmaDMPiGamma(Q2):
    Q = math.sqrt(Q2)
    if Q > mPi_:
        pcm = 0.5 * (Q2 - mPi_ ** 2) / Q
    else:
        return 0.0
    cDM = gDM_
    DMmed = cDM / (Q2 - mMed_ ** 2 + complex(0.0, 1.0) * mMed_ * wMed_)
    DMmed2 = abs(DMmed) ** 2
    temp = FPiGamma(Q2)
    return (
        1.0
        / 12
        / math.pi
        * DMmed2
        * (1 + 2 * mDM_ ** 2 / Q2)
        * Q
        * pcm ** 3
        * abs(temp) ** 2
        * Resonance.gev2nb
    )
