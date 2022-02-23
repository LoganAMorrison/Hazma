# Libraries to load
import math
import cmath


from . import Resonance
from . import alpha


# parametrization based on hep-ex/0605109

# set of parameters
# rho0, omega0,phi0,rho1,phi1
ResMasses_ = [0.77526, 0.78284, 1.01952, 1.465, 1.70]
ResWidths_ = [0.1491, 0.00868, 0.00421, 0.40, 0.30]
Amp_ = [0.0861, 0.00824, 0.0158, 0.0147, 0.0]
Phase_ = [0.0, 11.3, 170.0, 61.0, 0.0]
mEta_ = 0.547862
mPi_ = 0.13957061

# coupling modification depending on mediator quark couplings
cRho_ = 1.0
cOmega_ = 1.0
cPhi_ = 1.0
cRhoOmPhi_ = [cRho_, cOmega_, cPhi_, cRho_, cPhi_]

ii = complex(0.0, 1.0)

# Parameter set for DM part
gDM_ = 1.0
mDM_ = 0.41
mMed_ = 5
wMed_ = 10.0
cI1_ = 1.0
cI0_ = 1.0
cS_ = 1.0


def resetParameters(gDM, mDM, mMed, wMed, cMedu, cMedd, cMeds):
    """
    change rho, omega, phi contributions
    """
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
    cRhoOmPhi_ = [cI1_, cI0_, cS_, cI1_, cS_]


def Widths(Q2, ix):
    """
    width of the intermediate vector mesons
    """
    Q = math.sqrt(Q2)
    if ix == 0:
        pcm = 0.5 * (Q2 - mEta_ ** 2) / Q
        resWidths = (
            ResWidths_[0]
            * ResMasses_[0] ** 2
            / Q2
            * ((Q2 - 4.0 * mPi_ ** 2) / (ResMasses_[0] ** 2 - 4.0 * mPi_ ** 2)) ** 1.5
        )
    else:
        resWidths = ResWidths_[ix]
    return resWidths


# Form factor for Eta Gamma
def FEtaGamma(Q2):
    Q = math.sqrt(Q2)
    form = 0.0
    for i in range(0, len(ResMasses_)):
        Di = ResMasses_[i] ** 2 - Q2 - ii * Q * Widths(Q2, i)
        form += (
            cRhoOmPhi_[i]
            * Amp_[i]
            * ResMasses_[i] ** 2
            * cmath.exp(ii * math.radians(Phase_[i]))
            / Di
        )
    return form


# Decay rate of the dark mediator into EtaGamma
def GammaDM(mMed):
    Q2 = mMed ** 2
    if mMed > mEta_:
        pcm = 0.5 * (Q2 - mEta_ ** 2) / mMed
    else:
        return 0.0
    return 1.0 / 12.0 / math.pi * pcm ** 3 * abs(FEtaGamma(Q2)) ** 2 * Resonance.gev2nb


# SM cross section for Eta Gamma
def sigmaSMEtaGamma(Q2):
    Q = math.sqrt(Q2)
    if Q > mEta_:
        pcm = 0.5 * (Q2 - mEta_ ** 2) / Q
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
        * abs(FEtaGamma(Q2)) ** 2
        * Resonance.gev2nb
    )


# DM cross section for Eta Gamma
def sigmaDMEtaGamma(Q2):
    Q = math.sqrt(Q2)
    if Q > mEta_:
        pcm = 0.5 * (Q2 - mEta_ ** 2) / Q
    else:
        return 0.0
    cDM = gDM_
    DMmed = cDM / (Q2 - mMed_ ** 2 + complex(0.0, 1.0) * mMed_ * wMed_)
    DMmed2 = abs(DMmed) ** 2
    temp = FEtaGamma(Q2)
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
