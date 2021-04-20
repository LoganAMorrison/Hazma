# Libraries to load
import cmath
import math

from . import Resonance, alpha

# parametrization based on arXiv: 1601.08061

# Set of parameters obtained from the fit
ResMasses_ = [0.77526, 0.78284, 1.01952, 1.45, 1.70]
ResWidths_ = [0.1491, 0.00868, 0.00421, 0.40, 0.30]
Amp_ = [0.0426, 0.0434, 0.00303, 0.00523, 0.]
Phase_ = [-12.7, 0., 158., 180., 0.]
mPi_ = 0.13957061

cRho_ = 1.
cOmega_ = 1.
cPhi_ = 1.
cRhoOmPhi_ = [cRho_, cOmega_, cPhi_, cOmega_, cOmega_]

ii = complex(0., 1.)

# Parameter set for DM part
gDM_ = 1.
mDM_ = 0.41
mMed_ = 5
wMed_ = 10.
cI1_ = 1.
cI0_ = 1.
cS_ = 1.

# change rho, omega, phi contributions


def resetParameters(gDM, mDM, mMed, wMed, cMedu, cMedd, cMeds):
    global cI1_, cI0_, cS_
    global cRhoOmPhi_
    global gDM_, mDM_, mMed_, wMed_
    gDM_ = gDM
    mDM_ = mDM
    mMed_ = mMed
    wMed_ = wMed
    cI1_ = cMedu-cMedd
    cI0_ = 3*(cMedu+cMedd)
    cS_ = -3*cMeds
    cRhoOmPhi_ = [cI1_, cI0_, cS_, cI0_, cI0_]

# Width of the intermediate vector meson


def Widths(Q2, ix):
    Q = math.sqrt(Q2)
    if ix == 0:
        pcm = 0.5*(Q2-mPi_**2)/Q
        resWidths = ResWidths_[0]*ResMasses_[0]**2/Q2 * \
            ((Q2-4.*mPi_**2)/(ResMasses_[0]**2-4.*mPi_**2))**1.5
    else:
        resWidths = ResWidths_[ix]
    return resWidths

# Form factor of PiGamma


def FPiGamma(Q2):
    Q = math.sqrt(Q2)
    form = 0.
    for i in range(0, len(ResMasses_)):
        Di = ResMasses_[i]**2-Q2-ii*Q*Widths(Q2, i)
        form += cRhoOmPhi_[i]*Amp_[i]*ResMasses_[i]**2 * \
            cmath.exp(ii*math.radians(Phase_[i]))/Di
    return form

# Decay rate of the dark mediator into PiGamma


def GammaDM(mMed):
    Q2 = mMed**2
    if(mMed > 2*mPi_):
        pcm = 0.5*(Q2-mPi_**2)/mMed
    else:
        return 0.
    return 1./12./math.pi*pcm**3*abs(FPiGamma(Q2))**2*Resonance.gev2nb

# cross section for Pi Gamma


def sigmaSMPiGamma(Q2):
    Q = math.sqrt(Q2)
    if(Q > 2*mPi_):
        pcm = 0.5*(Q2-mPi_**2)/Q
    else:
        return 0.
    return 4.*math.pi*alpha.alphaEM(Q2)**2*pcm**3/3./Q/Q2*abs(FPiGamma(Q2))**2*Resonance.gev2nb

# cross section for Pi Gamma


def sigmaDMPiGamma(Q2):
    Q = math.sqrt(Q2)
    if(Q > 2*mPi_):
        pcm = 0.5*(Q2-mPi_**2)/Q
    else:
        return 0.
    cDM = gDM_
    DMmed = cDM/(Q2-mMed_**2+complex(0., 1.)*mMed_*wMed_)
    DMmed2 = abs(DMmed)**2
    temp = FPiGamma(Q2)
    return 1./12/math.pi*DMmed2*(1+2*mDM_**2/Q2)*Q*pcm**3*abs(temp)**2*Resonance.gev2nb
