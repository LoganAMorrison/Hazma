# Libraries to load
import cmath, math
from . import Resonance, alpha

# own parametrization, see arXiv:1911.11147

# amplitudes etc
amp = [0.045, 0.0315, 0.0]
phase = [180.0, 0.0, 180.0]
br4pi = [0.0, 0.33, 0.0]
# rho masses and widths
rhoMasses = [0.77526, 1.593, 1.909]
rhoWidths = [0.1491, 0.203, 0.048]
mPhi = 1.019461
mpi = 0.1349766
wgts = [
    amp[0] * cmath.exp(complex(0.0, phase[0] / 180.0 * math.pi)),
    amp[1] * cmath.exp(complex(0.0, phase[1] / 180.0 * math.pi)),
    amp[2] * cmath.exp(complex(0.0, phase[2] / 180.0 * math.pi)),
]

# coupling modification depending on mediator quark couplings
# Parameter set for DM part
gDM_ = 1.0
mDM_ = 0.41
mMed_ = 5
wMed_ = 10.0
cI1_ = 1.0
cI0_ = 1.0
cS_ = 1.0

# change rho, omega, phi contributions
def resetParameters(gDM, mDM, mMed, wMed, cMedu, cMedd, cMeds):
    global cI1_, cI0_, cS_
    global gDM_, mDM_, mMed_, wMed_
    global wgts
    gDM_ = gDM
    mDM_ = mDM
    mMed_ = mMed
    wMed_ = wMed
    cI1_ = cMedu - cMedd
    cI0_ = 3 * (cMedu + cMedd)
    cS_ = -3 * cMeds
    wgts = [weight * cI1_ for weight in wgts]


def GammaDM(M):
    sHat = M ** 2
    if sHat < (mpi + mPhi) ** 2 or cI1_ == 0:
        return 0
    ecms = math.sqrt(sHat)
    form = complex(0.0, 0.0)  # dimensions 1/E
    for ix in range(0, len(amp)):
        mR2 = rhoMasses[ix] ** 2
        wid = rhoWidths[ix] * (
            1.0
            - br4pi[ix]
            + br4pi[ix]
            * mR2
            / sHat
            * ((sHat - 16.0 * mpi ** 2) / (mR2 - 16.0 * mpi ** 2) ** 1.5)
        )
        form += wgts[ix] * mR2 / (mR2 - sHat - complex(0.0, 1.0) * ecms * wid)
    pcm = 0.5 * ecms * Resonance.beta(sHat, mPhi, mpi)
    output = 1 / 12.0 / math.pi * pcm ** 3 * abs(form) ** 2
    # return output * Resonance.gev2nb
    return output


def sigmaSMPhiPi(sHat):
    if sHat < (mpi + mPhi) ** 2 or cI1_ == 0:
        return 0
    ecms = math.sqrt(sHat)
    pre = complex(0.0, 0.0)  # dimensions 1/E
    for ix in range(0, len(amp)):
        mR2 = rhoMasses[ix] ** 2
        wid = rhoWidths[ix] * (
            1.0
            - br4pi[ix]
            + br4pi[ix]
            * mR2
            / sHat
            * ((sHat - 16.0 * mpi ** 2) / (mR2 - 16.0 * mpi ** 2) ** 1.5)
        )
        pre += wgts[ix] * mR2 / (mR2 - sHat - complex(0.0, 1.0) * ecms * wid)
    pcm = 0.5 * ecms * Resonance.beta(sHat, mPhi, mpi)
    # phase-space, |me|^2 factors
    output = 2.0 * pcm ** 3 / 8.0 / math.pi / ecms * abs(pre) ** 2 / 3
    # initial-state factors
    output *= 16.0 * math.pi ** 2 / sHat * alpha.alphaEM(sHat) ** 2
    return output * Resonance.gev2nb


# DM cross section for Eta Gamma
def sigmaDMPhiPi(sHat):
    if sHat < (mpi + mPhi) ** 2 or cI1_ == 0:
        return 0
    ecms = math.sqrt(sHat)
    form = complex(0.0, 0.0)  # dimensions 1/E
    for ix in range(0, len(amp)):
        mR2 = rhoMasses[ix] ** 2
        wid = rhoWidths[ix] * (
            1.0
            - br4pi[ix]
            + br4pi[ix]
            * mR2
            / sHat
            * ((sHat - 16.0 * mpi ** 2) / (mR2 - 16.0 * mpi ** 2) ** 1.5)
        )
        form += wgts[ix] * mR2 / (mR2 - sHat - complex(0.0, 1.0) * ecms * wid)
    pcm = 0.5 * ecms * Resonance.beta(sHat, mPhi, mpi)
    cDM = gDM_
    DMmed = cDM / (sHat - mMed_ ** 2 + complex(0.0, 1.0) * mMed_ * wMed_)
    DMmed2 = abs(DMmed) ** 2
    return (
        DMmed2
        / 12.0
        / math.pi
        * ecms
        * (1 + 2 * mDM_ ** 2 / sHat)
        * pcm ** 3
        * abs(form) ** 2
        * Resonance.gev2nb
    )
