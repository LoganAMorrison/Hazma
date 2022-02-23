# Load libraries that we need
import math

import numpy as np
import scipy
import scipy.integrate
import scipy.special

from . import alpha
from .Resonance import Hhat, dHhatds, H, BreitWignerGS, BreitWignerFW, gev2nb
from .masses import mpi

"""
Parameter set for hadronic current, parametrization taken from arXiv:1002.0279
"""

# truncation parameter
nMax_ = 2000
# omega parameters, relevant for the rho-omega mixing in the 0th order rho
# resonance
omegaMag_ = 0.00187
omegaPhase_ = 0.106
omegaMass_ = 0.7824
omegaWidth_ = 0.00833
omegaWgt_ = 0.0
beta_ = 2.148
# rho parameters, for 0th to 5th order rho resonances
rhoMag_ = [1.0, 1.0, 0.59, 4.8e-2, 0.40, 0.43]
rhoPhase_ = [0.0, 0.0, -2.2, -2.0, -2.9, 1.19]
rhoMasses_ = [0.77337, 1.490, 1.870, 2.12, 2.321, 2.567]
rhoWidths_ = [0.1471, 0.429, 0.357, 0.3, 0.444, 0.491]
rhoWgt_ = []
mass_ = []
width_ = []
coup_ = []
hres_ = []
h0_ = []
dh_ = []

# Parameter set for DM part
gDM_ = 1.0
mDM_ = 0.41
mMed_ = 5
wMed_ = 10.0
cI1_ = 1.0
cI0_ = 1.0
cS_ = 1.0


# function to reset parameters
def resetParameters(gDM, mDM, mMed, wMed, cMedu, cMedd, cMeds):
    global rhoWgt_, beta_, nMax_
    global rhoMag_, rhoPhase_, rhoMasses_, rhoWidths_
    global omegaMag_, omegaPhase_, omegaMass_, omegaWidth_
    global coup_
    global gDM_, mDM_, mMed_, wMed_, cI1_, cI0_
    gDM_ = gDM
    mDM_ = mDM
    mMed_ = mMed
    wMed_ = wMed
    cI1_ = cMedu - cMedd
    # taken out since omega mixes with rho and omega and phi are not coupling
    # to dark mediator

    # cI0_ = 3*(cMedu+cMedd)
    # cS_ = -3*cMeds
    # omegaMag_ =0.001867242111672990*cI0_
    # just to shorten things if no I1 contribution
    if cI1_ == 0:
        nMax_ = 1
        coup_ = [0] * nMax_
    else:
        # masses in vectors
        rhoWgt_ = []
        mass_ = []
        width_ = []
        coup_ = []
        hres_ = []
        h0_ = []
        dh_ = []
        initialize()


# initialize tower of couplings
def initialize():
    global rhoWgt_, omegaWgt_
    global mass_, width_, coup_
    global hres_, h0_, dh_
    rhoSum = 0.0
    # print rhoWgt_
    for ix in range(0, len(rhoMag_)):
        rhoWgt_.append(
            rhoMag_[ix]
            * (math.cos(rhoPhase_[ix]) + complex(0.0, 1.0) * math.sin(rhoPhase_[ix]))
        )
        if ix > 0:
            rhoSum += rhoWgt_[ix]
    omegaWgt_ = omegaMag_ * (
        math.cos(omegaPhase_) + complex(0.0, 1.0) * math.sin(omegaPhase_)
    )
    # set up the masses and widths of the rho resonances
    gamB = scipy.special.gamma(2.0 - beta_)
    cwgt = 0.0
    for ix in range(0, nMax_):
        # this is gam(2-beta+n)/gam(n+1)
        if ix > 0:
            gamB *= ((1.0 - beta_ + float(ix))) / float(ix)
        c_n = (
            scipy.special.gamma(beta_ - 0.5)
            / (0.5 + float(ix))
            / math.sqrt(np.pi)
            * math.sin(np.pi * (beta_ - 1.0 - float(ix)))
            / np.pi
            * gamB
        )
        if ix % 2 != 0:
            c_n *= -1.0
        if ix == 0:
            # print 'testing 0',c_n,1.087633403691967
            c_n = 1.087633403691967
        # set the masses and widths
        # calc for higher resonances
        if ix >= len(rhoMasses_):
            mass_.append(rhoMasses_[0] * math.sqrt(1.0 + 2.0 * float(ix)))
            width_.append(rhoWidths_[0] / rhoMasses_[0] * mass_[-1])
        # input for lower ones
        else:
            mass_.append(rhoMasses_[ix])
            width_.append(rhoWidths_[ix])
        if ix > 0 and ix < len(rhoWgt_):
            cwgt += c_n
        # parameters for the gs propagators
        hres_.append(Hhat(mass_[-1] ** 2, mass_[-1], width_[-1], mpi, mpi))
        dh_.append(dHhatds(mass_[-1], width_[-1], mpi, mpi))
        h0_.append(H(0.0, mass_[-1], width_[-1], mpi, mpi, dh_[-1], hres_[-1]))
        coup_.append(cI1_ * c_n)
    # fix up the early weights
    for ix in range(1, len(rhoWgt_)):
        # print ix
        coup_[ix] = cI1_ * rhoWgt_[ix] * cwgt / rhoSum


# form factor calculation
def Fpi(q2, imode):
    FPI = complex(0.0, 0.0)
    # print coup_[0]
    for ix in range(0, nMax_):
        term = coup_[ix] * BreitWignerGS(
            q2, mass_[ix], width_[ix], mpi, mpi, h0_[ix], dh_[ix], hres_[ix]
        )
        # include rho-omega if needed
        if ix == 0 and imode != 0:
            term *= (
                1.0
                / (1.0 + omegaWgt_)
                * (1.0 + omegaWgt_ * BreitWignerFW(q2, omegaMass_, omegaWidth_))
            )
        # sum
        FPI += term
    # factor for cc mode
    if imode == 0:
        FPI *= math.sqrt(2.0)
    return FPI


# Decay rate of mediator-> 2pions
def GammaDM(medMass):
    if medMass < 2 * mpi:
        return 0
    temp = Fpi(medMass ** 2, 1)
    return (
        1.0
        / 48.0
        / math.pi
        * medMass
        * (1 - 4 * mpi ** 2 / medMass ** 2) ** 1.5
        * abs(temp) ** 2
        * gev2nb
    )


# cross-section for e+e- -> pi+pi-, see cross-section formula eq. (6) and eq. (28) in
# low_energy.pdf
def sigmaSM(Q2):
    if Q2 < 4 * mpi ** 2:
        return 0
    alphaEM = alpha.alphaEM(Q2)
    temp = Fpi(Q2, 1)
    return (
        1.0
        / 3.0
        * math.pi
        * alphaEM ** 2
        / Q2
        * (1.0 - 4.0 * mpi ** 2 / Q2) ** 1.5
        * abs(temp) ** 2
        * gev2nb
    )


# cross section for DM annihilations, see cross-section formula eq. (6) and eq. (28) in
# low_energy.pdf
def sigmaDM(Q2):
    if Q2 < 4 * mpi ** 2:
        return 0
    cDM = gDM_
    DMmed = cDM / (Q2 - mMed_ ** 2 + complex(0.0, 1.0) * mMed_ * wMed_)
    DMmed2 = abs(DMmed) ** 2
    temp = Fpi(Q2, 1)
    return (
        1
        / 48.0
        / math.pi
        * DMmed2
        * Q2
        * (1 + 2 * mDM_ ** 2 / Q2)
        * (1.0 - 4.0 * mpi ** 2 / Q2) ** 1.5
        * abs(temp) ** 2
        * gev2nb
    )
