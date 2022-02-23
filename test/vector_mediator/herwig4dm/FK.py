import math

from scipy import special
import numpy as np

from .Resonance import (
    MeV,
    Hhat,
    H,
    dHhatds,
    BreitWignerFW,
    BreitWignerGS,
    BreitWignerPWave,
    gev2nb,
)
from . import alpha
from .masses import mpi, mk0, mk


# PDG mass values
# mk0 = 0.497611
# mk = 0.493677

# parametrization, taken from arXiv:1002.0279 with own fit values

# truncation parameter
nMax_ = 200
# initial parameters for the model
betaRho_ = 2.1968
betaOmega_ = 2.6936
betaPhi_ = 1.9452
etaPhi_ = 1.055
gammaOmega_ = 0.5
gammaPhi_ = 0.2
# rho parameters
rhoMag_ = [
    1.1148916618504967,
    -0.050374779737077324,
    -0.014908906283692132,
    -0.03902475997619905,
    -0.038341465215871416,
]
rhoPhase_ = [0, 0, 0, 0, 0]
rhoMasses_ = [
    775.49 * MeV,
    1520.6995754050117 * MeV,
    1740.9719246639341 * MeV,
    1992.2811314327789 * MeV,
]
rhoWidths_ = [
    149.4 * MeV,
    213.41728317817743 * MeV,
    84.12224414791908 * MeV,
    289.9733272437917 * MeV,
]
# omega parameters
omegaMag_ = [
    1.3653229680598022,
    -0.02775156567495144,
    -0.32497165559032715,
    1.3993153161869765,
]
omegaPhase_ = [0, 0, 0, 0, 0]
omegaMasses_ = [782.65 * MeV, 1414.4344268685891 * MeV, 1655.375231284883 * MeV]
omegaWidths_ = [8.49 * MeV, 85.4413887755723 * MeV, 160.31760444832305 * MeV]
# phi parameters
phiMag_ = [
    0.965842498579515,
    -0.002379766320723148,
    -0.1956211640216197,
    0.16527771485190898,
]
phiPhase_ = [0.0, 0.0, 0.0, 0, 0]
phiMasses_ = [
    1019.4209171596993 * MeV,
    1594.759278457624 * MeV,
    2156.971341201067 * MeV,
]
phiWidths_ = [
    4.252653332329334 * MeV,
    28.741821847408196 * MeV,
    673.7556174184005 * MeV,
]
# rho weights
rhoWgt_ = []
# omega weights
omegaWgt_ = []
# phi weights
phiWgt_ = []
# masses in vectors
mass_ = [[], [], []]
width_ = [[], [], []]
coup_ = [[], [], []]
hres_ = []
h0_ = []
dh_ = []

crhoextra_ = 0.0
comegaextra_ = 0.0
cphiextra_ = 0.0
c_extra_ = []

mtau = 1.77686
vud = 0.97420
brmu = 0.1739

# Parameter set for DM part
gDM_ = 1.0
mDM_ = 0.41
mMed_ = 5
wMed_ = 10.0
cI1_ = 1.0
cI0_ = 1.0
cS_ = 1.0


def spectral(Q):
    Q2 = Q ** 2
    fk = Fkaon(Q2, 0, mk0, mk)
    return (
        1.0
        / 24.0
        / np.pi
        * (1.0 - (mk0 - mk) ** 2 / Q2) ** 1.5
        * (1.0 - (mk0 + mk) ** 2 / Q2) ** 1.5
        * (fk * fk.conjugate()).real
    )


def dBR(Q):
    Q2 = Q ** 2
    pcm = math.sqrt(0.25 / Q2 * (Q2 - (mk0 + mk) ** 2) * (Q2 - (mk0 - mk) ** 2))
    pre = (
        brmu
        * 0.5
        * vud ** 2
        / mtau ** 2
        * (1 + 2.0 * Q2 / mtau ** 2)
        * (1 - Q2 / mtau ** 2) ** 2
        * (2.0 * pcm / Q) ** 2
        * Q
    )
    fk = Fkaon(Q2, 0, mk0, mk)
    return pre * (fk * fk.conjugate()).real


def c_0(b0):
    ratio = 2.0 / math.sqrt(np.pi)
    b1 = b0
    while b1 > 2.0:
        ratio *= (b1 - 1.5) / (b1 - 2.0)
        b1 -= 1.0
    ratio *= special.gamma(b1 - 0.5) / special.gamma(b1 - 1.0)
    return ratio


def findBeta(c0):
    betamin = 1.0
    betamid = 5.0
    betamax = 10.0
    eps = 1e-10
    cmid = c_0(betamid)
    while abs(cmid - c0) > eps:
        cmin = c_0(betamin)
        cmax = c_0(betamax)
        cmid = c_0(betamid)
        if c0 > cmin and c0 < cmid:
            betamax = betamid
        elif c0 > cmid and c0 < cmax:
            betamin = betamid
        elif c0 >= cmax:
            betamax *= 2.0
        else:
            print("bisect fails", betamin, betamid, betamax, c0)
            print("bisect fails", cmin, cmid, cmax, c0)
            quit()
        betamid = 0.5 * (betamin + betamax)
    return betamid


def resetParameters(gDM, mDM, mMed, wMed, cMedu, cMedd, cMeds):
    global cRho_, cOmega_, cPhi_
    global rhoWgt_, omegaWgt_, phiWgt_
    global mass_, width_, coup_
    global hres_, h0_, dh_
    global betaRho_, betaOmega_, betaPhi_
    global rhoMag_
    global omegaMag_
    global phiMag_
    global c_extra_
    global gDM_, mDM_, mMed_, wMed_, cI1_, cI0_, cS_
    # switch for rho, omega, phi contributions
    cI1_ = cMedu - cMedd
    cI0_ = 3 * (cMedu + cMedd)
    cS_ = -3 * cMeds
    # rest of parameters are calculated
    betaRho_ = findBeta(rhoMag_[0])
    betaOmega_ = findBeta(omegaMag_[0])
    betaPhi_ = findBeta(phiMag_[0])
    # rho weights
    rhoWgt_ = []
    # omega weights
    omegaWgt_ = []
    # phi weights
    phiWgt_ = []
    # calculated couplings
    c_extra_ = []
    # masses in vectors
    mass_ = [[], [], []]
    width_ = [[], [], []]
    coup_ = [[], [], []]
    hres_ = []
    h0_ = []
    dh_ = []
    initialize()


def initialize():
    global rhoWgt_, omegaWgt_, phiWgt_
    global mass_, width_, coup_
    global hres_, h0_, dh_, cphiextra_, comegaextra_, crhoextra_, c_extra_
    # rho weights
    rhoWgt_ = []
    for ix in range(0, len(rhoMag_)):
        rhoWgt_.append(
            rhoMag_[ix]
            * (math.cos(rhoPhase_[ix]) + complex(0.0, 1.0) * math.sin(rhoPhase_[ix]))
        )
    # omega weights
    omegaWgt_ = []
    for ix in range(0, len(omegaMag_)):
        omegaWgt_.append(
            omegaMag_[ix]
            * (
                math.cos(omegaPhase_[ix])
                + complex(0.0, 1.0) * math.sin(omegaPhase_[ix])
            )
        )
    # phi weights
    phiWgt_ = []
    for ix in range(0, len(phiMag_)):
        phiWgt_.append(
            phiMag_[ix]
            * (math.cos(phiPhase_[ix]) + complex(0.0, 1.0) * math.sin(phiPhase_[ix]))
        )
    # rho masses and couplings
    gamB = special.gamma(2.0 - betaRho_)
    # masses in vectors
    mass_ = [[], [], []]
    width_ = [[], [], []]
    coup_ = [[], [], []]
    hres_ = []
    h0_ = []
    dh_ = []
    for ix in range(0, nMax_):
        if cI1_ == 0:
            coup_[0] = [0.0] * nMax_
            mass_[0] = [1.0] * nMax_
            width_[0] = [1.0] * nMax_
            hres_ = [0.0] * nMax_
            dh_ = [0.0] * nMax_
            h0_ = [0.0] * nMax_
            break
        # this is gam(2-beta+n)/gam(n+1)
        if ix > 0:
            gamB *= ((1.0 - betaRho_ + float(ix))) / float(ix)
        c_n = (
            special.gamma(betaRho_ - 0.5)
            / (0.5 + float(ix))
            / math.sqrt(np.pi)
            * math.sin(np.pi * (betaRho_ - 1.0 - float(ix)))
            / np.pi
            * gamB
        )
        if ix % 2 != 0:
            c_n *= -1.0
        # set couplings, masses and width
        coup_[0].append(c_n)
        if ix < len(rhoMasses_):
            mass_[0].append(rhoMasses_[ix])
            width_[0].append(rhoWidths_[ix])
        else:
            mass_[0].append(rhoMasses_[0] * math.sqrt(1.0 + 2.0 * float(ix)))
            width_[0].append(rhoWidths_[0] / rhoMasses_[0] * mass_[0][-1])
        # parameters for the gs propagators
        hres_.append(Hhat(mass_[0][-1] ** 2, mass_[0][-1], width_[0][-1], mpi, mpi))
        dh_.append(dHhatds(mass_[0][-1], width_[0][-1], mpi, mpi))
        h0_.append(H(0.0, mass_[0][-1], width_[0][-1], mpi, mpi, dh_[-1], hres_[-1]))
    # reset the parameters for the low lying resonances
    # set the masses and widths
    # couplings
    total = sum(coup_[0])
    for i in range(0, len(rhoWgt_)):
        total += rhoWgt_[i] - coup_[0][i]
        coup_[0][i] = rhoWgt_[i]
    coup_[0][len(rhoWgt_)] = 1.0 - total + coup_[0][len(rhoWgt_)]
    crhoextra_ = 1.0 - total + coup_[0][len(rhoWgt_)]
    # omega masses and couplings
    gamB = special.gamma(2.0 - betaOmega_)
    for ix in range(0, nMax_):
        if cI0_ == 0:
            coup_[1] = [0.0] * nMax_
            mass_[1] = [1.0] * nMax_
            width_[1] = [1.0] * nMax_
            break
        #  this is gam(2-beta+n)/gam(n+1)
        if ix > 0:
            gamB *= ((1.0 - betaOmega_ + float(ix))) / float(ix)
        c_n = (
            special.gamma(betaOmega_ - 0.5)
            / (0.5 + float(ix))
            / math.sqrt(np.pi)
            * math.sin(np.pi * (betaOmega_ - 1.0 - float(ix)))
            / np.pi
            * gamB
        )
        if ix % 2 != 0:
            c_n *= -1.0
        coup_[1].append(c_n)
        # set the masses and widths
        mass_[1].append(omegaMasses_[0] * math.sqrt(1.0 + 2.0 * float(ix)))
        width_[1].append(gammaOmega_ * mass_[1][-1])
    # reset the parameters for the low lying resonances
    # set the masses and widths
    for i in range(0, len(omegaMasses_)):
        mass_[1][i] = omegaMasses_[i]
        width_[1][i] = omegaWidths_[i]
    # couplings
    total = sum(coup_[1])
    for i in range(0, len(omegaWgt_)):
        total += omegaWgt_[i] - coup_[1][i]
        coup_[1][i] = omegaWgt_[i]
    coup_[1][len(omegaWgt_)] = 1.0 - total + coup_[1][len(omegaWgt_)]
    comegaextra_ = 1.0 - total + coup_[1][len(omegaWgt_)]
    # phi masses and couplings
    gamB = special.gamma(2.0 - betaPhi_)
    for ix in range(0, nMax_):
        if cS_ == 0:
            coup_[2] = [0.0] * nMax_
            mass_[2] = [1.0] * nMax_
            width_[2] = [1.0] * nMax_
            break
        # this is gam(2-beta+n)/gam(n+1)
        if ix > 0:
            gamB *= ((1.0 - betaPhi_ + float(ix))) / float(ix)
        c_n = (
            special.gamma(betaPhi_ - 0.5)
            / (0.5 + float(ix))
            / math.sqrt(np.pi)
            * math.sin(np.pi * (betaPhi_ - 1.0 - float(ix)))
            / np.pi
            * gamB
        )
        if ix % 2 != 0:
            c_n *= -1.0
        # couplings
        coup_[2].append(c_n)
        # set the masses and widths
        mass_[2].append(phiMasses_[0] * math.sqrt(1.0 + 2.0 * float(ix)))
        width_[2].append(gammaPhi_ * mass_[2][-1])
    # reset the parameters for the low lying resonances
    # set the masses and widths
    for i in range(0, len(phiMasses_)):
        mass_[2][i] = phiMasses_[i]
        width_[2][i] = phiWidths_[i]
    # couplings
    total = sum(coup_[2])
    for i in range(0, len(phiWgt_)):
        total += phiWgt_[i] - coup_[2][i]
        coup_[2][i] = phiWgt_[i]
    coup_[2][len(phiWgt_)] = 1.0 - total + coup_[2][len(phiWgt_)]
    cphiextra_ = 1.0 - total + coup_[2][len(phiWgt_)]
    c_extra_ = [crhoextra_, comegaextra_, cphiextra_]


# calculate the form factor for 2Kaons, imode=0: neutral, imode=1: charged
def Fkaon(q2, imode):
    if imode == 0:
        mK = mk0
    if imode == 1:
        mK = mk
    FK = complex(0.0, 0.0)
    for ix in range(0, nMax_):
        # rho exchange
        term = (
            cI1_
            * coup_[0][ix]
            * BreitWignerGS(
                q2, mass_[0][ix], width_[0][ix], mpi, mpi, h0_[ix], dh_[ix], hres_[ix]
            )
        )
        if imode != 0:
            FK += 0.5 * term
        else:
            FK -= 0.5 * term
        # omega exchange
        term = cI0_ * coup_[1][ix] * BreitWignerFW(q2, mass_[1][ix], width_[1][ix])
        FK += 1.0 / 6.0 * term
        # phi exchange
        term = (
            cS_
            * coup_[2][ix]
            * BreitWignerPWave(q2, mass_[2][ix], width_[2][ix], mK, mK)
        )
        if ix == 0 and imode == 0:
            term *= etaPhi_
        FK += term / 3.0
    # factor for cc mode
    # if(imode==0) :
    #    FK *= math.sqrt(2.0)
    return FK


# Decay rate of mediator-> 2Kaons, imode=0: neutral, imode=1: charged
def GammaDM(medMass, imode):
    mK = 0.0
    if imode == 0:
        if medMass < 2 * mk0:
            return 0
        mK = mk0
        temp = Fkaon(medMass, imode)
    if imode == 1:
        if medMass < 2 * mk:
            return 0
        mK = mk
        temp = Fkaon(medMass ** 2, imode)
    return (
        1.0
        / 48.0
        / math.pi
        * medMass
        * (1 - 4 * mK ** 2 / medMass ** 2) ** 1.5
        * abs(temp) ** 2
        * gev2nb
    )


# SM cross section for e+e- annihilation to neutral Kaon production
def sigmaSM0(Q2):
    alphaEM = alpha.alphaEM(Q2)
    temp = Fkaon(Q2, 0)
    return (
        1.0
        / 3.0
        * math.pi
        * alphaEM ** 2
        / Q2
        * (1.0 - 4.0 * mk0 ** 2 / Q2) ** 1.5
        * abs(temp) ** 2
        * gev2nb
    )


# SM cross section for e+e- annihilation to charged Kaon production
def sigmaSMP(Q2):
    alphaEM = alpha.alphaEM(Q2)
    temp = Fkaon(Q2, 1)
    return (
        1.0
        / 3.0
        * math.pi
        * alphaEM ** 2
        / Q2
        * (1.0 - 4.0 * mk ** 2 / Q2) ** 1.5
        * abs(temp) ** 2
        * gev2nb
    )


# cross section for DM annihilations to two neutral Kaons
def sigmaDM0(Q2):
    if Q2 < 2 * mk ** 2:
        return 0
    cDM = gDM_
    DMmed = cDM / (Q2 - mMed_ ** 2 + complex(0.0, 1.0) * mMed_ * wMed_)
    DMmed2 = abs(DMmed) ** 2
    temp = Fkaon(Q2, 0)
    return (
        1
        / 48.0
        / math.pi
        * DMmed2
        * Q2
        * (1 + 2 * mDM_ ** 2 / Q2)
        * (1.0 - 4.0 * mk0 ** 2 / Q2) ** 1.5
        * abs(temp) ** 2
        * gev2nb
    )


# cross section for DM annihilations to two charged Kaons
def sigmaDMP(Q2):
    if Q2 < 4 * mk ** 2:
        return 0
    cDM = gDM_
    DMmed = cDM / (Q2 - mMed_ ** 2 + complex(0.0, 1.0) * mMed_ * wMed_)
    DMmed2 = abs(DMmed) ** 2
    temp = Fkaon(Q2, 1)
    return (
        1
        / 48.0
        / math.pi
        * DMmed2
        * Q2
        * (1 + 2 * mDM_ ** 2 / Q2)
        * (1.0 - 4.0 * mk ** 2 / Q2) ** 1.5
        * abs(temp) ** 2
        * gev2nb
    )
