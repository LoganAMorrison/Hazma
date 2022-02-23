# Libraries to load
import math
import cmath

from . import Resonance
from . import alpha


# own parametrization as in 1911.11147
# parameters from fit
mPhi_p_ = 1.67
mPhi_pp_ = 2.14
gPhi_p_ = 0.122
gPhi_pp_ = 0.0435
mPhi_ = 1.019461
mEta_ = 0.547862
a_Phi_p_ = 0.175
a_Phi_pp_ = 0.00409
phi_Phi_p_ = 0
phi_Phi_pp_ = 2.19

# coupling modification depending on mediator quark couplings
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
    global gDM_, mDM_, mMed_, wMed_
    gDM_ = gDM
    mDM_ = mDM
    mMed_ = mMed
    wMed_ = wMed
    cI1_ = cMedu - cMedd
    cI0_ = 3 * (cMedu + cMedd)
    cS_ = -3 * cMeds


def GammaDM(mMed):
    Q2 = mMed ** 2
    Q = math.sqrt(Q2)
    if Q > mEta_ + mPhi_:
        pcm = (
            0.5
            / Q
            * math.sqrt(
                Q2 ** 2
                + mPhi_ ** 4
                + mEta_ ** 4
                - 2.0 * Q2 * mEta_ ** 2
                - 2.0 * Q2 * mPhi_ ** 2
                - 2.0 * mEta_ ** 2 * mPhi_ ** 2
            )
        )
    else:
        return 0.0
    amp = Resonance.BreitWignerFW(Q2, mPhi_p_, gPhi_p_) * a_Phi_p_ * cmath.exp(
        complex(0.0, phi_Phi_p_)
    ) + Resonance.BreitWignerFW(Q2, mPhi_pp_, gPhi_pp_) * a_Phi_pp_ * cmath.exp(
        complex(0.0, phi_Phi_pp_)
    )
    amp *= cS_
    return 1 / 12.0 / math.pi * pcm ** 3 * abs(amp) ** 2


def sigmaSMEtaPhi(Q2):
    Q = math.sqrt(Q2)
    if Q > mEta_ + mPhi_:
        pcm = (
            0.5
            / Q
            * math.sqrt(
                Q2 ** 2
                + mPhi_ ** 4
                + mEta_ ** 4
                - 2.0 * Q2 * mEta_ ** 2
                - 2.0 * Q2 * mPhi_ ** 2
                - 2.0 * mEta_ ** 2 * mPhi_ ** 2
            )
        )
    else:
        return 0.0
    amp = Resonance.BreitWignerFW(Q2, mPhi_p_, gPhi_p_) * a_Phi_p_ * cmath.exp(
        complex(0.0, phi_Phi_p_)
    ) + Resonance.BreitWignerFW(Q2, mPhi_pp_, gPhi_pp_) * a_Phi_pp_ * cmath.exp(
        complex(0.0, phi_Phi_pp_)
    )
    amp *= cS_
    return (
        4.0
        * math.pi
        * alpha.alphaEM(Q2) ** 2
        * pcm ** 3
        / 3.0
        / Q
        / Q2
        * abs(amp) ** 2
        * Resonance.gev2nb
    )


def sigmaDMEtaPhi(Q2):
    Q = math.sqrt(Q2)
    if Q > mEta_ + mPhi_:
        pcm = (
            0.5
            / Q
            * math.sqrt(
                Q2 ** 2
                + mPhi_ ** 4
                + mEta_ ** 4
                - 2.0 * Q2 * mEta_ ** 2
                - 2.0 * Q2 * mPhi_ ** 2
                - 2.0 * mEta_ ** 2 * mPhi_ ** 2
            )
        )
    else:
        return 0.0
    amp = Resonance.BreitWignerFW(Q2, mPhi_p_, gPhi_p_) * a_Phi_p_ * cmath.exp(
        complex(0.0, phi_Phi_p_)
    ) + Resonance.BreitWignerFW(Q2, mPhi_pp_, gPhi_pp_) * a_Phi_pp_ * cmath.exp(
        complex(0.0, phi_Phi_pp_)
    )
    amp *= cS_
    cDM = gDM_
    DMmed = cDM / (Q2 - mMed_ ** 2 + complex(0.0, 1.0) * mMed_ * wMed_)
    DMmed2 = abs(DMmed) ** 2
    return (
        1
        / 12.0
        / math.pi
        * DMmed2
        * Q
        * (1 + 2 * mDM_ ** 2 / Q2)
        * pcm ** 3
        * abs(amp) ** 2
        * Resonance.gev2nb
    )
