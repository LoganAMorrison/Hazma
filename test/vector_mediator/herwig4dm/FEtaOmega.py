#libraries to load
import alpha,math,Resonance,cmath
import numpy,scipy,FK

# own fit values and parametrization, taken from 1911.11147
mOmega_p_  = 1.43
mOmega_pp_ = 1.67
gOmega_p_  = 0.215
gOmega_pp_ = 0.113
mOmega_ = 0.78265
mEta_   = 0.547862
a_Omega_p_  = 0.0862
a_Omega_pp_ = 0.0648
phi_Omega_p_  = 0
phi_Omega_pp_ = math.pi

# coupling modification depending on mediator quark couplings
# Parameter set for DM part
gDM_ = 1.
mDM_ = 0.41
mMed_ = 5
wMed_ = 10.
cI1_ = 1.
cI0_ = 1.
cS_ = 1.

#change rho, omega, phi contributions
def resetParameters(gDM,mDM,mMed,wMed,cMedu,cMedd,cMeds) :
    global cI1_,cI0_,cS_
    global gDM_,mDM_,mMed_,wMed_
    gDM_ = gDM
    mDM_ = mDM
    mMed_ = mMed
    wMed_ = wMed
    cI1_ = cMedu-cMedd
    cI0_ = 3*(cMedu+cMedd)
    cS_ = -3*cMeds

def GammaDM(mMed):
    Q2 = mMed**2
    Q = math.sqrt(Q2)
    if(Q>mEta_+mOmega_) :
        pcm = 0.5/Q*math.sqrt(Q2**2+mOmega_**4+mEta_**4-2.*Q2*mEta_**2-2.*Q2*mOmega_**2-2.*mEta_**2*mOmega_**2)
    else :
        return 0.
    amp = Resonance.BreitWignerFW(Q2,mOmega_p_ ,gOmega_p_ )*a_Omega_p_ *cmath.exp(complex(0.,phi_Omega_p_ ))+\
          Resonance.BreitWignerFW(Q2,mOmega_pp_,gOmega_pp_)*a_Omega_pp_*cmath.exp(complex(0.,phi_Omega_pp_))
    amp *=cI0_
    return 1/12./math.pi*pcm**3*abs(amp)**2*Resonance.gev2nb

def sigmaSMEtaOmega(Q2) :
    Q = math.sqrt(Q2)
    if(Q>mEta_+mOmega_) :
        pcm = 0.5/Q*math.sqrt(Q2**2+mOmega_**4+mEta_**4-2.*Q2*mEta_**2-2.*Q2*mOmega_**2-2.*mEta_**2*mOmega_**2)
    else :
        return 0.
    amp = Resonance.BreitWignerFW(Q2,mOmega_p_ ,gOmega_p_ )*a_Omega_p_ *cmath.exp(complex(0.,phi_Omega_p_ ))+\
          Resonance.BreitWignerFW(Q2,mOmega_pp_,gOmega_pp_)*a_Omega_pp_*cmath.exp(complex(0.,phi_Omega_pp_))
    amp *=cI0_
    return 4.*math.pi*alpha.alphaEM(Q2)**2*pcm**3/3./Q/Q2*abs(amp)**2*Resonance.gev2nb

def sigmaDMEtaOmega(Q2) :
    Q = math.sqrt(Q2)
    if(Q>mEta_+mOmega_) :
        pcm = 0.5/Q*math.sqrt(Q2**2+mOmega_**4+mEta_**4-2.*Q2*mEta_**2-2.*Q2*mOmega_**2-2.*mEta_**2*mOmega_**2)
    else :
        return 0.
    amp = Resonance.BreitWignerFW(Q2,mOmega_p_ ,gOmega_p_ )*a_Omega_p_ *cmath.exp(complex(0.,phi_Omega_p_ ))+\
          Resonance.BreitWignerFW(Q2,mOmega_pp_,gOmega_pp_)*a_Omega_pp_*cmath.exp(complex(0.,phi_Omega_pp_))
    amp *=cI0_
    cDM = gDM_
    DMmed = cDM/(Q2-mMed_**2+complex(0.,1.)*mMed_*wMed_)
    DMmed2 = abs(DMmed)**2
    return 1/12./math.pi*DMmed2*Q*(1+2*mDM_**2/Q2)*pcm**3*abs(amp)**2*Resonance.gev2nb


