# Libraries to load
import alpha,math,Resonance,cmath

ii = complex(0.,1.)

# set of parameters, taken from arXiv:1610.00235
gRhoOmPi_ = 15.9
Amp_ = [1.,0.175,0.014]
Phase_ = [0.,124.,-63.]
rhoMasses_ = [0.77526,1.510,1.720]
rhoWidths_ = [0.1491,0.44,0.25]
fRho_ = 5.06325#alphaEM evaluated (with alpha.alphaEM() fct) at energy that is considered
# taken from PDG
mOmega_ = 0.78265
mPi_ = 0.1349766
#brPi0Gamma_ = 0.0888
brPi0Gamma_ = 0.0828

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


# rho width, based on arXiv:1303.5198.
def gRhos(Q2,ix):
    Q = math.sqrt(Q2)
    if ix==0:
        pOm = 0.5/Q*math.sqrt(Q2**2+mPi_**4+mOmega_**4-2.*Q2*mOmega_**2-2.*Q2*mPi_**2-2.*mOmega_**2*mPi_**2)
        gRho = rhoWidths_[0]*rhoMasses_[0]**2/Q2*((Q2-4.*mPi_**2)/(rhoMasses_[0]**2-4.*mPi_**2))**1.5
        gRho+=gRhoOmPi_**2*pOm**3/12./math.pi
    else:
        gRho = rhoWidths_[ix]
    return gRho

# form factor for Omega Pion
def FOmPiGamma(Q2) :
    Q = math.sqrt(Q2)
    form=0.
    for i in range(0,len(rhoMasses_)):
        Di = rhoMasses_[i]**2-Q2-ii*Q*gRhos(Q2,i)
        form+=Amp_[i]*cmath.exp(ii*math.radians(Phase_[i]))*rhoMasses_[i]**2/Di
        #form+=Amp_[i]*cmath.exp(ii*Phase_[i]*2*math.pi/360.)*rhoMasses_[i]**2/Di
    form*=gRhoOmPi_*cI1_/fRho_
    return form

def GammaDM(mMed):
    Q = mMed
    Q2 = Q**2
    if(Q2>(mOmega_+mPi_)**2 and cI1_ != 0) :
        pcm = 0.5/Q*math.sqrt(Q2**2+mPi_**4+mOmega_**4-2.*Q2*mOmega_**2-2.*Q2*mPi_**2-2.*mOmega_**2*mPi_**2)
    else :
        return 0.
    return 1/12./math.pi*pcm**3*abs(FOmPiGamma(Q2))**2*Resonance.gev2nb

# cross section for Omega Pion
def sigmaSMOmegaPion(Q2) :
    Q = math.sqrt(Q2)
    #print "alpha: ", alpha.alphaEM(Q2), "at ", Q, " GeV"
    if(Q>mOmega_+mPi_) :
        pcm = 0.5/Q*math.sqrt(Q2**2+mPi_**4+mOmega_**4-2.*Q2*mOmega_**2-2.*Q2*mPi_**2-2.*mOmega_**2*mPi_**2)
    else :
        return 0.
    return 4.*math.pi*alpha.alphaEM(Q2)**2*pcm**3/3./Q/Q2*abs(FOmPiGamma(Q2))**2*Resonance.gev2nb

# cross section for Omega Pion
def sigmaDMOmegaPion(Q2) :
    Q = math.sqrt(Q2)
    #print "alpha: ", alpha.alphaEM(Q2), "at ", Q, " GeV"
    if(Q>mOmega_+mPi_) :
        pcm = 0.5/Q*math.sqrt(Q2**2+mPi_**4+mOmega_**4-2.*Q2*mOmega_**2-2.*Q2*mPi_**2-2.*mOmega_**2*mPi_**2)
    else :
        return 0.
    cDM = gDM_
    DMmed = cDM/(Q2-mMed_**2+complex(0.,1.)*mMed_*wMed_)
    DMmed2 = abs(DMmed)**2
    return DMmed2/12./math.pi*Q*(1+2*mDM_**2/Q2)*pcm**3*abs(FOmPiGamma(Q2))**2*Resonance.gev2nb
