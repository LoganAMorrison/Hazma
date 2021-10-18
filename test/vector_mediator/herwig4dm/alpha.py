import math
m_e   = 0.510998928e-3
m_mu  = 0.1056583715
m_tau = 1.77682
m_top = 173.21

def realPi(r) :
    fvthr = 1.666666666666667e0
    rmax  = 1.e6
    # use assymptotic formula
    if ( abs(r)<1e-3 ) :
        return -fvthr-math.log(r)
    # return zero for large values
    elif(abs(r)>rmax) :
        return 0.;
    elif(4.*r>1.) :
        beta=math.sqrt(4.*r-1.)
        return 1./3. -(1.+2.*r)*(2.-beta*math.acos(1.-1./(2.*r)))
    else :
        beta=math.sqrt(1.-4.*r)
        return 1./3.-(1.+2.*r)*(2.+beta*math.log(abs((beta-1.)/(beta+1.))))
    
def alphaEM(scale) :
    eps=1e-6
    a1=0.0
    b1=0.00835
    c1=1.000
    a2=0.0    
    b2=0.00238
    c2=3.927
    a3=0.00165
    b3=0.00299
    c3=1.000
    a4=0.00221
    b4=0.00293
    c4=1.000
    # alpha_EM at Q^2=0
    alem=7.2973525698e-3
    aempi = alem/(3.*math.pi)
    # return q^2=0 value for small scales
    if(scale<eps) :
        return alem
    # leptonic component
    repigg = aempi*(realPi(m_e**2/scale)+realPi(m_mu**2/scale)+realPi(m_tau**2/scale))
    # Hadronic component from light quarks
    if(scale<9e-2) :
        repigg+=a1+b1*math.log(1.+c1*scale)
    elif(scale<9.) :
        repigg+=a2+b2*math.log(1.+c2*scale);
    elif(scale<1.e4) :
        repigg+=a3+b3*math.log(1.+c3*scale);
    else :
        repigg+=a4+b4*math.log(1.+c4*scale);
    # Top Contribution
    repigg+=aempi*realPi(m_top**2/scale);
    # return the answer
    return alem/(1.-repigg);
