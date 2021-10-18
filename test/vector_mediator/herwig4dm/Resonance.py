import math
from math import pi

# units from H7
MeV = 1e-3
# fixed pion mass
mpi_       = .13957018
gev2nb = 389379.3656

def beta2(s,m1,m2) :
    return max(0.,(1.-(m1+m2)**2/s)*(1.-(m1-m2)**2/s))
    
def beta(s,m1,m2) :
    return math.sqrt(beta2(s,m1,m2))
  
def dHhatds(mRes,gamma,m1,m2) :
  v2 = beta2(mRes**2,m1,m2)
  v = math.sqrt(v2)
  r = (m1**2 + m2**2)/mRes**2
  return gamma/pi/mRes/v2*((3.-2.*v2- 3.*r)*math.log((1.+v)/(1.-v)) + 2.*v*(1.- r/(1.-v2)))
 
def Hhat(s,mRes,gamma,m1,m2) :
  vR = beta(mRes**2,m1,m2)
  v  = beta(    s    ,m1,m2)
  return gamma/mRes/pi*s*(v/vR)**3*math.log((1.+v)/(1.-v))

def H(s,mRes,gamma,m1,m2,dH,Hres) :
  if(s!=0.) :
    return Hhat(s,mRes,gamma,m1,m2) - Hres - (s-mRes**2)*dH
  else :
    return -2.*(m1+m2)**2/pi*gamma/mRes/beta(mRes**2,m1,m2)**3 - Hres + mRes**2*dH

def gammaP(s,mRes,gamma,m1,m2) :
  v2 = beta2(s,m1,m2)
  if(v2<=0.) : return 0.
  vR2 = beta2(mRes**2,m1,m2)
  if(vR2==0.) :
      rp=0.
  else :
      rp = math.sqrt(max(0.,v2/vR2))
  return math.sqrt(s)/mRes*rp**3*gamma

def BreitWignerGS(s,mRes,gamma,m1,m2,H0,dH,Hres) :
  mR2 = mRes**2
  return (mR2+H0)/(mR2-s+H(s,mRes,gamma,m1,m2,dH,Hres)
                   -complex(0.,1.)*math.sqrt(s)*gammaP(s,mRes,gamma,m1,m2))

def BreitWignerFW(s,mRes,gamma) :
    mR2 = mRes**2
    return mR2/(mR2-s-complex(0.,1.)*mRes*gamma)

def BreitWignerPWave(s,mRes,gamma,m1,m2) :
    mR2 = mRes**2
    return mR2/(mR2-s-complex(0.,1.)*math.sqrt(s)*gammaP(s,mRes,gamma,m1,m2))

def ga1(Q2) :
    if Q2<9.*mpi_**2 :
        return 0.
    elif(Q2>0.838968432668) :
        return 1.623*Q2+10.38-9.32/Q2+0.65/Q2**2
    else :
        delta = (Q2-9.*mpi_**2)
        return 4.1*delta**3*(1.-3.3*delta+5.8*delta**2)
    
def BreitWignera1(Q2,mRes,gamma) :
    mR2 = mRes**2
    return mR2/(mR2-Q2-complex(0.,1)*gamma*mRes*ga1(Q2)/ga1(mR2))
