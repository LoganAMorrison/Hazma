# Libraries to load
import math,alpha,cmath,numpy
import os,F3pi
import matplotlib.pyplot as plt
import Resonance

ii = complex(0.,1.)
# from GeV to nb units for data comparison
gev2nb    = 389379.3656

#scale=0.5#3*F3pi.mPi_

xSM=[]
ySM=[]
xDP=[]
yDP=[]
xBL=[]
yBL=[]
print("SM cross section")
# SM case
scale = 0.5#3.*F3pi.mPi_
while scale < 4.0 :
    Q2 = scale**2
    xSM.append(scale)
    ySM.append(F3pi.sigmaSM(Q2))
    if(scale<=1.1) :
        scale+=0.01
    else :
        scale+=0.01


# set DM parameters
# DM to mediator coupling
gDM = 1.
#DM mass
mDM = 0.25#F3pi.mPi_
# mediator mass
mMed = 5
# mediator width, file with total width will be added (work in progress)
# wMed


# Dark Photon case
# couplings of mediator to quarks
cMed_u = 2./3.
cMed_d = -1./3.
cMed_s = -1./3.
print("Dark Photon cross-section")
while mDM < 2.0 :
    energy = 2*mDM+0.0001
    Q2 = energy**2
    xDP.append(energy)
    # mediator width should be replaced by a function for the full decay width
    wMed = F3pi.GammaDM(mMed)
    F3pi.resetParameters(gDM,mDM,mMed,wMed,cMed_u,cMed_d,cMed_s)
    yDP.append(F3pi.sigmaDM(Q2))
    if(energy<=1.1) :
        mDM+=0.001
    else :
        mDM+=0.01


# B-L model
# couplings of mediator to quarks
cMed_u = 1./3.
cMed_d = 1./3.
cMed_s = 1./3.
print("B-L model")
# reset DM mass
mDM = 0.25#F2pi.mpi_
while mDM < 2.0 :
    energy = 2*mDM+0.0001
    Q2 = energy**2
    xBL.append(energy)
    # mediator width should be replaced by a function for the full decay width
    wMed = F3pi.GammaDM(mMed)
    F3pi.resetParameters(gDM,mDM,mMed,wMed,cMed_u,cMed_d,cMed_s)
    yBL.append(F3pi.sigmaDM(Q2))
    if(energy<=1.1) :
        mDM+=0.001
    else :
        mDM+=0.01

plt.plot(xSM,ySM,color="blue",label="SM")
plt.plot(xDP,yDP,color="red",label="DP")
plt.plot(xBL,yBL,color="green",label="BL")
plt.legend()
plt.yscale("log")
plt.xlabel("$\\sqrt{s}$/GeV")
plt.ylabel("$\\sigma$/nb")
#plt.show()
plt.savefig("plots/3pions.pdf")



