# Libraries to load
import FPiGamma,os
import numpy,scipy,math
import matplotlib.pyplot as plt

# set DM parameters
# DM to mediator coupling
gDM = 1.
#DM mass
mDM = FPiGamma.mPi_
# mediator mass
mMed = 10
# mediator width, own width.py will be added with all channels (work in progress)
# wMed


xSM=[]
ySM=[]
xDP=[]
yDP=[]
xBL=[]
yBL=[]

energy=2*FPiGamma.mPi_
while energy < 2.0:
    xSM.append(energy)
    ySM.append(FPiGamma.sigmaSMPiGamma(energy**2))
    energy+=0.001

# couplings of mediator to quarks - Dark Photon case
cMed_u = 2./3.
cMed_d = -1./3.
cMed_s = -1./3.
while mDM < 2.0:
    energy = 2*mDM+0.0001
    Q2 = energy**2
    xDP.append(energy)
    # mediator width should be replaced by a function for the full decay width
    wMed = FPiGamma.GammaDM(mMed)
    FPiGamma.resetParameters(gDM,mDM,mMed,wMed,cMed_u,cMed_d,cMed_s)
    yDP.append(FPiGamma.sigmaDMPiGamma(Q2))
    mDM+=0.001

# B-L model
cMed_u = 1./3.
cMed_d = 1./3.
cMed_s = 1./3.
mDM = FPiGamma.mPi_
while mDM < 2.0:
    energy = 2*mDM+0.0001
    Q2 = energy**2
    xBL.append(energy)
    # mediator width should be replaced by a function for the full decay width
    wMed = FPiGamma.GammaDM(mMed)
    FPiGamma.resetParameters(gDM,mDM,mMed,wMed,cMed_u,cMed_d,cMed_s)
    yBL.append(FPiGamma.sigmaDMPiGamma(Q2))
    mDM+=0.001
    



with open('txt_files/DPPiGamma.txt', 'w') as txtfile:
    for i in range(0,len(xDP)):
        txtfile.write("%s , %s\n" %(xDP[i],yDP[i]))
    txtfile.close()


plt.plot(xSM,ySM,color="blue",label="SM")
plt.plot(xDP,yDP,color="red",label="DP")
plt.plot(xBL,yBL,color="green",label="BL")
plt.title("$\\pi\\gamma$ final state")
plt.xlabel("$\\sqrt{s}$/GeV")
plt.ylabel("$\\sigma$/nb")
plt.xlim(0.28,2.0)
plt.yscale("log")
#plt.minorticks_on()
#plt.grid(which='minor', alpha=0.2)
#plt.grid(which='major', alpha=0.5)
plt.legend()
plt.savefig("plots/PiGamma.pdf")
plt.clf()
plt.cla()
plt.close()

