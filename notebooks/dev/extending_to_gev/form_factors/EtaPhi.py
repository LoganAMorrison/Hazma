# Libraries to load
import alpha,math,Resonance,cmath
import numpy,scipy,FK
import FEtaPhi
import os
import matplotlib.pyplot as plt


# set DM parameters
# DM to mediator coupling
gDM = 1.
#DM mass
mDM = (FEtaPhi.mEta_+FEtaPhi.mPhi_)/2.
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

#energy range
low_lim = FEtaPhi.mPhi_+FEtaPhi.mEta_
upp_lim = 2.0
step_size = 0.005


energy = low_lim
while energy < upp_lim :
    xSM.append(energy)
    ySM.append(FEtaPhi.sigmaSMEtaPhi(energy**2))
    energy+=step_size

with open('txt_files/EtaPhi.txt', 'w') as txtfile:
    for i in range(0,len(xSM)):
        txtfile.write("%s , %s\n" %(xSM[i],ySM[i]))
    txtfile.close()

# couplings of mediator to quarks
cMed_u = 2./3.
cMed_d = -1./3.
cMed_s = -1./3.
while mDM < upp_lim:
    energy = 2*mDM+0.0001
    Q2 = energy**2
    xDP.append(energy)
    wMed = FEtaPhi.GammaDM(mMed)
    FEtaPhi.resetParameters(gDM,mDM,mMed,wMed,cMed_u,cMed_d,cMed_s)
    yDP.append(FEtaPhi.sigmaDMEtaPhi(Q2))
    mDM+=step_size

# couplings of mediator to quarks
cMed_u = 1./3.
cMed_d = 1./3.
cMed_s = 1./3.
mDM = (FEtaPhi.mEta_+FEtaPhi.mPhi_)/2.
while mDM < upp_lim:
    energy = 2*mDM+0.0001
    Q2 = energy**2
    xBL.append(energy)
    wMed = FEtaPhi.GammaDM(mMed)
    FEtaPhi.resetParameters(gDM,mDM,mMed,wMed,cMed_u,cMed_d,cMed_s)
    yBL.append(FEtaPhi.sigmaDMEtaPhi(Q2))
    mDM+=step_size
    
plt.plot(xSM,ySM,color="blue",label="SM")
plt.plot(xDP,yDP,color="red",label="DP")
plt.plot(xBL,yBL,color="green",linestyle='--',label="BL")
plt.xlabel("$\\sqrt{s}$/GeV")
plt.ylabel("$\\sigma$/nb")
plt.yscale("log")
plt.xlim(FEtaPhi.mPhi_+FEtaPhi.mEta_,2.0)
plt.title("$\\eta\\phi$ final state")
plt.legend()
plt.savefig("plots/EtaPhi.pdf")
plt.clf()
plt.cla()
plt.close()
