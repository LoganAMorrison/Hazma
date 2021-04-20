# Libraries to load
import FEtaGamma,os
import numpy,scipy,math
import matplotlib.pyplot as plt

# set DM parameters
# DM to mediator coupling
gDM = 1.
#DM mass
mDM = FEtaGamma.mEta_/2.
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

low_lim=FEtaGamma.mEta_
upp_lim=2.0

energy=low_lim
while energy < upp_lim:
    xSM.append(energy)
    ySM.append(FEtaGamma.sigmaSMEtaGamma(energy**2))
    energy+=0.005

with open('txt_files/SMEtaGamma.txt', 'w') as txtfile:
    for i in range(0,len(xSM)):
        txtfile.write("%s , %s\n" %(xSM[i],ySM[i]))
    txtfile.close()

# couplings of mediator to quarks
cMed_u = 2./3.
cMed_d = -1./3.
cMed_s = -1./3.
while mDM < 1.0:
    energy = 2*mDM+0.0001
    Q2 = energy**2
    xDP.append(energy)
    # mediator width should be replaced by a function for the full decay width
    wMed = FEtaGamma.GammaDM(mMed)
    FEtaGamma.resetParameters(gDM,mDM,mMed,wMed,cMed_u,cMed_d,cMed_s)
    yDP.append(FEtaGamma.sigmaDMEtaGamma(Q2))
    mDM+=0.001

cMed_u = 1./3.
cMed_d = 1./3.
cMed_s = 1./3.
mDM = FEtaGamma.mEta_/2.
while mDM < 1.0:
    energy = 2*mDM+0.0001
    Q2 = energy**2
    xBL.append(energy)
    # mediator width should be replaced by a function for the full decay width
    wMed = FEtaGamma.GammaDM(mMed)
    FEtaGamma.resetParameters(gDM,mDM,mMed,wMed,cMed_u,cMed_d,cMed_s)
    yBL.append(FEtaGamma.sigmaDMEtaGamma(Q2))
    mDM+=0.001

plt.plot(xSM,ySM,color="blue",label="SM")
plt.plot(xDP,yDP,color="red",label="DP")
plt.plot(xBL,yBL,color="green",label="BL")
plt.title("$\\eta\\gamma$ final state")
plt.xlabel("$\\sqrt{s}$/GeV")
plt.ylabel("$\\sigma$/nb")
plt.ylim(1e-6,100)
plt.yscale("log")
#plt.minorticks_on()
#plt.grid(which='minor', alpha=0.2)
#plt.grid(which='major', alpha=0.5)
plt.legend()
plt.savefig("plots/EtaGamma.pdf")
plt.clf()
plt.cla()
plt.close()
