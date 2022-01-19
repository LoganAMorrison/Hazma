# Libraries to load
import F4pi,os,math
import matplotlib.pyplot as plt
import Resonance,alpha,numpy

br_omega_pi_gamma = 0.084

F4pi.readHadronic_Current()
xSM_charged=[]
ySM_charged=[]
xDP_charged=[]
yDP_charged=[]
xBL_charged=[]
yBL_charged=[]

xSM_neutral=[]
ySM_neutral=[]
xDP_neutral=[]
yDP_neutral=[]
xBL_neutral=[]
yBL_neutral=[]

# set DM parameters
# DM to mediator coupling
gDM = 1.
#DM mass
# mDM = 0.62
mDM= 0.62/2.

# mediator mass
mMed = 4.
# mediator width, own width.py will be added with all channels (work in progress)
# wMed



low_lim = 0.62 
upp_lim = 4.0

scale=low_lim
while scale<upp_lim:
    s=scale**2
    xSM_neutral.append(scale)
    xSM_charged.append(scale)
    ySM_neutral.append(F4pi.sigmaSM(s,"neutral"))
    ySM_charged.append(F4pi.sigmaSM(s,"charged"))
    scale+=0.01

# Dark Photon
    
cMedu = 2./3.
cMedd = -1./3.
cMeds = -1./3.
wMed = F4pi.GammaDM(mMed,"neutral")
F4pi.resetParameters(gDM,mDM,mMed,wMed,cMedu,cMedd,cMeds)
F4pi.readHadronic_Current()

mDM = 0.6125/2.
while mDM<2.:
    energy = 2*mDM+0.0001
    s=energy**2
    xDP_neutral.append(energy)
    xDP_charged.append(energy)
    wMed_n = F4pi.GammaDM(mMed,mode="neutral")
    F4pi.resetParameters(gDM,mDM,mMed,wMed_n,cMedu,cMedd,cMeds)
    yDP_neutral.append(F4pi.sigmaDM(s,mode="neutral"))
    wMed_c = F4pi.GammaDM(mMed,mode="charged")
    F4pi.resetParameters(gDM,mDM,mMed,wMed_c,cMedu,cMedd,cMeds)
    yDP_charged.append(F4pi.sigmaDM(s,mode="charged"))
    mDM+=0.01
    
# B-L model, in fact only baryon-number coupled model in general
    
cMedu = 1./3.
cMedd = 1./3.
cMeds = 1./3.
wMed = F4pi.GammaDM(mMed,"neutral")
F4pi.resetParameters(gDM,mDM,mMed,wMed,cMedu,cMedd,cMeds)
F4pi.readHadronic_Current()

mDM = 0.6125/2.
while mDM<2.:
    energy = 2*mDM+0.0001
    s=energy**2
    xBL_neutral.append(energy)
    xBL_charged.append(energy)
    wMed_n = F4pi.GammaDM(mMed,mode="neutral")
    F4pi.resetParameters(gDM,mDM,mMed,wMed_n,cMedu,cMedd,cMeds)
    yBL_neutral.append(F4pi.sigmaDM(s,mode="neutral"))
    wMed_c = F4pi.GammaDM(mMed,mode="charged")
    F4pi.resetParameters(gDM,mDM,mMed,wMed_c,cMedu,cMedd,cMeds)
    yBL_charged.append(F4pi.sigmaDM(s,mode="charged"))
    mDM+=0.01


###############
#### plots ####
###############

# neutral
plt.plot(xSM_neutral,ySM_neutral,color="blue",label="SM")
plt.plot(xDP_neutral,yDP_neutral,color="red",label="DP")
plt.plot(xBL_neutral,yBL_neutral,color="green",label="BL")
plt.yscale("log")
plt.xlabel("$\\sqrt{s}$/GeV")
plt.ylabel("$\\sigma$/nb")
plt.xlim(0.5,3.0)
plt.title("$e^+e^- \\to \\pi^+\\pi^- 2\\pi^0$")
plt.legend()
plt.savefig("plots/test_4pi_neutral.pdf")
#plt.show()
#plt.clf()
#plt.cla()
plt.close()

# charged
plt.plot(xSM_charged,ySM_charged,color="blue",label="SM")
plt.plot(xDP_charged,yDP_charged,color="red",label="DP")
plt.plot(xBL_charged,yBL_charged,color="green",label="BL")
plt.yscale("log")
plt.xlabel("$\\sqrt{s}$/GeV")
plt.ylabel("$\\sigma$/nb")
# plt.xlim(0.5,3.0)
# plt.ylim(0.005,100)
plt.title("$e^+e^- \\to 2\\pi^+2\\pi^-$")
plt.legend()
plt.savefig("plots/test_4pi_charged.pdf")
#plt.show()
# plt.clf()
# plt.cla()
plt.close()



