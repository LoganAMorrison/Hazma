# Libraries to load

from . import FPiGamma_new as FPiGamma

import matplotlib.pyplot as plt

# set DM parameters
# DM to mediator coupling
gDM = 1.0
# DM mass
mDM = FPiGamma.mPi_ / 2.0
# mediator mass
mMed = 10
# mediator width, own width.py will be added with all channels (work in progress)
# wMed


xSM = []
ySM = []
xDP = []
yDP = []
xBL = []
yBL = []

energy = FPiGamma.mPi_
while energy < 2.0:
    xSM.append(energy)
    ySM.append(FPiGamma.sigmaSMPiGamma(energy ** 2))
    energy += 0.001

# couplings of mediator to quarks - Dark Photon case
cMed_u = 2.0 / 3.0
cMed_d = -1.0 / 3.0
cMed_s = -1.0 / 3.0
while mDM < 2.0:
    energy = 2 * mDM + 0.0001
    Q2 = energy ** 2
    xDP.append(energy)
    # mediator width should be replaced by a function for the full decay width
    wMed = FPiGamma.GammaDM(mMed)
    FPiGamma.resetParameters(gDM, mDM, mMed, wMed, cMed_u, cMed_d, cMed_s)
    yDP.append(FPiGamma.sigmaDMPiGamma(Q2))
    mDM += 0.001

# B-L model
cMed_u = 1.0 / 3.0
cMed_d = 1.0 / 3.0
cMed_s = 1.0 / 3.0
mDM = FPiGamma.mPi_ / 2.0
while mDM < 2.0:
    energy = 2 * mDM + 0.0001
    Q2 = energy ** 2
    xBL.append(energy)
    # mediator width should be replaced by a function for the full decay width
    wMed = FPiGamma.GammaDM(mMed)
    FPiGamma.resetParameters(gDM, mDM, mMed, wMed, cMed_u, cMed_d, cMed_s)
    yBL.append(FPiGamma.sigmaDMPiGamma(Q2))
    mDM += 0.001


with open("txt_files/DPPiGamma.txt", "w") as txtfile:
    for i in range(0, len(xDP)):
        txtfile.write("%s , %s\n" % (xDP[i], yDP[i]))
    txtfile.close()


plt.plot(xSM, ySM, color="blue", label="SM")
plt.plot(xDP, yDP, color="red", label="DP")
plt.plot(xBL, yBL, color="green", label="BL")
plt.title("$\\pi\\gamma$ final state")
plt.xlabel("$\\sqrt{s}$/GeV")
plt.ylabel("$\\sigma$/nb")
plt.xlim(0.14, 2.0)
plt.yscale("log")
# plt.minorticks_on()
# plt.grid(which='minor', alpha=0.2)
# plt.grid(which='major', alpha=0.5)
plt.legend()
plt.savefig("plots/PiGamma.pdf")
plt.clf()
plt.cla()
plt.close()
