# Libraries to load

import matplotlib.pyplot as plt

from . import FOmegaPion


# set DM parameters
# DM to mediator coupling
gDM = 1.0
# DM mass
mDM = (FOmegaPion.mOmega_ + FOmegaPion.mPi_) / 2.0
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


energy = FOmegaPion.mOmega_ + FOmegaPion.mPi_  # 0.9
upp_lim = 2.000
step_size = 0.005
while energy < upp_lim:
    xSM.append(energy)
    Q2 = energy ** 2
    ySM.append(FOmegaPion.sigmaSMOmegaPion(Q2))
    energy += step_size

with open("txt_files/OmegaPion.txt", "w") as txtfile:
    for i in range(0, len(xSM)):
        txtfile.write("%s , %s\n" % (xSM[i], ySM[i]))
    txtfile.close()

# couplings of mediator to quarks
cMed_u = 2.0 / 3.0
cMed_d = -1.0 / 3.0
cMed_s = -1.0 / 3.0
while mDM < upp_lim:
    energy = 2 * mDM + 0.0001
    Q2 = energy ** 2
    xDP.append(energy)
    wMed = FOmegaPion.GammaDM(mMed)
    FOmegaPion.resetParameters(gDM, mDM, mMed, wMed, cMed_u, cMed_d, cMed_s)
    yDP.append(FOmegaPion.sigmaDMOmegaPion(Q2))
    mDM += step_size

# couplings of mediator to quarks
cMed_u = 1.0 / 3.0
cMed_d = 1.0 / 3.0
cMed_s = 1.0 / 3.0
mDM = (FOmegaPion.mOmega_ + FOmegaPion.mPi_) / 2.0
while mDM < upp_lim:
    energy = 2 * mDM + 0.0001
    Q2 = energy ** 2
    xBL.append(energy)
    wMed = FOmegaPion.GammaDM(mMed)
    FOmegaPion.resetParameters(gDM, mDM, mMed, wMed, cMed_u, cMed_d, cMed_s)
    yBL.append(FOmegaPion.sigmaDMOmegaPion(Q2))
    mDM += step_size

plt.plot(xSM, ySM, color="blue", label="SM")
plt.plot(xDP, yDP, color="red", label="DP")
plt.plot(xBL, yBL, color="green", label="BL")
plt.title("$\\omega\\pi$ final state")
plt.xlabel("$\\sqrt{s}$/GeV")
plt.ylabel("$\\sigma$/nb")
# plt.xlim(1.047,2.005)
plt.xlim(0.9, 2.005)
# plt.minorticks_on()
# plt.grid(which='minor', alpha=0.2)
# plt.grid(which='major', alpha=0.5)
plt.legend()
plt.savefig("plots/OmegaPion.pdf")
plt.clf()
plt.cla()
plt.close()
