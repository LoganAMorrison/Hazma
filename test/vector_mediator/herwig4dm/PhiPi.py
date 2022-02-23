# Libraries to load

import matplotlib.pyplot as plt

from . import FPhiPi


# set DM parameters
# DM to mediator coupling
gDM = 1.0
# DM mass
mDM = (FPhiPi.mpi + FPhiPi.mPhi) / 2.0
# mediator mass
mMed = 10
# mediator width, own width.py will be added with all channels (work in progress)
# wMed


# energy range
low_lim = FPhiPi.mPhi + FPhiPi.mpi  # 1.1
upp_lim = 2.0
step_size = 0.005

xSM = []
ySM = []
xDP = []
yDP = []
xBL = []
yBL = []

energy = low_lim
while energy < upp_lim:
    xSM.append(energy)
    ySM.append(FPhiPi.sigmaSMPhiPi(energy ** 2))
    energy += step_size

with open("txt_files/PhiPi.txt", "w") as txtfile:
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
    wMed = FPhiPi.GammaDM(mMed)
    FPhiPi.resetParameters(gDM, mDM, mMed, wMed, cMed_u, cMed_d, cMed_s)
    yDP.append(FPhiPi.sigmaDMPhiPi(Q2))
    mDM += step_size

# couplings of mediator to quarks
cMed_u = 1.0 / 3.0
cMed_d = 1.0 / 3.0
cMed_s = 1.0 / 3.0
mDM = (FPhiPi.mpi + FPhiPi.mPhi) / 2.0
while mDM < upp_lim:
    energy = 2 * mDM + 0.0001
    Q2 = energy ** 2
    xBL.append(energy)
    wMed = FPhiPi.GammaDM(mMed)
    FPhiPi.resetParameters(gDM, mDM, mMed, wMed, cMed_u, cMed_d, cMed_s)
    yBL.append(FPhiPi.sigmaDMPhiPi(Q2))
    mDM += step_size


plt.plot(xSM, ySM, color="blue", label="SM")
plt.plot(xDP, yDP, color="red", label="DP")
plt.plot(xBL, yBL, color="green", label="BL")
plt.xlabel("$\\sqrt{s}$/GeV")
plt.ylabel("$\\sigma$/nb")
plt.title("$\\phi\\pi$ final state")
plt.xlim(low_lim, 2.0)
plt.legend()
plt.savefig("plots/PhiPi.pdf")
plt.clf()
plt.cla()
