# Libraries to load

import matplotlib.pyplot as plt

from . import FKKpi, FPhiPi

analyses = {}
br00 = 0.342
brpp = 0.489

xSM = []
ySM_0 = []
ySM_1 = []
ySM_2 = []

xDP = []
yDP_0 = []
yDP_1 = []
yDP_2 = []

xBL = []
yBL_0 = []
yBL_1 = []
yBL_2 = []


# energy range
low_lim = 1.21
upp_lim = 3.0
FKKpi.readHadronic_Current()
# SM case
scale = low_lim
while scale < upp_lim:
    Q2 = scale ** 2
    xSM.append(scale)
    ySM_0.append(FKKpi.sigmaSM(Q2, 0))
    ySM_1.append(FKKpi.sigmaSM(Q2, 1) + br00 * FPhiPi.sigmaSMPhiPi(Q2))
    ySM_2.append(FKKpi.sigmaSM(Q2, 2) + brpp * FPhiPi.sigmaSMPhiPi(Q2))
    scale += 0.01


# set DM parameters
# DM to mediator coupling
gDM = 1.0
# DM mass
mDM = 1.21 / 2

# mediator mass
mMed = 10.0
# mediator width, file with total width will be added (work in progress)
# wMed


# Dark Photon case
# couplings of mediator to quarks
cMed_u = 2.0 / 3.0
cMed_d = -1.0 / 3.0
cMed_s = -1.0 / 3.0


# FKKpi.resetParameters(gDM,0.,0.,0.,cMed_u,cMed_d,cMed_s)
while mDM < 2.0:
    energy = 2 * mDM + 0.0001
    Q2 = energy ** 2
    xDP.append(energy)
    wMed_0 = FKKpi.GammaDM(mMed, 0)
    FKKpi.resetParameters(gDM, mDM, mMed, wMed_0, cMed_u, cMed_d, cMed_s)
    yDP_0.append(FKKpi.sigmaDM(Q2, 0))
    wMed_1 = FKKpi.GammaDM(mMed, 1)
    FKKpi.resetParameters(gDM, mDM, mMed, wMed_1, cMed_u, cMed_d, cMed_s)
    yDP_1.append(FKKpi.sigmaDM(Q2, 1))
    wMed_2 = FKKpi.GammaDM(mMed, 2)
    FKKpi.resetParameters(gDM, mDM, mMed, wMed_2, cMed_u, cMed_d, cMed_s)
    yDP_2.append(FKKpi.sigmaDM(Q2, 2))
    mDM += 0.05


# B-L model
# couplings of mediator to quarks
cMed_u = 1.0 / 3.0
cMed_d = 1.0 / 3.0
cMed_s = 1.0 / 3.0

# reset DM mass
mDM = 1.21 / 2

# FKKpi.resetParameters(gDM,0.,0.,0.,cMed_u,cMed_d,cMed_s)

while mDM < 2.0:
    energy = 2 * mDM + 0.0001
    Q2 = energy ** 2
    xBL.append(energy)
    wMed_0 = FKKpi.GammaDM(mMed, 0)
    FKKpi.resetParameters(gDM, mDM, mMed, wMed_0, cMed_u, cMed_d, cMed_s)
    yBL_0.append(FKKpi.sigmaDM(Q2, 0))
    wMed_1 = FKKpi.GammaDM(mMed, 1)
    FKKpi.resetParameters(gDM, mDM, mMed, wMed_1, cMed_u, cMed_d, cMed_s)
    yBL_1.append(FKKpi.sigmaDM(Q2, 1))
    wMed_2 = FKKpi.GammaDM(mMed, 2)
    FKKpi.resetParameters(gDM, mDM, mMed, wMed_2, cMed_u, cMed_d, cMed_s)
    yBL_2.append(FKKpi.sigmaDM(Q2, 2))
    mDM += 0.05


plt.plot(xSM, ySM_0, color="blue", label="SM")
plt.plot(xDP, yDP_0, color="red", label="DP")
plt.plot(xBL, yBL_0, color="green", label="BL")
plt.legend()
plt.yscale("log")
plt.xlim(1.2, 2.2)
plt.title("$e^+e^- \\to K^0_SK^0_L\\pi^0$")
plt.xlabel("$\\sqrt{s}$/GeV")
plt.ylabel("$\\sigma$/nb")
plt.savefig("plots/KKpi_K0K0Pi0.pdf")
# plt.show()
plt.close()


plt.plot(xSM, ySM_1, color="blue", label="SM")
plt.plot(xDP, yDP_1, color="red", label="DP")
plt.plot(xBL, yBL_1, color="green", label="BL")
plt.legend()
plt.yscale("log")
plt.xlim(1.2, 2.2)
plt.title("$e^+e^- \\to K^+K^-\\pi^0$")
plt.xlabel("$\\sqrt{s}$/GeV")
plt.ylabel("$\\sigma$/nb")
plt.savefig("plots/KKpi_KpKmPi0.pdf")
# plt.show()
plt.close()

plt.plot(xSM, ySM_2, color="blue", label="SM")
plt.plot(xDP, yDP_2, color="red", label="DP")
plt.plot(xBL, yBL_2, color="green", label="BL")
plt.legend()
plt.yscale("log")
plt.xlim(1.2, 2.2)
plt.title("$e^+e^- \\to K^\\pm K^0_S\\pi^\\mp$")
plt.xlabel("$\\sqrt{s}$/GeV")
plt.ylabel("$\\sigma$/nb")
plt.savefig("plots/KKpi_KpmK0pimp.pdf")
# plt.show()
plt.close()
