# Libraries to load

import matplotlib.pyplot as plt

from . import FOmPiPi

ii = complex(0.0, 1.0)

xSM = []
ySM_c = []
ySM_n = []

xDP = []
yDP_c = []
yDP_n = []

xBL = []
yBL_c = []
yBL_n = []


# SM case
scale = FOmPiPi.mOm_ + 2 * FOmPiPi.mPi_[0] + 0.1
while scale < 2.5:
    Q2 = scale ** 2
    xSM.append(scale)
    ySM_n.append(FOmPiPi.sigmaSM(Q2, 0))
    ySM_c.append(FOmPiPi.sigmaSM(Q2, 1))
    scale += 0.01

# set DM parameters
# DM to mediator coupling
gDM = 1.0
# DM mass
mDM = 0.58089518

# mediator mass
mMed = 10.0
# mediator width, file with total width will be added (work in progress)
# wMed


# B-L model
# couplings of mediator to quarks
cMed_u = 1.0 / 3.0
cMed_d = 1.0 / 3.0
cMed_s = 1.0 / 3.0

# reset DM mass
mDM = 0.58089518

while mDM < 2.0:
    energy = 2 * mDM + 0.0001
    Q2 = energy ** 2
    xBL.append(energy)
    # mediator width should be replaced by a function for the full decay width
    wMed_n = FOmPiPi.GammaDM(mMed, 0)
    FOmPiPi.resetParameters(gDM, mDM, mMed, wMed_n, cMed_u, cMed_d, cMed_s)
    yBL_n.append(FOmPiPi.sigmaDM(Q2, 0))
    wMed_c = FOmPiPi.GammaDM(mMed, 1)
    FOmPiPi.resetParameters(gDM, mDM, mMed, wMed_c, cMed_u, cMed_d, cMed_s)
    yBL_c.append(FOmPiPi.sigmaDM(Q2, 1))
    mDM += 0.01


# plots

plt.plot(xSM, ySM_n, color="blue", label="SM")
plt.plot(xDP, yDP_n, color="red", label="DP")
plt.plot(xBL, yBL_n, color="green", label="BL")
plt.legend()
plt.yscale("log")
plt.xlim(1.125, 2.0)
plt.title("$e^+e^- \\to\\omega\\pi^0\\pi^0$")
plt.xlabel("$\\sqrt{s}$/GeV")
plt.ylabel("$\\sigma$/nb")
plt.savefig("plots/OmegaPiPi_neutral.pdf")
# plt.show()
plt.close()


plt.plot(xSM, ySM_c, color="blue", label="SM")
plt.plot(xBL, yBL_c, color="green", label="BL")
plt.plot(xDP, yDP_c, color="red", label="DP")
plt.legend()
plt.yscale("log")
plt.xlim(1.125, 2.0)
plt.title("$e^+e^- \\to\\omega\\pi^+\\pi^-$")
plt.xlabel("$\\sqrt{s}$/GeV")
plt.ylabel("$\\sigma$/nb")
plt.savefig("plots/OmegaPiPi_charged.pdf")
# plt.show()
plt.close()
