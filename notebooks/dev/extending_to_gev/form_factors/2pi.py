# based on https://arxiv.org/pdf/1002.0279.pdf

import numpy, scipy, F2pi, math
import matplotlib.pyplot as plt
import os, alpha, Resonance
from Resonance import MeV

# initialize and plot the form factor
F2pi.initialize()

xSM = []
ySM = []
xDP = []
yDP = []
xBL = []
yBL = []

# SM case
scale = 2.0 * F2pi.mpi_
while scale < 4.0:
    Q2 = scale ** 2
    xSM.append(scale)
    ySM.append(F2pi.sigmaSM(Q2))
    if scale <= 1.1:
        scale += 0.001
    else:
        scale += 0.01

# set DM parameters
# DM to mediator coupling
gDM = 1.0
# DM mass
mDM = F2pi.mpi_
# mediator mass
mMed = 5
# mediator width, file with total width will be added (work in progress)
# wMed


# Dark Photon case
# couplings of mediator to quarks
cMed_u = 2.0 / 3.0
cMed_d = -1.0 / 3.0
cMed_s = -1.0 / 3.0

while mDM < 2.0:
    energy = 2 * mDM + 0.0001
    Q2 = energy ** 2
    xDP.append(energy)
    # mediator width should be replaced by a function for the full decay width
    wMed = F2pi.GammaDM(mMed)
    F2pi.resetParameters(gDM, mDM, mMed, wMed, cMed_u, cMed_d, cMed_s)
    yDP.append(F2pi.sigmaDM(Q2))
    if energy <= 1.1:
        mDM += 0.001
    else:
        mDM += 0.01

# B-L model
# couplings of mediator to quarks
cMed_u = 1.0 / 3.0
cMed_d = 1.0 / 3.0
cMed_s = 1.0 / 3.0

# reset DM mass
mDM = F2pi.mpi_
while mDM < 2.0:
    energy = 2 * mDM + 0.0001
    Q2 = energy ** 2
    xBL.append(energy)
    # mediator width should be replaced by a function for the full decay width
    wMed = F2pi.GammaDM(mMed)
    F2pi.resetParameters(gDM, mDM, mMed, wMed, cMed_u, cMed_d, cMed_s)
    yBL.append(F2pi.sigmaDM(Q2))
    if energy <= 1.1:
        mDM += 0.001
    else:
        mDM += 0.01

plt.plot(xSM, ySM, color="blue", label="SM")
plt.plot(xDP, yDP, color="red", label="DP")
plt.plot(xBL, yBL, color="green", label="BL")
plt.xlabel("$\\sqrt{s}$/GeV")
plt.ylabel("$\\sigma$/nb")
plt.title("$\\pi\\pi$ final state")
plt.xlim(2.0 * F2pi.mpi_, 1.0)
# plt.ylim(0.,50)
plt.minorticks_on()
plt.grid(which="minor", alpha=0.2)
plt.grid(which="major", alpha=0.5)
plt.legend()
plt.savefig("plots/2pi_rho.pdf")
plt.clf()
plt.cla()
plt.close()

# plt.yscale("log")
plt.plot(xSM, ySM, color="blue", label="SM")
plt.plot(xDP, yDP, color="red", label="DP")
plt.plot(xBL, yBL, color="green", label="BL")
plt.xlabel("$\\sqrt{s}$/GeV")
plt.ylabel("$\\sigma$/nb")
plt.title("$\\pi\\pi$ final state")
plt.xlim(1.0, 4.0)
plt.ylim(-1.0, 60)
plt.minorticks_on()
plt.grid(which="minor", alpha=0.2)
plt.grid(which="major", alpha=0.5)
plt.legend()
plt.savefig("plots/2pi_high.pdf")
plt.clf()
plt.cla()
plt.close()
