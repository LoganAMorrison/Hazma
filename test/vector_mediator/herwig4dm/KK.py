# Libraries to load
import math
import os

import numpy
import scipy
import matplotlib.pyplot as plt

from . import FK


# energy range SM
low_lim = 2.0 * FK.mK0
upp_lim = 2.0

# set DM parameters
# DM to mediator coupling
gDM = 1.0
# DM mass
mDM = FK.mK0
# mediator mass
mMed = 10
# mediator width, own width.py will be added with all channels (work in progress)
# wMed
# couplings of mediator to quarks
cMed_u = 2.0 / 3.0
cMed_d = -1.0 / 3.0
cMed_s = -1.0 / 3.0

########
# plot arrays
########

xSM = []
y_sigmaSM00 = []
y_sigmaSMpm = []

xDP = []
y_sigmaDP00 = []
y_sigmaDPpm = []

xBL = []
y_sigmaBL00 = []
y_sigmaBLpm = []

xBL = []
y_sigmaBL00 = []
y_sigmaBLpm = []

FK.initialize()
# SM case
energy = low_lim
while energy < upp_lim:
    xSM.append(energy)
    sigmaN = FK.sigmaSM0(energy ** 2)
    sigmaP = FK.sigmaSMP(energy ** 2)
    y_sigmaSM00.append(sigmaN)
    y_sigmaSMpm.append(sigmaP)
    energy += 0.001
    if energy > 1.10:
        energy += 0.005

# Dark Photon case
while mDM < upp_lim:
    energy = 2 * mDM + 0.0001
    xDP.append(energy)
    wMed = FK.GammaDM(mMed, 0)
    FK.resetParameters(gDM, mDM, mMed, wMed, cMed_u, cMed_d, cMed_s)
    sigmaN = FK.sigmaDM0(energy ** 2)
    wMed = FK.GammaDM(mMed, 1)
    FK.resetParameters(gDM, mDM, mMed, wMed, cMed_u, cMed_d, cMed_s)
    sigmaP = FK.sigmaDMP(energy ** 2)
    y_sigmaDP00.append(sigmaN)
    y_sigmaDPpm.append(sigmaP)
    mDM += 0.0005
    if mDM > 1.10:
        step_size = 0.005

# reset DM mass
mDM = FK.mK0
# B-L case
cMed_u = 1.0 / 3.0
cMed_d = 1.0 / 3.0
cMed_s = 1.0 / 3.0
while mDM < upp_lim:
    energy = 2 * mDM + 0.0001
    xBL.append(energy)
    wMed = FK.GammaDM(mMed, 0)
    FK.resetParameters(gDM, mDM, mMed, wMed, cMed_u, cMed_d, cMed_s)
    sigmaN = FK.sigmaDM0(energy ** 2)
    wMed = FK.GammaDM(mMed, 1)
    FK.resetParameters(gDM, mDM, mMed, wMed, cMed_u, cMed_d, cMed_s)
    sigmaP = FK.sigmaDMP(energy ** 2)
    y_sigmaBL00.append(sigmaN)
    y_sigmaBLpm.append(sigmaP)
    mDM += 0.0005
    if mDM > 1.10:
        step_size = 0.005

with open("txt_files/K0K0.txt", "w") as txtfile:
    for i in range(0, len(xSM)):
        txtfile.write("%s , %s\n" % (xSM[i], y_sigmaSM00[i]))
    txtfile.close()

with open("txt_files/KpKm.txt", "w") as txtfile:
    for i in range(0, len(xSM)):
        txtfile.write("%s , %s\n" % (xSM[i], y_sigmaSMpm[i]))
    txtfile.close()

plt.plot(xSM, y_sigmaSMpm, color="blue", label="SM")
plt.plot(xDP, y_sigmaDPpm, color="red", label="DP")
plt.plot(xBL, y_sigmaBLpm, color="green", label="BL")
plt.yscale("log")
plt.xlabel("$\\sqrt{s}$/GeV")
plt.ylabel("$\\sigma$/nb")
plt.xlim(2.0 * FK.mK0, 1.1)
plt.ylim(1.0, 3.0e3)
plt.minorticks_on()
plt.grid(which="minor", alpha=0.2)
plt.grid(which="major", alpha=0.5)
plt.title("$K^{+}K^{-}$ final state")
plt.legend()
plt.savefig("plots/KpKm-phi.pdf")
plt.clf()
plt.cla()
plt.close()


plt.plot(xSM, y_sigmaSM00, color="blue", label="SM")
plt.plot(xDP, y_sigmaDP00, color="red", label="DP")
plt.plot(xBL, y_sigmaBL00, color="green", label="BL")
plt.yscale("log")
plt.xlabel("$\\sqrt{s}$/GeV")
plt.ylabel("$\\sigma$/nb")
# plt.xlim(2.*FK.mK0,1.1)
plt.xlim(low_lim, 1.1)
plt.ylim(1, 3.0e3)
plt.minorticks_on()
plt.grid(which="minor", alpha=0.2)
plt.grid(which="major", alpha=0.5)
plt.title("$K^{0}K^{0}$ final state")
plt.legend()
plt.savefig("plots/K0K0-phi.pdf")
plt.clf()
plt.cla()
plt.close()

plt.xlim(1.1, 2.0)
plt.semilogy(xSM, y_sigmaSM00, color="blue", label="SM")
plt.semilogy(xDP, y_sigmaDP00, color="red", label="DP")
plt.semilogy(xBL, y_sigmaBL00, color="green", label="BL")
plt.xlabel("$\\sqrt{s}$/GeV")
plt.ylabel("$\\sigma$/nb")
plt.yscale("log")
plt.minorticks_on()
plt.grid(which="minor", alpha=0.2)
plt.grid(which="major", alpha=0.5)
plt.title("$K^{0}K^{0}$ final state")
plt.legend()
plt.savefig("plots/K0K0-cont.pdf")
plt.clf()
plt.cla()
plt.close()

plt.xlim(1.1, 2.0)
plt.ylim(1e-3, 100.0)
plt.semilogy(xSM, y_sigmaSMpm, color="blue", label="SM")
plt.semilogy(xDP, y_sigmaDPpm, color="red", label="DP")
plt.semilogy(xBL, y_sigmaBLpm, color="green", label="BL")
plt.xlabel("$\\sqrt{s}$/GeV")
plt.ylabel("$\\sigma$/nb")
plt.yscale("log")
plt.minorticks_on()
plt.grid(which="minor", alpha=0.2)
plt.grid(which="major", alpha=0.5)
plt.title("$K^{+}K^{-}$ final state")
plt.legend()
plt.savefig("plots/KpKm-cont.pdf")
plt.clf()
plt.cla()
plt.close()
