# Libraries to load
import matplotlib.pyplot as plt

from . import FEtaOmega


# set DM parameters
# DM to mediator coupling
gDM = 1.0
# DM mass
mDM = (FEtaOmega.mEta_ + FEtaOmega.mOmega_) / 2.0
# mediator mass
mMed = 10
# mediator width, own width.py will be added with all channels
# (work in progress)
# wMed

xSM = []
ySM = []
xDP = []
yDP = []
xBL = []
yBL = []

# energy range
low_lim = FEtaOmega.mOmega_ + FEtaOmega.mEta_
upp_lim = 2.0
step_size = 0.005


energy = low_lim
while energy < upp_lim:
    xSM.append(energy)
    # resetParameters(mOmega_p,mOmega_pp,gOmega_p,gOmega_pp,a_Omega_p,a_Omega_pp,phi_Omega_p,phi_Omega_pp)
    ySM.append(FEtaOmega.sigmaSMEtaOmega(energy ** 2))
    energy += step_size

with open("txt_files/EtaOmega.txt", "w") as txtfile:
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
    wMed = FEtaOmega.GammaDM(mMed)
    FEtaOmega.resetParameters(gDM, mDM, mMed, wMed, cMed_u, cMed_d, cMed_s)
    yDP.append(FEtaOmega.sigmaDMEtaOmega(Q2))
    mDM += step_size

# couplings of mediator to quarks
cMed_u = 1.0 / 3.0
cMed_d = 1.0 / 3.0
cMed_s = 1.0 / 3.0
mDM = (FEtaOmega.mEta_ + FEtaOmega.mOmega_) / 2.0
while mDM < upp_lim:
    energy = 2 * mDM + 0.0001
    Q2 = energy ** 2
    xBL.append(energy)
    wMed = FEtaOmega.GammaDM(mMed)
    FEtaOmega.resetParameters(gDM, mDM, mMed, wMed, cMed_u, cMed_d, cMed_s)
    yBL.append(FEtaOmega.sigmaDMEtaOmega(Q2))
    mDM += step_size

plt.plot(xSM, ySM, color="blue", label="SM")
plt.plot(xDP, yDP, color="red", label="DP")
plt.plot(xBL, yBL, color="green", linestyle="--", label="BL")
plt.xlabel("$\\sqrt{s}$/GeV")
plt.ylabel("$\\sigma$/nb")
plt.xlim(FEtaOmega.mOmega_ + FEtaOmega.mEta_, 2.0)
plt.title("$e^+e^- \\to\\eta\\omega$")
plt.legend()
plt.savefig("plots/EtaOmega.pdf")
plt.clf()
plt.cla()
plt.close()
