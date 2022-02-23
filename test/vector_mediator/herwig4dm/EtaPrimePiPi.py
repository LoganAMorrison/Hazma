# Libraries to load
import matplotlib.pyplot as plt

from . import FEtaPrimePiPi


xSM = []
ySM = []
xDP = []
yDP = []
xBL = []
yBL = []

# SM case
scale = FEtaPrimePiPi.mEta_ + 2 * FEtaPrimePiPi.mPi_ + 0.1
while scale < 4.0:
    Q2 = scale ** 2
    xSM.append(scale)
    ySM.append(FEtaPrimePiPi.sigmaSM(Q2))
    scale += 0.01


# set DM parameters
# DM to mediator coupling
gDM = 1.0
# DM mass
mDM = 0.25  # F3pi.mPi_
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
    wMed = FEtaPrimePiPi.GammaDM(mMed)
    FEtaPrimePiPi.resetParameters(gDM, mDM, mMed, wMed, cMed_u, cMed_d, cMed_s)
    yDP.append(FEtaPrimePiPi.sigmaDM(Q2))
    mDM += 0.01

# xDPm=[]
# wDP= []
# wDP2 = []

# FEtaPrimePiPi.resetParameters(gDM,0.,0.,0.,cMed_u,cMed_d,cMed_s)
# mDM = FEtaPrimePiPi.mEta_+2*FEtaPrimePiPi.mPi_+0.1
# while mDM < 2. :
#     # energy = 2*mDM+0.0001
#     # Q2 = energy**2
#     xDPm.append(mDM)
#     yNormSM = (FEtaPrimePiPi.sigmaSM(mDM**2)/xSection2e2mu(mDM**2))
#     wMed = FEtaPrimePiPi.GammaDM(mDM)
#     wDP.append(wMed)
#     wDP2.append(GammaDMf(1.,gDM,mDM,-1,mMu_)*yNormSM)
#     mDM+=0.01

# B-L model
# couplings of mediator to quarks
cMed_u = 1.0 / 3.0
cMed_d = 1.0 / 3.0
cMed_s = 1.0 / 3.0

# reset DM mass
mDM = 0.25  # F2pi.mpi_
while mDM < 2.0:
    energy = 2 * mDM + 0.0001
    Q2 = energy ** 2
    xBL.append(energy)
    # mediator width should be replaced by a function for the full decay width
    wMed = FEtaPrimePiPi.GammaDM(mMed)
    FEtaPrimePiPi.resetParameters(gDM, mDM, mMed, wMed, cMed_u, cMed_d, cMed_s)
    yBL.append(FEtaPrimePiPi.sigmaDM(Q2))
    mDM += 0.01

plt.plot(xSM, ySM, color="blue", label="SM")
plt.plot(xDP, yDP, color="red", label="DP")
plt.plot(xBL, yBL, color="green", label="BL")
plt.legend()
plt.yscale("log")
plt.xlabel("$\\sqrt{s}$/GeV")
plt.ylabel("$\\sigma$/nb")
plt.savefig("plots/EtaPrimePiPi.pdf")
