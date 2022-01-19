# Library to load
import scipy.integrate as integrate
import Resonance, alpha, cmath, math
from scipy.interpolate import interp1d
import numpy as np


# masses and width from PDG
mKp = 0.493677  # charged Kaon
mK0 = 0.497648  # neutral Kaon
mpi0 = 0.1349766  # neutral pion
mpip = 0.13957018  # charged pion
mKS = 0.8956  # KStar mass
gKS = 0.047  # KStar width

#############################################
# own parametrization, see arXiv:1911.11147 #
#############################################

# masses and amplitudes, I=0, phi resonances
isoScalarMasses = [
    1019.461 * Resonance.MeV,
    1633.4 * Resonance.MeV,
    1957 * Resonance.MeV,
]
isoScalarWidths = [4.249 * Resonance.MeV, 218 * Resonance.MeV, 267 * Resonance.MeV]
isoScalarAmp = [0.0, 0.233, 0.0405]
isoScalarPhase = [0, 1.1e-07, 5.19]
# masses and amplitudes, I=1
isoVectorMasses = [
    775.26 * Resonance.MeV,
    1470 * Resonance.MeV,
    1720 * Resonance.MeV,
]  # ,1900*Resonance.MeV]
isoVectorWidths = [
    149.1 * Resonance.MeV,
    400 * Resonance.MeV,
    250 * Resonance.MeV,
]  # ,100*Resonance.MeV]
isoVectorAmp = [-2.34, 0.594, -0.0179]
isoVectorPhase = [0, 0.317, 2.57]
# K* K pi coupling
# g2=math.sqrt(6.*math.pi*mKS**2/(0.5*mKS*Resonance.beta(mKS**2,mKp,mpip))**3*gKS)
g2 = 5.37392360229
# masses for the integrals
M = 0.0
m1 = 0.0
m2 = 0.0
m3 = 0.0

# coupling strength to rho, phi contributions
cI1_ = 1.0
cS_ = 1.0
cI0_ = 0.0

hadronic_interpolator_0 = None
hadronic_interpolator_1 = None
hadronic_interpolator_2 = None


def resetParameters(gDM, mDM, mMed, wMed, cMedu, cMedd, cMeds):
    global gDM_, mDM_, mMed_, wMed_, cI1_, cI0_, cS_
    cI1_ = cMedu - cMedd
    cI0_ = 3 * (cMedu + cMedd)
    cS_ = -3 * cMeds
    gDM_ = gDM
    mDM_ = mDM
    mMed_ = mMed
    wMed_ = wMed
    readHadronic_Current()


###############################
### Phase Space Calculation ###
###############################
# To check form of integral see eg. PDG eq. (49.22)
# in Kinematics review of https://pdg.lbl.gov/2021/reviews/contents_sports.html
# check also eq. (139) of low_energy.pdf notes

# The tricky part is that KKpi consists of several contribution,
# e.g. K+K0pi- could come from K+(K*- -> K0 pi-) or K0 (K*0 -> K+ pi-)
# Hence, we cannot simply integrate over the phase space due to the intermediate K* structure,
# and we have to consider interference terms between both contributions

# low integrand limit on m_23^2
def m23_2_low(m12_2):
    m12 = math.sqrt(m12_2)
    E2s = 0.5 * (m12_2 - m1 ** 2 + m2 ** 2) / m12
    E3s = 0.5 * (M ** 2 - m12_2 - m3 ** 2) / m12
    return (E2s + E3s) ** 2 - (
        math.sqrt(E2s ** 2 - m2 ** 2) + math.sqrt(E3s ** 2 - m3 ** 2)
    ) ** 2


# upper integrand limit on m_23^2
def m23_2_upp(m12_2):
    m12 = math.sqrt(m12_2)
    E2s = 0.5 * (m12_2 - m1 ** 2 + m2 ** 2) / m12
    E3s = 0.5 * (M ** 2 - m12_2 - m3 ** 2) / m12
    return (E2s + E3s) ** 2 - (
        math.sqrt(E2s ** 2 - m2 ** 2) - math.sqrt(E3s ** 2 - m3 ** 2)
    ) ** 2


# the square of the numerator piece times the phase-space prefactors, see PDG (49.22)
# the numerator is the contracted Levi-Civita tensor as in eq. (138) of the low_energy.pdf notes
# it will be useful for the interference term where we cannot simply integrate over one m_ij^2
def phase_space(m212, m223):
    num = 0.25 * (
        -(M ** 4 * m2 ** 2)
        - m1 ** 4 * m3 ** 2
        - m223 * (m212 * (m212 + m223 - m3 ** 2) + m2 ** 2 * (-m212 + m3 ** 2))
        + m1 ** 2
        * (
            m3 ** 2 * (m223 - m3 ** 2)
            + m2 ** 2 * (-m212 + m3 ** 2)
            + m212 * (m223 + m3 ** 2)
        )
        + M ** 2
        * (
            -(m2 ** 4)
            + m212 * (m223 - m3 ** 2)
            + m1 ** 2 * (m2 ** 2 - m223 + m3 ** 2)
            + m2 ** 2 * (m212 + m223 + m3 ** 2)
        )
    )
    pre = 1.0 / (2.0 * math.pi) ** 3 / 32.0 / M ** 3
    return pre * num


# full integrand for the phase space of interference term
# real part
def dsigma_int_re(m212, m223):
    # Resonance for K* contribution
    out = (
        phase_space(m212, m223)
        * Resonance.BreitWignerPWave(m212, mKS, gKS, m1, m2)
        / mKS ** 2
        * Resonance.BreitWignerPWave(m223, mKS, gKS, m2, m3).conjugate()
        / mKS ** 2
    )
    return out.real


# imaginary part
def dsigma_int_im(m212, m223):
    # phase space and resonance for K* contribution
    out = (
        phase_space(m212, m223)
        * Resonance.BreitWignerPWave(m212, mKS, gKS, m1, m2)
        / mKS ** 2
        * Resonance.BreitWignerPWave(m223, mKS, gKS, m2, m3).conjugate()
        / mKS ** 2
    )
    return out.imag


# analytic integral of dsigma_12 over m23
def I1(m122, m1, m2, m3):
    m12 = math.sqrt(m122)
    # see for example eq. (139) in low_energy.pdf notes
    num = (
        8.0
        / 3.0
        / m12
        * M ** 3
        * (0.5 * M * Resonance.beta(M ** 2, m12, m3)) ** 3
        * (0.5 * m12 * Resonance.beta(m122, m1, m2)) ** 3
    )
    num *= abs(Resonance.BreitWignerPWave(m122, mKS, gKS, m1, m2) / mKS ** 2) ** 2
    return 1.0 / (2.0 * math.pi) ** 3 / 32.0 / M ** 3 * num


# Calculate phase space integral
def calculateIntegrals(shat, imode):
    global M, m1, m2, m3
    M = math.sqrt(shat)
    # K0K0pi0
    if imode == 0:
        m1 = mK0
        m2 = mpi0
        m3 = mK0
    # K+ K- pi0
    elif imode == 1:
        m1 = mKp
        m2 = mpi0
        m3 = mKp
    # K+pi-K0
    elif imode == 2:
        m1 = mK0
        m2 = mpip
        m3 = mKp
    # phase space integrals
    # non-interfering terms (with one analytic integral)
    I12 = integrate.quad(I1, (m1 + m2) ** 2, (M - m3) ** 2, args=(m1, m2, m3))[0]
    I23 = integrate.quad(I1, (m1 + m2) ** 2, (M - m3) ** 2, args=(m3, m2, m1))[0]
    # interference terms
    Iint_re = integrate.dblquad(
        dsigma_int_re, (m1 + m2) ** 2, (M - m3) ** 2, m23_2_low, m23_2_upp
    )[0]
    Iint_im = integrate.dblquad(
        dsigma_int_im, (m1 + m2) ** 2, (M - m3) ** 2, m23_2_low, m23_2_upp
    )[0]
    Iint = complex(Iint_re, Iint_im)
    return [I12, I23, Iint]


# add up all isospin 0/1 components
def isoSpinAmplitudes(shat):
    # I=0, phi contribution
    A0 = 0.0
    for ix in range(0, len(isoScalarMasses)):
        A0 += (
            cS_
            * isoScalarAmp[ix]
            * cmath.exp(complex(0.0, isoScalarPhase[ix]))
            * Resonance.BreitWignerFW(shat, isoScalarMasses[ix], isoScalarWidths[ix])
        )
    # I=1, rho contribution
    A1 = 0.0
    for ix in range(0, len(isoVectorMasses)):
        A1 += (
            cI1_
            * isoVectorAmp[ix]
            * cmath.exp(complex(0.0, isoVectorPhase[ix]))
            * Resonance.BreitWignerFW(shat, isoVectorMasses[ix], isoVectorWidths[ix])
        )
    return (A0, A1)


######################
### CROSS SECTIONS ###
######################

# cross section for subprocess K* K,
# either isospin 0 or 1 can be chosen
def sigmaKK(sHat, isospin):
    ecms = math.sqrt(sHat)
    A0, A1 = isoSpinAmplitudes(sHat)
    pre = 0.0
    if isospin == 0:
        pre = A0
    elif isospin == 1:
        pre = A1
    pcm = 0.5 * ecms * Resonance.beta(sHat, mKS, mKp)
    # phase-space, |me|^2 factors
    output = 2.0 * pcm ** 3 / 8.0 / math.pi / ecms * abs(pre) ** 2 / 3
    # initial-state factors
    output *= 32.0 * math.pi ** 2 / sHat * alpha.alphaEM(sHat) ** 2
    return output * Resonance.gev2nb


# cross section for KKpi process, phase space integral is input
def sigmaKKPi(shat, imode, I):
    # prefactor of leptonic current
    pre = 16.0 * math.pi ** 2 * alpha.alphaEM(shat) ** 2 / 3.0 / shat
    # the rest is the hadronic current
    pre *= 4.0 * g2 ** 2 / math.sqrt(shat)
    # amplitudes
    A0, A1 = isoSpinAmplitudes(shat)
    amp_12 = 0.0
    amp_23 = 0.0
    # Used A0, A1 relations like in 1010.4180, although irrelevant since I1_amp's take care of sign
    # K_L K_S pi0
    if imode == 0:
        # amp_12  = 1./math.sqrt(6.)*(A0-A1)
        amp_12 = 1.0 / math.sqrt(6.0) * (A0 + A1)
        amp_23 = amp_12
    # K+ K- pi0
    elif imode == 1:
        # amp_12  = 1./math.sqrt(6.)*(A0+A1)
        amp_12 = 1.0 / math.sqrt(6.0) * (A0 - A1)
        amp_23 = amp_12
    # K+pi-K0
    elif imode == 2:
        # factor 2 as two charged modes contribute
        pre *= 2
        amp_12 = 1.0 / math.sqrt(6.0) * (A0 + A1)
        amp_23 = 1.0 / math.sqrt(6.0) * (A0 - A1)
    # put everything together
    Itotal = (
        I[0] * abs(amp_12) ** 2
        + I[1] * abs(amp_23) ** 2
        + 2.0 * (I[2] * amp_12 * amp_23.conjugate()).real
    )
    return pre * Itotal * Resonance.gev2nb


################################################
### cross sections with precalculated values ###
################################################

# use precalculated values to speed up calculation
def readHadronic_Current():
    for imode in range(0, 3):
        [energies, integral_values] = np.load(
            "KKpi/KKpi_coefficients_%d.npy" % imode, allow_pickle=True
        )
        integrals = {}
        for xen in range(0, len(energies)):
            integrals[energies[xen]] = integral_values[xen]
        x = []
        y = []
        for energy in energies:
            x.append(energy)
            s = energy ** 2
            I = integrals[energy]
            pre = 4.0 * g2 ** 2
            # amplitudes
            A0, A1 = isoSpinAmplitudes(s)
            amp_12 = 0.0
            amp_23 = 0.0
            # Used A0, A1 relations like in 1010.4180, although irrelevant since I1_amp's take care of sign
            # K_L K_S pi0
            if imode == 0:
                # amp_12  = 1./math.sqrt(6.)*(A0-A1)
                amp_12 = 1.0 / math.sqrt(6.0) * (A0 + A1)
                amp_23 = amp_12
            # K+ K- pi0
            elif imode == 1:
                # amp_12  = 1./math.sqrt(6.)*(A0+A1)
                amp_12 = 1.0 / math.sqrt(6.0) * (A0 - A1)
                amp_23 = amp_12
            # K+pi-K0
            elif imode == 2:
                # as two charge modes
                pre *= 2
                amp_12 = 1.0 / math.sqrt(6.0) * (A0 + A1)
                amp_23 = 1.0 / math.sqrt(6.0) * (A0 - A1)
            # put everything together
            Itotal = (
                I[0] * abs(amp_12) ** 2
                + I[1] * abs(amp_23) ** 2
                + 2.0 * (I[2] * amp_12 * amp_23.conjugate()).real
            )
            y.append(pre * Itotal)
        global hadronic_interpolator_0, hadronic_interpolator_1, hadronic_interpolator_2
        if imode == 0:
            hadronic_interpolator_0 = interp1d(
                x, y, kind="cubic", fill_value="extrapolate"
            )
        if imode == 1:
            hadronic_interpolator_1 = interp1d(
                x, y, kind="cubic", fill_value="extrapolate"
            )
        if imode == 2:
            hadronic_interpolator_2 = interp1d(
                x, y, kind="cubic", fill_value="extrapolate"
            )


# SM cross section, e+e- -> KKpi
def sigmaSM(s, imode):
    if s <= (2 * mK0 + mpip) ** 2:
        return 0
    en = math.sqrt(s)
    # leptonic initial current contracted
    pre = 16.0 * math.pi ** 2 * alpha.alphaEM(s) ** 2 / 3.0 / s
    pre *= 1.0 / math.sqrt(s)
    if imode == 0:
        had = abs(hadronic_interpolator_0(en))
    if imode == 1:
        had = abs(hadronic_interpolator_1(en))
    if imode == 2:
        had = abs(hadronic_interpolator_2(en))
    return pre * had * Resonance.gev2nb


# DM DM -> KK pi
def sigmaDM(s, imode):
    if s <= (2 * mK0 + mpip) ** 2:
        return 0
    en = math.sqrt(s)
    # Dark prefactor
    cDM = gDM_
    DMmed = cDM / (s - mMed_ ** 2 + complex(0.0, 1.0) * mMed_ * wMed_)
    DMmed2 = abs(DMmed) ** 2
    pre = DMmed2 * s * (1 + 2 * mDM_ ** 2 / s) / 3.0
    # phase space
    pre *= 1.0 / math.sqrt(s)
    if imode == 0:
        had = abs(hadronic_interpolator_0(en))
    if imode == 1:
        had = abs(hadronic_interpolator_1(en))
    if imode == 2:
        had = abs(hadronic_interpolator_2(en))
    return pre * had * Resonance.gev2nb


# Mediator -> KK pi
def GammaDM(mMed, imode):
    if mMed ** 2 <= (2 * mK0 + mpip) ** 2:
        return 0
    # vector spin average
    pre = 1 / 3.0
    # phase space
    pre *= 1.0
    if imode == 0:
        had = abs(hadronic_interpolator_0(mMed))
    if imode == 1:
        had = abs(hadronic_interpolator_1(mMed))
    if imode == 2:
        had = abs(hadronic_interpolator_2(mMed))
    return pre * had


##########################################################
# cross sections where we don't use precalculated values #
##########################################################

# # cross-section for several modes, mode 0: KSKLpi0, mode 1: KpKmpi0 , mode 2: Kp pi- K0/Km pi+ K0
# def sigmaSM(shat,imode) :
#     I = calculateIntegrals(shat,imode)
#     return sigmaKKPi(shat,imode,I)

# def GammaDM(mMed, imode) :
#     if mMed**2<=(2*mK0+mpip)**2: return 0
#     I = calculateIntegrals(mMed**2,imode)
#     # vector spin average
#     pre = 1/3.
#     #phase space
#     pre *= (4.*g2**2/mMed)*mMed
#     # amplitudes
#     A0,A1 = isoSpinAmplitudes(mMed**2)
#     amp_12  = 0.
#     amp_23  = 0.
#     # Used A0, A1 relations like in 1010.4180, although irrelevant since I1_amp's take care of sign
#     # K_L K_S pi0
#     if(imode==0) :
#         #amp_12  = 1./math.sqrt(6.)*(A0-A1)
#         amp_12  = 1./math.sqrt(6.)*(A0+A1)
#         amp_23  = amp_12
#     # K+ K- pi0
#     elif(imode==1) :
#         #amp_12  = 1./math.sqrt(6.)*(A0+A1)
#         amp_12  = 1./math.sqrt(6.)*(A0-A1)
#         amp_23  = amp_12
#     # K+pi-K0
#     elif(imode==2) :
#         # as two charge modes
#         pre *=2
#         amp_12 = 1./math.sqrt(6.)*(A0+A1)
#         amp_23 = 1./math.sqrt(6.)*(A0-A1)
#     # put everything together
#     Itotal = I[0]*abs(amp_12)**2+I[1]*abs(amp_23)**2+2.*(I[2]*amp_12*amp_23.conjugate()).real
#     return pre*Itotal*Resonance.gev2nb


# def sigmaDM(shat,imode) :
#     I = calculateIntegrals(shat,imode)
#     return sigmaKKPiDM(shat,imode,I)

# def sigmaKKPiDM(shat,imode,I) :
#     # Dark prefactor
#     cDM = gDM_
#     DMmed = cDM/(shat-mMed_**2+complex(0.,1.)*mMed_*wMed_)
#     DMmed2 = abs(DMmed)**2
#     pre = DMmed2*shat*(1+2*mDM_**2/shat)/3.
#     #phase space
#     pre *= 4.*g2**2/math.sqrt(shat)
#     # amplitudes
#     A0,A1 = isoSpinAmplitudes(shat)
#     amp_12  = 0.
#     amp_23  = 0.
#     # Used A0, A1 relations like in 1010.4180, although irrelevant since I1_amp's take care of sign
#     # K_L K_S pi0
#     if(imode==0) :
#         #amp_12  = 1./math.sqrt(6.)*(A0-A1)
#         amp_12  = 1./math.sqrt(6.)*(A0+A1)
#         amp_23  = amp_12
#     # K+ K- pi0
#     elif(imode==1) :
#         #amp_12  = 1./math.sqrt(6.)*(A0+A1)
#         amp_12  = 1./math.sqrt(6.)*(A0-A1)
#         amp_23  = amp_12
#     # K+pi-K0
#     elif(imode==2) :
#         # as two charge modes
#         pre *=2
#         amp_12 = 1./math.sqrt(6.)*(A0+A1)
#         amp_23 = 1./math.sqrt(6.)*(A0-A1)
#     # put everything together
#     Itotal = I[0]*abs(amp_12)**2+I[1]*abs(amp_23)**2+2.*(I[2]*amp_12*amp_23.conjugate()).real
#     return pre*Itotal*Resonance.gev2nb
