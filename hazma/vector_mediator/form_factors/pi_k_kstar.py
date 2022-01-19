from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad

from hazma.utils import kallen_lambda
from hazma.vector_mediator.form_factors.utils import (
    MPI_GEV,
    MK_GEV,
    MPI0_GEV,
    MK0_GEV,
    MOMEGA_GEV,
    RealArray,
    breit_wigner_fw,
    breit_wigner_pwave,
    beta,
)


KS_MASS_GEV = 0.8956  # KStar mass
KS_WDITH_GEV = 0.047  # KStar width


@dataclass
class FormFactorPiPiOmega:

    iso_scalar_masses: RealArray = np.array([1019.461e-3, 1633.4e-3, 1957e-3])
    iso_scalar_widths: RealArray = np.array([4.249e-3, 218e-3, 267e-3])
    iso_scalar_amps: RealArray = np.array([0.0, 0.233, 0.0405])
    iso_scalar_phases: RealArray = np.array([0, 1.1e-07, 5.19])

    iso_vector_masses: RealArray = np.array([775.26e-3, 1470e-3, 1720e-3])
    iso_vector_widths: RealArray = np.array([149.1e-3, 400e-3, 250e-3])
    iso_vector_amps: RealArray = np.array([-2.34, 0.594, -0.0179])
    iso_vector_phases: RealArray = np.array([0, 0.317, 2.57])

    def __iso_spin_amplitudes(self, s, gvuu, gvdd, gvss):
        ci1 = gvuu - gvdd
        cs = -3 * gvss

        a0 = np.sum(
            cs
            * self.iso_scalar_amps
            * np.exp(1j * self.iso_scalar_phases)
            * breit_wigner_fw(s, self.iso_scalar_masses, self.iso_scalar_widths)
        )
        a1 = np.sum(
            ci1
            * self.iso_vector_amps
            * np.exp(1j * self.iso_vector_phases)
            * breit_wigner_fw(s, self.iso_vector_masses, self.iso_vector_widths)
        )
        return (a0, a1)

    def __i1(self, m122, m, mu1, mu2, mu3):
        m12 = np.sqrt(m122)
        num = (
            8.0
            / 3.0
            / m12
            * m ** 3
            * (0.5 * m * beta(m ** 2, m12, mu3)) ** 3
            * (0.5 * m12 * beta(m122, mu1, mu2)) ** 3
        )
        num *= (
            abs(
                breit_wigner_pwave(m122, KS_MASS_GEV, KS_WDITH_GEV, mu1, mu2)
                / KS_MASS_GEV ** 2
            )
            ** 2
        )
        return 1.0 / (2.0 * np.pi) ** 3 / 32.0 / m ** 3 * num

    def __phase_space(self, m212, m223, m, m1, m2, m3):
        """
        Notes
        -----
        The square of the numerator piece times the phase-space prefactors, see PDG
        (49.22) the numerator is the contracted Levi-Civita tensor as in eq. (138) of
        the low_energy.pdf notes it will be useful for the interference term where we
        cannot simply integrate over one m_ij^2
        """
        num = 0.25 * (
            -(m ** 4 * m2 ** 2)
            - m1 ** 4 * m3 ** 2
            - m223 * (m212 * (m212 + m223 - m3 ** 2) + m2 ** 2 * (-m212 + m3 ** 2))
            + m1 ** 2
            * (
                m3 ** 2 * (m223 - m3 ** 2)
                + m2 ** 2 * (-m212 + m3 ** 2)
                + m212 * (m223 + m3 ** 2)
            )
            + m ** 2
            * (
                -(m2 ** 4)
                + m212 * (m223 - m3 ** 2)
                + m1 ** 2 * (m2 ** 2 - m223 + m3 ** 2)
                + m2 ** 2 * (m212 + m223 + m3 ** 2)
            )
        )
        pre = 1.0 / (2.0 * np.pi) ** 3 / 32.0 / m ** 3
        return pre * num

    # full integrand for the phase space of interference term
    # real part
    def dsigma_int_re(self, m212, m223, m, m1, m2, m3):
        # Resonance for K* contribution
        mKS = KS_MASS_GEV
        gKS = KS_WDITH_GEV
        out = (
            self.__phase_space(m212, m223, m, m1, m2, m3)
            * breit_wigner_pwave(m212, mKS, gKS, m1, m2)
            / mKS ** 2
            * breit_wigner_pwave(m223, mKS, gKS, m2, m3).conjugate()
            / mKS ** 2
        )
        return out.real

    # imaginary part
    def dsigma_int_im(self, m212, m223, m, m1, m2, m3):
        # phase space and resonance for K* contribution
        mKS = KS_MASS_GEV
        gKS = KS_WDITH_GEV
        out = (
            self.__phase_space(m212, m223, m, m1, m2, m3)
            * breit_wigner_pwave(m212, mKS, gKS, m1, m2)
            / mKS ** 2
            * breit_wigner_pwave(m223, mKS, gKS, m2, m3).conjugate()
            / mKS ** 2
        )
        return out.imag

    def __calculate_integrals(self, s, imode):
        m = np.sqrt(s)

        if imode == 0:
            m1, m2, m3 = MK0_GEV, MPI0_GEV, MK0_GEV
        elif imode == 1:
            m1, m2, m3 = MK_GEV, MPI0_GEV, MK_GEV
        elif imode == 2:
            m1, m2, m3 = MK0_GEV, MPI_GEV, MK_GEV
        else:
            raise ValueError(f"Invalid imode = {imode}")

        i12 = quad(
            lambda x: self.__i1(x, m, m1, m2, m3), (m1 + m2) ** 2, (m - m3) ** 2
        )[0]
        i23 = quad(
            lambda x: self.__i1(x, m, m3, m2, m1), (m1 + m2) ** 2, (m - m3) ** 2
        )[0]

    def form_factor(self, cme, gvuu, gvdd):
        """
        Compute the form factor for a vector decaying into two charged pions and
        an eta-prime.
        """
        ci0 = 3 * (gvuu + gvdd)
        return ci0 * np.sum(
            self.amps
            * np.exp(1j * self.phases)
            * self.masses ** 2
            / ((self.masses ** 2 - cme ** 2) - 1j * cme * self.widths)
        )

    def integrated_form_factor(
        self, cme: float, gvuu: float, gvdd: float, imode
    ) -> float:
        """
        Compute the form factor for a vector decaying into two charged pions and
        an eta-prime integrated over the three-body phase-space.
        """
        mup = MPI_GEV / cme
        muo = MOMEGA_GEV / cme
        jac = cme ** 2 / (1536.0 * np.pi ** 3 * muo ** 2)
        f2 = np.abs(self.form_factor(cme, gvuu, gvdd)) ** 2

        def integrand(z):
            k1 = kallen_lambda(z, mup ** 2, mup ** 2)
            k2 = kallen_lambda(1, z, muo ** 2)
            p = (1 + 10 * muo ** 2 + muo ** 4 - 2 * (1 + muo ** 2) * z + z ** 2) / z
            return p * np.sqrt(k1 * k2) * f2

        lb = (2 * mup) ** 2
        ub = (1.0 - muo) ** 2
        res = jac * quad(integrand, lb, ub)[0]

        if imode == 0:
            return res / 2.0
        return res
