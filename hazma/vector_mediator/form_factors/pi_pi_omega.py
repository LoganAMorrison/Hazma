from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad

from hazma import parameters
from hazma.utils import kallen_lambda
from hazma.vector_mediator.form_factors.utils import MOMEGA_GEV, MPI_GEV, RealArray


@dataclass
class FormFactorPiPiOmega:

    masses: RealArray = np.array([0.783, 1.420, 1.6608543573197])
    widths: RealArray = np.array([0.00849, 0.315, 0.3982595005228462])
    amps: RealArray = np.array([0.0, 0.0, 2.728870588760009])
    phases: RealArray = np.array([0.0, np.pi, 0.0])

    def __form_factor(self, cme, gvuu, gvdd):
        """
        Compute the form factor for a vector decaying into two charged pions and
        an eta-prime.
        """
        ci0 = 3 * (gvuu + gvdd)
        return ci0 * np.sum(
            self.amps
            * np.exp(1j * self.phases)
            * self.masses**2
            / ((self.masses**2 - cme**2) - 1j * cme * self.widths)
        )

    def __integrated_form_factor(
        self, *, cme: float, gvuu: float, gvdd: float, imode
    ) -> float:
        """
        Compute the form factor for a vector decaying into two charged pions and
        an eta-prime integrated over the three-body phase-space.
        """
        mup = MPI_GEV / cme
        muo = MOMEGA_GEV / cme
        jac = cme**2 / (1536.0 * np.pi**3 * muo**2)
        f2 = np.abs(self.__form_factor(cme, gvuu, gvdd)) ** 2

        def integrand(z):
            k1 = kallen_lambda(z, mup**2, mup**2)
            k2 = kallen_lambda(1, z, muo**2)
            p = (1 + 10 * muo**2 + muo**4 - 2 * (1 + muo**2) * z + z**2) / z
            return p * np.sqrt(k1 * k2) * f2

        lb = (2 * mup) ** 2
        ub = (1.0 - muo) ** 2
        res = jac * quad(integrand, lb, ub)[0]

        if imode == 0:
            return res / 2.0
        return res

    def form_factor(self, *, cme: float, gvuu: float, gvdd: float, imode) -> float:
        """
        Compute the form factor for a vector decaying into two charged pions and
        an eta-prime integrated over the three-body phase-space.
        """
        cme_gev = cme * 1e-3
        integral = self.__integrated_form_factor(
            cme=cme_gev, gvuu=gvuu, gvdd=gvdd, imode=imode
        )
        return integral * 1e6 / cme**2

    def width(self, *, mv: float, gvuu: float, gvdd: float, imode: int) -> float:
        if mv < 2 * parameters.charged_pion_mass + parameters.omega_mass:
            return 0.0
        ff = self.form_factor(cme=mv, gvuu=gvuu, gvdd=gvdd, imode=imode)
        return ff * mv / 2

    def energy_distributions(self, *, cme, imode, nbins: int = 25):
        if imode == 1:
            mpi = parameters.charged_pion_mass
        elif imode == 0:
            mpi = parameters.neutral_pion_mass
        else:
            raise ValueError(f"Invalid imode={imode}. Must be 0 or 1.")

        mw = parameters.omega_mass
        mup = mpi / cme
        muw = mw / cme

        bounds_p = mup, 0.5 * (1.0 + mup**2 - (mup + muw) ** 2)
        bounds_w = muw, 0.5 * (1.0 - 4 * mup**2 + muw**2)

        bs_p = cme * np.linspace(bounds_p[0], bounds_p[1], nbins + 1)
        bs_w = cme * np.linspace(bounds_w[0], bounds_w[1], nbins + 1)
        es_p = 0.5 * (bs_p[1:] + bs_p[:-1])
        es_w = 0.5 * (bs_w[1:] + bs_w[:-1])

        if cme < 2 * mpi + mw:
            return [
                (es_p, np.zeros_like(es_p)),
                (es_p, np.zeros_like(es_p)),
                (es_w, np.zeros_like(es_w)),
            ]

        def dnde_pi(e):
            z = 1 - 2 * e / cme + mup**2
            p1 = kallen_lambda(z, mup**2, 1.0)
            p2 = kallen_lambda(z, mup**2, muw**2)
            f = (
                (z - mup**2) ** 2 * (1 + z + z**2 - 2 * (1 + z) * mup**2 + mup**4)
                + (
                    z * (1 + z * (28 + z))
                    - 2 * (1 + 2 * z * (1 + z)) * mup**2
                    + (4 + 5 * z) * mup**4
                    - 2 * mup**6
                )
                * muw**2
                + (1 + z + z**2 - 2 * (1 + z) * mup**2 + mup**4) * muw**4
            ) / (18.0 * z**3 * muw**2)

            return np.sqrt(p1 * p2) * f

        def dnde_omega(e):
            z = 1 - 2 * e / cme + muw**2

            p1 = kallen_lambda(z, mup**2, mup**2)
            p2 = kallen_lambda(z, muw**2, 1.0)
            f = ((-1 + z) ** 2 - 2 * (-5 + z) * muw**2 + muw**4) / (6.0 * z * muw**2)

            return f * np.sqrt(p1 * p2)

        dist_p = dnde_pi(es_p)
        dist_w = dnde_omega(es_w)

        dist_p = dist_p / np.trapz(dist_p, x=es_p)
        dist_w = dist_w / np.trapz(dist_w, x=es_w)

        return [(es_p, dist_p), (es_p, dist_p), (es_w, dist_w)]
