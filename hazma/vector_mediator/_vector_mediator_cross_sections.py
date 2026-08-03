from hazma.parameters import charged_pion_mass as mpi
from hazma.parameters import neutral_pion_mass as mpi0
from hazma.parameters import muon_mass as mmu
from hazma.parameters import electron_mass as me
from hazma.parameters import fpi, qe
from scipy.integrate import quad

from hazma.vector_mediator._c_vector_mediator_cross_sections import (
    sigma_xx_to_v_to_ff as sig_ff,
)
from hazma.vector_mediator._c_vector_mediator_cross_sections import (
    sigma_xx_to_v_to_pipi as sig_pipi,
)
from hazma.vector_mediator._c_vector_mediator_cross_sections import (
    sigma_xx_to_v_to_pi0g as sig_pi0g,
)
from hazma.vector_mediator._c_vector_mediator_cross_sections import (
    sigma_xx_to_v_to_pi0v as sig_pi0v,
)
from hazma.vector_mediator._c_vector_mediator_cross_sections import (
    sigma_xx_to_vv as sig_vv,
)

from hazma.vector_mediator._c_vector_mediator_cross_sections import (
    thermal_cross_section as tcs,
)

from numpy.polynomial.legendre import leggauss
import warnings
import numpy as np


class VectorMediatorCrossSections:
    def sigma_xx_to_v_to_ff(self, e_cm, f):
        """
        Returns the cross section for xbar x to fbar f.

        Parameters
        ----------
        e_cm : float
            Center of mass energy.
        f : float
            Name of final state fermion: "e" or "mu".
        self : object
            Class containing the vector mediator parameters.

        Returns
        -------
        cross_section : float
            Cross section for xbar + x -> v -> fbar + f.
        """
        assert f == "e" or f == "mu"
        ml = me if f == "e" else mmu
        gvll = self.gvee if f == "e" else self.gvmumu

        return sig_ff(e_cm, self.mx, self.mv, self.gvxx, gvll, self.width_v, ml)

    def sigma_xx_to_v_to_pipi(self, e_cm):
        """
        Returns the cross section for xbar x to pi+ pi-.

        Parameters
        ----------
        e_cm : float
            Center of mass energy.
        self : object
            Class containing the vector mediator parameters.

        Returns
        -------
        cross_section : float
            Cross section for xbar + x -> v -> f + f.
        """
        return sig_pipi(
            e_cm,
            self.mx,
            self.mv,
            self.gvxx,
            self.gvuu,
            self.gvdd,
            self.gvss,
            self.gvee,
            self.gvmumu,
            self.width_v,
        )

    def sigma_xx_to_v_to_pi0g(self, e_cm):
        """
        Returns the cross section for xbar x to pi0 g.

        Parameters
        ----------
        e_cm : float
            Center of mass energy.
        self : object
            Class containing the vector mediator parameters.

        Returns
        -------
        cross_section : float
            Cross section for xbar + x -> v -> pi0 g
        """
        return sig_pi0g(
            e_cm,
            self.mx,
            self.mv,
            self.gvxx,
            self.gvuu,
            self.gvdd,
            self.gvss,
            self.gvee,
            self.gvmumu,
            self.width_v,
        )

    def sigma_xx_to_v_to_pi0v(self, e_cm):
        """
        Returns the cross section for xbar x to pi0 v.

        Parameters
        ----------
        e_cm : float
            Center of mass energy.
        self : object
            Class containing the vector mediator parameters.

        Returns
        -------
        cross_section : float
            Cross section for xbar + x -> v -> pi0 v
        """
        return sig_pi0v(
            e_cm,
            self.mx,
            self.mv,
            self.gvxx,
            self.gvuu,
            self.gvdd,
            self.gvss,
            self.gvee,
            self.gvmumu,
            self.width_v,
        )

    def sigma_xx_to_vv(self, e_cm):
        """
        Returns the cross section for xbar x to v v.

        Parameters
        ----------
        e_cm : float or array-like
            Center of mass energy(ies).
        self : object
            Class containing the vector mediator parameters.

        Returns
        -------
        cross_section : float or array-like
            Cross section for xbar + x -> v -> pi0 v
        """
        return sig_vv(
            e_cm,
            self.mx,
            self.mv,
            self.gvxx,
            self.gvuu,
            self.gvdd,
            self.gvss,
            self.gvee,
            self.gvmumu,
            self.width_v,
        )

    def sigma_xx_to_v_to_xx(self, e_cm):
        """
        Returns the DM annihilation cross section into DM.
        """
        # see sigma_xx_to_s_to_ff for explaination of this context mangager
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", r"invalid value encountered in true_divide"
            )
            warnings.filterwarnings(
                "ignore", r"divide by zero encountered in true_divide"
            )
            warnings.filterwarnings("ignore", r"invalid value encountered in multiply")
            warnings.filterwarnings("ignore", r"invalid value encountered in sqrt")
            warnings.filterwarnings("ignore", r"invalid value encountered in power")
            warnings.filterwarnings("ignore", r"invalid value encountered in subtract")
            warnings.filterwarnings("ignore", r"invalid value encountered in add")

            e_cms = np.array(e_cm) if hasattr(e_cm, "__len__") else e_cm
            mask = e_cms > 2.0 * self.mx

            rv = self.mv / e_cms
            rx = self.mx / e_cms
            gvxx = self.gvxx
            rwv = self.width_v / e_cms

            def msqrd(z):
                return (
                    gvxx**4
                    * (
                        9
                        - 56 * rx**2
                        + 2
                        * (
                            9 * rv**2 * (-1 + rv**2 + rwv**2)
                            + 4 * rv**2 * (11 + 2 * rv**2 + 2 * rwv**2) * rx**2
                            + 8 * (7 + 6 * rv**2 * (-4 + rv**2 + rwv**2)) * rx**4
                            + 64 * rx**6
                        )
                        - 2
                        * (-1 + 4 * rx**2)
                        * (
                            3 * rv**2 * (-3 + 2 * rv**2 + 2 * rwv**2)
                            + 4
                            * (3 + 6 * rv**4 + 2 * rv**2 * (-7 + 3 * rwv**2))
                            * rx**2
                            - 32 * (-2 + rv**2) * rx**4
                        )
                        * z
                        + 2 * (3 * rv**2 - 4 * rx**2) * (-1 + 4 * rx**2) ** 3 * z**3
                        + (1 - 4 * rx**2) ** 4 * z**4
                        + 2
                        * (
                            3
                            + 5 * rv**4
                            + 12 * rx**2
                            + 8 * rx**4
                            + rv**2 * (-3 + 5 * rwv**2 - 20 * rx**2)
                        )
                        * (z - 4 * rx**2 * z) ** 2
                    )
                ) / (
                    (1 + rv**4 + rv**2 * (-2 + rwv**2))
                    * (
                        4 * rv**4
                        + 4 * rv**2 * (rwv**2 + (-1 + 4 * rx**2) * (-1 + z))
                        + (1 - 4 * rx**2) ** 2 * (-1 + z) ** 2
                    )
                )

            # Compute the nodes and weights for Legendre-Gauss quadrature
            nodes, weights = leggauss(10)
            nodes, weights = (np.reshape(nodes, (10, 1)), np.reshape(weights, (10, 1)))
            ret_val = mask * np.nan_to_num(
                np.sum(weights * msqrd(nodes), axis=0) / (32.0 * np.pi * e_cms)
            )

        return ret_val.real

    def dsigma_ds_xx_to_v_to_pi0pipi(self, s, e_cm):
        mx = self.mx

        if (
            e_cm > 2.0 * mpi + mpi0
            and e_cm > 2.0 * mx
            and 4.0 * mpi**2 < s < (e_cm - mpi0) ** 2
        ):
            gvuu = self.gvuu
            gvdd = self.gvdd
            gvxx = self.gvxx
            mv = self.mv
            width_v = self.width_v

            ret_val = (
                3.0
                * (
                    (gvdd + gvuu) ** 2
                    * gvxx**2
                    * np.sqrt(s * (-4.0 * mpi**2 + s))
                    * np.sqrt(
                        e_cm**4 + (mpi0**2 - s) ** 2 - 2.0 * e_cm**2 * (mpi0**2 + s)
                    )
                    * (
                        -24.0 * mpi**6 * s
                        + mpi**4 * (-2.0 * mpi0**4 + 28.0 * mpi0**2 * s + 22.0 * s**2)
                        + 2.0 * mpi**2 * (mpi0**6 - 4.0 * s**3)
                        + s
                        * (-2.0 * mpi0**6 - 4.0 * mpi0**4 * s - mpi0**2 * s**2 + s**3)
                        + e_cm**4
                        * (
                            -2.0 * mpi**4
                            + 2.0 * mpi**2 * (mpi0**2 - s)
                            + s * (-2.0 * mpi0**2 + s)
                        )
                        + e_cm**2
                        * (
                            4.0 * mpi**4 * (mpi0**2 + s)
                            + s * (4.0 * mpi0**4 + 5.0 * mpi0**2 * s - 2.0 * s**2)
                            - 4.0 * mpi**2 * (mpi0**4 + 3.0 * mpi0**2 * s - s**2)
                        )
                    )
                )
                / (
                    294912.0
                    * fpi**6
                    * np.pi**7
                    * np.sqrt(e_cm**2)
                    * np.sqrt(-4.0 * mx**2 + e_cm**2)
                    * s**2
                    * (mv**4 - 2.0 * mv**2 * e_cm**2 + e_cm**4 + mv**2 * width_v**2)
                )
            )

            assert ret_val.imag == 0
            assert ret_val.real >= 0

            return ret_val.real
        else:
            return 0.0

    def sigma_xx_to_v_to_pi0pipi(self, e_cm):
        if e_cm > 2.0 * mpi + mpi0 and e_cm > 2.0 * self.mx:
            s_min = 4.0 * mpi**2
            s_max = (e_cm - mpi0) ** 2

            ret_val = quad(
                self.dsigma_ds_xx_to_v_to_pi0pipi, s_min, s_max, args=(e_cm)
            )[0]

            assert ret_val.imag == 0
            assert ret_val.real >= 0

            return ret_val.real
        else:
            return 0.0

    def annihilation_cross_section_funcs(self):
        return {
            "mu mu": lambda e_cm: self.sigma_xx_to_v_to_ff(e_cm, "mu"),
            "e e": lambda e_cm: self.sigma_xx_to_v_to_ff(e_cm, "e"),
            "pi pi": self.sigma_xx_to_v_to_pipi,
            "pi0 g": self.sigma_xx_to_v_to_pi0g,
            "pi0 v": self.sigma_xx_to_v_to_pi0v,
            "v v": self.sigma_xx_to_vv,
        }

    def thermal_cross_section(self, x):
        """
        Compute the thermally average cross section for vector mediator
        model.

        Parameters
        ----------
        x: float
            Mass of the dark matter divided by its temperature.

        Returns
        -------
        tcs: float
            Thermally average cross section.
        """
        return tcs(
            x,
            self.mx,
            self.mv,
            self.gvxx,
            self.gvuu,
            self.gvdd,
            self.gvss,
            self.gvee,
            self.gvmumu,
            self.width_v,
        )
