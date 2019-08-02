from cmath import sqrt, pi, atanh
from hazma.parameters import charged_pion_mass as mpi
from hazma.parameters import neutral_pion_mass as mpi0
from hazma.parameters import muon_mass as mmu
from hazma.parameters import electron_mass as me
from hazma.parameters import fpi, qe
from scipy.integrate import quad


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
        if f == "e":
            mf = me
            gvll = self.gvee
        elif f == "mu":
            mf = mmu
            gvll = self.gvmumu

        mx = self.mx

        if e_cm >= 2.0 * mf and e_cm >= 2.0 * mx:
            gvxx = self.gvxx
            mv = self.mv
            width_v = self.width_v

            ret_val = (
                gvll ** 2
                * gvxx ** 2
                * sqrt(e_cm ** 2 - 4.0 * mf ** 2)
                * (e_cm ** 2 + 2.0 * mf ** 2)
                * (e_cm ** 2 + 2 * mx ** 2)
            ) / (
                12.0
                * e_cm ** 2
                * sqrt(e_cm ** 2 - 4.0 * mx ** 2)
                * pi
                * (
                    e_cm ** 4
                    - 2.0 * e_cm ** 2 * mv ** 2
                    + mv ** 4
                    + mv ** 2 * width_v ** 2
                )
            )

            assert ret_val.imag == 0
            assert ret_val.real >= 0

            return ret_val.real
        else:
            return 0.0

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
        mx = self.mx

        if e_cm >= 2.0 * mpi and e_cm >= 2.0 * mx:
            gvuu = self.gvuu
            gvdd = self.gvdd
            gvxx = self.gvxx
            mv = self.mv
            width_v = self.width_v

            ret_val = (
                (gvdd - gvuu) ** 2
                * gvxx ** 2
                * (-4.0 * mpi ** 2 + e_cm ** 2) ** 1.5
                * (2.0 * mx ** 2 + e_cm ** 2)
            ) / (
                48.0
                * pi
                * e_cm ** 2
                * sqrt(-4.0 * mx ** 2 + e_cm ** 2)
                * (
                    mv ** 4
                    - 2.0 * mv ** 2 * e_cm ** 2
                    + e_cm ** 4
                    + mv ** 2 * width_v ** 2
                )
            )

            assert ret_val.imag == 0
            assert ret_val.real >= 0

            return ret_val.real
        else:
            return 0.0

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
        mx = self.mx

        if e_cm >= mpi0 and e_cm >= 2.0 * mx:
            gvuu = self.gvuu
            gvdd = self.gvdd
            gvxx = self.gvxx
            mv = self.mv
            width_v = self.width_v

            ret_val = (
                3.0
                * (
                    (gvdd + 2.0 * gvuu) ** 2.0
                    * gvxx ** 2
                    * (-mpi0 ** 2 + e_cm ** 2) ** 3
                    * (2.0 * mx ** 2 + e_cm ** 2)
                    * qe ** 2
                )
                / (
                    13824.0
                    * fpi ** 2
                    * pi ** 5
                    * e_cm ** 3
                    * sqrt(-4.0 * mx ** 2 + e_cm ** 2)
                    * (
                        mv ** 4
                        - 2.0 * mv ** 2 * e_cm ** 2
                        + e_cm ** 4
                        + mv ** 2 * width_v ** 2
                    )
                )
            )

            assert ret_val.imag == 0
            assert ret_val.real >= 0

            return ret_val.real
        else:
            return 0.0

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
        mx = self.mx
        mv = self.mv

        if e_cm >= mpi0 + mv and e_cm >= 2.0 * mx:
            gvuu = self.gvuu
            gvdd = self.gvdd
            gvxx = self.gvxx
            width_v = self.width_v

            ret_val = (
                (gvdd - gvuu) ** 2
                * (gvdd + gvuu) ** 2
                * gvxx ** 2
                * (
                    (-mpi0 - mv + e_cm)
                    * (mpi0 - mv + e_cm)
                    * (-mpi0 + mv + e_cm)
                    * (mpi0 + mv + e_cm)
                )
                ** 1.5
                * (2 * mx ** 2 + e_cm ** 2)
            ) / (
                1536.0
                * fpi ** 2
                * pi ** 5
                * e_cm ** 3
                * sqrt(-4 * mx ** 2 + e_cm ** 2)
                * ((-mv ** 2 + e_cm ** 2) ** 2 + mv ** 2 * width_v ** 2)
            )

            assert ret_val.imag == 0
            assert ret_val.real >= 0

            return ret_val.real
        else:
            return 0.0

    def dsigma_ds_xx_to_v_to_pi0pipi(self, s, e_cm):
        mx = self.mx

        if (
            e_cm > 2.0 * mpi + mpi0
            and e_cm > 2.0 * mx
            and 4.0 * mpi ** 2 < s < (e_cm - mpi0) ** 2
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
                    * gvxx ** 2
                    * sqrt(s * (-4.0 * mpi ** 2 + s))
                    * sqrt(
                        e_cm ** 4
                        + (mpi0 ** 2 - s) ** 2
                        - 2.0 * e_cm ** 2 * (mpi0 ** 2 + s)
                    )
                    * (
                        -24.0 * mpi ** 6 * s
                        + mpi ** 4
                        * (-2.0 * mpi0 ** 4 + 28.0 * mpi0 ** 2 * s + 22.0 * s ** 2)
                        + 2.0 * mpi ** 2 * (mpi0 ** 6 - 4.0 * s ** 3)
                        + s
                        * (
                            -2.0 * mpi0 ** 6
                            - 4.0 * mpi0 ** 4 * s
                            - mpi0 ** 2 * s ** 2
                            + s ** 3
                        )
                        + e_cm ** 4
                        * (
                            -2.0 * mpi ** 4
                            + 2.0 * mpi ** 2 * (mpi0 ** 2 - s)
                            + s * (-2.0 * mpi0 ** 2 + s)
                        )
                        + e_cm ** 2
                        * (
                            4.0 * mpi ** 4 * (mpi0 ** 2 + s)
                            + s * (4.0 * mpi0 ** 4 + 5.0 * mpi0 ** 2 * s - 2.0 * s ** 2)
                            - 4.0
                            * mpi ** 2
                            * (mpi0 ** 4 + 3.0 * mpi0 ** 2 * s - s ** 2)
                        )
                    )
                )
                / (
                    294912.0
                    * fpi ** 6
                    * pi ** 7
                    * sqrt(e_cm ** 2)
                    * sqrt(-4.0 * mx ** 2 + e_cm ** 2)
                    * s ** 2
                    * (
                        mv ** 4
                        - 2.0 * mv ** 2 * e_cm ** 2
                        + e_cm ** 4
                        + mv ** 2 * width_v ** 2
                    )
                )
            )

            assert ret_val.imag == 0
            assert ret_val.real >= 0

            return ret_val.real
        else:
            return 0.0

    def sigma_xx_to_v_to_pi0pipi(self, e_cm):
        if e_cm > 2.0 * mpi + mpi0 and e_cm > 2.0 * self.mx:
            s_min = 4.0 * mpi ** 2
            s_max = (e_cm - mpi0) ** 2

            ret_val = quad(
                self.dsigma_ds_xx_to_v_to_pi0pipi, s_min, s_max, args=(e_cm)
            )[0]

            assert ret_val.imag == 0
            assert ret_val.real >= 0

            return ret_val.real
        else:
            return 0.0

    def sigma_xx_to_vv(self, e_cm):
        mx = self.mx
        mv = self.mv

        if e_cm >= 2.0 * mv and e_cm >= 2.0 * mx:
            gvxx = self.gvxx

            ret_val = (
                gvxx ** 4
                * sqrt(-4 * mv ** 2 + e_cm ** 2)
                * (
                    -2
                    - (2 * (mv ** 2 + 2 * mx ** 2) ** 2)
                    / (mv ** 4 - 4 * mv ** 2 * mx ** 2 + mx ** 2 * e_cm ** 2)
                    + (
                        4
                        * (
                            4 * mv ** 4
                            - 8 * mv ** 2 * mx ** 2
                            - 8 * mx ** 4
                            + 4 * mx ** 2 * e_cm ** 2
                            + e_cm ** 4
                        )
                        * atanh(
                            (
                                sqrt(4 * mv ** 2 - e_cm ** 2)
                                * sqrt(4 * mx ** 2 - e_cm ** 2)
                            )
                            / (2 * mv ** 2 - e_cm ** 2)
                        )
                    )
                    / (
                        (2 * mv ** 2 - e_cm ** 2)
                        * sqrt(4 * mv ** 2 - e_cm ** 2)
                        * sqrt(4 * mx ** 2 - e_cm ** 2)
                    )
                )
            ) / (8.0 * pi * e_cm ** 2 * sqrt(-4 * mx ** 2 + e_cm ** 2))

            assert ret_val.imag == 0
            assert ret_val.real >= 0

            return ret_val.real
        else:
            return 0.0

    def sigma_xx_to_v_to_xx(self, e_cm):
        """
        Returns the DM annihilation cross section into DM.
        """
        rv = self.mv / e_cm
        rx = self.mx / e_cm
        gvxx = self.gvxx
        rwv = self.width_v / e_cm

        if e_cm > 2.0 * self.mx:

            def msqrd(z):
                return (
                    gvxx ** 4
                    * (
                        9
                        - 56 * rx ** 2
                        + 2
                        * (
                            9 * rv ** 2 * (-1 + rv ** 2 + rwv ** 2)
                            + 4 * rv ** 2 * (11 + 2 * rv ** 2 + 2 * rwv ** 2) * rx ** 2
                            + 8
                            * (7 + 6 * rv ** 2 * (-4 + rv ** 2 + rwv ** 2))
                            * rx ** 4
                            + 64 * rx ** 6
                        )
                        - 2
                        * (-1 + 4 * rx ** 2)
                        * (
                            3 * rv ** 2 * (-3 + 2 * rv ** 2 + 2 * rwv ** 2)
                            + 4
                            * (3 + 6 * rv ** 4 + 2 * rv ** 2 * (-7 + 3 * rwv ** 2))
                            * rx ** 2
                            - 32 * (-2 + rv ** 2) * rx ** 4
                        )
                        * z
                        + 2
                        * (3 * rv ** 2 - 4 * rx ** 2)
                        * (-1 + 4 * rx ** 2) ** 3
                        * z ** 3
                        + (1 - 4 * rx ** 2) ** 4 * z ** 4
                        + 2
                        * (
                            3
                            + 5 * rv ** 4
                            + 12 * rx ** 2
                            + 8 * rx ** 4
                            + rv ** 2 * (-3 + 5 * rwv ** 2 - 20 * rx ** 2)
                        )
                        * (z - 4 * rx ** 2 * z) ** 2
                    )
                ) / (
                    (1 + rv ** 4 + rv ** 2 * (-2 + rwv ** 2))
                    * (
                        4 * rv ** 4
                        + 4 * rv ** 2 * (rwv ** 2 + (-1 + 4 * rx ** 2) * (-1 + z))
                        + (1 - 4 * rx ** 2) ** 2 * (-1 + z) ** 2
                    )
                )

            ret_val = quad(msqrd, -1, 1)[0] / (32.0 * pi * e_cm)

            assert ret_val.imag == 0.0
            assert ret_val.real >= 0.0

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
