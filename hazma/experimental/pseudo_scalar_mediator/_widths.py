import numpy as np

from hazma.parameters import vh, b0, alpha_em, fpi
from hazma.parameters import muon_mass as mmu
from hazma.parameters import electron_mass as me
from hazma.parameters import up_quark_mass as muq
from hazma.parameters import down_quark_mass as mdq
from hazma.parameters import charged_pion_mass as mpi

from scipy.integrate import quad


class PseudoScalarMediatorWidths:
    def width_p_to_gg(self):
        """Returns the partial decay width of the pseudoscalar decaying into
        photons.
        """
        beta = self.beta
        gpFF = self.gpFF
        mp = self.mp

        ret = -(
            alpha_em**2
            * mp**3
            * ((1 + beta) * fpi * gpFF - beta * vh)
            * ((-1 + beta) * fpi * gpFF + beta * vh)
        ) / (128.0 * fpi**2 * np.pi**3 * vh**2)

        assert ret.imag == 0
        assert ret.real >= 0

        return ret.real

    def width_p_to_xx(self):
        mp = self.mp
        rx = self.mx / mp

        if 2.0 * rx < 1:
            ret = -(
                (-1 + self.beta**2) * self.gpxx**2 * mp * np.sqrt(1 - 4 * rx**2)
            ) / (32.0 * np.pi)

            assert ret.imag == 0
            assert ret.real >= 0

            return ret.real
        else:
            return 0.0

    def width_p_to_ff(self, f):
        mp = self.mp

        if f == "e":
            rf = me / mp
            gpff = self.gpee
        elif f == "mu":
            rf = mmu / mp
            gpff = self.gpmumu

        if 2.0 * rf < 1:
            ret = -(
                (-1 + self.beta**2) * gpff**2 * mp * np.sqrt(1 - 4 * rf**2)
            ) / (8.0 * np.pi)

            assert ret.imag == 0
            assert ret.real >= 0

            return ret.real
        else:
            return 0.0

    def dwidth_ds_p_to_pi0pi0pi0(self, s):
        mp = self.mp
        mpi0 = self.mpi0  # use shifted pion mass!

        if mp >= 3.0 * mpi0:
            gpuu = self.gpuu
            gpdd = self.gpdd
            gpGG = self.gpGG
            beta = self.beta

            ret = -(
                b0**2
                * np.sqrt(s * (-4 * mpi0**2 + s))
                * np.sqrt(mp**4 + (mpi0**2 - s) ** 2 - 2 * mp**2 * (mpi0**2 + s))
                * (
                    -(beta**2 * (mdq + muq) ** 2 * vh**2)
                    + 2
                    * beta
                    * fpi
                    * (mdq + muq)
                    * vh
                    * (gpGG * (mdq - muq) + (gpdd - gpuu) * vh)
                    + (-1 + 10 * beta**2)
                    * fpi**2
                    * (gpGG * (mdq - muq) + (gpdd - gpuu) * vh) ** 2
                )
            ) / (256.0 * fpi**4 * mp**3 * np.pi**3 * s * vh**2)

            assert ret.imag == 0
            assert ret.real >= 0

            return ret
        else:
            return 0.0

    def width_p_to_pi0pi0pi0(self):
        """
        Returns the width for the pseudoscalar's decay into three neutral
        pions.

        Parameters
        ----------
        self : PseudoScalarMediator or PseudoScalarMediatorParameters object
            Object containing the parameters of the pseudo-scalar mediator
            model. Can be a PseudoScalarMediator or a
            PseudoScalarMediatorParameters object.

        Returns
        -------
        gamma : float
            The width for P -> pi0 pi0 pi0.
        """
        mp = self.mp
        mpi0 = self.mpi0  # use shifted pion mass!

        smax = (mp - mpi0) ** 2
        smin = 4.0 * mpi0**2

        res = quad(self.dwidth_ds_p_to_pi0pi0pi0, smin, smax)

        return res[0]

    def dwidth_ds_p_to_pi0pipi(self, s):
        mp = self.mp
        mpi0 = self.mpi0  # use shifted pion mass!

        if mp >= 2.0 * mpi + mpi0:
            gpuu = self.gpuu
            gpdd = self.gpdd
            gpGG = self.gpGG
            beta = self.beta

            ret = (
                np.sqrt(s * (-4 * mpi**2 + s))
                * np.sqrt(mp**4 + (mpi0**2 - s) ** 2 - 2 * mp**2 * (mpi0**2 + s))
                * (
                    beta**2 * (2 * mpi**2 + mpi0 - 3 * s) ** 2 * vh**2
                    + 2
                    * b0
                    * beta
                    * (2 * mpi**2 + mpi0 - 3 * s)
                    * vh
                    * (
                        -(beta * (mdq + muq) * vh)
                        + fpi * (gpGG * (mdq - muq) + (gpdd - gpuu) * vh)
                    )
                    + b0**2
                    * (
                        beta**2 * (mdq + muq) ** 2 * vh**2
                        - 2
                        * beta
                        * fpi
                        * (mdq + muq)
                        * vh
                        * (gpGG * (mdq - muq) + (gpdd - gpuu) * vh)
                        - (-1 + 4 * beta**2)
                        * fpi**2
                        * (gpGG * (mdq - muq) + (gpdd - gpuu) * vh) ** 2
                    )
                )
            ) / (2304.0 * fpi**4 * mp**3 * np.pi**3 * s * vh**2)

            ret = (
                np.sqrt(s * (-4 * mpi**2 + s))
                * np.sqrt(mp**4 + (mpi0**2 - s) ** 2 - 2 * mp**2 * (mpi0**2 + s))
                * (
                    beta**2 * (2 * mpi**2 + mpi0**2 - 3 * s) ** 2 * vh**2
                    + 2
                    * b0
                    * beta
                    * (2 * mpi**2 + mpi0**2 - 3 * s)
                    * vh
                    * (
                        -(beta * (mdq + muq) * vh)
                        + fpi * (gpGG * (mdq - muq) + (gpdd - gpuu) * vh)
                    )
                    + b0**2
                    * (
                        beta**2 * (mdq + muq) ** 2 * vh**2
                        - 2
                        * beta
                        * fpi
                        * (mdq + muq)
                        * vh
                        * (gpGG * (mdq - muq) + (gpdd - gpuu) * vh)
                        - (-1 + 4 * beta**2)
                        * fpi**2
                        * (gpGG * (mdq - muq) + (gpdd - gpuu) * vh) ** 2
                    )
                )
            ) / (2304.0 * fpi**4 * mp**3 * np.pi**3 * s * vh**2)

            assert ret.imag == 0
            assert ret.real >= 0

            return ret
        else:
            return 0.0

    def width_p_to_pi0pipi(self):
        """
        Returns the width for the pseudoscalar's decay into a neutral pion and
        two charged pions.

        Parameters
        ----------
        self : PseudoScalarMediator or PseudoScalarMediatorParameters object
            Object containing the parameters of the pseudo-scalar mediator
            model. Can be a PseudoScalarMediator or a
            PseudoScalarMediatorParameters object.

        Returns
        -------
        gamma : float
            The width for P -> pi0 pi+ pi-.
        """
        mp = self.mp
        mpi0 = self.mpi0  # use shifted pion mass!

        smax = (mp - mpi0) ** 2
        smin = 4.0 * mpi**2

        res = quad(self.dwidth_ds_p_to_pi0pipi, smin, smax)

        return res[0]

    def partial_widths(self):
        """
        Returns a dictionary for the partial decay widths of the pseudoscalar
        mediator.

        Returns
        -------
        width_dict : dictionary
            Dictionary of all of the individual decay widths of the
            pseudoscalar mediator as well as the total decay width.
        """
        w_gg = self.width_p_to_gg()
        w_xx = self.width_p_to_xx()

        w_ee = self.width_p_to_ff("e")
        w_mumu = self.width_p_to_ff("mu")

        w_pi0pipi = self.width_p_to_pi0pipi()
        w_pi0pi0pi0 = self.width_p_to_pi0pi0pi0()

        total = w_gg + w_xx + w_ee + w_mumu + w_pi0pipi + w_pi0pi0pi0

        width_dict = {
            "g g": w_gg,
            "x x": w_xx,
            "e e": w_ee,
            "mu mu": w_mumu,
            "pi0 pi pi": w_pi0pipi,
            "pi0 pi0 pi0": w_pi0pi0pi0,
            "total": total,
        }

        return width_dict
