from cmath import sqrt, pi

from hazma.parameters import (
    charged_pion_mass as mpi,
    neutral_pion_mass as mpi0,
    electron_mass as me,
    muon_mass as mmu,
    qe,
    fpi,
)


class VectorMediatorWidths:
    def width_v_to_pipi(self):
        mv = self.mv

        if mv > 2.0 * mpi:
            gvuu = self.gvuu
            gvdd = self.gvdd
            ret_val = ((gvdd - gvuu) ** 2 * (-4.0 * mpi ** 2 + mv ** 2) ** 1.5) / (
                16.0 * mv ** 2 * pi
            )

            assert ret_val.imag == 0
            assert ret_val.real >= 0

            return ret_val.real
        else:
            return 0.0

    def width_v_to_pi0g(self):
        mv = self.mv

        if mv > mpi0:
            gvuu = self.gvuu
            gvdd = self.gvdd

            ret_val = (
                3.0
                * (
                    (gvdd + 2.0 * gvuu) ** 2
                    * (-mpi0 ** 2 + mv ** 2)
                    * (
                        -mpi0 ** 6
                        + 4.0 * mpi ** 4 * mv ** 2
                        + 12.0 * mpi0 ** 4 * mv ** 2
                        - 21.0 * mpi0 ** 2 * mv ** 4
                        + 4.0 * mv ** 6
                        + mpi ** 2
                        * (mpi0 ** 4 - 12.0 * mpi0 ** 2 * mv ** 2 + 13.0 * mv ** 4)
                    )
                    * qe ** 2
                )
                / (18432.0 * fpi ** 2 * mv ** 5 * pi ** 5)
            )

            assert ret_val.imag == 0
            assert ret_val.real >= 0

            return ret_val.real
        else:
            return 0.0

    def width_v_to_xx(self):
        mv = self.mv
        mx = self.mx

        if mv > 2.0 * mx:
            gvxx = self.gvxx

            ret_val = (
                gvxx ** 2
                * sqrt(mv ** 2 - 4.0 * mx ** 2).real
                * (mv ** 2 + 2.0 * mx ** 2)
            ) / (4.0 * mv ** 2 * pi)

            assert ret_val.imag == 0
            assert ret_val.real >= 0

            return ret_val.real
        else:
            return 0.0

    def width_v_to_ff(self, f):
        if f == "e":
            mf = me
            gvll = self.gvee
        elif f == "mu":
            mf = mmu
            gvll = self.gvmumu
        else:
            return 0.0

        mv = self.mv

        if mv > 2.0 * mf:
            ret_val = (
                gvll ** 2
                * sqrt(mv ** 2 - 4.0 * mf ** 2).real
                * (2.0 * mf ** 2 + mv ** 2)
            ) / (4.0 * mv ** 2 * pi)

            assert ret_val.imag == 0
            assert ret_val.real >= 0

            return ret_val.real
        else:
            return 0.0

    def partial_widths(self):
        w_pipi = self.width_v_to_pipi()
        w_pi0g = self.width_v_to_pi0g()
        w_xx = self.width_v_to_xx()
        w_ee = self.width_v_to_ff("e")
        w_mumu = self.width_v_to_ff("mu")

        total = w_pipi + w_pi0g + w_xx + w_ee + w_mumu

        return {
            "pi pi": w_pipi,
            "pi0 g": w_pi0g,
            "x x": w_xx,
            "e e": w_ee,
            "mu mu": w_mumu,
            "total": total,
        }
