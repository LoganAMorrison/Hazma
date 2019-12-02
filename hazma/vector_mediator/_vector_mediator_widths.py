from cmath import sqrt, pi

from hazma.parameters import (
    charged_pion_mass as mpi,
    neutral_pion_mass as mpi0,
    electron_mass as me,
    muon_mass as mmu,
    qe,
    fpi,
    alpha_em,
)


class VectorMediatorWidths:
    def width_v_to_pipi(self):
        mv = self.mv

        if mv > 2.0 * mpi:
            val = (
                ((self.gvdd - self.gvuu) ** 2 * (-4 * mpi ** 2 + mv ** 2) ** 1.5)
                / (48.0 * mv ** 2 * pi)
            ).real

            assert val >= 0

            return val
        else:
            return 0.0

    def width_v_to_pi0g(self):
        mv = self.mv

        if mv > mpi0:
            val = (
                (
                    alpha_em
                    * (self.gvdd + 2 * self.gvuu) ** 2
                    * (-mpi0 ** 2 + mv ** 2) ** 3
                )
                / (3456.0 * fpi ** 2 * mv ** 3 * pi ** 4)
            ).real

            assert val >= 0

            return val
        else:
            return 0.0

    def width_v_to_xx(self):
        mv = self.mv
        mx = self.mx

        if mv > 2.0 * mx:
            val = (
                (self.gvxx ** 2 * sqrt(mv ** 2 - 4 * mx ** 2) * (mv ** 2 + 2 * mx ** 2))
                / (12.0 * mv ** 2 * pi)
            ).real

            assert val >= 0

            return val
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
            val = (
                (gvll ** 2 * sqrt(-4 * mf ** 2 + mv ** 2) * (2 * mf ** 2 + mv ** 2))
                / (12.0 * mv ** 2 * pi)
            ).real

            assert val >= 0

            return val
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
