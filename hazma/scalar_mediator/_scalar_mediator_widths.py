from typing import Dict
from cmath import sqrt, pi

from hazma.parameters import vh, b0, alpha_em
from hazma.parameters import charged_pion_mass as mpi
from hazma.parameters import neutral_pion_mass as mpi0
from hazma.parameters import up_quark_mass as muq
from hazma.parameters import down_quark_mass as mdq
from hazma.parameters import electron_mass as me
from hazma.parameters import muon_mass as mmu


def width_s_to_gg(self):
    """
    Returns the partial decay width of the scalar decaying into photon.
    """
    return (alpha_em**2 * self.gsFF**2 * self.ms**3) / (64.0 * self.lam**2 * pi**3)


def width_s_to_pi0pi0(self):
    """
    Returns the partial decay width of the scalar decaying into
    neutral pions.
    """
    ms = self.ms

    if ms > 2.0 * mpi0:
        gsff = self.gsff
        gsGG = self.gsGG
        vs = self.vs
        lam = self.lam

        val = (
            (
                sqrt(-4 * mpi0**2 + ms**2)
                * (
                    -162 * gsGG * lam**3 * (-2 * mpi0**2 + ms**2) * vh**2
                    + b0
                    * (mdq + muq)
                    * (9 * lam + 4 * gsGG * vs)
                    * (-3 * lam * vh + 3 * gsff * lam * vs + 2 * gsGG * vh * vs)
                    * (
                        2 * gsGG * vh * (9 * lam - 4 * gsGG * vs)
                        + 9 * gsff * lam * (3 * lam + 4 * gsGG * vs)
                    )
                )
                ** 2
            )
            / (209952.0 * lam**6 * ms**2 * pi * vh**4 * (9 * lam + 4 * gsGG * vs) ** 2)
        ).real

        assert val >= 0

        return val
    else:
        return 0.0


def width_s_to_pipi(self):
    """
    Returns the partial decay width of the scalar decaying into
    charged pion.
    """
    ms = self.ms

    if ms > 2.0 * mpi:
        gsff = self.gsff
        gsGG = self.gsGG
        vs = self.vs
        lam = self.lam

        val = (
            (
                sqrt(-4 * mpi**2 + ms**2)
                * (
                    -162 * gsGG * lam**3 * (-2 * mpi**2 + ms**2) * vh**2
                    + b0
                    * (mdq + muq)
                    * (9 * lam + 4 * gsGG * vs)
                    * (-3 * lam * vh + 3 * gsff * lam * vs + 2 * gsGG * vh * vs)
                    * (
                        2 * gsGG * vh * (9 * lam - 4 * gsGG * vs)
                        + 9 * gsff * lam * (3 * lam + 4 * gsGG * vs)
                    )
                )
                ** 2
            )
            / (104976.0 * lam**6 * ms**2 * pi * vh**4 * (9 * lam + 4 * gsGG * vs) ** 2)
        ).real

        assert val >= 0

        return val
    else:
        return 0.0


def width_s_to_xx(self):
    """
    Returns the partial decay width of the scalar decaying into
    two fermions x.
    """
    ms = self.ms
    mx = self.mx

    if ms > 2.0 * mx:
        gsxx = self.gsxx

        val = ((gsxx**2 * (ms**2 - 4 * mx**2) ** 1.5) / (8.0 * ms**2 * pi)).real

        assert val >= 0

        return val
    else:
        return 0.0


def width_s_to_ff(self, f):
    """
    Returns the partial decay width of the scalar decaying into
    two fermions x.

    Parameters
    ----------
    f : string
        Name of the final state fermion.
    """
    ms = self.ms

    if f == "e":
        mf = me
    elif f == "mu":
        mf = mmu
    else:
        raise ValueError(f"Invalid fermion string {f}. Expected 'e' or 'mu'.")

    if ms > 2.0 * mf:
        val = (
            (self.gsff**2 * mf**2 * (-4 * mf**2 + ms**2) ** 1.5)
            / (8.0 * ms**2 * pi * vh**2)
        ).real

        assert val >= 0

        return val
    else:
        return 0.0


def partial_widths(self) -> Dict[str, float]:
    """
    Returns a dictionary for the partial decay widths of the scalar
    mediator.

    Returns
    -------
    width_dict : dictionary
        Dictionary of all of the individual decay widths of the scalar
        mediator as well as the total decay width. The possible decay
        modes of the scalar mediator are 'g g', 'pi0 pi0', 'pi pi', 'x x'
        and 'f f'. The total decay width has the key 'total'.
    """
    w_gg = self.width_s_to_gg()
    w_pi0pi0 = self.width_s_to_pi0pi0()
    w_pipi = self.width_s_to_pipi()
    w_xx = self.width_s_to_xx()

    w_ee = self.width_s_to_ff("e")
    w_mumu = self.width_s_to_ff("mu")

    total = w_gg + w_pi0pi0 + w_pipi + w_xx + w_ee + w_mumu

    width_dict = {
        "g g": w_gg,
        "pi0 pi0": w_pi0pi0,
        "pi pi": w_pipi,
        "x x": w_xx,
        "e e": w_ee,
        "mu mu": w_mumu,
        "total": total,
    }

    return width_dict
