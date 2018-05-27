from cmath import sqrt, pi

from ..parameters import (charged_pion_mass as mpi,
                          neutral_pion_mass as mpi0,
                          electron_mass as me,
                          muon_mass as mmu,
                          qe, fpi)


def width_v_to_pipi(params):
    mv = params.mv

    if mv > 2. * mpi:
        gvuu = params.gvuu
        gvdd = params.gvdd
        return ((gvdd - gvuu)**2 * (-4. * mpi**2 + mv**2)**1.5) / \
            (16. * mv**2 * pi)
    else:
        return 0.


def width_v_to_pi0g(params):
    mv = params.mv

    if mv > mpi0:
        gvuu = params.gvuu
        gvdd = params.gvdd

        ret_val = 3. * ((gvdd + 2. * gvuu)**2 * (-mpi0**2 + mv**2) *
                        (-mpi0**6 + 4. * mpi**4 * mv**2 +
                         12. * mpi0**4 * mv**2 -
                         21. * mpi0**2 * mv**4 + 4. * mv**6 + mpi**2 *
                         (mpi0**4 - 12. * mpi0**2 * mv**2 +
                          13. * mv**4)) * qe**2) / \
            (18432. * fpi**2 * mv**5 * pi**5)

        assert ret_val.imag == 0
        assert ret_val.real >= 0

        return ret_val
    else:
        return 0.


def width_v_to_xx(params):
    mv = params.mv
    mx = params.mx

    if mv > 2. * mx:
        gvxx = params.gvxx

        return (gvxx**2 * sqrt(mv**2 - 4. * mx**2) * (mv**2 + 2. * mx**2)) / \
            (4. * mv**2 * pi)
    else:
        return 0.0


def width_v_to_ff(f, params):
    if f == "e":
        mf = me
        gvll = params.gvee
    elif f == "mu":
        mf = mmu
        gvll = params.gvmumu
    else:
        return 0.

    mv = params.mv

    if mv > 2. * mf:
        return (gvll**2 * sqrt(mv**2 - 4. * mf**2) * (2. * mf**2 + mv**2)) / \
            (4. * mv**2 * pi)
    else:
        return 0.


def partial_widths(params):
    w_pipi = width_v_to_pipi(params).real
    w_pi0g = width_v_to_pi0g(params).real
    w_xx = width_v_to_xx(params).real
    w_ee = width_v_to_ff("e", params).real
    w_mumu = width_v_to_ff("mu", params).real

    total = w_pipi + w_pi0g + w_xx + w_ee + w_mumu

    width_dict = {'pi pi': w_pipi,
                  'pi0 g': w_pi0g,
                  'x x': w_xx,
                  'e e': w_ee,
                  'mu mu': w_mumu,
                  'total': total}

    return width_dict
