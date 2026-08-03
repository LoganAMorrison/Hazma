import numpy as np

from hazma.parameters import vh, b0, alpha_em, fpi
from hazma.parameters import muon_mass as mmu
from hazma.parameters import electron_mass as me
from hazma.parameters import up_quark_mass as muq
from hazma.parameters import down_quark_mass as mdq
from hazma.parameters import charged_pion_mass as mpi

from scipy import integrate

from ._proto import PseudoScalarMediatorBase


def sigma_xx_to_p_to_ff(model: PseudoScalarMediatorBase, Q, f):
    """Returns the cross section for two DM particles into two fermions.

    Parameters
    ----------
    Q : float
        Center of mass energy.
    f : string
        Name of final state fermions: 'e' or 'mu'.
    model : PseudoScalarMediatorParameters
        Object of the pseudoscalar parameters class.

    Returns
    -------
    cross_section : float
        Cross section for xx -> p -> ff.
    """
    if f == "e":
        rf = me / Q
        gpff = model.gpee
    elif f == "mu":
        rf = mmu / Q
        gpff = model.gpmumu
    else:
        raise ValueError(f"Invalid fermion {f}. Use 'e' or 'mu'.")

    beta = model.beta
    gpxx = model.gpxx
    mp = model.mp
    width_p = model.width_p
    rx = model.mx / Q

    if 2.0 * rf < 1 and 2.0 * rx < 1:
        ret = (
            (1 - 2 * beta**2) * gpff**2 * gpxx**2 * Q**2 * np.sqrt(1 - 4 * rf**2)
        ) / (
            16.0
            * np.pi
            * np.sqrt(1 - 4 * rx**2)
            * ((mp**2 - Q**2) ** 2 + mp**2 * width_p**2)
        )

        assert ret.imag == 0
        assert ret.real >= 0

        return ret.real
    else:
        return 0.0


def sigma_xx_to_p_to_gg(model: PseudoScalarMediatorBase, Q):
    """Returns the cross section for DM annihilating into two photons.

    Parameters
    ----------
    Q : float
        Center of mass energy.

    Returns
    -------
    cross_section : float
        Cross section for xx -> p -> gg.
    """
    beta = model.beta
    mx = model.mx

    if Q >= 2.0 * mx:
        gpFF = model.gpFF
        gpxx = model.gpxx
        mp = model.mp
        rx = mx / Q
        width_p = model.width_p

        ret = (
            alpha_em**2
            * gpxx**2
            * Q**4
            * (
                (1 - 2 * beta**2) * fpi**2 * gpFF**2
                - 2 * beta * fpi * gpFF * vh
                + beta**2 * vh**2
            )
        ) / (
            256.0
            * fpi**2
            * np.pi**3
            * np.sqrt(1.0 - 4 * rx**2)
            * vh**2
            * ((mp**2 - Q**2) ** 2 + mp**2 * width_p**2)
        )

        assert ret.imag == 0
        assert ret.real >= 0

        return ret
    else:
        return 0.0


def sigma_xx_to_pp(model: PseudoScalarMediatorBase, Q):
    """Returns the cross section for DM annihilating into two mediators.

    Parameters
    ----------
    Q : float
        Center of mass energy.
    model : PseudoScalarMediatorParameters
        Object of the pseudoscalar parameters class.

    Returns
    -------
    cross_section : float
        Cross section for xx -> pp.
    """
    mx = model.mx
    mp = model.mp
    beta = model.beta

    if Q > 2.0 * mp and Q >= 2.0 * mx:
        gpxx = model.gpxx
        rp = mp / Q
        rx = mx / Q

        ret = (
            (-1 + 2 * beta**2)
            * gpxx**4
            * (
                (
                    2
                    * np.sqrt((-1 + 4 * rp**2) * (-1 + 4 * rx**2))
                    * (3 * rp**4 + 2 * rx**2 - 8 * rp**2 * rx**2)
                )
                / (rp**4 + rx**2 - 4 * rp**2 * rx**2)
                + (
                    2
                    * (1 - 4 * rp**2 + 6 * rp**4)
                    * (
                        1j * np.pi
                        + 2
                        * np.arctanh(
                            (1 - 2 * rp**2)
                            / np.sqrt((-1 + 4 * rp**2) * (-1 + 4 * rx**2))
                        )
                    )
                )
                / (-1 + 2 * rp**2)
            )
        ) / (64.0 * Q**2 * np.pi * (1 - 4 * rx**2))

        assert ret.imag == 0
        assert ret.real >= 0

        return ret
    else:
        return 0.0


def dsigma_ds_xx_to_p_to_pi0pi0pi0(model: PseudoScalarMediatorBase, s, Q):
    """Returns the dsigma/ds for DM annihilation into three neutral pions.

    Parameters
    ----------
    Q : float
        Center of mass energy.

    Returns
    -------
    dsigma_ds : float
        dsigma/ds for xx -> p -> pi0 pi0 pi0, where s=(P-q)^2, with P the
        center of mass momentum and q the momentum of one of the pi0s.
    """
    mx = model.mx
    mpi0 = model.mpi0  # use shifted pion mass!

    if Q > 2.0 * mx and Q >= 3.0 * mpi0:
        beta = model.beta
        gpxx = model.gpxx
        gpuu = model.gpuu
        gpdd = model.gpdd
        gpGG = model.gpGG
        mp = model.mp
        width_p = model.width_p

        ret = -(
            b0**2
            * gpxx**2
            * np.sqrt(s * (-4 * mpi0**2 + s))
            * np.sqrt(mpi0**4 + (Q**2 - s) ** 2 - 2 * mpi0**2 * (Q**2 + s))
            * (
                -(beta**2 * (mdq + muq) ** 2 * vh**2)
                + 2
                * beta
                * fpi
                * (mdq + muq)
                * vh
                * (gpGG * (mdq - muq) + (gpdd - gpuu) * vh)
                + (-1 + 11 * beta**2)
                * fpi**2
                * (gpGG * (mdq - muq) + (gpdd - gpuu) * vh) ** 2
            )
        ) / (
            512.0
            * fpi**4
            * np.pi**3
            * Q
            * np.sqrt(-4 * mx**2 + Q**2)
            * s
            * vh**2
            * (mp**4 + Q**4 + mp**2 * (-2 * Q**2 + width_p**2))
        )

        assert ret.imag == 0
        assert ret.real >= 0

        return ret
    else:
        return 0.0


def sigma_xx_to_p_to_pi0pi0pi0(model: PseudoScalarMediatorBase, Q):
    """Returns the DM annihilation cross section into three neutral pions.

    Notes
    -----
    Integrates dsigma/ds as given by `dsigma_ds_xx_to_p_to_pi0pi0pi0`
    over s.

    Parameters
    ----------
    model : PseudoScalarMediatorParameters
        Object of the pseudoscalar parameters class.
    Q : float
        Center of mass energy.

    Returns
    -------
    cross_section : float
        The DM annihilation cross section xx -> p -> pi0 pi0 pi0.
    """

    mpi0 = model.mpi0  # use shifted pion mass!
    smax = (Q - mpi0) ** 2
    smin = 4.0 * mpi0**2

    def msqrd(s):
        return dsigma_ds_xx_to_p_to_pi0pi0pi0(model, s, Q)

    return integrate.quad(msqrd, smin, smax, args=(Q))[0]


def dsigma_ds_xx_to_p_to_pi0pipi(model: PseudoScalarMediatorBase, s, Q):
    """Returns the dsigma/ds for DM annihilation into a neutral pion and two
    charged pions.

    Parameters
    ----------
    Q : float
        Center of mass energy.
    model : PseudoScalarMediatorParameters
        Object of the pseudoscalar parameters class.

    Returns
    -------
    dsigma_ds : float
        dsigma/ds for xx -> p -> pi0 pi pi, where s=(P-q)^2, with P the
        center of mass momentum and q the momentum of the pi0.
    """
    mx = model.mx
    mpi0 = model.mpi0  # use shifted pion mass!

    if Q > 2.0 * mx and Q >= 2.0 * mpi + mpi0:
        beta = model.beta
        gpxx = model.gpxx
        gpuu = model.gpuu
        gpdd = model.gpdd
        gpGG = model.gpGG
        mp = model.mp
        width_p = model.width_p

        ret = (
            gpxx**2
            * np.sqrt(s * (-4 * mpi**2 + s))
            * np.sqrt(mpi0**4 + (Q**2 - s) ** 2 - 2 * mpi0**2 * (Q**2 + s))
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
                    - (-1 + 5 * beta**2)
                    * fpi**2
                    * (gpGG * (mdq - muq) + (gpdd - gpuu) * vh) ** 2
                )
            )
        ) / (
            4608.0
            * fpi**4
            * np.pi**3
            * Q
            * np.sqrt(-4 * mx**2 + Q**2)
            * s
            * vh**2
            * (mp**4 + Q**4 + mp**2 * (-2 * Q**2 + width_p**2))
        )

        assert ret.imag == 0
        assert ret.real >= 0

        return ret
    else:
        return 0.0


def sigma_xx_to_p_to_pi0pipi(model: PseudoScalarMediatorBase, Q):
    """Returns the DM annihilation cross section into a neutral pion and two
    charged pions.

    Notes
    -----
    Integrates dsigma/ds as given by `dsigma_ds_xx_to_p_to_pi0pipi` over s.

    Parameters
    ----------
    Q : float
        Center of mass energy.
    model : PseudoScalarMediatorParameters
        Object of the pseudoscalar parameters class.

    Returns
    -------
    cross_section : float
        The DM annihilation cross section xx -> p -> pi0 pi pi.
    """

    mpi0 = model.mpi0  # use shifted pion mass!
    smax = (Q - mpi0) ** 2
    smin = 4.0 * mpi**2

    def msqrd(s):
        return dsigma_ds_xx_to_p_to_pi0pipi(model, s, Q)

    return integrate.quad(msqrd, smin, smax, args=(Q))[0]


def annihilation_cross_sections(model: PseudoScalarMediatorBase, Q):
    """
    Compute the total cross section DM annihilation.

    Parameters
    ----------
    Q : float
        Center of mass energy.

    Returns
    -------
    cs : dict
        Dictionary containing the theory's cross sections. The keys are
        'total', 'mu mu', 'e e', 'pi0 pi0 pi0', 'pi0 pi pi', 'g g', 'p p'.
    """
    css = {}
    css["mu mu"] = sigma_xx_to_p_to_ff(model, Q, "mu")
    css["e e"] = sigma_xx_to_p_to_ff(model, Q, "e")
    css["g g"] = sigma_xx_to_p_to_gg(model, Q)
    css["pi0 pi pi"] = sigma_xx_to_p_to_pi0pipi(model, Q)
    css["pi0 pi0 pi0"] = sigma_xx_to_p_to_pi0pi0pi0(model, Q)
    css["p p"] = sigma_xx_to_pp(model, Q)
    css["total"] = sum(css.values())
    return css


def annihilation_branching_fractions(model: PseudoScalarMediatorBase, Q):
    """
    Compute the branching fractions for DM annihilation.

    Parameters
    ----------
    Q : float
        Center of mass energy.

    Returns
    -------
    bfs : dictionary
        Dictionary of the branching fractions. The keys are 'total',
        'mu mu', 'e e', 'pi0 pi0 pi0', 'pi0 pi pi', 'g g', 'p p'.
    """
    css = annihilation_cross_sections(model, Q)
    bfs = {key: val / css["total"] for key, val in css if not key == "total"}
    return bfs
