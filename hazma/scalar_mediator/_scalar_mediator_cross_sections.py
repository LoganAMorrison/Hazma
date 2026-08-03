from hazma.scalar_mediator._c_scalar_mediator_cross_sections import (
    sigma_xx_to_s_to_ff as sig_ff,
    sigma_xx_to_s_to_gg as sig_gg,
    sigma_xx_to_s_to_pi0pi0 as sig_pi0pi0,
    sigma_xx_to_s_to_pipi as sig_pipi,
    sigma_xx_to_ss as sig_ss,
    sigma_ss_to_xx as sig_ss_to_xx,
    sigma_xl_to_xl as sig_xl,
    sigma_xpi_to_xpi as sig_xpi,
    sigma_xpi0_to_xpi0 as sig_xpi0,
    sigma_xg_to_xg as sig_xg,
    sigma_xs_to_xs as sig_xs,
    thermal_cross_section as tcs,
)

from hazma.parameters import muon_mass as mmu
from hazma.parameters import electron_mass as me

from numpy.polynomial.legendre import leggauss
import warnings
import numpy as np


def sigma_xx_to_s_to_ff(self, e_cm, f):
    """Returns the spin-averaged, cross section for dark matter
    annihilating into a pair of fermions, *f* through a scalar mediator in
    the s-channel.

    Parameters
    ----------
    e_cm : float or array-like
        Center of mass energy(ies).
    f: str
        String for the final state fermion: f = 'e' for electron and
        f = 'mu' for muon.

    Returns
    -------
    sigma : float or array-like
        Cross section for x + x -> s* -> f + f.
    """
    assert f == "e" or f == "mu"
    mf = me if f == "e" else mmu

    return sig_ff(
        e_cm,
        self.mx,
        self.ms,
        self.gsxx,
        self.gsff,
        self.gsGG,
        self.gsFF,
        self.lam,
        self.width_s,
        self.vs,
        mf,
    )


def sigma_xx_to_s_to_gg(self, e_cm):
    """Returns the spin-averaged, cross section for dark matter
    annihilating into a pair of photons through a scalar mediator in the
    s-channel.

    Parameters
    ----------
    e_cm : float or array-like
        Center of mass energy(ies).

    Returns
    -------
    sigma : float or array-like
        Cross section for x + x -> s* -> g + g.
    """
    return sig_gg(
        e_cm,
        self.mx,
        self.ms,
        self.gsxx,
        self.gsff,
        self.gsGG,
        self.gsFF,
        self.lam,
        self.width_s,
        self.vs,
    )


def sigma_xx_to_s_to_pi0pi0(self, e_cm):
    """Returns the spin-averaged, cross section for dark matter
    annihilating into a pair of neutral pion through a scalar mediator in
    the s-channel.

    Parameters
    ----------
    e_cm : float or array-like
        Center of mass energy(ies).

    Returns
    -------
    sigma : float or array-like
        Cross section for x + x -> s* -> pi0 + pi0.
    """
    return sig_pi0pi0(
        e_cm,
        self.mx,
        self.ms,
        self.gsxx,
        self.gsff,
        self.gsGG,
        self.gsFF,
        self.lam,
        self.width_s,
        self.vs,
    )


def sigma_xx_to_s_to_pipi(self, e_cm):
    """Returns the spin-averaged, cross section for dark matter
    annihilating into a pair of charged pions through a scalar mediator in
    the s-channel.

    Parameters
    ----------
    e_cm : float or array-like
        Center of mass energy(ies).

    Returns
    -------
    sigma : float or array-like
        Cross section for x + x -> s* -> np.pi + np.pi.
    """
    return sig_pipi(
        e_cm,
        self.mx,
        self.ms,
        self.gsxx,
        self.gsff,
        self.gsGG,
        self.gsFF,
        self.lam,
        self.width_s,
        self.vs,
    )


def sigma_xx_to_ss(self, e_cm):
    """Returns the spin-averaged, cross section for dark matter
    annihilating into a pair of scalar mediator through the t and u
    channels.

    Parameters
    ----------
    e_cm : float or array-like
        Center of mass energy(ies).

    Returns
    -------
    sigma : float or array-like
        Cross section for x + x -> s + s.
    """
    return sig_ss(
        e_cm,
        self.mx,
        self.ms,
        self.gsxx,
        self.gsff,
        self.gsGG,
        self.gsFF,
        self.lam,
        self.width_s,
        self.vs,
    )


def sigma_xx_to_s_to_xx(self, e_cm):
    """Returns the spin-averaged, self interaction cross section for dark
    matter.

    Parameters
    ----------
    e_cm : float or array-like
        Center of mass energy(ies).

    Returns
    -------
    sigma : float or array-like
        Cross section for x + x -> x + x
    """
    # see sigma_xx_to_s_to_ff for explaination of this context mangager
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"invalid value encountered in true_divide")
        warnings.filterwarnings("ignore", r"divide by zero encountered in true_divide")
        warnings.filterwarnings("ignore", r"invalid value encountered in multiply")
        warnings.filterwarnings("ignore", r"invalid value encountered in sqrt")
        warnings.filterwarnings("ignore", r"invalid value encountered in power")
        warnings.filterwarnings("ignore", r"invalid value encountered in subtract")
        warnings.filterwarnings("ignore", r"invalid value encountered in add")

        e_cms = np.array(e_cm) if hasattr(e_cm, "__len__") else e_cm
        mask = e_cms > 2.0 * self.mx

        rxs = self.mx / e_cms
        rss = self.ms / e_cms
        rwss = self.width_s / e_cms

        def msqrd(z):
            return (
                (
                    self.gsxx**4
                    * (
                        3 * (-1 + z) ** 2
                        + 256 * rxs**8 * (-1 + z) ** 2
                        - 64 * rxs**6 * (5 - 8 * z + 3 * z**2)
                        + 32 * rxs**4 * (5 - 6 * z + 3 * z**2)
                        - 4 * rxs**2 * (5 - 12 * z + 7 * z**2)
                        + rss**4
                        * (
                            3
                            + z**2
                            - 8 * rxs**2 * (1 + z**2)
                            + 16 * rxs**4 * (3 + z**2)
                        )
                        + rss**2
                        * (
                            3
                            - 3 * z**2
                            - 64 * rxs**6 * (3 - 4 * z + z**2)
                            - 16 * rxs**4 * (-9 + 12 * z + z**2)
                            + 4 * rxs**2 * (-17 + 8 * z + 5 * z**2)
                            + rwss**2
                            * (
                                3
                                + z**2
                                - 8 * rxs**2 * (1 + z**2)
                                + 16 * rxs**4 * (3 + z**2)
                            )
                        )
                    )
                )
                / (
                    (1 + rss**4 + rss**2 * (-2 + rwss**2))
                    * (
                        4 * rss**4
                        + 4 * rss**2 * (rwss**2 + (-1 + 4 * rxs**2) * (-1 + z))
                        + (1 - 4 * rxs**2) ** 2 * (-1 + z) ** 2
                    )
                )
            ).real

        # Compute the nodes and weights for Legendre-Gauss quadrature
        nodes, weights = leggauss(10)
        nodes, weights = (np.reshape(nodes, (10, 1)), np.reshape(weights, (10, 1)))
        ret_val = mask * np.nan_to_num(
            np.sum(weights * msqrd(nodes), axis=0) / (32.0 * np.pi * e_cms)
        )

    return ret_val.real


def sigma_xpi_to_xpi(self, e_cm):
    """
    Returns the spin-averaged, elastic scattering cross section of DM off
    charged-pions for the scalar mediator model. Note only considers
    both a pi^+ or pi^- (i.e. sums over charges.)

    Parameters
    ----------
    e_cm : float
        Center of mass energy.

    Returns
    -------
    sigma : float
        Cross section for x + pi -> x + pi
    """
    return sig_xpi(
        e_cm,
        self.mx,
        self.ms,
        self.gsxx,
        self.gsff,
        self.gsGG,
        self.gsFF,
        self.lam,
        self.width_s,
        self.vs,
    )


def sigma_xpi0_to_xpi0(self, e_cm):
    """
    Returns the spin-averaged, elastic scattering cross section of DM off
    neutral pion for the scalar mediator model.

    Parameters
    ----------
    e_cm : float
        Center of mass energy.

    Returns
    -------
    sigma : float
        Cross section for x + pi^0 -> x + pi^0
    """
    return sig_xpi0(
        e_cm,
        self.mx,
        self.ms,
        self.gsxx,
        self.gsff,
        self.gsGG,
        self.gsFF,
        self.lam,
        self.width_s,
        self.vs,
    )


def sigma_xl_to_xl(self, e_cm, f):
    """
    Returns the spin-averaged, elastic scattering cross section of DM off
    leptons for the scalar mediator model. Note this considers
    both a l + or lbar (i.e. it sums over charges.)

    Parameters
    ----------
    e_cm : float
        Center of mass energy.
    f : string
        String labeling final state lepton: either 'e' or 'mu'.

    Returns
    -------
    sigma : float
        Cross section for x + l -> x + l
    """
    assert f == "e" or f == "mu"
    ml = me if f == "e" else mmu

    return sig_xl(
        e_cm,
        self.mx,
        self.ms,
        self.gsxx,
        self.gsff,
        self.gsGG,
        self.gsFF,
        self.lam,
        self.width_s,
        self.vs,
        ml,
    )


def sigma_xg_to_xg(self, e_cm):
    """
    Returns the spin-averaged, elastic scattering cross section of DM off
    photons for the scalar mediator model.

    Parameters
    ----------
    e_cm : float
        Center of mass energy.

    Returns
    -------
    sigma : float
        Cross section for x + g -> x + g
    """
    return sig_xg(
        e_cm,
        self.mx,
        self.ms,
        self.gsxx,
        self.gsff,
        self.gsGG,
        self.gsFF,
        self.lam,
        self.width_s,
        self.vs,
    )


def sigma_xs_to_xs(self, e_cm):
    """
    Returns the spin-averaged, elastic scattering cross section of DM off
    scalar mediators for the scalar mediator model.

    Parameters
    ----------
    e_cm : float
        Center of mass energy.

    Returns
    -------
    sigma : float
        Cross section for x + s -> x + s
    """
    return sig_xs(
        e_cm,
        self.mx,
        self.ms,
        self.gsxx,
        self.gsff,
        self.gsGG,
        self.gsFF,
        self.lam,
        self.width_s,
        self.vs,
    )


def sigma_ss_to_xx(self, e_cm):
    """
    Cross section for mediator annihilations into DM.
    """
    return sig_ss_to_xx(
        e_cm,
        self.mx,
        self.ms,
        self.gsxx,
        self.gsff,
        self.gsGG,
        self.gsFF,
        self.lam,
        self.width_s,
        self.vs,
    )


def thermal_cross_section(self, x):
    """
    Compute the thermally average cross section.

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
        self.ms,
        self.gsxx,
        self.gsff,
        self.gsGG,
        self.gsFF,
        self.lam,
        self.width_s,
        self.vs,
    )


def annihilation_cross_section_funcs(self):
    return {
        "mu mu": lambda e_cm: self.sigma_xx_to_s_to_ff(e_cm, "mu"),
        "e e": lambda e_cm: self.sigma_xx_to_s_to_ff(e_cm, "e"),
        "g g": self.sigma_xx_to_s_to_gg,
        "pi0 pi0": self.sigma_xx_to_s_to_pi0pi0,
        "pi pi": self.sigma_xx_to_s_to_pipi,
        "s s": self.sigma_xx_to_ss,
    }


def elastic_scattering_cross_sections(self, e_cm):
    return {
        "pi": self.sigma_xpi_to_xpi(e_cm),
        "pi0": self.sigma_xpi0_to_xpi0(e_cm),
        "e": self.sigma_xl_to_xl(e_cm, "e"),
        "mu": self.sigma_xl_to_xl(e_cm, "mu"),
        "g": self.sigma_xg_to_xg(e_cm),
        "s": self.sigma_xs_to_xs(e_cm),
    }
