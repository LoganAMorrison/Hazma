import numpy as np

from hazma import spectra
from hazma.parameters import muon_mass as mmu
from hazma.parameters import electron_mass as me

from ._proto import PseudoScalarMediatorBase
from ._fsr import dnde_xx_to_p_to_ffg
from ._msqrd import msqrd_xx_to_p_to_000, msqrd_xx_to_p_to_pm0

# Stuff needed to compute fsr from x xbar -> P -> pip pim pi0
# from ..gamma_ray_decay import gamma_ray_fsr
# from .pseudo_scalar_mediator_mat_elem_sqrd_rambo import msqrd_xx_to_p_to_pm0g


# TODO: pp spectrum. Gonna need Logan to do this since it
# requires cython...
def dnde_pp(model: PseudoScalarMediatorBase, egams, Q, mode="total"):
    # eng_p = Q / 2.
    pass


def dnde_ee(_: PseudoScalarMediatorBase, egams, cme):
    """Computes spectrum from DM annihilation into electrons."""
    return dnde_xx_to_p_to_ffg(egams, cme, me)


def dnde_mumu(_: PseudoScalarMediatorBase, egams, cme):
    """Computes spectrum from DM annihilation into muons."""
    fsr = dnde_xx_to_p_to_ffg(egams, cme, mmu)
    decay = 2.0 * spectra.dnde_photon_muon(egams, cme / 2.0)
    return fsr + decay


def dnde_pi0pipi(model: PseudoScalarMediatorBase, photon_energies, cme):
    """Computes spectrum from DM annihilation into a neutral pion and two
    charged pions.

    Notes
    -----
    This function uses RAMBO to "convolve" the pions' spectra with the
    matrix element over the pi0 pi pi phase space.
    """

    def msqrd(momenta):
        return msqrd_xx_to_p_to_pm0(momenta, model)

    return spectra.dnde_photon(
        photon_energies=photon_energies,
        cme=cme,
        final_states=["pi0", "pi", "pi"],
        msqrd=msqrd,
        msqrd_signature="momenta",
    )


def dnde_pi0pi0pi0(model: PseudoScalarMediatorBase, photon_energies, cme):
    """Return the gamma ray spectrum for dark matter annihilations into
    three neutral pions.

    Notes
    -----
    This function uses RAMBO to "convolve" the pions' spectra with the
    matrix element over the pi0 pi0 pi0 phase space.
    """

    def msqrd(momenta):
        return msqrd_xx_to_p_to_000(momenta, model)

    return spectra.dnde_photon(
        photon_energies=photon_energies,
        cme=cme,
        final_states=["pi0", "pi0", "pi0"],
        msqrd=msqrd,
        msqrd_signature="momenta",
    )


def spectrum_funcs(model):
    """
    Returns a dictionary of all the avaiable spectrum functions for
    a pair of initial state fermions with mass `mx` annihilating into
    each available final state.

    Each argument of the spectrum functions in `eng_gams`, an array
    of the gamma ray energies to evaluate the spectra at and `cme`, the
    center of mass energy of the process.
    """

    def mk_dnde(fn):
        def dnde(photon_energies, cme):
            return fn(model, photon_energies, cme)

        return dnde

    return {
        "mu mu": mk_dnde(dnde_mumu),
        "e e": mk_dnde(dnde_ee),
        "pi0 pi pi": mk_dnde(dnde_pi0pipi),
        "pi0 pi0 pi0": mk_dnde(dnde_pi0pi0pi0),
        "p p": mk_dnde(dnde_pp),
    }


def gamma_ray_lines(model, cme):
    bf = annihilation_branching_fractions(model, cme)["g g"]
    return {"g g": {"energy": cme / 2.0, "bf": bf}}
