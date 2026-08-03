"""
This file contains the decay spectra from a right-handed neutrino at rest.
"""

from typing import Callable, Tuple
import logging

import numpy as np

from hazma import spectra
from hazma.form_factors.vector import VectorFormFactorPiPi
from hazma.parameters import standard_model_masses as sm_masses

from ._proto import SingleRhNeutrinoModel, Generation
from . import _widths as rhn_widths


_LEPTON_STRS = ["e", "mu", "tau"]
_NEUTRINO_STRS = ["ve", "vm", "vt"]


def dnde_zero(e, _):
    return np.zeros_like(e)


def _dnde_two_body(
    model: SingleRhNeutrinoModel,
    product_energies,
    product: str,
    states: Tuple[str, str],
):
    """
    Compute the gamma-ray spectrum from the decay of a right-handed
    neutrino into a neutral pion and neutrino.

    Parameters
    ----------
    photon_energies: float or np.array
        Photon energies where the spectrum should be computed.
    """
    if model.mx < sum(sm_masses[s] for s in states):
        return np.zeros_like(product_energies)

    if product == "photon":
        return spectra.dnde_photon(
            photon_energies=product_energies,
            cme=model.mx,
            final_states=states,
            include_fsr=True,
            average_fsr=True,
        )
    if product == "positron":
        return spectra.dnde_positron(
            positron_energies=product_energies, cme=model.mx, final_states=states
        )
    if product in ["neutrino", "ve", "vm", "vt"]:
        return spectra.dnde_neutrino(
            neutrino_energies=product_energies, cme=model.mx, final_states=states
        )

    raise ValueError(f"Invalid product {product}.")


def _dnde_three_body(
    model: SingleRhNeutrinoModel,
    product_energies,
    product: str,
    states: Tuple[str, str, str],
    msqrd: Callable,
    nbins: int,
    three_body_integrator: str,
):
    if model.mx < sum(sm_masses[s] for s in states):
        return np.zeros_like(product_energies)

    if product == "photon":
        return spectra.dnde_photon(
            photon_energies=product_energies,
            cme=model.mx,
            final_states=states,
            msqrd=msqrd,
            nbins=nbins,
            three_body_integrator=three_body_integrator,
            include_fsr=True,
            average_fsr=True,
            msqrd_signature="st",
        )
    if product == "positron":
        return spectra.dnde_positron(
            positron_energies=product_energies,
            cme=model.mx,
            final_states=states,
            msqrd=msqrd,
            nbins=nbins,
            three_body_integrator=three_body_integrator,
            msqrd_signature="st",
        )
    if product in ["neutrino", "ve", "vm", "vt"]:
        return spectra.dnde_neutrino(
            neutrino_energies=product_energies,
            cme=model.mx,
            final_states=states,
            msqrd=msqrd,
            nbins=nbins,
            three_body_integrator=three_body_integrator,
            msqrd_signature="st",
        )

    raise ValueError(f"Invalid product {product}.")


def dnde_v_pi0(model: SingleRhNeutrinoModel, product_energies, product: str):
    """
    Compute the spectrum of a specified product from the decay of a right-handed
    neutrino into a neutral pion and neutrino.

    Parameters
    ----------
    model: SingleRhNeutrinoModel
        Object containing the model parameters. Should implement the
        `SingleRhNeutrinoModel` protocol.
    product_energies: float or np.array
         Energies of the product where the spectrum should be computed.
    product: str
         The product to compute spectrum for.

    Returns
    -------
    dnde: float or array-like
        Differential energy spectrum evaluated at the input energies.
    """
    nu = _NEUTRINO_STRS[model.gen]
    return _dnde_two_body(model, product_energies, product, (nu, "pi0"))


def dnde_v_eta(model: SingleRhNeutrinoModel, product_energies, product: str):
    """
    Compute the spectrum of a specified product from the decay of a right-handed
    neutrino into a eta and neutrino.

    Parameters
    ----------
    model: SingleRhNeutrinoModel
        Object containing the model parameters. Should implement the
        `SingleRhNeutrinoModel` protocol.
    product_energies: float or np.array
         Energies of the product where the spectrum should be computed.
    product: str
         The product to compute spectrum for.

    Returns
    -------
    dnde: float or array-like
        Differential energy spectrum evaluated at the input energies.
    """
    nu = _NEUTRINO_STRS[model.gen]
    return _dnde_two_body(model, product_energies, product, (nu, "pi0"))


def dnde_v_rho(model: SingleRhNeutrinoModel, product_energies, product: str):
    """
    Compute the spectrum of a specified product from the decay of a right-handed
    neutrino into a neutral rho and neutrino.

    Parameters
    ----------
    model: SingleRhNeutrinoModel
        Object containing the model parameters. Should implement the
        `SingleRhNeutrinoModel` protocol.
    product_energies: float or np.array
         Energies of the product where the spectrum should be computed.
    product: str
         The product to compute spectrum for.

    Returns
    -------
    dnde: float or array-like
        Differential energy spectrum evaluated at the input energies.
    """
    nu = _NEUTRINO_STRS[model.gen]
    return _dnde_two_body(model, product_energies, product, (nu, "rho0"))


def dnde_v_omega(model: SingleRhNeutrinoModel, product_energies, product: str):
    """
    Compute the spectrum of a specified product from the decay of a right-handed
    neutrino into an omega and neutrino.

    Parameters
    ----------
    model: SingleRhNeutrinoModel
        Object containing the model parameters. Should implement the
        `SingleRhNeutrinoModel` protocol.
    product_energies: float or np.array
         Energies of the product where the spectrum should be computed.
    product: str
         The product to compute spectrum for.

    Returns
    -------
    dnde: float or array-like
        Differential energy spectrum evaluated at the input energies.
    """
    nu = _NEUTRINO_STRS[model.gen]
    return _dnde_two_body(model, product_energies, product, (nu, "omega"))


def dnde_v_phi(model: SingleRhNeutrinoModel, product_energies, product: str):
    """
    Compute the spectrum of a specified product from the decay of a right-handed
    neutrino into a phi and neutrino.

    Parameters
    ----------
    model: SingleRhNeutrinoModel
        Object containing the model parameters. Should implement the
        `SingleRhNeutrinoModel` protocol.
    product_energies: float or np.array
         Energies of the product where the spectrum should be computed.
    product: str
         The product to compute spectrum for.

    Returns
    -------
    dnde: float or array-like
        Differential energy spectrum evaluated at the input energies.
    """
    nu = _NEUTRINO_STRS[model.gen]
    return _dnde_two_body(model, product_energies, product, (nu, "phi"))


def dnde_l_pi(model: SingleRhNeutrinoModel, product_energies, product: str):
    """Compute the spectrum of a specified product from the decay of a
    right-handed neutrino into a charged pion and lepton.

    Parameters
    ----------
    model: SingleRhNeutrinoModel
        Object containing the model parameters. Should implement the
        `SingleRhNeutrinoModel` protocol.
    product_energies: float or np.array
         Energies of the product where the spectrum should be computed.
    product: str
         The product to compute spectrum for.

    Returns
    -------
    dnde: float or array-like
        Differential energy spectrum evaluated at the input energies.
    """
    ell = _LEPTON_STRS[model.gen]
    return _dnde_two_body(
        model=model,
        product_energies=product_energies,
        product=product,
        states=(ell, "pi"),
    )


def dnde_l_k(model: SingleRhNeutrinoModel, product_energies, product: str):
    r"""Compute the spectrum of a specified product from the decay of a
    right-handed neutrino into a charged kaon and lepton.

    Parameters
    ----------
    model: SingleRhNeutrinoModel
        Object containing the model parameters. Should implement the
        `SingleRhNeutrinoModel` protocol.
    product_energies: float or np.array
         Energies of the product where the spectrum should be computed.
    product: str
         The product to compute spectrum for.

    Returns
    -------
    dnde: float or array-like
        Differential energy spectrum evaluated at the input energies.
    """
    ell = _LEPTON_STRS[model.gen]
    return _dnde_two_body(model, product_energies, product, (ell, "k"))


def dnde_l_rho(model: SingleRhNeutrinoModel, product_energies, product: str):
    r"""Compute the spectrum of a product from the decay of a right-handed
    neutrino into a charged rho and lepton.

    Parameters
    ----------
    model: SingleRhNeutrinoModel
        Object containing the model parameters. Should implement the
        `SingleRhNeutrinoModel` protocol.
    product_energies: float or np.array
         Energies of the product where the spectrum should be computed.
    product: str
         The product to compute spectrum for.

    Returns
    -------
    dnde: float or array-like
        Differential energy spectrum evaluated at the input energies.
    """
    ell = _LEPTON_STRS[model.gen]
    return _dnde_two_body(model, product_energies, product, (ell, "k"))


def dnde_v_l_l(
    model: SingleRhNeutrinoModel,
    product_energies,
    product: str,
    genv: Generation,
    genl1: Generation,
    genl2: Generation,
    nbins: int = 30,
    three_body_integrator: str = "quad",
):
    r"""Compute the spectrum of a specified product from the decay of a
    right-handed neutrino into an active neutrino and two charged leptons.

    Parameters
    ----------
    model: SingleRhNeutrinoModel
        Object containing the model parameters. Should implement the
        `SingleRhNeutrinoModel` protocol.
    product_energies: float or np.array
         Energies of the product where the spectrum should be computed.
    product: str
         The product to compute spectrum for.
    form_factor: VectorFormFactorPiPi
        Pion electromagnetic form factor.
    genv: Generation
        Generations of the final state neutrinos 2 and 3. Generation of the
        first neutrino is assumed to be equal to the generation of the RHN.
    nbins: int
        Number of bins used to construct the underlying energy distributions.
        Default is 30.
    method: int
        Method used to integrate over distributions. Default is "quad".

    Returns
    -------
    dnde: float or array-like
        Differential energy spectrum evaluated at the input energies.
    """
    msqrd, _ = rhn_widths._make_msqrd_and_masses_v_l_l(model, genv, genl1, genl2)

    states = (_NEUTRINO_STRS[genv], _LEPTON_STRS[genl1], _LEPTON_STRS[genl2])
    logging.debug(f"dnde_v_l_l: states = {states}")
    return _dnde_three_body(
        model=model,
        product_energies=product_energies,
        product=product,
        states=states,
        msqrd=msqrd,
        nbins=nbins,
        three_body_integrator=three_body_integrator,
    )


def dnde_l_pi0_pi(
    model: SingleRhNeutrinoModel,
    product_energies,
    product: str,
    form_factor: VectorFormFactorPiPi,
    nbins: int = 30,
    three_body_integrator: str = "quad",
):
    r"""Compute the spectrum of a specified product from the decay of a
    right-handed neutrino into a charged pion, neutral pion and charged lepton.

    Parameters
    ----------
    model: SingleRhNeutrinoModel
        Object containing the model parameters. Should implement the
        `SingleRhNeutrinoModel` protocol.
    product_energies: float or np.array
         Energies of the product where the spectrum should be computed.
    product: str
         The product to compute spectrum for.
    form_factor: VectorFormFactorPiPi
        Pion electromagnetic form factor.
    genv: Generation
        Generations of the final state neutrinos 2 and 3. Generation of the
        first neutrino is assumed to be equal to the generation of the RHN.
    nbins: int
        Number of bins used to construct the underlying energy distributions.
        Default is 30.
    method: int
        Method used to integrate over distributions. Default is "quad".

    Returns
    -------
    dnde: float or array-like
        Differential energy spectrum evaluated at the input energies.
    """

    def msqrd(s, t):
        return rhn_widths.msqrd_l_pi0_pi(s, t, model, form_factor)

    states = (str(model.gen), "pi0", "pi")
    return _dnde_three_body(
        model=model,
        product_energies=product_energies,
        product=product,
        states=states,
        msqrd=msqrd,
        nbins=nbins,
        three_body_integrator=three_body_integrator,
    )


def dnde_v_pi_pi(
    model: SingleRhNeutrinoModel,
    product_energies,
    product: str,
    form_factor: VectorFormFactorPiPi,
    nbins: int = 30,
    three_body_integrator: str = "quad",
):
    """
    Compute the spectrum from the decay of a right-handed
    neutrino into an active neutrino and two charged pions.

    Parameters
    ----------
    model: SingleRhNeutrinoModel
        Object containing the model parameters. Should implement the
        `SingleRhNeutrinoModel` protocol.
    product_energies: float or np.array
         Energies of the product where the spectrum should be computed.
    product: str
         The product to compute spectrum for.
    form_factor: VectorFormFactorPiPi
        Pion electromagnetic form factor.
    genv: Generation
        Generations of the final state neutrinos 2 and 3. Generation of the
        first neutrino is assumed to be equal to the generation of the RHN.
    nbins: int
        Number of bins used to construct the underlying energy distributions.
        Default is 30.
    method: int
        Method used to integrate over distributions. Default is "quad".

    Returns
    -------
    dnde: float or array-like
        Differential energy spectrum evaluated at the input energies.
    """

    def msqrd(s, t):
        return rhn_widths.msqrd_v_pi_pi(s, t, model, form_factor)

    states = (_NEUTRINO_STRS[model.gen], "pi", "pi")
    return _dnde_three_body(
        model=model,
        product_energies=product_energies,
        product=product,
        states=states,
        msqrd=msqrd,
        nbins=nbins,
        three_body_integrator=three_body_integrator,
    )


def dnde_v_v_v(
    model: SingleRhNeutrinoModel,
    product_energies,
    product: str,
    genv: Generation,
    nbins: int = 30,
    three_body_integrator: str = "quad",
):
    """
    Compute the spectrum from the decay of a right-handed
    neutrino into three active neutrinos.

    Parameters
    ----------
    model: SingleRhNeutrinoModel
        Object containing the model parameters. Should implement the
        `SingleRhNeutrinoModel` protocol.
    product_energies: float or np.array
         Energies of the product where the spectrum should be computed.
    product: str
         The product to compute spectrum for.
    genv: Generation
        Generations of the final state neutrinos 2 and 3. Generation of the
        first neutrino is assumed to be equal to the generation of the RHN.
    nbins: int
        Number of bins used to construct the underlying energy distributions.
        Default is 30.
    method: int
        Method used to integrate over distributions. Default is "quad".

    Returns
    -------
    dnde: float or array-like
        Differential energy spectrum evaluated at the input energies.
    """

    def msqrd(s, t):
        return rhn_widths.msqrd_v_v_v(s, t, model, genv)

    states = (_NEUTRINO_STRS[model.gen], _NEUTRINO_STRS[genv], _NEUTRINO_STRS[genv])
    return _dnde_three_body(
        model=model,
        product_energies=product_energies,
        product=product,
        states=states,
        msqrd=msqrd,
        nbins=nbins,
        three_body_integrator=three_body_integrator,
    )
