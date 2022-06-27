"""
This file contains the decay spectra from a right-handed neutrino at rest.
"""
from typing import Callable, Tuple

import numpy as np

from hazma import spectra
from hazma.form_factors.vector import VectorFormFactorPiPi

from ._proto import SingleRhNeutrinoModel
from . import _widths as rhn_widths


_NEUTRINO_STRS = {"e": "ve", "mu": "vm", "tau": "vt"}
_DISPATCH = {
    "photon": spectra.dnde_photon,
    "positron": spectra.dnde_positron,
    "neutrino": spectra.dnde_neutrino,
}


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
    dnde = _DISPATCH.get(product)
    assert dnde is not None, f"Invalid product {product}."
    return dnde(product_energies, model.mx, states)


def _dnde_three_body(
    model: SingleRhNeutrinoModel,
    product_energies,
    product: str,
    states: Tuple[str, str, str],
    msqrd: Callable,
    nbins: int,
    method: str,
):
    dnde = _DISPATCH.get(product)
    assert dnde is not None, f"Invalid product {product}."
    return dnde(
        product_energies, model.mx, states, msqrd=msqrd, nbins=nbins, method=method
    )


def dnde_v_pi0(model: SingleRhNeutrinoModel, product_energies, product: str):
    """
    Compute the gamma-ray spectrum from the decay of a right-handed
    neutrino into a neutral pion and neutrino.

    Parameters
    ----------
    photon_energies: float or np.array
        Photon energies where the spectrum should be computed.
    """
    nu = _NEUTRINO_STRS[model.flavor]
    return _dnde_two_body(model, product_energies, product, (nu, "pi0"))


def dnde_v_eta(model: SingleRhNeutrinoModel, product_energies, product: str):
    """
    Compute the gamma-ray spectrum from the decay of a right-handed
    neutrino into a eta and neutrino.

    Parameters
    ----------
    photon_energies: float or np.array
        Photon energies where the spectrum should be computed.
    """
    nu = _NEUTRINO_STRS[model.flavor]
    return _dnde_two_body(model, product_energies, product, (nu, "pi0"))


def dnde_l_pi(model: SingleRhNeutrinoModel, product_energies, product: str):
    """
    Compute the gamma-ray spectrum from the decay of a right-handed
    neutrino into a charged pion and lepton.

    Parameters
    ----------
    photon_energies: float or np.array
        Photon energies where the spectrum should be computed.
    """
    return _dnde_two_body(model, product_energies, product, (model.flavor, "pi"))


def dnde_l_k(model: SingleRhNeutrinoModel, product_energies, product: str):
    """
    Compute the gamma-ray spectrum from the decay of a right-handed
    neutrino into a charged kaon and lepton.

    Parameters
    ----------
    photon_energies: float or np.array
        Photon energies where the spectrum should be computed.
    """
    return _dnde_two_body(model, product_energies, product, (model.flavor, "k"))


def dnde_v_l_l(model: SingleRhNeutrinoModel, photon_energies, j, n, m, nbins: int):
    """
    Compute the gamma-ray spectrum from the decay of a right-handed
    neutrino into an active neutrino and two charged leptons.

    Parameters
    ----------
    photon_energies: float or np.array
        Photon energies where the spectrum should be computed.
    """
    return np.zeros_like(photon_energies)


def dnde_l_pi0_pi(
    model: SingleRhNeutrinoModel,
    product_energies,
    form_factor: VectorFormFactorPiPi,
    product: str,
    nbins: int = 30,
    method: str = "quad",
):
    """
    Compute the gamma-ray spectrum from the decay of a right-handed
    neutrino into a charged pion, neutral pion and charged lepton.

    Parameters
    ----------
    photon_energies: float or np.array
        Photon energies where the spectrum should be computed.
    """

    def msqrd(s, t):
        return rhn_widths.msqrd_l_pi0_pi(model, s, t, form_factor)

    states = (model.flavor, "pi0", "pi")
    return _dnde_three_body(
        model=model,
        product_energies=product_energies,
        product=product,
        states=states,
        msqrd=msqrd,
        nbins=nbins,
        method=method,
    )


def dnde_nu_pi_pi(
    model: SingleRhNeutrinoModel,
    product_energies,
    form_factor: VectorFormFactorPiPi,
    product: str,
    nbins: int = 30,
    method: str = "quad",
):
    """
    Compute the spectrum from the decay of a right-handed
    neutrino into an active neutrino and two charged pions.

    Parameters
    ----------
    photon_energies: float or np.array
        Photon energies where the spectrum should be computed.
    """

    def msqrd(s, t):
        return rhn_widths.msqrd_v_pi_pi(model, s, t, form_factor)

    states = (_NEUTRINO_STRS[model.flavor], "pi", "pi")
    return _dnde_three_body(
        model=model,
        product_energies=product_energies,
        product=product,
        states=states,
        msqrd=msqrd,
        nbins=nbins,
        method=method,
    )
