r"""Approximate final-state radiation in the Altarelli-Parisi approximation.

.. note::

    The functions in this module give the FSR from a **single** radiating
    particle, and so carry a prefactor of :math:`\alpha_{\mathrm{em}}/2\pi`.
    Eq. (4.6) of the hazma paper (arXiv:1907.11846), and the deprecated
    ``hazma.utils.dnde_altarelli_parisi_*`` functions, instead give the result
    summed over a **pair** of oppositely-charged particles and carry
    :math:`\alpha_{\mathrm{em}}/\pi`. To reproduce Eq. (4.6), add the spectrum
    once per charged final-state particle, as ``hazma.spectra`` does.
"""

from typing import Any, Callable

import numpy as np

from hazma import parameters


def _scalar_splitting(x):
    return 2.0 * (1.0 - x) / x


def _fermion_splitting(x):
    return (1.0 + (1.0 - x) ** 2) / x


def _dndx_photon_fsr(  # noqa: PLR0913 — internal, mirrors the public signature
    x,
    s: float,
    m: float,
    split: Callable[[Any], Any],
    *,
    q=1.0,
    partner_mass: float | None = None,
):
    pre = q**2 * parameters.alpha_em / (2.0 * np.pi)
    xm = 1.0 - x
    kernel = np.zeros_like(x)

    # The spectrum stops at the photon endpoint x = 1 - (m + m_Y)²/s of a
    # radiator recoiling against a partner of mass m_Y, or earlier, at
    # x = 1 - e m²/s, where the bracket turns negative. An equal pair stops
    # at its endpoint with the bracket still ln 4 - 1 > 0, so its spectrum
    # ends in a step.
    m_y = m if partner_mass is None else partner_mass
    cutoff = max(np.e, (1.0 + m_y / m) ** 2)
    mask = s * xm / m**2 >= cutoff
    kernel[mask] = split(x[mask]) * (np.log(s * xm[mask] / m**2) - 1.0)

    return pre * kernel


def _dnde_photon_fsr(  # noqa: PLR0913 — internal, mirrors the public signature
    e,
    s: float,
    m: float,
    split: Callable[[Any], Any],
    *,
    q=1.0,
    partner_mass: float | None = None,
):
    e_to_x = 2.0 / np.sqrt(s)
    x = e * e_to_x
    return _dndx_photon_fsr(x, s, m, split, q=q, partner_mass=partner_mass) * e_to_x


def dndx_photon_ap_fermion(
    x, s: float, mass: float, charge=1.0, partner_mass: float | None = None
):
    """
    Compute dndx from the FSR off a charged fermion f from a
    process of the form X -> (f + Y + gamma) + Z.

    Parameters
    ----------
    x: array_like
        Scaled energies of the fermion, x = 2E/sqrt(s).
    s: float
        Squared center-of-mass energy flowing through of the radiating
        fermion and Y in the process X -> (f + Y) + Z.
    mass: float
        Mass of the radiating fermion.
    charge: float
        Charge of the radiating fermion. Default is 1.
    partner_mass: float or None, optional
        Mass of Y, the particle the radiator recoils against, in MeV. It sets
        the photon endpoint x = 1 - (mass + partner_mass)²/s. Default is
        None, which takes Y to have the radiator's own mass, as in a pair of
        oppositely-charged particles.
    """
    return _dndx_photon_fsr(
        x, s, mass, _fermion_splitting, q=charge, partner_mass=partner_mass
    )


def dndx_photon_ap_scalar(x, s, mass, charge=1.0, partner_mass: float | None = None):
    """
    Compute dndx from the FSR off a charged scalar f from a
    process of the form X -> (f + Y + gamma) + Z.

    Parameters
    ----------
    x: array_like
        Scaled energies of the fermion, x = 2E/sqrt(s).
    s: float
        Squared center-of-mass energy flowing through of the radiating
        scalar and Y in the process X -> (f + Y) + Z.
    mass: float
        Mass of the radiating scalar.
    charge: float
        Charge of the radiating scalar. Default is 1.
    partner_mass: float or None, optional
        Mass of Y, the particle the radiator recoils against, in MeV. It sets
        the photon endpoint x = 1 - (mass + partner_mass)²/s. Default is
        None, which takes Y to have the radiator's own mass, as in a pair of
        oppositely-charged particles.
    """
    return _dndx_photon_fsr(
        x, s, mass, _scalar_splitting, q=charge, partner_mass=partner_mass
    )


def dnde_photon_ap_fermion(e, s, mass, charge=1.0, partner_mass: float | None = None):
    """
    Compute dN/dE from the FSR off a charged fermion f from a
    process of the form X -> (f + Y + gamma) + Z.

    Parameters
    ----------
    e: array_like
        Photon energies.
    s: float
        Squared center-of-mass energy flowing through of the radiating
        fermion and Y in the process X -> (f + Y) + Z.
    mass: float
        Mass of the radiating fermion.
    charge: float
        Charge of the radiating fermion. Default is 1.
    partner_mass: float or None, optional
        Mass of Y, the particle the radiator recoils against, in MeV. It sets
        the photon endpoint x = 1 - (mass + partner_mass)²/s. Default is
        None, which takes Y to have the radiator's own mass, as in a pair of
        oppositely-charged particles.
    """
    return _dnde_photon_fsr(
        e, s, mass, _fermion_splitting, q=charge, partner_mass=partner_mass
    )


def dnde_photon_ap_scalar(e, s, mass, charge=1.0, partner_mass: float | None = None):
    """
    Compute dN/dE from the FSR off a charged scalar f from a
    process of the form X -> (f + Y + gamma) + Z.

    Parameters
    ----------
    e: array_like
        Photon energy.
    s: float
        Squared center-of-mass energy flowing through of the radiating
        scalar and Y in the process X -> (f + Y) + Z.
    mass: float
        Mass of the radiating scalar.
    charge: float
        Charge of the radiating scalar. Default is 1.
    partner_mass: float or None, optional
        Mass of Y, the particle the radiator recoils against, in MeV. It sets
        the photon endpoint x = 1 - (mass + partner_mass)²/s. Default is
        None, which takes Y to have the radiator's own mass, as in a pair of
        oppositely-charged particles.
    """
    return _dnde_photon_fsr(
        e, s, mass, _scalar_splitting, q=charge, partner_mass=partner_mass
    )
