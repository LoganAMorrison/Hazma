"""
Module for computing the vector form factor for pi+pi.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np  # type:ignore
from scipy.special import gamma  # type:ignore

from hazma.vector_mediator.form_factors.utils import (MPI_GEV, breit_wigner_fw,
                                                      breit_wigner_gs, dhhatds,
                                                      gamma_generator, h, hhat)


@dataclass(frozen=True)
class FormFactorPiPiParameters:
    """
    Class for storing the parameters needed to compute the form factor for
    V-pi-pi.
    """
    omega_mag: complex
    omega_phase: complex
    omega_mass: complex
    omega_width: complex
    omega_weight: complex
    mass: np.ndarray
    width: np.ndarray
    coup: np.ndarray
    hres: np.ndarray
    h0: np.ndarray
    dh: np.ndarray


def compute_pipi_form_factor_parameters(
        n_max: int = 2000
) -> FormFactorPiPiParameters:
    """
    Compute the parameters needed for computing the V-pi-pi form factor.

    Parameters
    ----------
    n_max: int

    Returns
    -------
    """
    # Set up the parameters.
    omega_mag = 0.00187 + 0j
    omega_phase = 0.106 + 0j
    omega_mass = 0.7824 + 0j
    omega_width = 0.00833 + 0j
    omega_wgt = 0j

    mass = np.zeros(n_max, dtype=np.float64)
    width = np.zeros(n_max, dtype=np.float64)
    coup = np.zeros(n_max, dtype=np.complex128)
    hres = np.zeros(n_max, dtype=np.float64)
    h0 = np.zeros(n_max, dtype=np.float64)
    dh = np.zeros(n_max, dtype=np.float64)

    # Set the rho-parameters
    rho_mag = np.array([1.0, 1.0, 0.59, 4.8e-2, 0.40, 0.43], dtype=np.float64)
    rho_phase = np.array([0.0, 0.0, -2.2, -2.0, -2.9, 1.19], dtype=np.float64)
    rho_masses = np.array([0.77337, 1.490, 1.870, 2.12,
                           2.321, 2.567], dtype=np.float64)
    rho_widths = np.array(
        [0.1471, 0.429, 0.357, 0.3, 0.444, 0.491], dtype=np.float64)
    rho_wgt = rho_mag * np.exp(1j * rho_phase)

    beta = 2.148

    # Compute the two of couplings
    omega_wgt = omega_mag * np.exp(1j * omega_phase)
    # set up the masses and widths of the rho resonances
    ixs = np.arange(n_max)
    gam_b = np.array([val for val in gamma_generator(beta, n_max)])
    gam_0 = gamma(beta - 0.5)

    coup = (
        gam_0
        / (0.5 + ixs)
        / np.sqrt(np.pi)
        * np.sin(np.pi * (beta - 1.0 - ixs))
        / np.pi
        * gam_b
        + 0j
    )
    coup[0] = 1.087633403691967
    coup[1::2] *= -1

    # set the masses and widths
    # calc for higher resonances
    mass = rho_masses[0] * np.sqrt(1.0 + 2.0 * ixs)
    mass[:len(rho_masses)] = rho_masses

    width = rho_widths[0] / rho_masses[0] * mass
    width[:len(rho_widths)] = rho_widths

    # parameters for the gs propagators
    hres = hhat(mass ** 2, mass, width, MPI_GEV, MPI_GEV)  # type: ignore
    dh = dhhatds(mass, width, MPI_GEV, MPI_GEV)  # type: ignore
    h0 = h(0.0, mass, width, MPI_GEV, MPI_GEV, dh, hres)  # type: ignore

    # fix up the early weights
    cwgt = np.sum(coup[1:len(rho_wgt)])
    rho_sum = np.sum(rho_wgt[1:])
    coup[1:len(rho_wgt) + 1] = rho_wgt * cwgt / rho_sum

    return FormFactorPiPiParameters(
        omega_mag,
        omega_phase,
        omega_mass,
        omega_width,
        omega_wgt,
        mass,
        width,
        coup,
        hres,
        h0,
        dh
    )


def form_factor_pipi(
        s: float,
        params: FormFactorPiPiParameters,
        ci1: float,
        imode: Optional[int] = 1
):
    if hasattr(s, '__len__'):
        ss = np.array(s)
    else:
        ss = np.array([s])

    ff = ci1 * params.coup * breit_wigner_gs(
        ss,
        params.mass,
        params.width,
        MPI_GEV,
        MPI_GEV,
        params.h0,
        params.dh,
        params.hres,
        reshape=True,
    )

    # include rho-omega if needed
    if imode != 0:
        ff[:, 0] *= (
            1.0
            / (1.0 + params.omega_weight)
            * (
                1.0
                + params.omega_weight
                * breit_wigner_fw(
                    s,
                    params.omega_mass,
                    params.omega_weight
                )
            )
        )
    # sum
    ff = np.sum(ff, axis=1)
    # factor for cc mode
    if imode == 0:
        ff *= np.sqrt(2.0)
    if len(ff) == 1:
        return ff[0]
    return ff
