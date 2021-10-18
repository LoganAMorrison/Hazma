"""
Module for computing the V-K-K form factor.
"""


from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from scipy.special import gamma  # type:ignore

from hazma.vector_mediator.form_factors.utils import (
    MK0_GEV,
    MKP_GEV,
    MPI_GEV,
    breit_wigner_fw,
    breit_wigner_gs,
    breit_wigner_pwave,
    dhhatds,
    gamma_generator,
    h,
    hhat,
)


@dataclass(frozen=True)
class FormFactorKKParameters:
    """
    Class for storing the parameters needed to compute the form factor for
    V-K-K.
    """

    rho_mass: npt.NDArray[np.float64]
    rho_width: npt.NDArray[np.float64]
    rho_coup: npt.NDArray[np.complex128]

    hres: npt.NDArray[np.float64]
    h0: npt.NDArray[np.float64]
    dh: npt.NDArray[np.float64]

    omega_mass: npt.NDArray[np.float64]
    omega_width: npt.NDArray[np.float64]
    omega_coup: npt.NDArray[np.complex128]

    phi_mass: npt.NDArray[np.float64]
    phi_width: npt.NDArray[np.float64]
    phi_coup: npt.NDArray[np.complex128]


def _find_beta(c0):
    """
    Find the beta parameter.
    """

    def c_0(b0):
        ratio = 2.0 / np.sqrt(np.pi)
        b1 = b0
        while b1 > 2.0:
            ratio *= (b1 - 1.5) / (b1 - 2.0)
            b1 -= 1.0
        ratio *= gamma(b1 - 0.5) / gamma(b1 - 1.0)
        return ratio

    betamin = 1.0
    betamid = 5.0
    betamax = 10.0
    eps = 1e-10
    cmid = c_0(betamid)
    while abs(cmid - c0) > eps:
        cmin = c_0(betamin)
        cmax = c_0(betamax)
        cmid = c_0(betamid)
        if c0 > cmin and c0 < cmid:
            betamax = betamid
        elif c0 > cmid and c0 < cmax:
            betamin = betamid
        elif c0 >= cmax:
            betamax *= 2.0
        else:
            print(f"bisect fails: {betamin}, {betamid}, {betamax}, {c0}")
            print(f"bisect fails: {cmin}, {cmid}, {cmax}, {c0}")
            raise RuntimeError("Failed to find beta.")
        betamid = 0.5 * (betamin + betamax)
    return betamid


def _compute_masses_width_couplings(
    masses: np.ndarray,
    widths: np.ndarray,
    magnitudes: np.ndarray,
    phases: np.ndarray,
    n_max: int,
    gam: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    # Get beta
    beta = _find_beta(magnitudes[0])

    # weights
    wgt = magnitudes * np.exp(1j * phases)

    # compute the couplings
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
    coup[1::2] *= -1

    # set up the masses and widths of the resonances
    mass = masses[0] * np.sqrt(1.0 + 2.0 * ixs)
    mass[: len(masses)] = masses

    width = mass * (widths[0] / masses[0] if gam is None else gam)
    width[: len(widths)] = widths

    # reset the parameters for the low lying resonances
    # set the masses and widths
    # couplings
    coup[: len(wgt)] = wgt
    total = np.sum(coup)
    coup[len(wgt)] += 1.0 - total

    return {
        "mass": mass,
        "width": width,
        "coup": coup,
    }


def _compute_h_parameters(
    mass: npt.NDArray[np.float64], width: npt.NDArray[np.float64], m: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Hhat, H(0) and the derivative of Hhat needed to compute the
    Gounaris-Sakurai Breit-Wigner function with pion loop corrections included.
    See ArXiv:1002.0279 Eqn.(2) for details.

    Parameters
    ----------
    mass: np.ndarray
        Masses of the resonances.
    width: np.ndarray
        Widths of the resonances.
    m: float
    """
    hres = np.array(hhat(mass ** 2, mass, width, m, m))
    dh = np.array(dhhatds(mass, width, m, m))
    h0 = np.array(h(0.0, mass, width, m, m, dh, hres))
    return hres, dh, h0


def compute_kk_form_factor_parameters(n_max: int = 200) -> FormFactorKKParameters:
    """
    Compute the parameters needed for computing the V-K-K form factor.

    Parameters
    ----------
    n_max: int
        Number of resonances to include.

    Returns
    -------
    params: FormFactorKKParameters
        Parameters of the resonances for the V-K-K form factor.
    """
    # initial parameters for the model
    # beta_rho = 2.1968
    # beta_omega = 2.6936
    # beta_phi = 1.9452
    gamma_omega = 0.5
    gamma_phi = 0.2

    # rho parameters
    rho_mag = np.array(
        [
            1.1148916618504967,
            -0.050374779737077324,
            -0.014908906283692132,
            -0.03902475997619905,
            -0.038341465215871416,
        ]
    )
    rho_phase = np.zeros(5, np.float64)
    rho_masses = np.array(
        [0.77549, 1.5206995754050117, 1.7409719246639341, 1.9922811314327789]
    )
    rho_widths = np.array(
        [0.1494, 0.21341728317817743, 0.08412224414791908, 0.2899733272437917]
    )
    # omega parameters
    omega_mag = np.array(
        [
            1.3653229680598022,
            -0.02775156567495144,
            -0.32497165559032715,
            1.3993153161869765,
        ]
    )
    omega_phase = np.zeros(len(omega_mag), np.float64)
    omega_masses = np.array([0.78265, 1.4144344268685891, 1.655375231284883])
    omega_widths = np.array([8.49e-3, 0.0854413887755723, 0.16031760444832305])
    # phi parameters
    phi_mag = np.array(
        [
            0.965842498579515,
            -0.002379766320723148,
            -0.1956211640216197,
            0.16527771485190898,
        ]
    )
    phi_phase = np.zeros(len(phi_mag), np.float64)
    phi_masses = np.array([1.0194209171596993, 1.594759278457624, 2.156971341201067])
    phi_widths = np.array(
        [4.252653332329334e-3, 0.028741821847408196, 0.6737556174184005]
    )

    rho_params = _compute_masses_width_couplings(
        rho_masses,
        rho_widths,
        rho_mag,
        rho_phase,
        n_max,
    )
    hres, dh, h0 = _compute_h_parameters(
        rho_params["mass"],
        rho_params["width"],
        MPI_GEV,
    )

    omega_params = _compute_masses_width_couplings(
        omega_masses,
        omega_widths,
        omega_mag,
        omega_phase,
        n_max,
        gam=gamma_omega,
    )

    phi_params = _compute_masses_width_couplings(
        phi_masses,
        phi_widths,
        phi_mag,
        phi_phase,
        n_max,
        gam=gamma_phi,
    )

    return FormFactorKKParameters(
        rho_params["mass"],
        rho_params["width"],
        rho_params["coup"],
        hres,
        h0,
        dh,
        omega_params["mass"],
        omega_params["width"],
        omega_params["coup"],
        phi_params["mass"],
        phi_params["width"],
        phi_params["coup"],
    )


def form_factor_kk(
    s: float,
    params: FormFactorKKParameters,
    gvuu: float,
    gvdd: float,
    gvss: float,
    imode: int,
) -> Union[complex, np.ndarray]:
    """
    Compute the form factor for the V-K-K interaction.

    Parameters
    ----------
    s: Union[float, np.ndarray]
        Center-of-mass energies squared.
    params: FormFactorKKParameters
        Parameters for computing the V-K-K form factor.
    gvuu: float
        Coupling of the vector to up-quarks.
    gvdd: float
        Coupling of the vector to down-quarks.
    gvss: float
        Coupling of the vector to strange-quarks.
    imode: int
        If imode = 0, the form factor is for V-K0-K0 and if imode = 1 then
        the form factor is for V-Kp-Km.

    Returns
    -------
    fk: Union[complex, np.ndarray]
        Form factor for V-K-K.
    """
    mk = MK0_GEV if imode == 0 else MKP_GEV
    eta_phi = 1.055

    # TODO Check these couplings. They look wrong.
    ci0 = 3.0 * (gvuu + gvdd)
    ci1 = gvuu - gvdd
    cs = -3.0 * gvss

    # Force s into an array. This makes vectorization in the case where s is
    # an array easier.
    if hasattr(s, "__len__"):
        ss = np.array(s)
    else:
        ss = np.array([s])

    # Rho exchange
    fk = (
        ci1
        * params.rho_coup
        * breit_wigner_gs(
            ss,
            params.rho_mass,
            params.rho_width,
            MPI_GEV,
            MPI_GEV,
            params.h0,
            params.dh,
            params.hres,
            reshape=True,
        )
    )
    fk *= 0.5 * (-1.0 if imode == 0 else 1.0)

    # Omega exchange
    fk += (
        1.0
        / 6.0
        * ci0
        * params.omega_coup
        * breit_wigner_fw(
            ss,
            params.omega_mass,
            params.omega_width,
            reshape=True,
        )
    )

    # Phi-exchange
    bwp = breit_wigner_pwave(
        ss,
        params.phi_mass,
        params.phi_width,
        mk,
        mk,
        reshape=True,
    )

    phi_terms = 1.0 / 3.0 * cs * params.phi_coup * bwp
    phi_terms[0] *= eta_phi if imode == 0 else 1.0
    fk += phi_terms

    fk = np.sum(fk, axis=1)

    if len(fk) == 1:
        return fk[0]
    return fk
