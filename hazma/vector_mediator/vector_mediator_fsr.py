from ..parameters import alpha_em, qe
from ..parameters import charged_pion_mass as mpi
from ..parameters import electron_mass as me
from ..parameters import muon_mass as mmu

from cmath import sqrt, log, pi
import numpy as np


def __dnde_xx_to_v_to_ffg(egam, Q, f, params):
    """Return the fsr spectra for fermions from decay of vector mediator.

    Computes the final state radiaton spectrum value dNdE from a vector
    mediator given a gamma ray energy of `egam`, center of mass energy `Q`
    and final state fermion `f`.

    Paramaters
    ----------
    egam : float
        Gamma ray energy.
    Q: float
        Center of mass energy of mass of off-shell vector mediator.
    f : float
        Name of the final state fermion: "e" or "mu".

    Returns
    -------
    spec_val : float
        Spectrum value dNdE from vector mediator.
    """
    if f == "e":
        mf = me
    elif f == "mu":
        mf = mmu

    e, m = egam / Q, mf / Q

    s = Q**2 - 2. * Q * egam

    ret_val = 0.0

    if 4. * mf**2 <= s <= Q**2 and Q > 2. * mf and Q > 2. * params.mx:
        ret_val = -(alpha_em *
                    (4. * sqrt(1. - 2. * e - 4. * m**2) *
                     (1. - 2. * m**2 + 2. * e * (-1 + e + 2. * m**2)) +
                     sqrt(1. - 2. * e) * (1. + 2. * (-1 + e) * e -
                                          4. * e * m**2 - 4. * m**4) *
                     (log(1. - 2. * e) - 4. *
                      log(sqrt(1. - 2. * e) +
                          sqrt(1. - 2. * e - 4. * m**2)) +
                        2. * log((sqrt(1. - 2. * e) -
                                  sqrt(1. - 2. * e - 4. * m**2)) *
                                 (1. - sqrt(1. + (4. * m**2) /
                                            (-1 + 2. * e))))))) / \
            (2. * e * (1. + 2. * m**2) *
             sqrt((-1 + 2. * e) * (-1 + 4. * m**2)) * pi * Q)

        assert ret_val.imag == 0.

        ret_val = ret_val.real

        assert ret_val >= 0

    return ret_val


def dnde_xx_to_v_to_ffg(egam, Q, f, params):
    """Return the fsr spectra for fermions from decay of vector mediator.

    Computes the final state radiaton spectrum value dNdE from a vector
    mediator given a gamma ray energy of `egam`, center of mass energy `Q`
    and final state fermion `f`.

    Paramaters
    ----------
    egam : float
        Gamma ray energy.
    Q: float
        Center of mass energy of mass of off-shell vector mediator.
    f : float
        Mass of the final state fermion.

    Returns
    -------
    spec_val : float
        Spectrum value dNdE from vector mediator.
    """
    if hasattr(egam, '__len__'):
        return np.array([__dnde_xx_to_v_to_ffg(e, Q, f, params)
                         for e in egam])
    else:
        return __dnde_xx_to_v_to_ffg(egam, Q, f, params)


def __dnde_xx_to_v_to_pipig(egam, Q, params):
    """Unvectorized dnde_xx_to_v_to_pipig"""
    mx = params.mx
    mv = params.mv

    # Make sure the COM energy is large enough for this process to occur
    if Q < 2. * mpi or Q < 2. * mx:
        return 0.
    # Make sure the photon energy is in bounds
    if egam <= 0 or egam >= (Q**2 - 4. * mpi**2) / (2. * Q):
        return 0.

    ret_val = -(Q * (-(mv**2 * Q) + Q**3)**2 * qe**2 *
                ((2 * sqrt((2 * egam**2 - 2 * egam * Q + Q**2) *
                           (2 * egam**2 - 4 * mpi**2 -
                            2 * egam * Q + Q**2)) *
                  (4 * egam**2 * (egam**2 + 2 * mpi**2) -
                   8 * egam * (egam**2 + mpi**2) * Q + 2 *
                   (egam**2 + 2 * mpi**2) * Q**2 +
                   2 * egam * Q**3 - Q**4)) /
                 (egam * (egam - Q) * (2 * egam**2 - 2 * egam * Q + Q**2)) +
                 (2 * (-4 * mpi**2 + Q**2) *
                  (2 * egam**2 - 2 * mpi**2 -
                   2 * egam * Q + Q**2) *
                  log((2 * egam**2 - 2 * egam * Q + Q**2 -
                       sqrt((2 * egam**2 - 2 * egam * Q + Q**2) *
                            (2 * egam**2 - 4 * mpi**2 -
                             2 * egam * Q + Q**2))) /
                      (2 * egam**2 - 2 * egam * Q + Q**2 +
                       sqrt((2 * egam**2 - 2 * egam * Q + Q**2) *
                            (2 * egam**2 - 4 * mpi**2 -
                             2 * egam * Q + Q**2))))) /
                 (egam * (-egam + Q)))) / \
        (4. * pi**2 * (-4 * mpi**2 + Q**2)**1.5 *
         (-mv**2 + Q**2)**2 * (2 * mx**2 * Q + Q**3))

    # assert ret_val >= 0.

    return ret_val


def dnde_xx_to_v_to_pipig(eng_gams, Q, params):
    """
    Returns the gamma ray energy spectrum for two fermions annihilating
    into two charged pions and a photon.

    Parameters
    ----------
    eng_gam : numpy.ndarray or double
        Gamma ray energy.
    Q : double
        Center of mass energy, or sqrt((ppip + ppim + pg)^2).

    Returns
    -------
    Returns gamma ray energy spectrum for
    :math:`\chi\bar{\chi}\to\pi^{+}\pi^{-}\gamma` evaluated at the gamma
    ray energy(ies).

    """

    if hasattr(eng_gams, '__len__'):
        return np.array([__dnde_xx_to_v_to_pipig(eng_gam, Q, params)
                         for eng_gam in eng_gams])
    else:
        return __dnde_xx_to_v_to_pipig(eng_gams, Q, params)


# TODO: FSR for pi0 pi pi final state!
