from ..fsr_helper_functions.scalar_mediator_fsr import fermion
import numpy as np

MASS_E = 0.510998928


def decay_spectra(eng_gam, eng_mu):
    """
    Returns zero. Electron is stable.
    """
    return 0.0


def fsr(eng_gam, cme, mediator='scalar'):
    """
    Compute muon fsr spectrum.

    Compute final state radiation spectrum dN/dE from decay of an off-shell
    mediator (scalar, psuedo-scalar, vector or axial-vector) into a pair of
    electronss.

    Paramaters
        eng_gam (float or np.ndarray) :
            Gamma ray energy(ies) in laboratory frame.
        cme (float) :
            Center of mass energy or mass of the off-shell mediator.
        mediator (string) :
            Mediator type : scalar, psuedo-scalar, vector or axial-vector.

    Returns
        spec (np.ndarray) :
            List of gamma ray spectrum values, dNdE, evaluated at `eng_gams`
            given a center of mass energy `cme`.

    Examples
        dNdE for a single gamma ray energy from scalar mediator.

        >>> from hazma import electron
        >>> eng_gam, cme = 200., 1000.
        >>> spec = electron.fsr(eng_gam, cme, 'scalar')

        dNdE for list of gamma ray energies from vector mediator.

        >>> from hazma import electron
        >>> eng_gams = np.logspace(0.0, 3.0, num=1000, dtype=float)
        >>> cme = 1000.
        >>> spec = electron.fsr(eng_gams, cme, 'scalar')
    """
    if mediator == 'scalar':
        if hasattr(eng_gam, "__len__"):
            fermionvec = np.vectorize(fermion)
            return fermionvec(eng_gam, cme, MASS_E)
        return fermion(eng_gam, cme, MASS_E)
    if mediator == 'psuedo-scalar':
        raise ValueError('psuedo-scalar mediator not yet availible.')
    if mediator == 'vector':
        raise ValueError('vector mediator not yet availible.')
    if mediator == 'axial-vector':
        raise ValueError('axial-vector mediator not yet availible.')
    else:
        raise ValueError(
            '''Invalid mediator: {}. Use scalar, psuedo-scalar,\
               vector or axial-vector.''')
