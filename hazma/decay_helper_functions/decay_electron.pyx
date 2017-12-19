import numpy as np
cimport numpy as np
import cython
include "parameters.pxd"

cdef double CSpectrumPoint(double eng_gam, double eng_e):
    """
    Compute dN_{\gamma}/dE_{\gamma} from mu -> e nu nu gamma in the
    laborartory frame.

    Keyword arguments::
        eng_gam (float) -- Gamma ray energy in laboratory frame.
        eng_mu (float) -- Muon energy in laboratory frame.
    """
    if eng_e < MASS_E:
        raise ValueError('Energy of electron cannot be less than the electron mass.')

    cdef double result = 0.0

    return result


@cython.cdivision(True)
cdef np.ndarray CSpectrum(np.ndarray eng_gams, double eng_e):
    """
    Compute dN/dE from mu -> e nu nu gamma in the laborartory frame.

    Paramaters
    ----------
    eng_gams : np.ndarray
        List of gamma ray energies in laboratory frame.
    eng_mu : float
        Muon energy in laboratory frame.

    Returns
    -------
    spec : np.ndarray
        List of gamma ray spectrum values, dNdE, evaluated at `eng_gams`
        given muon energy `eng_mu`.
    """
    if eng_e < MASS_E:
        raise ValueError('Energy of electron cannot be less than the electron mass.')
    cdef np.ndarray spec = np.zeros(len(eng_gams), dtype=np.float64)

    return spec
