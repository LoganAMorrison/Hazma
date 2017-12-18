import numpy as np
cimport numpy as np
import cython


cdef double CSpectrumPoint(double eng_gam, double eng_mu):
    """
    Compute dN_{\gamma}/dE_{\gamma} from mu -> e nu nu gamma in the
    laborartory frame.

    Keyword arguments::
        eng_gam (float) -- Gamma ray energy in laboratory frame.
        eng_mu (float) -- Muon energy in laboratory frame.
    """
    cdef double result = 0.0

    return result


@cython.cdivision(True)
cdef np.ndarray CSpectrum(np.ndarray eng_gams, double eng_mu):
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
    cdef np.ndarray spec = np.zeros(len(eng_gams), dtype=np.float64)

    return spec
