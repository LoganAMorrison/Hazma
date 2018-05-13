"""
Module for computing positron spectra.

@author: Logan Morrison and Adam Coogan
@date: May 2018

"""

from positron_helper_functions import positron_muon


def muon(ee, emu):
    """
    Returns the positron spectrum from a muon.

    Parameters
    ----------
    ee : float or array-like
        Energy of the positron.
    emu : float or array-like
        Energy of the muon.

    Returns
    -------
    dnde : float or array-like
        The value of the spectrum given a positron energy `ee`
        and muon energy `emu`
    """
    if hasattr(ee, "__len__"):
        return positron_muon.Spectrum(ee, emu)
    return positron_muon.SpectrumPoint(ee, emu)


"""
def __dnde_cpion(ee):
    emu = (mmu**2 / mpi + mpi) / 2.
    gamma = emu / mmu
    beta = np.sqrt(1. - 1. / gamma**2)

    def integrand(c2):
        ee1 = gamma * ee * (1. - beta * c2)
        return dnde_muon(ee1) / 2. / gamma / abs(1. - beta * c2)

    return quad(integrand, -1., 1.)[0]


def dnde_cpion(ee):
    if hasattr(ee, "__len__"):
        return np.array([__dnde_cpion(e) for e in ee])
    return __dnde_cpion(ee)
"""
