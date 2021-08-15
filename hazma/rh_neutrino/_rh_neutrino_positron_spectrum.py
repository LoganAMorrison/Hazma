"""
This file contains function for computing the positron spectrum from
right-handed neutrino decays.
"""

from hazma.positron_spectra import charged_pion as pspec_charged_pion
from hazma.parameters import charged_pion_mass as mpi


def dnde_pos_pi_l(self, positron_energies):
    """
    Compute the positron spectrum from the decay of a right-handed neutrino
    into a charged pion and lepton.

    Parameters
    ----------
    positron_energies: float or np.array
        Energies of the positrons.

    Returns
    -------
    spec: float or np.array
        Positron spectrum.
    """
    epi = (self.mx ** 2 + mpi ** 2 - self.ml ** 2) / (2.0 * self.mx)
    return pspec_charged_pion(positron_energies, epi)
