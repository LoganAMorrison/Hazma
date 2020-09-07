"""
This file contains implemenations of the FSR spectra from RH neutrino decays.
"""

import numpy as np
from hazma.parameters import fpi, GF, qe, Vud, charged_pion_mass as mpi


class RHNeutrinoFSR:
    def __dnde_pi_l_fsr(self, photon_eng):
        """
        Compute the FSR spectra from a right-handed neutrino decay into a
        charged pion and lepton.

        Parameters
        ----------
        photon_eng: float
           The energy of the final state photon in MeV.
        mvr: float
            Right-handed neutrino mass in MeV.
        stheta: float
            Mixing angle between right-handed and active neutrinos.
        ml: float
            Mass of the final-state lepton in MeV.

        Returns
        -------
        dnde: float
            Photon spectrum.
        """
        mvr = self.mx
        ml = self.ml
        s = mvr ** 2 - 2.0 * mvr * photon_eng
        if mvr < mpi + ml or (mpi + ml) ** 2 >= s or mvr ** 2 <= s:
            return 0.0

        return (
            qe ** 2
            * (
                (
                    -2 * ml ** 6
                    + 2 * ml ** 4 * (2 * mvr ** 2 + s)
                    + ml ** 2
                    * (
                        2 * mpi ** 4
                        - mvr ** 4
                        + mpi ** 2 * (6 * mvr ** 2 - 2 * s)
                        - 6 * mvr ** 2 * s
                        + s ** 2
                    )
                    + mvr ** 2
                    * (
                        2 * mpi ** 4
                        + mvr ** 4
                        + s ** 2
                        - 2 * mpi ** 2 * (mvr ** 2 + s)
                    )
                )
                * np.log(
                    (
                        ml ** 2
                        - mpi ** 2
                        + s
                        + np.sqrt(
                            ml ** 4
                            + (mpi ** 2 - s) ** 2
                            - 2 * ml ** 2 * (mpi ** 2 + s)
                        )
                    )
                    / (
                        ml ** 2
                        - mpi ** 2
                        + s
                        - np.sqrt(
                            ml ** 4
                            + (mpi ** 2 - s) ** 2
                            - 2 * ml ** 2 * (mpi ** 2 + s)
                        )
                    )
                )
                + 2
                * (
                    ml ** 4
                    - mpi ** 2 * mvr ** 2
                    + mvr ** 4
                    - ml ** 2 * (mpi ** 2 + 2 * mvr ** 2)
                )
                * (
                    -2
                    * np.sqrt(
                        ml ** 4
                        + (mpi ** 2 - s) ** 2
                        - 2 * ml ** 2 * (mpi ** 2 + s)
                    )
                    + (ml ** 2 + mpi ** 2 - s)
                    * np.log(
                        (
                            -(ml ** 2)
                            + mpi ** 2
                            + s
                            - np.sqrt(
                                ml ** 4
                                + (mpi ** 2 - s) ** 2
                                - 2 * ml ** 2 * (mpi ** 2 + s)
                            )
                        )
                        / (
                            -(ml ** 2)
                            + mpi ** 2
                            + s
                            + np.sqrt(
                                ml ** 4
                                + (mpi ** 2 - s) ** 2
                                - 2 * ml ** 2 * (mpi ** 2 + s)
                            )
                        )
                    )
                )
            )
        ) / (
            8.0
            * np.sqrt(
                -(ml ** 2)
                + (ml ** 2 - mpi ** 2 + mvr ** 2) ** 2 / (4.0 * mvr ** 2)
            )
            * (
                ml ** 4
                - mpi ** 2 * mvr ** 2
                + mvr ** 4
                - ml ** 2 * (mpi ** 2 + 2 * mvr ** 2)
            )
            * np.pi ** 2
            * (mvr ** 2 - s)
        )

    def dnde_pi_l_fsr(self, photon_energies):
        """
        Compute the FSR spectra from a right-handed neutrino decay into a
        charged pion and lepton.

        Parameters
        ----------
        photon_energies: float or np.array
           The energy of the final state photon in MeV.

        Returns
        -------
        dnde: float or np.array
            Photon spectrum.
        """
        if hasattr(photon_energies, "__len__"):
            return np.array([self.__dnde_pi_l_fsr(e) for e in photon_energies])
        else:
            return self.__dnde_pi_l_fsr(photon_energies)
