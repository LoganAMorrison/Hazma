from abc import ABCMeta, abstractmethod

import numpy as np

from hazma.parameters import convolved_spectrum_fn
from hazma.theory._theory_cmb import TheoryCMB
from hazma.theory._theory_constrain import TheoryConstrain
from hazma.theory._theory_gamma_ray_limits import TheoryGammaRayLimits


class Theory(TheoryGammaRayLimits, TheoryCMB, TheoryConstrain):
    """
    Represents a sub-GeV DM theory.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def description(self):
        pass

    @classmethod
    @abstractmethod
    def list_annihilation_final_states(cls):
        """Returns a list of the annihilation final states.

        Excludes states that are suppressed by (eg) factors of alpha_EM or
        |m_d-m_u|.

        Returns
        -------
        list(str)
            Possible annihilation final states.
        """
        pass

    @abstractmethod
    def annihilation_cross_sections(self, e_cm):
        """Gets DM annihilation cross sections.

        Arguments
        ---------
        e_cm : float
            Center of mass energy for the annihilation in MeV.

        Returns
        -------
        dict(str, float)
            Annihilation cross section into each final state in MeV**-2 as well
            as the total cross section.
        """
        pass

    @abstractmethod
    def partial_widths(self):
        """Gets mediator decay widths.

        Returns
        -------
        dict(str, float)
            Mediator partial widths in MeV as the total cross decay width.
        """
        pass

    @abstractmethod
    def annihilation_branching_fractions(self, e_cm):
        """Gets DM annihilation branching fractions.

        Arguments
        ---------
        e_cm : float
            Center of mass energy for the annihilation in MeV.

        Returns
        -------
        dict(str, float)
            Annihilation branching fractions into each final state.
        """
        pass

    @abstractmethod
    def spectra(self, e_gams, e_cm):
        pass

    @abstractmethod
    def spectrum_functions(self):
        pass

    def total_spectrum(self, e_gams, e_cm):
        """Returns total gamma ray spectrum.

        Parameters
        ----------
        e_gams : float or float numpy.array
            Photon energy or energies at which to compute the spectrum.
        e_cm : float
            DM center of mass energy.

        Returns
        -------
        tot_spec : float numpy.array
            Array containing the total annihilation spectrum.
        """
        if hasattr(e_gams, "__len__"):
            return self.spectra(e_gams, e_cm)["total"]
        else:
            return self.spectra(np.array([e_gams]), e_cm)["total"]

    @abstractmethod
    def gamma_ray_lines(self, e_cm):
        """Returns the energies of and branching fractions into monochromatic
        gamma rays produces by this theory.
        """
        pass

    def total_conv_spectrum_fn(
        self, e_gam_min, e_gam_max, e_cm, energy_res, n_pts=1000
    ):
        """Applies `convolved_spectrum_fn` to obtain the convolved gamma-ray
        spectrum. See documentation for that function.
        """
        return convolved_spectrum_fn(
            e_gam_min,
            e_gam_max,
            e_cm,
            energy_res,
            self.total_spectrum,
            self.gamma_ray_lines,
            n_pts,
        )

    @abstractmethod
    def positron_spectra(self, e_ps, e_cm):
        pass

    def total_positron_spectrum(self, e_ps, e_cm):
        """Returns total positron ray spectrum.

        Parameters
        ----------
        e_ps : float or float numpy.array
            Positron energy or energies at which to compute the spectrum.
        e_cm : float
            DM center of mass energy.

        Returns
        -------
        tot_spec : float numpy.array
            Array containing the total annihilation positron spectrum.
        """
        if hasattr(e_ps, "__len__"):
            return self.positron_spectra(e_ps, e_cm)["total"]
        else:
            return self.positron_spectra(np.array([e_ps]), e_cm)["total"]

    @abstractmethod
    def positron_lines(self, e_cm):
        pass

    def total_conv_positron_spectrum_fn(
        self, e_p_min, e_p_max, e_cm, energy_res, n_pts=1000
    ):
        """Applies `convolved_spectrum_fn` to obtain the convolved positron
        spectrum. See documentation for that function.
        """
        return convolved_spectrum_fn(
            e_p_min,
            e_p_max,
            e_cm,
            energy_res,
            self.total_positron_spectrum,
            self.positron_lines,
            n_pts,
        )

    @abstractmethod
    def constraints(self):
        """Get a dictionary of all available constraints.

        Notes
        -----
        Each key in the dictionary is the name of a constraint. Each value is a
        function that is positive when the constraint is satisfied and negative
        when it is not.
        """
        pass
