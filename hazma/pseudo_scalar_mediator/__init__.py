from ..theory import Theory

from .pseudo_scalar_mediator_parameters import PseudoScalarMediatorParameters

from .pseudo_scalar_mediator_cross_sections import branching_fractions as bfs
from .pseudo_scalar_mediator_cross_sections import cross_sections as cs

from .pseudo_scalar_mediator_spectra import dnde_mumu, dnde_ee
from .pseudo_scalar_mediator_spectra import spectra as specs

import warnings
import numpy as np
from ..hazma_errors import PreAlphaWarning


class PseudoScalarMediator(Theory, PseudoScalarMediatorParameters):
    r"""
    WARNING: This class is pre-alpha.

    Create a vector mediator model object.
    """

    def __init__(self, mx, mp, gpxx, gpff):
        super(PseudoScalarMediator, self).__init__(mx, mp, gpxx, gpff)

        self.params = PseudoScalarMediatorParameters(mx, mp, gpxx, gpff)

    def description(self):
        warnings.warn("", PreAlphaWarning)
        pass

    @classmethod
    def list_final_states(cls):
        """
        WARNING: This function is pre-alpha.

        Return a list of the available final states.

        Returns
        -------
        fs : array-like
            Array of the available final states.
        """
        warnings.warn("", PreAlphaWarning)

        return ['mu mu', 'e e']

    def cross_sections(self, cme):
        """
        WARNING: This function is pre-alpha.

        Compute the all the cross sections of the theory.

        Parameters
        ----------
        cme : float
            Center of mass energy.

        Returns
        -------
        cs : dictionary
            Dictionary of the cross sections of the theory.
        """
        warnings.warn("", PreAlphaWarning)
        return cs(cme, self.params)

    def branching_fractions(self, cme):
        """
        WARNING: This function is pre-alpha.

        Compute the branching fractions for two fermions annihilating through a
        psuedo-scalar mediator to mesons and leptons.

        Parameters
        ----------
        Q : float
            Center of mass energy.

        Returns
        -------
        bfs : dictionary
            Dictionary of the branching fractions.
        """
        warnings.warn("", PreAlphaWarning)
        return bfs(cme, self.params)

    def gamma_ray_lines(self, cme):
        warnings.warn("", PreAlphaWarning)
        return [np.array([]), np.array([])]

    def spectra(self, egams, cme):
        """
        WARNING: This function is pre-alpha.

        Compute the total spectrum from two fermions annihilating through a
        vector mediator to mesons and leptons.

        Parameters
        ----------
        egams : array-like, optional
            Gamma ray energies to evaluate the spectrum at.
        cme : float
            Center of mass energy.

        Returns
        -------
        specs : dictionary
            Dictionary of the spectra
        """
        warnings.warn("", PreAlphaWarning)
        return specs(egams, cme, self.params)

    def spectrum_functions(self):
        """
        WARNING: This function is pre-alpha.

        Returns a dictionary of all the avaiable spectrum functions for
        a pair of initial state fermions with mass `mx` annihilating into
        each available final state.

        Each argument of the spectrum functions in `eng_gams`, an array
        of the gamma ray energies to evaluate the spectra at and `cme`, the
        center of mass energy of the process.
        """
        warnings.warn("", PreAlphaWarning)

        def mumu(eng_gams, cme):
            return dnde_mumu(eng_gams, cme, self.params)

        def ee(eng_gams, cme):
            return dnde_ee(eng_gams, cme, self.params)

        return {'mu mu': mumu, 'e e': ee}

    def positron_spectra(self, eng_es, cme):
        pass

    def positron_lines(self, cme):
        pass
