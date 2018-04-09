from ..theory import Theory

from .vector_mediator_parameters import VectorMediatorParameters

from vector_mediator_cross_sections import branching_fractions as bfs
from vector_mediator_cross_sections import cross_sections as cs

from vector_mediator_spectra import spectra as specs
from vector_mediator_spectra import dnde_mumu, dnde_ee

import warnings
from hazma_errors import PreAlphaWarning


class VectorMediator(Theory, VectorMediatorParameters):
    r"""
    WARNING: This class is pre-alpha.

    Create a vector mediator model object.
    """

    def __init__(self, mx, mv, gvxx, gvff):
        super(VectorMediator, self).__init__(mx, mv, gvxx, gvff)

        self.params = VectorMediatorParameters(mx, mv, gvxx, gvff)

    def description(self):
        warnings.warn("", PreAlphaWarning)
        pass

    def list_final_states(self):
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
        vector mediator to mesons and leptons.

        Parameters
        ----------
        Q : float
            Center of mass energy.

        Returns
        -------
        bfs : dictionary
            Dictionary of the branching fractions. The keys are 'total',
            'mu mu', 'e e', 'pi0 pi0', 'pi pi', 'k k', 'k0 k0'.
        """
        warnings.warn("", PreAlphaWarning)
        return bfs(cme, self.params)

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
            Dictionary of the spectra. The keys are 'total', 'mu mu', 'e e',
            'pi0 pi0', 'pi pi', 'k k', 'k0 k0'.
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
