from ..theory import Theory

from .vector_mediator_parameters import VectorMediatorParameters

from vector_mediator_cross_sections import branching_fractions as bfs
from vector_mediator_cross_sections import cross_sections as cs

from vector_mediator_spectra import spectra as specs
from vector_mediator_spectra import dnde_mumu, dnde_ee, dnde_pipi, dnde_pi0g

from ..parameters import neutral_pion_mass as mpi0

import warnings
from ..hazma_errors import PreAlphaWarning


class VectorMediator(Theory, VectorMediatorParameters):
    r"""
    Create a vector mediator model object.
    """

    def __init__(self, mx, mv, gvxx, gvuu, gvdd, gvss, gvee, gvmumu):
        super(VectorMediator, self).__init__(mx, mv, gvxx, gvuu, gvdd, gvss,
                                             gvee, gvmumu)

    def description(self):
        warnings.warn("", PreAlphaWarning)
        pass

    def list_final_states(self):
        """
        Return a list of the available final states.

        Returns
        -------
        fs : array-like
            Array of the available final states.
        """
        return ['mu mu', 'e e', 'pi pi', 'pi0 g', "pi0 pi pi"]

    def cross_sections(self, cme):
        """
        Compute the all the cross sections of the theory.

        Parameters
        ----------
        cme : float
            Center of mass energy.

        Returns
        -------
        cs : dictionary
            Dictionary of the cross sections of the theory. The keys are
            'total', 'mu mu', 'e e', 'pi0 g', 'pi pi', 'pi0 pi pi'.
        """
        return cs(cme, self)

    def branching_fractions(self, cme):
        """
        Compute the branching fractions for two fermions annihilating through a
        vector mediator to mesons and leptons.

        Parameters
        ----------
        Q : float
            Center of mass energy.

        Returns
        -------
        bfs : dictionary
            Dictionary of the branching fractions of the theory. The keys are
            'total', 'mu mu', 'e e', 'pi0 g', 'pi pi'.
        """
        return bfs(cme, self)

    def gamma_ray_line_energies(self, cme):
        bfs = self.branching_fractions(cme)

        return [(cme**2 - mpi0**2) / (2. * cme), bfs["pi0 g"]]

    def spectra(self, egams, cme):
        """
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
        return specs(egams, cme, self)

    def spectrum_functions(self):
        """
        Returns a dictionary of all the avaiable spectrum functions for
        a pair of initial state fermions with mass `mx` annihilating into
        each available final state.

        Each argument of the spectrum functions in `eng_gams`, an array
        of the gamma ray energies to evaluate the spectra at and `cme`, the
        center of mass energy of the process.
        """
        return {'mu mu': lambda egams, cme: dnde_mumu(egams, cme, self),
                'e e': lambda egams, cme: dnde_ee(egams, cme, self),
                'pi pi': lambda egams, cme: dnde_pipi(egams, cme, self),
                'pi0 g': lambda egams, cme: dnde_pi0g(egams, cme, self)}
