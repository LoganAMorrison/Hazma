from ..theory import Theory

from .vector_mediator_parameters import VectorMediatorParameters

from vector_mediator_cross_sections import branching_fractions as bfs
from vector_mediator_cross_sections import cross_sections as cs

from vector_mediator_spectra import spectra as specs
from vector_mediator_spectra import dnde_mumu, dnde_ee, dnde_pipi, dnde_pi0g

from vector_mediator_positron_spectra import positron_spectra
from vector_mediator_positron_spectra import positron_lines as pls

from ..parameters import neutral_pion_mass as mpi0

import warnings
import numpy as np
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
        return ['mu mu', 'e e', 'pi pi', 'pi0 g', "pi0 pi pi", 'v v']

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

    def gamma_ray_lines(self, cme):
        """
        Parameters
        ----------
        cme : float
            Center of mass energy

        Returns
        -------
        line_es : numpy.array
        line_bfs : numpy.array
            The energies for gamma ray lines produced in DM annihilations and
            the branching fractions into the final states producing these
            lines. In this case, the relevant final state is pi0 g.
        """
        bfs = self.branching_fractions(cme)

        return [np.array([(cme**2 - mpi0**2) / (2. * cme)]),
                np.array([bfs["pi0 g"]])]

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
            'pi0 g', 'pi pi'
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

        Note
        ----
        This does not return a function for computing the spectrum for the pi0
        pi pi final state since it always contributes orders of magnitude less
        than the pi pi and pi0 g final states.
        """
        return {'mu mu': lambda egams, cme: dnde_mumu(egams, cme, self),
                'e e': lambda egams, cme: dnde_ee(egams, cme, self),
                'pi pi': lambda egams, cme: dnde_pipi(egams, cme, self),
                'pi0 g': lambda egams, cme: dnde_pi0g(egams, cme, self)}

    def positron_spectra(self, eng_ps, cme):
        """
        Compute the total positron spectrum from two fermions annihilating
        through a vector mediator to mesons and leptons.

        Parameters
        ----------
        eng_ps : array-like, optional
            Positron energies to evaluate the spectrum at.
        cme : float
            Center of mass energy.

        Returns
        -------
        specs : dictionary
            Dictionary of the spectra. The keys are 'total', 'mu mu', 'e e',
            'pi pi'.
        """
        return positron_spectra(eng_ps, cme, self)

    def positron_lines(self, cme):
        """
        Returns a dictionary of the energies and branching fractions of
        positron lines

        Parameters
        ----------
        eng_ps : array-like, optional
            Positron energies to evaluate the spectrum at.
        cme : float
            Center of mass energy.

        Returns
        -------
        lines : dictionary
            Dictionary of the lines. The keys are 'e e'.
        """
        return pls(cme, self)
