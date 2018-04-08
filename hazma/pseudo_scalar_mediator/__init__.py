from ..theory import Theory

from .pseudo_scalar_mediator_parameters import PseudoScalarMediatorParameters

import warnings
from hazma_errors import PreAlphaWarning


class PseudoScalarMediator(Theory, PseudoScalarMediatorParameters):
    r"""
    Create a vector mediator model object.
    """

    def __init__(self, mx, mp, gpxx, gpff):
        super(PseudoScalarMediator, self).__init__(mx, mp, gpxx, gpff)

        self.params = PseudoScalarMediatorParameters(mx, mp, gpxx, gpff)

    def description(self):
        warnings.warn("", PreAlphaWarning)
        pass

    def list_final_states(self):
        warnings.warn("", PreAlphaWarning)
        pass

    def cross_sections(self, cme):
        warnings.warn("", PreAlphaWarning)
        pass

    def branching_fractions(self, cme):
        warnings.warn("", PreAlphaWarning)
        pass

    def spectra(self, eng_gams, cme):
        warnings.warn("", PreAlphaWarning)
        pass

    def spectrum_functions(self):
        warnings.warn("", PreAlphaWarning)
        pass
