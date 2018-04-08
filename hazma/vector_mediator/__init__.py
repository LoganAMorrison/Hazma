from ..theory import Theory

from .vector_mediator_parameters import VectorMediatorParameters

import warnings
from hazma_errors import PreAlphaWarning


class VectorMediator(Theory, VectorMediatorParameters):
    r"""
    Create a vector mediator model object.
    """

    def __init__(self, mx, mv, gvxx, gvff):
        super(VectorMediator, self).__init__(mx, mv, gvxx, gvff)

        self.params = VectorMediatorParameters(mx, mv, gvxx, gvff)

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
