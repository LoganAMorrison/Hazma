from ..theory import Theory

from .axial_vector_mediator_parameters import AxialVectorMediatorParameters

import warnings
from ..hazma_errors import PreAlphaWarning


class AxialVectorMediator(Theory, AxialVectorMediatorParameters):
    r"""
    Create a vector mediator model object.
    """

    def __init__(self, mx, ma, gaxx, gaff):
        super(AxialVectorMediator, self).__init__(mx, ma, gaxx, gaff)

        self.params = AxialVectorMediatorParameters(mx, ma, gaxx, gaff)

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

    def positron_spectra(self, eng_es, cme):
        pass

    def positron_lines(self, cme):
        pass
