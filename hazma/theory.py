from abc import ABCMeta, abstractmethod


class Theory(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def description(self):
        pass

    @abstractmethod
    def list_final_states(self):
        pass

    @abstractmethod
    def cross_sections(self, cme):
        pass

    @abstractmethod
    def branching_fractions(self, cme):
        pass

    @abstractmethod
    def spectra(self, eng_gams, cme):
        pass

    @abstractmethod
    def spectrum_functions(self):
        pass
