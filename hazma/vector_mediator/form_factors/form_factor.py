import functools
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from hazma.rambo import PhaseSpace


class FormFactor(ABC):
    def __init__(self, fsp_masses):
        self.fsp_masses = fsp_masses

    @abstractmethod
    def msqrd(self, momenta, *, cme, gvuu: float, gvdd: float, gvss: float) -> float:
        pass

    @abstractmethod
    def width(self, *, mv, gvuu, gvdd, gvss, **kwargs):
        pass

    @abstractmethod
    def cross_section(self, *, cme, mx, mv, gvuu, gvdd, gvss, **kwargs):
        pass

    def energy_distributions(
        self,
        *,
        cme: float,
        gvuu: float,
        gvdd: float,
        gvss: float,
        npts: int,
        nbins: int = 25
    ):
        def msqrd(momenta):
            return self.msqrd(momenta, cme=cme, gvuu=gvuu, gvdd=gvdd, gvss=gvss)

        phase_space = PhaseSpace(cme, masses=np.array(self.fsp_masses), msqrd=msqrd)
        return phase_space.energy_distributions(n=npts, nbins=nbins)
