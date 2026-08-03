import functools
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from hazma.phase_space import Rambo


class FormFactor(ABC):
    def __init__(self, fsp_masses):
        self.fsp_masses = fsp_masses

    @abstractmethod
    def msqrd(self, momenta, *, cme, gvuu: float, gvdd: float, gvss: float) -> float:
        pass

    @abstractmethod
    def width(self, *, mv, gvuu, gvdd, gvss, **kwargs):
        pass

    def cross_section(self, *, cme, mx, mv, width_v, gvxx, gvuu, gvdd, gvss, **kwargs):
        gvxx = gvxx
        mux2 = (mx / cme) ** 2
        mug2 = (width_v / cme) ** 2
        muv2 = (mv / cme) ** 2

        num = gvxx**2 * (1 + 2 * mux2)
        den = (1 + (mug2 - 2) * muv2 + muv2**2) * np.sqrt(1.0 - 4.0 * mux2) * cme**1.5
        # Factor to transform width to cross section
        fact = num / den / cme**1.5

        partial_width = self.width(mv=cme, gvuu=gvuu, gvdd=gvdd, gvss=gvss, **kwargs)

        return partial_width * fact

    def energy_distributions(
        self,
        *,
        cme: float,
        gvuu: float,
        gvdd: float,
        gvss: float,
        npts: int,
        nbins: int = 25,
    ):
        def msqrd(momenta):
            return self.msqrd(momenta, cme=cme, gvuu=gvuu, gvdd=gvdd, gvss=gvss)

        phase_space = Rambo(cme, masses=np.array(self.fsp_masses), msqrd=msqrd)
        return phase_space.energy_distributions(n=npts, nbins=nbins)
