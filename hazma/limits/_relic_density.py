from copy import copy
import warnings

import numpy as np
from scipy import optimize

from hazma import parameters
from hazma.relic_density import relic_density

from ._abstract import AbstractLimit


class RelicDensityLimit(AbstractLimit):
    """Class for computing limits on dark-matter models from relic-density."""

    def __init__(
        self,
        prop: str,
        prop_min: float,
        prop_max: float,
        vx: float = 1e-3,
        log: bool = True,
        method: str = "brentq",
    ):
        """
        Create a constrainer object for constraining the dark-matter
        annihilation cross section by varying a specified property
        such that the model yields the correct relic-density.

        Parameters
        ----------
        prop: str
            String specifying the property to vary in order fix the
            dark-matter relic-density.
        prop_min: float
            Minimum value of the property.
        prop_max: float
            Maximum value of the property.
        vx: float, optional
            The dark-matter velocity used to compute the annihilation cross
            section. Default is 1e-3.
        log: bool, optional
            If true, the property is varied logarithmically.
        """
        super().__init__()
        self.prop = prop
        self.prop_min = prop_min
        self.prop_max = prop_max
        self.vx = vx
        self.log = log
        self.method = method

    @property
    def description(self):
        return "[dark_violet] Relic Density"

    @property
    def name(self):
        return "relic-density"

    def _sigmav(self, model, vx: float = 1e-3):
        """Compute <σv> for the given model."""
        cme = 2 * model.mx * (1.0 + 0.5 * vx**2)
        sig = model.annihilation_cross_sections(cme)["total"]
        return sig * vx * parameters.sv_inv_MeV_to_cm3_per_s

    def _setprop(self, model, val):
        if self.log:
            setattr(model, self.prop, 10**val)
        else:
            setattr(model, self.prop, val)

    def _constrain(self, model):
        model_ = copy(model)
        lb = self.prop_min if not self.log else np.log10(self.prop_min)
        ub = self.prop_max if not self.log else np.log10(self.prop_max)

        def f(val):
            self._setprop(model_, val)
            return relic_density(model_, semi_analytic=True) - parameters.omega_h2_cdm

        try:
            root: optimize.RootResults = optimize.root_scalar(
                f, bracket=[lb, ub], method="brentq"
            )
            if not root.converged:
                warnings.warn(f"root_scalar did not converge. Flag: {root.flag}")
            self._setprop(model_, root.root)
            return self._sigmav(model_, self.vx)
        except ValueError as e:
            warnings.warn(f"Error encountered: {e}. Returning nan", RuntimeWarning)
            return np.nan
