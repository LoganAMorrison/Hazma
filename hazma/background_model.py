import numpy as np
from scipy.interpolate import interp1d
from hazma.parameters import load_interp


class BackgroundModel(object):
    """Represents a gamma ray background model.

    Attributes
    ----------
    e_range : [float, float]
        Minimum and maximum photon energies for which this model is valid, in
        MeV.
    """

    def __init__(self, e_range, flux_fn):
        """
        Parameters
        ----------
        e_range : [float, float]
            Photon energies (MeV) between which this background model is valid.
            Note that these bounds are inclusive.
        flux_fn : np.array
            Background gamma ray flux (MeV^-1 sr^-1 m^-2 s^-1) as a function of
            photon energy (MeV). This function must be vectorized (ie, able to
            map a 1D numpy.array of energies to a 1D numpy.array of fluxes).
        """
        self.e_range = e_range
        self.__dPhi_dEdOmega = flux_fn

    @classmethod
    def from_file(cls, rf_name):
        """Factory method to create a BackgroundModel from a data file.

        Parameters
        ----------
        rf_name : resource_filename
            Name of file whose lines are comma-separated pairs of photon
            energies and background fluxes, in MeV and MeV^-1 sr^-1 m^-2 s^-1.

        Returns
        -------
        bg_model : BackgroundModel
        """
        flux_fn = load_interp(rf_name, bounds_error=True, fill_value=np.nan)
        e_range = flux_fn.x[[0, -1]]

        return cls(e_range, flux_fn)

    @classmethod
    def from_vals(cls, es, fluxes):
        """Factory method to create a BackgroundModel from flux data points.

        Parameters
        ----------
        es : np.array
            Photon energies (MeV).
        fluxes : np.array
            Background gamma ray flux (MeV^-1 sr^-1 m^-2 s^-1) at the energies
            in es.

        Returns
        -------
        bg_model : BackgroundModel
        """
        return cls(es[[0, -1]], interp1d(es, fluxes))

    def dPhi_dEdOmega(self, es):
        """Computes this background model's gamma ray flux.

        Parameters
        ----------
        es : float or np.array
            Photon energy/energies at which to compute

        Returns
        -------
        d^2Phi / dE dOmega
            Background gamma ray flux, in MeV^-1 sr^-1 m^-2 s^-1. For any
            energies outside of self.e_range, np.nan is returned.
        """
        if hasattr(es, "__len__"):
            # Check if any energies are out of bounds
            es_out_of_bounds = es[(es < self.e_range[0]) |
                                  (es > self.e_range[1])]

            if len(es_out_of_bounds) == 0:
                return self.__dPhi_dEdOmega(es)
            else:
                raise ValueError("The gamma ray background model is not "
                                 "applicable for energy %f MeV." %
                                 es_out_of_bounds[0])
        else:
            if es < self.e_range[0] or es > self.e_range[1]:
                raise ValueError("The gamma ray background model is not "
                                 "applicable for energy %f MeV." %
                                 es_out_of_bounds[0])
            else:
                return self.__dPhi_dEdOmega(es)
