import warnings


class BackgroundModel(object):
    """

    Represents a gamma ray background model, which is required for computing
    projected limits for planned gamma-ray detectors.

    Parameters
    ----------
    e_range : [float, float]
        Minimum and maximum photon energies for which this model is valid, in
        MeV.
    dPhi_dEdOmega : np.array
        Background gamma ray flux (MeV^-1 sr^-1 cm^-2 s^-1) as a function of
        photon energy (MeV). This function must be vectorized.

    """

    def __init__(self, e_range, dPhi_dEdOmega):
        self.e_range = e_range
        self.__dPhi_dEdOmega = dPhi_dEdOmega

    @classmethod
    def from_interp(cls, interp):
        return cls(interp.x[[0, -1]], interp)

    def dPhi_dEdOmega(self, es):
        """Computes this background model's gamma ray flux.

        Parameters
        ----------
        es : float or np.array
            Photon energy/energies at which to compute

        Returns
        -------
        dPhi_dEdOmega : np.array
            Background gamma ray flux, in MeV^-1 sr^-1 cm^-2 s^-1. For any
            energies outside of ``self.e_range``, ``np.nan`` is returned.
        """
        if hasattr(es, "__len__"):
            # Check if any energies are out of bounds
            es_out_of_bounds = es[(es < self.e_range[0]) | (es > self.e_range[1])]

            if len(es_out_of_bounds) > 0:
                warnings.warn(
                    "The gamma ray background model is not applicable for energy %f MeV."
                    % es_out_of_bounds[0]
                )
            return self.__dPhi_dEdOmega(es)
        else:
            if es < self.e_range[0] or es > self.e_range[1]:
                warnings.warn(
                    "The gamma ray background model is not applicable for energy %f MeV."
                    % es
                )
            return self.__dPhi_dEdOmega(es)
