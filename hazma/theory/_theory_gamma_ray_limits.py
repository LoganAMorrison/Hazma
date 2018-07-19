from ..gamma_ray_limits.gamma_ray_limit_parameters import A_eff_e_astrogam
from ..gamma_ray_limits.gamma_ray_limit_parameters import T_obs_e_astrogam
from ..gamma_ray_limits.gamma_ray_limit_parameters import draco_params
from ..gamma_ray_limits.gamma_ray_limit_parameters import default_bg_model
from ..gamma_ray_limits.gamma_ray_limit_parameters import energy_res_e_astrogam
from ..gamma_ray_limits.compute_limits import (unbinned_limit as ubl,
                                               binned_limit as bl)

import numpy as np


class TheoryGammaRayLimits:
    def binned_limit(self, measurement, n_sigma=2.):
        return bl(self.total_spectrum, self.gamma_ray_lines, self.mx, False,
                  measurement, n_sigma)

    def binned_limits(self, mxs, measurement, n_sigma=2.):
        lims = []

        for mx in mxs:
            self.mx = mx
            lims.append(self.binned_limit(measurement, n_sigma))

        return np.array(lims)

    def unbinned_limit(self, A_eff=A_eff_e_astrogam,
                       energy_res=energy_res_e_astrogam,
                       T_obs=T_obs_e_astrogam, target_params=draco_params,
                       bg_model=default_bg_model, n_sigma=5.,
                       debug_msgs=False):
        """Computes smallest value of <sigma v> detectable for given target and
        experiment parameters.

        Notes
        -----
        We define a signal to be detectable if

        .. math:: N_S / sqrt(N_B) >= n_\sigma,

        where :math:`N_S` and :math:`N_B` are the number of signal and
        background photons in the energy window of interest and
        :math:`n_\sigma` is the significance in number of standard deviations.
        Note that :math:`N_S \propto \langle \sigma v \rangle`. While the
        photon count statistics are properly taken to be Poissonian and using a
        confidence interval would be more rigorous, this procedure provides a
        good estimate and is simple to compute.

        Parameters
        ----------
        dN_dE_DM : float -> float
            Photon spectrum per dark matter annihilation as a function of
            photon energy
        mx : float
            Dark matter mass
        dPhi_dEdOmega_B : float -> float
            Background photon spectrum per solid angle as a function of photon
            energy
        self_conjugate : bool
            True if DM is its own antiparticle; false otherwise
        n_sigma : float
            Number of standard deviations the signal must be above the
            background to be considered detectable
        delta_Omega : float
            Angular size of observation region in sr
        J_factor : float
            J factor for target in MeV^2 / cm^5
        A_eff : float
            Effective area of experiment in cm^2
        T_obs : float
            Experiment's observation time in s

        Returns
        -------
        <sigma v> : float
            Smallest detectable thermally averaged total cross section in units
            of cm^3 / s
        """
        return ubl(self.total_spectrum, self.gamma_ray_lines, self.mx, False,
                   A_eff, energy_res, T_obs, target_params, bg_model, n_sigma,
                   debug_msgs=debug_msgs)

    def unbinned_limits(self, mxs, A_eff=A_eff_e_astrogam,
                        energy_res=energy_res_e_astrogam,
                        T_obs=T_obs_e_astrogam, target_params=draco_params,
                        bg_model=default_bg_model, n_sigma=5.,
                        debug_msgs=False):
        """Computes gamma ray constraints over a range of DM masses.

        See documentation for :func:`unbinned_limit`.
        """
        lims = []

        for mx in mxs:
            self.mx = mx
            lims.append(self.unbinned_limit(A_eff, energy_res, T_obs,
                                            target_params, bg_model, n_sigma,
                                            debug_msgs))

        return np.array(lims)

