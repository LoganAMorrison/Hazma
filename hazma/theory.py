from .gamma_ray_limits.gamma_ray_limit_parameters import (eASTROGAM_params,
                                                          dSph_params)
from .gamma_ray_limits.compute_limits import compute_limit
from .parameters import neutral_pion_mass as mpi0

import numpy as np
from scipy.interpolate import interp1d
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

    def get_e_gam_window(self):
        """Return best gamma ray energy range over which to probe this model.

        Notes
        -----
        This could probably be significantly optimized. For a start, centering
        it on the pi0 peak would be a big improvement...

        Returns
        -------
        e_gam_min, e_gam_max : (float, float)
            Boundaries of energy window
        """
        if self.mx < mpi0:
            e_gam_min = 5.
        else:
            e_gam_min = mpi0 / 2.

        return e_gam_min, self.mx

    def compute_limit(self, n_sigma=5., exp_params=eASTROGAM_params,
                      target_params=dSph_params):
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
        e_gam_min : float
            Lower bound for energy window used to set limit
        e_gam_max : float
            Upper bound for energy window used to set limit
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
        # Choose energy window in which to search for signal
        e_gam_min, e_gam_max = self.get_e_gam_window()

        # Create function to interpolate spectrum over energy window
        e_gams = np.logspace(np.log10(e_gam_min), np.log10(e_gam_max), 100)
        dN_dE_DM = interp1d(e_gams,
                            self.spectra(e_gams, 2.001*self.mx)["total"])

        return compute_limit(dN_dE_DM, self.mx, e_gam_min, e_gam_max,
                             n_sigma=n_sigma, exp_params=exp_params,
                             target_params=target_params)

    def compute_limits(self, mxs, n_sigma=5., exp_params=eASTROGAM_params,
                       target_params=dSph_params):
        """Computes gamma ray constraints over a range of DM masses.

        See documentation for :func:`compute_limit`.
        """
        limits = []

        for mx in mxs:
            self.mx = mx
            limits.append(self.compute_limit(n_sigma, exp_params,
                                             target_params))

        return np.array(limits)
