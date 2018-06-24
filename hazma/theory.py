from .gamma_ray_limits.gamma_ray_limit_parameters import A_eff_e_astrogam
from .gamma_ray_limits.gamma_ray_limit_parameters import T_obs_e_astrogam
from .gamma_ray_limits.gamma_ray_limit_parameters import draco_params
from .gamma_ray_limits.gamma_ray_limit_parameters import default_bg_model
from .gamma_ray_limits.gamma_ray_limit_parameters import energy_res_e_astrogam
from .gamma_ray_limits.compute_limits import unbinned_limit, binned_limit
from .cmb import f_eff, cmb_limit

import numpy as np
from skimage import measure
from abc import ABCMeta, abstractmethod


class Theory(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def description(self):
        pass

    @classmethod
    @abstractmethod
    def list_final_states(cls):
        pass

    @abstractmethod
    def cross_sections(self, cme):
        pass

    @abstractmethod
    def branching_fractions(self, cme):
        pass

    @abstractmethod
    def gamma_ray_lines(self, cme):
        """Returns the energies of and branching fractions into monochromatic
        gamma rays produces by this theory.
        """
        pass

    @abstractmethod
    def spectra(self, eng_gams, cme):
        pass

    @abstractmethod
    def spectrum_functions(self):
        pass

    @abstractmethod
    def positron_spectra(self, eng_es, e_cm):
        pass

    @abstractmethod
    def positron_lines(self, e_cm):
        pass

    def binned_limit(self, measurement, n_sigma=2.):
        def spec_fn(e_gams, e_cm):
            if hasattr(e_gams, "__len__"):
                return self.spectra(e_gams, e_cm)["total"]
            else:
                return self.spectra(np.array([e_gams]), e_cm)["total"]

        return binned_limit(spec_fn, self.gamma_ray_lines, self.mx, False,
                            measurement, n_sigma)

    def binned_limits(self, mxs, measurement, n_sigma=2.):
        def binned_limit_change_mass(mx):
            self.mx = mx
            return self.binned_limit(measurement, n_sigma)

        return np.vectorize(binned_limit_change_mass)(mxs)

    def unbinned_limit(self, A_eff=A_eff_e_astrogam,
                       energy_res=energy_res_e_astrogam,
                       T_obs=T_obs_e_astrogam, target_params=draco_params,
                       bg_model=default_bg_model, n_sigma=5.):
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
        def spec_fn(e_gams, e_cm):
            if hasattr(e_gams, "__len__"):
                return self.spectra(e_gams, e_cm)["total"]
            else:
                return self.spectra(np.array([e_gams]), e_cm)["total"]

        return unbinned_limit(spec_fn, self.gamma_ray_lines, self.mx, False,
                              A_eff, energy_res, T_obs, target_params,
                              bg_model, n_sigma)

    def unbinned_limits(self, mxs, A_eff=A_eff_e_astrogam,
                        energy_res=energy_res_e_astrogam,
                        T_obs=T_obs_e_astrogam, target_params=draco_params,
                        bg_model=default_bg_model, n_sigma=5.):
        """Computes gamma ray constraints over a range of DM masses.

        See documentation for :func:`unbinned_limit`.
        """
        def unbinned_limit_change_mass(mx):
            self.mx = mx
            return self.unbinned_limit(A_eff, energy_res, T_obs, target_params,
                                       bg_model, n_sigma)

        return np.vectorize(unbinned_limit_change_mass)(mxs)

    def cmb_limit(self, x_kd=1.0e-4):
        """Computes CMB limit on <sigma v>.

        Parameters
        ----------
        x_kd: float
            T_kd / m_x, where T_kd is the dark matter's kinetic decoupling
            temperature.

        Returns
        -------
        <sigma v> : float
            Upper bound on <sigma v>.
        """
        def spec_fn(e_gams, e_cm):
            if hasattr(e_gams, "__len__"):
                return self.spectra(e_gams, e_cm)["total"]
            else:
                return self.spectra(np.array([e_gams]), e_cm)["total"]

        def pos_spec_fn(e_ps, e_cm):
            if hasattr(e_ps, "__len__"):
                return self.positron_spectra(e_ps, e_cm)["total"]
            else:
                return self.positron_spectra(np.array([e_ps]), e_cm)["total"]

        f_eff_dm = f_eff(spec_fn, self.gamma_ray_lines, pos_spec_fn,
                         self.positron_lines, self.mx, x_kd)

        return cmb_limit(self.mx, f_eff_dm)

    def cmb_limits(self, mxs, x_kd=1.0e-4):  # TODO: clean this up...
        """Computes CMB limit on <sigma v>.

        Parameters
        ----------
        mxs : np.array
            DM masses at which to compute the CMB limits.
        x_kd: float
            T_kd / m_x, where T_kd is the dark matter's kinetic decoupling
            temperature.

        Returns
        -------
        svs : np.array
            Array of upper bounds on <sigma v> for each mass in mxs.
        """
        def cmb_limit_change_mass(mx):
            self.mx = mx
            return self.cmb_limit(x_kd)

        return np.vectorize(cmb_limit_change_mass)(mxs)

    def f_eff(self, x_kd=1.0e-4):  # TODO: clean this up...
        def spec_fn(e_gams, e_cm):
            if hasattr(e_gams, "__len__"):
                return self.spectra(e_gams, e_cm)["total"]
            else:
                return self.spectra(np.array([e_gams]), e_cm)["total"]

        def pos_spec_fn(e_ps, e_cm):
            if hasattr(e_ps, "__len__"):
                return self.positron_spectra(e_ps, e_cm)["total"]
            else:
                return self.positron_spectra(np.array([e_ps]), e_cm)["total"]

        return f_eff(spec_fn, self.gamma_ray_lines, pos_spec_fn,
                     self.positron_lines, self.mx, x_kd)

    def f_effs(self, mxs, x_kd=1.0e-4):
        def f_eff_change_mass(mx):
            self.mx = mx
            return self.f_eff(x_kd)

        return np.vectorize(f_eff_change_mass)(mxs)

    def constrain(self, p1, p1_vals, p2, p2_vals, ls_or_img="image"):
        """Computes constraints over 2D slice of parameter space.

        Parameters
        ----------
        p1 : string
            Name of a parameter to constraint.
        p1_vals : np.array
            Values of p1 at which to compute constraints. Must be sorted.
        p2 : string
            Name of the other parameter to constraint. Must be different than
            p1.
        p2_vals : np.array
            Values of p2 at which to compute constraints. Must be sorted.
        ls_or_img : "image" or "ls"
            Controls whether this function returns level sets or images.

        Returns
        -------
        constrs : dict
            A dictionary containing the constraints on the theory in the (p1,
            p2) plane.

            If ls_or_img is "ls", the values are level sets. A level set is a
            list of curves, where each curve is a list of values of (p1, p2)
            defining the parameter values that saturate the constraint. If
            ls_or_img is "image", each value is a 2D numpy.array I(x,y) such
            that I_ij > 0 when (p1_vals[i], p2_vals[j]) is not excluded by the
            corresponding constraint and I_ij < 0 if (p1_vals[i], p2_vals[j])
            is excluded by the constraint.
        """
        if p1 == p2:
            raise ValueError("Parameters being constrained must not be the "
                             "same. Both are %s." % p1)

        n_p1s, n_p2s = len(p1_vals), len(p2_vals)
        constraints = self.constraints()

        # Store the constraint images. Note that p1 and p2 must be swapped
        # so we can use Cartesian rather than matrix indexing.
        imgs = {cn: np.zeros([n_p2s, n_p1s]) for cn in constraints.keys()}

        # Loop over the parameter values
        for idx_p1, p1_val in np.ndenumerate(p1_vals):
            for idx_p2, p2_val in np.ndenumerate(p2_vals):
                setattr(self, p1, p1_val)
                setattr(self, p2, p2_val)
                # TODO: remove this
                self.gsGG = self.gsff
                self.gsFF = self.gsff

                # Compute all constraints at this point in parameter space
                for cn, fn in constraints.iteritems():
                    imgs[cn][idx_p2[0], idx_p1[0]] = fn()

        if ls_or_img == "image":
            return imgs
        elif ls_or_img == "ls":
            return {cn: _img_to_ls(img) for cn, img in imgs}

    @abstractmethod
    def constraints(self):
        """Get a dictionary of all available constraints.

        Notes
        -----
        Each key in the dictionary is the name of a constraint. Each value is a
        function that is positive when the constraint is satisfied and negative
        when it is not.
        """
        pass


def _img_to_ls(p1_vals, p2_vals, img):
    """Finds levels sets for an image.
    """
    contours_raw = measure.find_contours(img, level=0)
    contours = []

    # Convert from indices to values of p1 and p2
    for c in contours_raw:
        p1s = c[:, 1] / len(p1_vals) * (p1_vals[-1] - p1_vals[0]) + p1_vals[0]
        p2s = c[:, 0] / len(p2_vals) * (p2_vals[-1] - p2_vals[0]) + p2_vals[0]
        contours.append(np.array([p1s, p2s]))

    return contours
