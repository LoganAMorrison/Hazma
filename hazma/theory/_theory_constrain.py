from hazma.constraint_parameters import sv_inv_MeV_to_cm3_per_s
from skimage import measure

import numpy as np


class TheoryConstrain:
    def custom_constrain(self, param_grid, ls_or_img="image"):
        """Computes constraints over grid of parameter values.

        Parameters
        ----------
        param_grid : 2D array of parameters
            Parameter values at which to compute constraints.
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
        n_rows, n_cols = param_grid.shape
        constraints = self.constraints()

        # Store the constraint images. Note that p1 and p2 must be swapped
        # so we can use Cartesian rather than matrix indexing.
        imgs = {cn: np.zeros([n_rows, n_rows]) for cn in constraints.keys()}

        # Loop over the parameter grid
        for i in range(n_rows):
            for j in range(n_cols):
                # Set this theory's parameters to the values at this point
                self.__dict__.update(param_grid[i, j].__dict__)

                # Compute all constraints at this point in parameter space
                for cn, fn in constraints.items():
                    imgs[cn][i, j] = fn()

        if ls_or_img == "image":
            return imgs
        elif ls_or_img == "ls":
            return {cn: self._img_to_ls(img) for cn, img in imgs}

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
            raise ValueError(
                "Parameters being constrained must not be the "
                "same. Both are %s." % p1
            )

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

                # Compute all constraints at this point in parameter space
                for cn, fn in constraints.items():
                    imgs[cn][idx_p2[0], idx_p1[0]] = fn()

        if ls_or_img == "image":
            return imgs
        elif ls_or_img == "ls":
            return {cn: self._img_to_ls(img) for cn, img in imgs}

    def constrain_gamma_helper(self, p2, p2_vals, measurement, n_sigma=2):
        """Computes constraints on a parameter from gamma ray experiments.

        Notes
        -----
        * p2 must not depend on mx or the center of mass energy.
        * TODO: make sure this is working correctly, it currently doesn't
        account for gamma ray lines!!!
        """
        vx = 1.0e-3
        e_cm = 2.0 * self.mx * (1 + 0.5 * vx ** 2)  # TODO: should depend on target

        # Factor to convert dN/dE to Phi/<sigma v>
        dm_flux_factor = (
            measurement.target.J
            * measurement.target.dOmega
            / (2.0 * 4.0 * np.pi * self.mx ** 2)
        )

        # Energy range over which to compute convolved spectrum
        e_bin_min, e_bin_max = measurement.bins[0][0], measurement.bins[-1][1]

        def get_bin_fluxes(spec_fn, line_fn):
            """Gets Phi/<sigma v> for a particular channel.
            """
            dnde_det = self.get_detected_spectrum_function(
                e_bin_min, e_bin_max, e_cm, measurement.energy_res, 500
            )
            return np.array(
                [
                    dm_flux_factor * dnde_det.integral(bl, br)
                    for bl, br in measurement.bins
                ]
            )

        # Compute Phi/<sigma v> in each bin for each final state
        fs_bin_fluxes = {
            fs: get_bin_fluxes(spec_fn, lambda e_cm: {})
            for fs, spec_fn in self.spectrum_funcs().items()
            if fs != "total"
        }
        # line_bin_fluxes = {fs: get_bin_fluxes(None, line_fn) for fs, line_fn
        #                    in self.gamma_ray_lines(cme) if fs != "total"}

        def flux_difference(p2, p2_val):
            """Compute difference between Phi_obs+N*sigma - Phi_th."""
            setattr(self, p2, p2_val)

            # Compute cross sections
            css = self.annihilation_cross_sections(e_cm)
            # Get fluxes by multiplying <sigma v>
            bin_fluxes = np.array(
                [
                    bf * css[fs] * vx * sv_inv_MeV_to_cm3_per_s
                    for fs, bf in fs_bin_fluxes.items()
                ]
            )

            return np.min(
                measurement.fluxes
                + n_sigma * measurement.upper_errors
                - bin_fluxes.sum(axis=0)
            )

        return np.array([flux_difference(p2, p2v) for p2v in p2_vals])

    def constrain_gamma(self, p1, p1_vals, p2, p2_vals, measurement, n_sigma=2):
        """Computes constraints from gamma ray experiments in the p1-p2 plane.

        Notes
        -----
        p1 must not depend on mx or the center of mass energy.
        """
        img = np.zeros([len(p2_vals), len(p1_vals)])

        for idx_p1, p1_val in enumerate(p1_vals):
            setattr(self, p1, p1_val)

            # Compute constraint function along the current column
            img[idx_p1, :] = self.constrain_gamma_helper(
                p2, p2_vals, measurement, n_sigma
            )

        return img

    def _img_to_ls(self, p1_vals, p2_vals, img):
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
