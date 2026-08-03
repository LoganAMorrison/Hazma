from abc import abstractmethod
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
            raise NotImplementedError("currently does not work")

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
            raise NotImplementedError("currently does not work")

    def _constrain_binned_gamma_helper(self, measurement, n_sigma=2, method="1bin"):
        """
        TODO: refactor! Lots of code duplication...
        """
        e_min, e_max = measurement.e_lows[0], measurement.e_highs[-1]

        if self.kind == "ann":  # type: ignore
            e_cm = 2 * self.mx * (1 + 0.5 * measurement.target.vx**2)  # type: ignore
            f_dm = 2.0
            # Approximate <sigma v> ~ sigma * v
            sv = (
                self.annihilation_cross_sections(e_cm)["total"]  # type: ignore
                * measurement.target.vx
            )
            dm_flux_factor = (
                sv
                * measurement.target.J
                * measurement.target.dOmega
                / (2 * f_dm * self.mx**2 * 4 * np.pi)  # type: ignore
            )
            dnde_conv = self.total_conv_spectrum_fn(  # type: ignore
                e_min, e_max, e_cm, measurement.energy_res
            )
        elif self.kind == "dec":  # type: ignore
            raise NotImplementedError()

        def bin_constraint(e_low, e_high, phi, sigma):
            """Subroutine to compute limit in a single bin."""
            # Compute integrated flux from DM annihilation
            phi_dm = dm_flux_factor * dnde_conv.integral(e_low, e_high)

            # If flux is finite and nonzero, set a limit
            if not np.isnan(phi_dm) and phi_dm > 0:
                # Compute maximum allow integrated flux
                phi_max = (
                    measurement.target.dOmega
                    * (e_high - e_low)
                    * (n_sigma * sigma + phi)
                )
                assert phi_max > 0
                return phi_max - phi_dm
            else:
                return np.inf

        # Compute limits for each bin
        bin_constraints = []

        for e_low, e_high, phi, sigma in zip(
            measurement.e_lows,
            measurement.e_highs,
            measurement.fluxes,
            measurement.upper_errors,
        ):
            bin_constraints.append(bin_constraint(e_low, e_high, phi, sigma))

        if method == "1bin":
            return min(bin_constraints)
        else:
            raise NotImplementedError()

    def constrain_binned_gamma(
        self,
        p1,
        p1_vals,
        p2,
        p2_vals,
        measurement,
        n_sigma=2,
        method="1bin",
        ls_or_img="image",
    ):
        """Computes constraints from gamma ray experiments in the p1-p2 plane."""
        img = np.zeros([len(p2_vals), len(p1_vals)])

        ls_or_img  # making linter shut up

        # Loop over the parameter values
        for idx_p1, p1_val in np.ndenumerate(p1_vals):
            for idx_p2, p2_val in np.ndenumerate(p2_vals):
                setattr(self, p1, p1_val)
                setattr(self, p2, p2_val)

                # Compute all constraints at this point in parameter space
                img[idx_p2[0], idx_p1[0]] = self._constrain_binned_gamma_helper(
                    measurement, n_sigma, method
                )

        return img

    def _img_to_ls(self, p1_vals, p2_vals, img):
        """Finds levels sets for an image."""
        contours_raw = measure.find_contours(img, level=0)
        contours = []

        # Convert from indices to values of p1 and p2
        for c in contours_raw:
            p1s = c[:, 1] / len(p1_vals) * (p1_vals[-1] - p1_vals[0]) + p1_vals[0]
            p2s = c[:, 0] / len(p2_vals) * (p2_vals[-1] - p2_vals[0]) + p2_vals[0]
            contours.append(np.array([p1s, p2s]))

        return contours

    @abstractmethod
    def constraints(self):
        r"""
        Get a dictionary of all available constraints.

        Subclasses must implement this method.

        Notes
        -----
        Each key in the dictionary is the name of a constraint. Each value is a
        function that is positive when the constraint is satisfied and negative
        when it is not.
        """
        raise NotImplementedError()
