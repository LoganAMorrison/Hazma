from scipy.interpolate import interp1d
from scipy.integrate import quad
from hazma.cmb import vx_cmb, f_eff_g, f_eff_ep, p_ann_planck_temp_pol

import numpy as np


class TheoryCMB:
    def cmb_limit(self, x_kd=1.0e-4, p_ann=p_ann_planck_temp_pol):
        r"""
        Computes the CMB limit on <sigma v>.

        This is derived by requiring that

        .. math::
            f_{\mathrm{eff}} \langle \sigma v \rangle / m_{\chi} < p_{\mathrm{ann}},

        where :math:`f_{\mathrm{eff}}` is the efficiency with which dark matter
        annihilations around recombination inject energy into the plasma and
        :math:`p_{\mathrm{ann}}` is derived from CMB observations.

        Parameters
        ----------
        x_kd: float
            T_kd / m_x, where T_kd is the dark matter's kinetic decoupling
            temperature. This will be computed self-consistently in future
            versions of ``hazma``.
        p_ann : float
            Constraint on energy release per DM annihilation in cm^3 s^-1
            MeV^-1.

        Returns
        -------
        <sigma v> : float
            Upper bound on <sigma v>, in cm^3 s^-1.
        """
        # TODO: account for non-self-conjugate DM. See discussion in Gondolo
        # and Gelmini.
        # if self.self_conjugate:
        #     factor = 1.0
        # else:
        #     factor = 0.5

        return p_ann * self.mx / self.f_eff(x_kd)

    def _f_eff_helper(self, fs, x_kd=1e-4, mode="quad"):
        """Computes f_eff^gg or f_eff^ep for DM annihilation.

        Parameters
        ----------
        fs : string
            "g g" or "e e", depending on which f_eff the user wants to compute.
        x_kd: float
            T_kd / m_x, where T_kd is the dark matter's kinetic decoupling
            temperature.
        mode : string
            "quad" or "interp". The first mode should be used if fs is "g g"
            ("e e") and none of the gamma ray spectrum functions (positron
            spectrum functions) use RAMBO as it is more accurate.

        Returns
        -------
        f_eff_dm : float
            f_eff for photons or electrons and positrons.
        """
        # Center of mass energy
        e_cm = 2.0 * self.mx * (1.0 + 0.5 * vx_cmb(self.mx, x_kd) ** 2)

        if fs == "g g":
            f_eff_base = f_eff_g
            lines = self.gamma_ray_lines(e_cm)
            spec_fn = self.total_spectrum
        elif fs == "e e":
            f_eff_base = f_eff_ep
            lines = self.positron_lines(e_cm)

            def spec_fn(es, e_cm):
                return 2.0 * self.total_positron_spectrum(es, e_cm)

        # Lower bound on integrals. Upper bound is many GeV, so we don't need
        # to do error checking.
        e_min = f_eff_base.x[0]

        # Continuum contributions from photons. Create an interpolator to avoid
        # recomputing spectrum.
        if mode == "interp":
            # If RAMBO is needed to compute the spectrum, it is prohibitively
            # time-consuming to try integrating the spectrum function. Instead,
            # simultaneously compute the spectrum over a grid of points.
            es = np.geomspace(e_min, e_cm / 2, 1000)
            dnde_tot = spec_fn(es, e_cm)
            spec_interp = interp1d(es, dnde_tot, bounds_error=False, fill_value=0.0)

            def integrand(e):
                return e * spec_interp(e) * f_eff_base(e)

            f_eff_dm = quad(integrand, e_min, e_cm / 2, epsabs=0, epsrel=1e-3)[0] / e_cm
        elif mode == "quad":
            # If RAMBO is not needed to compute the spectrum, this will give
            # much cleaner results.
            def integrand(e):
                return e * spec_fn(e, e_cm) * f_eff_base(e)

            f_eff_dm = quad(integrand, e_min, e_cm / 2, epsabs=0, epsrel=1e-3)[0] / e_cm

        # Sum up line contributions
        f_eff_line_dm = 0.0
        for ch, line in lines.items():
            energy = line["energy"]

            # Make sure the base f_eff is defined at this energy
            if energy > e_min:
                bf = line["bf"]
                multiplicity = 2.0 if ch == fs else 1.0
                f_eff_line_dm += energy * bf * f_eff_base(energy) * multiplicity / e_cm

        return f_eff_dm + f_eff_line_dm

    def f_eff_g(self, x_kd=1e-4):
        return self._f_eff_helper("g g", x_kd, "quad")

    def f_eff_ep(self, x_kd=1e-4):
        return self._f_eff_helper("e e", x_kd, "quad")

    def f_eff(self, x_kd=1.0e-4):
        """
        Computes :math:`f_{\mathrm{eff}}` the efficiency with which dark matter
        annihilations around recombination inject energy into the thermal
        plasma.
        """
        return self.f_eff_ep(x_kd) + self.f_eff_g(x_kd)
