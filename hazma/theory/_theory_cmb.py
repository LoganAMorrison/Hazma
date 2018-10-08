from hazma.cmb import vx_cmb, f_eff_g, f_eff_ep
from scipy.interpolate import interp1d
from scipy.integrate import quad

import numpy as np


class TheoryCMB(object):
    def cmb_limit(self, x_kd=1.0e-4):
        """Computes the CMB limit on <sigma v>.

        Notes
        -----
        We use the constraint from the Planck collaboration:
            f_eff <sigma v> / m_x < 4.1e-31 cm^3 s^-1 MeV^-1

        Parameters
        ----------
        mx : float
            Dark matter mass in MeV.
        f_eff : float
            Efficiency with which energy is deposited into the CMB by DM
            annihilations.

        Returns
        -------
        <sigma v> : float
            Upper bound on <sigma v>, in cm^3 s^-1.
        """
        return 4.1e-31 * self.mx / self.f_eff(x_kd)

    def cmb_limits(self, mxs, x_kd=1.0e-4):
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
            Array of upper bounds on <sigma v> (in cm^3/s) for each mass in
            mxs.
        """
        lims = []

        for mx in mxs:
            self.mx = mx
            lims.append(self.cmb_limit(x_kd))

        return np.array(lims)

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
        e_cm = 2. * self.mx * (1. + 0.5 * vx_cmb(self.mx, x_kd)**2)

        # Lower bound on integrals
        if fs == "g g":
            f_eff_base = f_eff_g
        elif fs == "e e":
            f_eff_base = f_eff_ep

        e_min = f_eff_base.x[0]

        # Continuum contributions from photons. Create an interpolator to avoid
        # recomputing spectrum.
        if fs == "g g":
            spec_fn = self.total_spectrum
        elif fs == "e e":
            def spec_fn(es, e_cm):
                return 2. * self.total_positron_spectrum(es, e_cm)

        if mode == "interp":
            # If RAMBO is needed to compute the spectrum, it is prohibitively
            # time-consuming to try integrating the spectrum function. Instead,
            # simultaneously compute the spectrum over a grid of points.
            es = np.logspace(np.log10(e_min), np.log10(self.mx), 1000)
            dnde_tot = spec_fn(es, e_cm)
            spec_interp = interp1d(es, dnde_tot, bounds_error=False,
                                   fill_value=0.)

            def integrand(e):
                return e * spec_interp(e) * f_eff_base(e)

            f_eff_dm = quad(integrand, e_min, self.mx, epsabs=0,
                            epsrel=1e-3)[0] / (2. * self.mx)
        elif mode == "quad":
            # If RAMBO is not needed to compute the spectrum, this will give
            # much cleaner results.
            def integrand(e):
                return e * spec_fn(e, e_cm) * f_eff_base(e)

            f_eff_dm = quad(integrand, e_min, self.mx, epsabs=0,
                            epsrel=1e-3)[0] / (2. * self.mx)

        # Line contributions from the relevant final state
        if fs == "g g":
            lines = self.gamma_ray_lines(e_cm)
        elif fs == "e e":
            lines = self.positron_lines(e_cm)

        f_eff_line_dm = 0.

        for ch, line in lines.iteritems():
            energy = line["energy"]

            # Make sure the base f_eff is defined at this energy
            if energy > e_min:
                bf = line["bf"]
                multiplicity = 2. if ch == fs else 1.
                f_eff_line_dm += (energy * bf * f_eff_base(energy) *
                                  multiplicity / (2. * self.mx))

        return f_eff_dm + f_eff_line_dm

    def f_eff_g(self, x_kd=1e-4):
        return self._f_eff_helper("g g", x_kd, "quad")

    def f_eff_ep(self, x_kd=1e-4):
        return self._f_eff_helper("e e", x_kd, "quad")

    def f_eff(self, x_kd=1.0e-4):
        return self.f_eff_ep(x_kd) + self.f_eff_g(x_kd)

    def f_effs(self, mxs, x_kd=1.0e-4):
        f_eff_vals = []

        for mx in mxs:
            self.mx = mx
            f_eff_vals.append(self.f_eff(x_kd))

        return np.array(f_eff_vals)
