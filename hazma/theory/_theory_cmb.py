from scipy.interpolate import interp1d
from scipy.integrate import quad
from hazma.cmb import vx_cmb, f_eff_g, f_eff_ep, p_ann_planck_temp_pol
from hazma.parameters import electron_mass

import numpy as np


class TheoryCMB:
    def cmb_limit(self, x_kd=1.0e-4, p_ann=p_ann_planck_temp_pol):
        r"""
        Computes the CMB limit on <sigma v>.

        This is derived by requiring that

        .. math::
            f_{\mathrm{eff}} \langle \sigma v \rangle / (2 m_{\chi})
            < p_{\mathrm{ann}},

        where :math:`f_{\mathrm{eff}}` is the efficiency with which dark matter
        annihilations around recombination inject energy into the plasma and
        :math:`p_{\mathrm{ann}}` is derived from CMB observations. The factor
        of two is the population factor of symmetric Dirac dark matter, whose
        particles and antiparticles each make up half the density, as Hazma's
        gamma-ray limits also assume.

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
            Upper bound on <sigma v>, in cm^3 s^-1. It is ``inf`` when the
            annihilation products deposit no energy, as below the electron
            mass with no photon above 5 keV.
        """
        f_eff = self.f_eff(x_kd)
        if f_eff == 0.0:
            return np.inf
        return 2.0 * p_ann * self.mx / f_eff  # type: ignore

    def _f_eff_helper(self, fs, x_kd=1e-4, mode="quad"):
        """Computes f_eff^gg or f_eff^ep, one term of the fold f_eff states.

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
        mx = self.mx  # type: ignore
        e_cm = 2.0 * mx * (1.0 + 0.5 * vx_cmb(mx, x_kd) ** 2)

        # Upper bounds on integrals are many GeV, so we don't need to do error
        # checking there.
        if fs == "g g":
            lines = self.gamma_ray_lines(e_cm)  # type: ignore
            spec_fn = self.total_spectrum  # type: ignore
            # Photons below the table's first point are not counted.
            e_min = f_eff_g.x[0]
            f_eff_base = f_eff_g
        elif fs == "e e":
            lines = self.positron_lines(e_cm)  # type: ignore

            def spec_fn(es, e_cm):
                return 2.0 * self.total_positron_spectrum(es, e_cm)  # type: ignore

            # Every pair member is counted. The table is indexed by kinetic
            # energy and is flat at its first point, 5 keV, which stands in for
            # every kinetic energy below it.
            e_min = electron_mass
            k_min = f_eff_ep.x[0]  # type: ignore

            def f_eff_base(e: float) -> float:
                return f_eff_ep(max(e - electron_mass, k_min))  # type: ignore

        # Continuum contributions. Create an interpolator to avoid recomputing
        # spectrum.
        if e_cm / 2 <= e_min:
            # No particle of this species can be injected.
            f_eff_dm = 0.0
        elif mode == "interp":
            # If RAMBO is needed to compute the spectrum, it is prohibitively
            # time-consuming to try integrating the spectrum function. Instead,
            # simultaneously compute the spectrum over a grid of points.
            es = np.geomspace(e_min, e_cm / 2, 1000)
            dnde_tot = spec_fn(es, e_cm)  # type: ignore
            spec_interp = interp1d(es, dnde_tot, bounds_error=False, fill_value=0.0)

            def integrand(e):
                return e * spec_interp(e) * f_eff_base(e)  # type: ignore

            f_eff_dm = quad(integrand, e_min, e_cm / 2, epsabs=0, epsrel=1e-3)[0]
        elif mode == "quad":
            # If RAMBO is not needed to compute the spectrum, this will give
            # much cleaner results.
            def integrand(e):
                # The spectrum functions return an array even for one energy.
                dnde = spec_fn(np.array([e]), e_cm)[0]
                return e * dnde * f_eff_base(e)  # type: ignore

            f_eff_dm = quad(integrand, e_min, e_cm / 2, epsabs=0, epsrel=1e-3)[0]

        # Sum up line contributions. A line in the final state that names the
        # species carries two of its particles: two photons, or an electron
        # and the positron the line counts.
        f_eff_line_dm = 0.0
        for ch, line in lines.items():  # type: ignore
            energy = line["energy"]

            if energy > e_min:
                bf = line["bf"]
                multiplicity = 2.0 if ch == fs else 1.0
                f_eff_line_dm += energy * bf * f_eff_base(energy) * multiplicity

        return (f_eff_dm + f_eff_line_dm) / (2.0 * mx)  # type: ignore

    def f_eff_g(self, x_kd=1e-4):
        return self._f_eff_helper("g g", x_kd, "quad")

    def f_eff_ep(self, x_kd=1e-4):
        return self._f_eff_helper("e e", x_kd, "quad")

    def f_eff(self, x_kd=1.0e-4):
        r"""
        Computes the CMB energy-deposition efficiency :math:`f_{\mathrm{eff}}`.

        :math:`f_{\mathrm{eff}}` is the efficiency with which dark matter
        annihilations around recombination inject energy into the thermal
        plasma. The injected spectrum is folded with the deposition efficiency curves
        of arXiv:1506.03811 and normalized to the rest energy of the
        annihilating pair:

        .. math::
            f_{\mathrm{eff}} = \frac{1}{2 m_{\chi}} \int dE \, E
            \left[f_{\gamma}(E) \frac{dN_{\gamma}}{dE}
            + f_{e^+e^-}(E - m_e) \frac{dN_{e^\pm}}{dE}\right].

        :math:`E` is the total energy of the injected particle. The
        :math:`e^+e^-` curve is tabulated in the kinetic energy of one pair
        member, and :math:`dN_{e^\pm}/dE` is the charge-summed spectrum: twice
        the positron spectrum, which counts positrons only, because the final
        states of dark matter annihilation are charge-symmetric. Lines are
        included with their branching fractions.

        Parameters
        ----------
        x_kd: float
            T_kd / m_x, where T_kd is the dark matter's kinetic decoupling
            temperature.

        Returns
        -------
        f_eff : float
            Fraction of the pair's rest energy deposited in the plasma.
        """
        return self.f_eff_ep(x_kd) + self.f_eff_g(x_kd)
