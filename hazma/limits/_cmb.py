import importlib_resources

from scipy.interpolate import interp1d
import numpy as np
from scipy.integrate import quad

from hazma.parameters import temp_cmb_formation

from ._abstract import AbstractLimit


"""
Functions required for computing CMB limits and related quantities.
"""

# Get paths to files inside the module
_f_eff_ep_ref = importlib_resources.files("hazma.limits.data") / "f_eff_ep.dat"
_f_eff_g_ref = importlib_resources.files("hazma.limits.data") / "f_eff_g.dat"

# Load f_eff^{e+ e-}
with importlib_resources.as_file(_f_eff_ep_ref) as path:
    f_eff_ep_data = np.loadtxt(path, delimiter=",").T
    f_eff_ep = interp1d(f_eff_ep_data[0] / 1.0e6, f_eff_ep_data[1])  # eV -> MeV

# Load f_eff^{e+ e-}
with importlib_resources.as_file(_f_eff_g_ref) as path:
    f_eff_g_data = np.loadtxt(path, delimiter=",").T
    f_eff_g = interp1d(f_eff_g_data[0] / 1.0e6, f_eff_g_data[1])  # eV -> MeV


#: Planck 2018 95% upper limit on p_ann from temperature + polarization
#: measurements, in cm^3 s^-1 MeV^-1
p_ann_planck_temp_pol = 3.5e-31  # temperature + polarization
#: Planck 2018 95% upper limit on p_ann from temperature + polarization +
#: lensing measurements, in cm^3 s^-1 MeV^-1
p_ann_planck_temp_pol_lensing = 3.3e-31  # temp + pol + lensing
#: Planck 2018 95% upper limit on p_ann from temperature + polarization +
#: lensing + BAO measurements, in cm^3 s^-1 MeV^-1
p_ann_planck_temp_pol_lensing_bao = 3.2e-31  # temp + pol + lensing + BAO


class CMBLimit(AbstractLimit):
    def __init__(self, x_kd=1e-4, p_ann=p_ann_planck_temp_pol):
        super().__init__()
        self._x_kd = x_kd
        self._p_ann = p_ann

    @property
    def x_kd(self):
        return self._x_kd

    @property
    def p_ann(self):
        return self._p_ann

    @property
    def description(self):
        return "[purple] CMB"

    @property
    def name(self):
        return "cmb"

    @staticmethod
    def vx_cmb(mx, x_kd):
        """Computes the DM relative velocity at CMB using eq. 28 from `this
        reference <https://arxiv.org/abs/1309.4091>`_.

        Parameters
        ----------
        mx : float
            Dark matter mass in MeV.
        x_kd: float
            T_kd / m_x, where T_kd is the dark matter's kinetic decoupling
            temperature.

        Returns
        -------
        v_x : float
            The DM relative velocity at the time of CMB formation.
        """
        return 2.0e-4 * 10e6 * temp_cmb_formation / mx * np.sqrt(1.0e-4 / x_kd)

    def _constrain(self, model):
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
        return self.p_ann * model.mx / self.f_eff(self.x_kd)  # type: ignore

    def _f_eff_helper(self, model, fs, mode="quad"):
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
        vx = self.vx_cmb(model.mx, self.x_kd) ** 2
        e_cm = 2.0 * model.mx * (1.0 + 0.5 * vx**2)

        if fs == "g g":
            f_eff_base = f_eff_g
            lines = model.gamma_ray_lines(e_cm)

            def spec_fn(es, e_cm):
                return model.total_spectrum(es, e_cm)

        elif fs == "e e":
            f_eff_base = f_eff_ep
            lines = model.positron_lines(e_cm)

            def spec_fn(es, e_cm):
                return 2.0 * model.total_positron_spectrum(es, e_cm)

        # Lower bound on integrals. Upper bound is many GeV, so we don't need
        # to do error checking.
        e_min = f_eff_base.x[0]  # type: ignore

        # Continuum contributions from photons. Create an interpolator to avoid
        # recomputing spectrum.
        if mode == "interp":
            # If RAMBO is needed to compute the spectrum, it is prohibitively
            # time-consuming to try integrating the spectrum function. Instead,
            # simultaneously compute the spectrum over a grid of points.
            es = np.geomspace(e_min, e_cm / 2, 1000)
            dnde_tot = spec_fn(es, e_cm)  # type: ignore
            spec_interp = interp1d(es, dnde_tot, bounds_error=False, fill_value=0.0)

            def integrand(e):
                return e * spec_interp(e) * f_eff_base(e)  # type: ignore

            f_eff_dm = quad(integrand, e_min, e_cm / 2, epsabs=0, epsrel=1e-3)[0] / e_cm

        elif mode == "quad":
            # If RAMBO is not needed to compute the spectrum, this will give
            # much cleaner results.
            def integrand(e):
                return e * spec_fn(e, e_cm) * f_eff_base(e)  # type: ignore

            f_eff_dm = quad(integrand, e_min, e_cm / 2, epsabs=0, epsrel=1e-3)[0] / e_cm

        # Sum up line contributions
        f_eff_line_dm = 0.0
        for ch, line in lines.items():  # type: ignore
            energy = line["energy"]

            # Make sure the base f_eff is defined at this energy
            if energy > e_min:
                bf = line["bf"]
                multiplicity = 2.0 if ch == fs else 1.0
                f_eff_line_dm += (
                    energy
                    * bf
                    * f_eff_base(energy)  # type: ignore
                    * multiplicity
                    / e_cm
                )

        return f_eff_dm + f_eff_line_dm  # type: ignore

    def f_eff_g(self, model):
        return self._f_eff_helper(model, "g g", "quad")

    def f_eff_ep(self, model):
        return self._f_eff_helper(model, "e e", "quad")

    def f_eff(self, model):
        r"""
        Computes :math:`f_{\mathrm{eff}}` the efficiency with which dark matter
        annihilations around recombination inject energy into the thermal
        plasma.
        """
        return self.f_eff_ep(model) + self.f_eff_g(model)
