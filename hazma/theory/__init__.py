from abc import ABCMeta, abstractmethod

import numpy as np

from hazma.parameters import convolved_spectrum_fn
from hazma.theory._theory_cmb import TheoryCMB
from hazma.theory._theory_constrain import TheoryConstrain
from hazma.theory._theory_gamma_ray_limits import TheoryGammaRayLimits


class TheoryAnn(TheoryGammaRayLimits, TheoryCMB, TheoryConstrain):
    """
    Represents a sub-GeV DM theory.
    """

    __metaclass__ = ABCMeta

    kind = "ann"

    @staticmethod
    @abstractmethod
    def list_annihilation_final_states():
        r"""
        Lists annihilation final states.

        Subclasses must implement this method.

        Returns
        -------
        fss : list(str)
            Possible annihilation final states.
        """
        pass

    @abstractmethod
    def annihilation_cross_section_funcs(self):
        r"""
        Gets functions to compute annihilation cross sections.

        Subclasses must implement this method.

        Returns
        -------
        sigma_fns : dict(str, (float) -> float)
            A map specifying a function for each final state which takes a
            center of mass energy and returns the corresponding annihilation
            cross section in :math:`\mathrm{MeV}^{-2}`.
        """
        pass

    def annihilation_cross_sections(self, e_cm):
        r"""
        Computes annihilation cross sections.

        Parameters
        ---------
        e_cm : float
            Center of mass energy for the annihilation in MeV.

        Returns
        -------
        sigmas : dict(str, float)
            Annihilation cross section into each final state in
            :math:`\mathrm{MeV}^{-2}` as well as the total cross section.
        """
        sigmas = {
            fs: sigma_fn(e_cm)
            for fs, sigma_fn in self.annihilation_cross_section_funcs().items()
        }
        sigmas["total"] = sum(sigmas.values())
        return sigmas

    def annihilation_branching_fractions(self, e_cm):
        r"""
        Computes annihilation branching fractions.

        Parameters
        ---------
        e_cm : float
            Center of mass energy for the annihilation in MeV.

        Returns
        -------
        bfs : dict(str, float)
            Annihilation branching fractions into each final state.
        """
        cs = self.annihilation_cross_sections(e_cm)

        if cs["total"] == 0:
            return {fs: 0.0 for fs in cs if fs != "total"}
        else:
            return {
                fs: sigma / cs["total"]
                for fs, sigma in cs.items()
                if fs != "total"
            }

    @abstractmethod
    def partial_widths(self):
        """
        Computes mediator decay widths.

        Subclasses must implement this method.

        Returns
        -------
        widths : dict(str, float)
            Mediator partial widths in MeV as the total cross decay width.
        """
        pass

    @abstractmethod
    def _spectrum_funcs(self):
        r"""
        Gets a function computing the continuum gamma-ray spectrum for
        annihilations into each relevant final state.

        Subclasses must implement this method.

        Returns
        -------
        spec_fns : dict(str, (float or np.array, float) -> float)
            :math:`dN/dE_\gamma` as a function of photon energies and the
            annihilation center of mass energy for annihilation into each final
            state that produces a continuum spectrum.
        """
        pass

    def spectrum_funcs(self):
        """
        Wraps _spectrum_functions so the function for a final state returns
        zero when annihilation into that state is not permitted.
        """
        sigma_ann_fns = self.annihilation_cross_section_funcs()
        dndes_wrapped = {}

        for fs, dnde in self._spectrum_funcs().items():

            def dnde_wrapped(e_gams, e_cm):
                if sigma_ann_fns[fs](e_cm) > 0:
                    return dnde(e_gams, e_cm)
                else:
                    return np.zeros_like(e_gams)

            dndes_wrapped[fs] = dnde_wrapped

        return dndes_wrapped

    def spectra(self, e_gams, e_cm):
        r"""
        Gets the contributions to the continuum gamma-ray annihilation spectrum
        for each final state.

        Parameters
        ---------
        e_gams : float or float numpy.array
            Photon energy or energies at which to compute the spectrum.
        e_cm : float
            Center of mass energy for the annihilation in MeV.

        Returns
        -------
        specs : dict(str, float)
            Contribution to :math:`dN/dE_\gamma` at the given photon energies
            and center-of-mass energy for each relevant final state. More
            specifically, this is the spectrum for annihilation into each
            channel rescaled by the corresponding branching fraction into that
            channel.
        """
        bfs = self.annihilation_branching_fractions(e_cm)
        specs = {}

        for fs, dnde_func in self.spectrum_funcs().items():
            # Only compute the spectrum if the channel is accessible
            if bfs[fs] != 0:
                specs[fs] = bfs[fs] * dnde_func(e_gams, e_cm)
            else:
                if type(e_gams) == float:
                    specs[fs] = 0
                else:
                    specs[fs] = np.zeros(e_gams.shape)

        specs["total"] = sum(specs.values())

        return specs

    def total_spectrum(self, e_gams, e_cm):
        r"""
        Computes total continuum gamma-ray spectrum.

        Parameters
        ----------
        e_gams : float or float numpy.array
            Photon energy or energies at which to compute the spectrum.
        e_cm : float
            Annihilation center of mass energy.

        Returns
        -------
        spec : float numpy.array
            Array containing the total annihilation gamma-ray spectrum.
        """
        if hasattr(e_gams, "__len__"):
            return self.spectra(e_gams, e_cm)["total"]
        else:
            return self.spectra(np.array([e_gams]), e_cm)["total"]

    @abstractmethod
    def _gamma_ray_line_energies(self, e_cm):
        r"""
        Gets information about annihilation into gamma-ray lines.

        Subclasses must implement this method.

        Parameters
        ----------
        e_cm : float
            Annihilation center of mass energy.

        Returns
        -------
        lines : dict(str, float)
            Energy of the photon for each final state containing a
            monochromatic photon.
        """
        pass

    def gamma_ray_lines(self, e_cm):
        """
        Returns
        -------
        lines : dict(str, dict(str, float))
            For each final state containing a monochromatic photon, gives the
            energy of the photon and branching fraction into that final state.
        """
        bfs = self.annihilation_branching_fractions(e_cm)
        lines = {}

        for fs, e in self._gamma_ray_line_energies(e_cm).items():
            lines[fs] = {"energy": e, "bf": bfs[fs]}

        return lines

    def total_conv_spectrum_fn(
        self, e_gam_min, e_gam_max, e_cm, energy_res, n_pts=1000
    ):
        r"""
        Computes the total gamma-ray spectrum convolved with an energy
        resolution function.

        Parameters
        ----------
        e_min : float
            Lower bound of energy range over which to perform convolution.
        e_max : float
            Upper bound of energy range over which to perform convolution.
        e_cm : float
            Center of mass energy for DM annihilation.
        energy_res : float -> float
            The detector's energy resolution (Delta E / E) as a function of
            photon energy in MeV.
        n_pts : float
            Number of points to use to create resulting interpolating function.
            More points gives higher accuracy at the cost of computing time,
            but is necessary if the continuum spectrum contains very sharp
            features.

        Returns
        -------
        dnde_conv : InterpolatedUnivariateSpline
            An interpolator giving the DM annihilation spectrum smeared by the
            energy resolution function. Using photon energies outside the range
            [e_min, e_max] will produce a ``bounds_error``.
        """
        return convolved_spectrum_fn(
            e_gam_min,
            e_gam_max,
            energy_res,
            lambda e_gams: self.total_spectrum(e_gams, e_cm),
            self.gamma_ray_lines(e_cm),
            n_pts,
        )

    @abstractmethod
    def _positron_spectrum_funcs(self):
        r"""
        Gets a function computing the continuum positron spectrum for
        annihilations into each relevant final state.

        Subclasses must implement this method.

        Returns
        -------
        spec_fns : dict(str, (float or np.array, float) -> float)
            :math:`dN/dE_{e^+}` as a function of positron energies and the
            annihilation center of mass energy for annihilation into each final
            state that produces a continuum spectrum.
        """
        pass

    def positron_spectrum_funcs(self):
        """
        Wraps _positron_spectrum_functions so the function for a final state
        returns zero when annihilation into that state is not permitted.
        """
        sigma_ann_fns = self.annihilation_cross_section_funcs()
        dndes_wrapped = {}

        for fs, dnde in self._positron_spectrum_funcs().items():

            def dnde_wrapped(e_ps, e_cm):
                if sigma_ann_fns[fs](e_cm) > 0:
                    return dnde(e_ps, e_cm)
                else:
                    return np.zeros_like(e_ps)

            dndes_wrapped[fs] = dnde_wrapped

        return dndes_wrapped

    def positron_spectra(self, e_ps, e_cm):
        r"""
        Gets the contributions to the continuum positron annihilation spectrum
        for each final state.

        Parameters
        ---------
        e_ps : float or float numpy.array
            Positron energy or energies at which to compute the spectrum.
        e_cm : float
            Center of mass energy for the annihilation in MeV.

        Returns
        -------
        specs : dict(str, float)
            Contribution to :math:`dN/dE_\gamma` at the given positron energies
            and center-of-mass energy for each relevant final state. More
            specifically, this is the spectrum for annihilation into each
            channel rescaled by the corresponding branching fraction into that
            channel.
        """
        bfs = self.annihilation_branching_fractions(e_cm)
        specs = {}

        for fs, dnde_pos_func in self.positron_spectrum_funcs().items():
            specs[fs] = bfs[fs] * dnde_pos_func(e_ps, e_cm)

        specs["total"] = sum(specs.values())

        return specs

    def total_positron_spectrum(self, e_ps, e_cm):
        r"""
        Computes the total positron ray spectrum.

        Parameters
        ----------
        e_ps : float or float numpy.array
            Positron energy or energies at which to compute the spectrum.
        e_cm : float
            Annihilation center of mass energy.

        Returns
        -------
        spec : float numpy.array
            Array containing the total annihilation positron spectrum.
        """
        if hasattr(e_ps, "__len__"):
            return self.positron_spectra(e_ps, e_cm)["total"]
        else:
            return self.positron_spectra(np.array([e_ps]), e_cm)["total"]

    @abstractmethod
    def _positron_line_energies(self, e_cm):
        r"""
        Gets information about annihilation into monochromatic positrons.

        Subclasses must implement this method.

        Parameters
        ----------
        e_cm : float
            Annihilation center of mass energy.

        Returns
        -------
        lines : dict(str, float)
            Energy of positron in each final state containing a monochromatic
            positron.
        """
        pass

    def positron_lines(self, e_cm):
        """
        Returns
        -------
        lines : dict(str, dict(str, float))
            For each final state containing a monochromatic photon, gives the
            energy of the photon and branching fraction into that final state.
        """
        bfs = self.annihilation_branching_fractions(e_cm)
        lines = {}

        for fs, e in self._positron_line_energies(e_cm).items():
            lines[fs] = {"energy": e, "bf": bfs[fs]}

        return lines

    def total_conv_positron_spectrum_fn(
        self, e_p_min, e_p_max, e_cm, energy_res, n_pts=1000
    ):
        r"""
        Computes the total positron spectrum convolved with an energy
        resolution function.

        Parameters
        ----------
        e_min : float
            Lower bound of energy range over which to perform convolution.
        e_max : float
            Upper bound of energy range over which to perform convolution.
        e_cm : float
            Center of mass energy for DM annihilation.
        energy_res : float -> float
            The detector's energy resolution (Delta E / E) as a function of
            positron energy in MeV.
        n_pts : float
            Number of points to use to create resulting interpolating function.
            More points gives higher accuracy at the cost of computing time,
            but is necessary if the continuum spectrum contains very sharp
            features.

        Returns
        -------
        dnde_conv : InterpolatedUnivariateSpline
            An interpolator giving the DM annihilation spectrum smeared by the
            energy resolution function. Using positron energies outside the
            range [e_min, e_max] will produce a ``bounds_error``.
        """
        return convolved_spectrum_fn(
            e_p_min,
            e_p_max,
            e_cm,
            energy_res,
            lambda e_ps: self.total_positron_spectrum(e_ps, e_cm),
            self.positron_lines(e_cm),
            n_pts,
        )


class TheoryDec(TheoryGammaRayLimits, TheoryCMB, TheoryConstrain):

    __metaclass__ = ABCMeta

    kind = "dec"

    @staticmethod
    @abstractmethod
    def list_decay_final_states():
        pass

    @abstractmethod
    def _decay_widths(self):
        """
        Decay width into each final state.
        """
        pass

    def decay_widths(self):
        """
        Wraps `_decay_widths`, adding the total width in.
        """
        widths = {fs: width for fs, width in self._decay_widths().items()}
        widths["total"] = sum(widths.values())
        return widths

    def decay_branching_fractions(self):
        """
        Branching fraction for decay into each final state.
        """
        widths = self.decay_widths()

        if widths["total"] == 0:
            return {fs: 0.0 for fs in widths if fs != "total"}
        else:
            return {
                fs: width / widths["total"]
                for fs, width in widths.items()
                if fs != "total"
            }

    @abstractmethod
    def _spectrum_funcs(self):
        """
        Gets a function taking a photon energy and returning the continuum
        gamma ray spectrum dN/dE for each relevant decay final state.
        """
        pass

    def spectrum_funcs(self):
        widths = self.decay_widths()
        dndes_wrapped = {}

        def dnde_null(e_gams):
            return np.zeros_like(e_gams)

        for fs, dnde in self._spectrum_funcs().items():
            if widths[fs] > 0.0:
                dndes_wrapped[fs] = dnde
            else:
                dndes_wrapped[fs] = dnde_null

        return dndes_wrapped

    def spectra(self, e_gams):
        """
        Computes spectra for each final state at the provided photon energies.
        """
        bfs = self.decay_branching_fractions()
        specs = {}

        for fs, dnde_func in self.spectrum_funcs().items():
            specs[fs] = bfs[fs] * dnde_func(e_gams)

        specs["total"] = sum(specs.values())

        return specs

    def total_spectrum(self, e_gams):
        if hasattr(e_gams, "__len__"):
            return self.spectra(e_gams)["total"]
        else:
            return self.spectra(np.array([e_gams]))["total"]

    @abstractmethod
    def _gamma_ray_line_energies(self):
        """
        Returns dict of final states and photon energies for final states
        containing monochromatic gamma ray lines.
        """
        pass

    def gamma_ray_lines(self):
        bfs = self.decay_branching_fractions()
        lines = {}

        for fs, e in self._gamma_ray_line_energies().items():
            lines[fs] = {"energy": e, "bf": bfs[fs]}

        return lines

    def total_conv_spectrum_fn(
        self, e_gam_min, e_gam_max, energy_res, n_pts=1000
    ):
        return convolved_spectrum_fn(
            e_gam_min,
            e_gam_max,
            energy_res,
            self.total_spectrum,
            self.gamma_ray_lines(),
            n_pts,
        )

    @abstractmethod
    def _positron_spectrum_funcs(self):
        """
        Returns functions `float -> float` giving the continuum positron
        spectrum for each final state.
        """
        pass

    def positron_spectrum_funcs(self):
        widths = self.decay_widths()
        dndes_wrapped = {}

        for fs, dnde in self._positron_spectrum_funcs().items():

            def dnde_wrapped(e_ps):
                if widths[fs] > 0:
                    return dnde(e_ps)
                else:
                    return np.zeros_like(e_ps)

            dndes_wrapped[fs] = dnde_wrapped

        return dndes_wrapped

    def positron_spectra(self, e_ps):
        bfs = self.decay_branching_fractions()
        specs = {}

        for fs, dnde_pos_func in self.positron_spectrum_funcs().items():
            specs[fs] = bfs[fs] * dnde_pos_func(e_ps)

        specs["total"] = sum(specs.values())

        return specs

    def total_positron_spectrum(self, e_ps):
        if hasattr(e_ps, "__len__"):
            return self.positron_spectra(e_ps)["total"]
        else:
            return self.positron_spectra(np.array([e_ps]))["total"]

    @abstractmethod
    def _positron_line_energies(self):
        pass

    def positron_lines(self):
        bfs = self.decay_branching_fractions()
        lines = {}

        for fs, e in self._positron_line_energies().items():
            lines[fs] = {"energy": e, "bf": bfs[fs]}

        return lines

    def total_conv_positron_spectrum_fn(
        self, e_p_min, e_p_max, energy_res, n_pts=1000
    ):
        return convolved_spectrum_fn(
            e_p_min,
            e_p_max,
            energy_res,
            self.total_positron_spectrum,
            self.positron_lines(),
            n_pts,
        )
