from ..theory import Theory

from scalar_mediator_cross_sections import branching_fractions as bfs
from scalar_mediator_cross_sections import cross_sections as cs

from scalar_mediator_positron_spectra import positron_spectra
from scalar_mediator_positron_spectra import positron_lines as pls

from scalar_mediator_spectra import spectra as specs
from scalar_mediator_spectra import gamma_ray_lines as gls
from scalar_mediator_spectra import dnde_mumu, dnde_ee, dnde_s
from scalar_mediator_spectra import dnde_neutral_pion, dnde_charged_pion

from scalar_mediator_widths import partial_widths as pws

from .scalar_mediator_parameters import ScalarMediatorParameters


class ScalarMediator(Theory, ScalarMediatorParameters):
    r"""
    Create a scalar mediator model object.

    Creates an object for the scalar mediator model given UV couplings from
    common UV complete models of a real scalar extension of the SM. The UV
    complete models are:

        1) Scalar mediator coupling to a new heavy quark. When the heavy quark
           is integrated out of the theory, the scalar obtains an effective
           coupling to gluons, leading to a coupling to pions through a
           dialation current.

        2) Scalar mediator mixing with the standard model higgs. The scalar
           mediator obtains couplings to the massive standard model states
           which will be `sin(theta) m / v_h` where theta is the mixing angle
           between the higgs and the scalar, m is the mass of the massive state
           and v_h is the higgs vev.  The scalar mediator also gets an
           effective coupling to gluons when the top quark is integrated out.

    Attributes
    ----------
    mx : float
        Mass of the initial state fermion.
    ms : float
        Mass of the scalar mediator.
    gsxx : float
        Coupling of scalar mediator to initial state fermions.
    gsff : float
        Coupling of scalar mediator to standard model fermions. This is
        the sine of the mixing angle between scalar mediator and the higgs.
    gsGG : float
        Coupling of the scalar mediator to gluons.
    gsFF : float
        Coupling of the scalar mediator to photons.
    """

    def __init__(self, mx, ms, gsxx, gsff, gsGG, gsFF):
        """
        Initialize scalar mediator model parameters.

        Parameters
        ----------
        mx : float
            Mass of the initial state fermion.
        ms : float
            Mass of the scalar mediator.
        gsxx : float
            Coupling of scalar mediator to initial state fermions.
        gsff : float
            Coupling of scalar mediator to standard model fermions. This is
            the sine of the mixing angle between scalar mediator and the higgs.
        gsGG : float
            Coupling of the scalar mediator to gluons.
        gsFF : float
            Coupling of the scalar mediator to photons.
        """
        super(ScalarMediator, self).__init__(mx, ms, gsxx, gsff, gsGG, gsFF)

    def description(self):
        """
        Returns a string giving the details of the model.
        """
        return '''
        The UV complete models are: \n \n

        \t 1) Scalar mediator coupling to a new heavy quark. When the heavy \n
        \t    quark is integrated out of the theory, the scalar obtains an \n
        \t    effective coupling to gluons, leading to a coupling to pions \n
        \t    through a dialation current. \n \n

        \t 2) Scalar mediator mixing with the standard model higgs. The \n
        \t    scalar mediator obtains couplings to the massive standard \n
        \t    model states which will be `sin(theta) m / v_h` where theta \n
        \t    is the mixing angle between the higgs and the scalar, m is the \n
        \t    mass of the massive state and v_h is the higgs vev.  The \n
        \t    scalar mediator also gets an effective coupling to gluons when \n
        \t    the top quark is integrated out. \n

        Attributes \n
        ---------- \n
        mx : float \n
        \t Mass of the initial state fermion. \n
        ms : float \n
        \t Mass of the scalar mediator. \n
        gsxx : float \n
        \t Coupling of scalar mediator to initial state fermions. \n
        gsff : float \n
        \t Coupling of scalar mediator to standard model fermions. This is \n
        \t the sine of the mixing angle between scalar mediator and the
        \t higgs. \n
        gsGG : float \n
        \t Coupling of the scalar mediator to gluons. \n
        gsFF : float \n
        \t Coupling of the scalar mediator to photons. \n

        Methods \n
        ------- \n
        list_final_states : \n
        \t Return a list of the available final states. \n
        cross_sections : \n
        \t Computes the all the cross sections of the theory and returns \n
        \t a dictionary containing the cross sections. \n
        branching_fractions : \n
        \t Computes the all the branching fractions of the theory and \n
        \t returns a dictionary containing the branching fractions. \n
        spectra : \n
        \t Computes all spectra of the theory for a pair of initial \n
        \t state fermions annihilating into each available final state \n
        \t and returns a dictionary of arrays containing the spectra. \n
        spectrum_functions :
        \t Returns a dictionary of all the avaiable spectrum functions for \n
        \t a pair of initial state fermions with mass `mx` \n
        \t annihilating into each available final state. \n
        partial_widths : \n
        \t Returns a dictionary for the partial decay widths of the scalar \n
        \t mediator. \n
        '''

    @classmethod
    def list_final_states(cls):
        """
        Return a list of the available final states.

        Returns
        -------
        fs : array-like
            Array of the available final states.
        """
        return ['mu mu', 'e e', 'g g', 'pi0 pi0', 'pi pi', 's s']

    def cross_sections(self, cme):
        """
        Compute the all the cross sections of the theory.

        Parameters
        ----------
        cme : float
            Center of mass energy.

        Returns
        -------
        cs : dictionary
            Dictionary of the cross sections of the theory.
        """
        return cs(cme, self)

    def branching_fractions(self, cme):
        """
        Compute the branching fractions for two fermions annihilating through a
        scalar mediator to mesons and leptons.

        Parameters
        ----------
        Q : float
            Center of mass energy.

        Returns
        -------
        bfs : dictionary
            Dictionary of the branching fractions. The keys are 'total',
            'mu mu', 'e e', 'pi0 pi0', 'pi pi'
        """
        return bfs(cme, self)

    def gamma_ray_lines(self, cme):
        return gls(cme, self)

    def spectra(self, egams, cme):
        """
        Compute the total spectrum from two fermions annihilating through a
        scalar mediator to mesons and leptons.

        Parameters
        ----------
        egams : array-like, optional
            Gamma ray energies to evaluate the spectrum at.
        cme : float
            Center of mass energy.

        Returns
        -------
        specs : dictionary
            Dictionary of the spectra. The keys are 'total', 'mu mu', 'e e',
            'pi0 pi0', 'pi pi'
        """
        return specs(egams, cme, self)

    def spectrum_functions(self):
        """
        Returns a dictionary of all the avaiable spectrum functions for
        a pair of initial state fermions with mass `mx` annihilating into
        each available final state.

        Each argument of the spectrum functions in `eng_gams`, an array
        of the gamma ray energies to evaluate the spectra at and `cme`, the
        center of mass energy of the process.
        """
        return {'mu mu': lambda e_gams, cme: dnde_mumu(e_gams, cme, self),
                'e e': lambda e_gams, cme: dnde_ee(e_gams, cme, self),
                'pi0 pi0': lambda e_gams, cme:
                    dnde_neutral_pion(e_gams, cme, self),
                'pi pi': lambda e_gams, cme:
                    dnde_charged_pion(e_gams, cme, self),
                's s': lambda e_gams, cme: dnde_s(e_gams, cme, self)}

    def partial_widths(self):
        """
        Returns a dictionary for the partial decay widths of the scalar
        mediator.

        Returns
        -------
        width_dict : dictionary
            Dictionary of all of the individual decay widths of the scalar
            mediator as well as the total decay width. The possible decay
            modes of the scalar mediator are 'g g', 'pi0 pi0', 'pi pi', 'x x'
            and 'f f'. The total decay width has the key
            'total'.
        """
        return pws(self)

    def positron_spectra(self, eng_ps, cme):
        """
        Compute the total positron spectrum from two fermions annihilating
        through a scalar mediator to mesons and leptons.

        Parameters
        ----------
        eng_ps : array-like, optional
            Positron energies to evaluate the spectrum at.
        cme : float
            Center of mass energy.

        Returns
        -------
        specs : dictionary
            Dictionary of the spectra. The keys are 'total', 'mu mu',
            'pi pi'.
        """
        return positron_spectra(eng_ps, cme, self)

    def positron_lines(self, cme):
        """
        Returns a dictionary of the energies and branching fractions of
        positron lines

        Parameters
        ----------
        eng_ps : array-like, optional
            Positron energies to evaluate the spectrum at.
        cme : float
            Center of mass energy.

        Returns
        -------
        lines : dictionary
            Dictionary of the lines. The keys are 'e e'.
        """
        return pls(cme, self)

    from scalar_mediator_constraints import (width_B_k_s,
                                             width_k_pi_s,
                                             width_kl_pi0_s,
                                             width_B_xs_s,
                                             constraint_B_k_invis,
                                             constraint_B_k_mu_mu,
                                             constraint_B_k_e_e,
                                             constraint_k_pi_invis,
                                             constraint_kl_pi0_mu_mu,
                                             constraint_kl_pi0_e_e,
                                             constrain_beam_dump,
                                             constraints)
