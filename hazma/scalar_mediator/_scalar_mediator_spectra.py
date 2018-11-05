import numpy as np

from hazma.decay import muon
from hazma.decay import neutral_pion, charged_pion

from hazma.parameters import muon_mass as mmu
from hazma.parameters import electron_mass as me

from hazma.scalar_mediator.scalar_mediator_decay_spectrum \
    import dnde_decay_s, dnde_decay_s_pt


class ScalarMediatorSpectra:
    def dnde_ee(self, e_gams, e_cm, spectrum_type='all'):
        if spectrum_type == 'all':
            return (self.dnde_ee(e_gams, e_cm, 'fsr') +
                    self.dnde_ee(e_gams, e_cm, 'decay'))
        elif spectrum_type == 'fsr':
            return self.dnde_xx_to_s_to_ffg(e_gams, e_cm, me)
        elif spectrum_type == 'decay':
            return np.array([0.0 for _ in range(len(e_gams))])
        else:
            raise ValueError("Type {} is invalid. Use 'all', 'fsr' or \
                             'decay'".format(spectrum_type))

    def dnde_mumu(self, e_gams, e_cm, spectrum_type='all'):
        if spectrum_type == 'all':
            return (self.dnde_mumu(e_gams, e_cm, 'fsr') +
                    self.dnde_mumu(e_gams, e_cm, 'decay'))
        elif spectrum_type == 'fsr':
            return self.dnde_xx_to_s_to_ffg(e_gams, e_cm, mmu)
        elif spectrum_type == 'decay':
            return 2. * muon(e_gams, e_cm / 2.0)
        else:
            raise ValueError("Type {} is invalid. Use 'all', 'fsr' or \
                             'decay'".format(spectrum_type))

    def dnde_pi0pi0(self, e_gams, e_cm, spectrum_type='all'):
        if spectrum_type == 'all':
            return (self.dnde_pi0pi0(e_gams, e_cm, 'fsr') +
                    self.dnde_pi0pi0(e_gams, e_cm, 'decay'))
        if spectrum_type == 'fsr':
            return np.array([0.0 for _ in range(len(e_gams))])
        if spectrum_type == 'decay':
            return 2.0 * neutral_pion(e_gams, e_cm / 2.0)
        else:
            raise ValueError("Type {} is invalid. Use 'all', 'fsr' or \
                             'decay'".format(spectrum_type))

    def dnde_pipi(self, e_gams, e_cm, spectrum_type='all'):
        if spectrum_type == 'all':
            return (self.dnde_pipi(e_gams, e_cm, 'fsr') +
                    self.dnde_pipi(e_gams, e_cm, 'decay'))
        elif spectrum_type == 'fsr':
            return self.dnde_xx_to_s_to_pipig(e_gams, e_cm)
        elif spectrum_type == 'decay':
            return 2. * charged_pion(e_gams, e_cm / 2.0)
        else:
            raise ValueError("Type {} is invalid. Use 'all', 'fsr' or \
                             'decay'".format(spectrum_type))

    def dnde_ss(self, e_gams, e_cm, fs="total"):
        # Each scalar gets half the COM energy
        eng_s = e_cm / 2.

        ms = self.ms
        pws = self.partial_widths()
        pw_array = np.zeros(5, dtype=float)

        pw_array[0] = pws["e e"] / pws["total"]
        pw_array[1] = pws["mu mu"] / pws["total"]
        pw_array[2] = pws["pi0 pi0"] / pws["total"]
        pw_array[3] = pws["pi pi"] / pws["total"]
        pw_array[4] = pws["g g"] / pws["total"]

        if hasattr(e_gams, "__len__"):
            return 2. * dnde_decay_s(e_gams, eng_s, ms, pw_array, fs)
        return 2. * dnde_decay_s_pt(e_gams, eng_s, ms, pw_array, fs)

    def spectra(self, e_gams, e_cm):
        """
        Compute the total spectrum from two fermions annihilating through a
        scalar mediator to mesons and leptons.

        Parameters
        ----------
        e_cm : float
            Center of mass energy.
        e_gams : array-like, optional
            Gamma ray energies to evaluate the spectrum at.

        Returns
        -------
        specs : dictionary
            Dictionary of the spectra. The keys are 'total', 'mu mu', 'e e',
            'pi0 pi0', 'pi pi'
        """

        # Compute branching fractions
        bfs = self.annihilation_branching_fractions(e_cm)

        # Only compute the spectrum if the channel's branching fraction is
        # nonzero
        def spec_helper(bf, specfn):
            if bf != 0:
                return bf * specfn(e_gams, e_cm)
            else:
                return np.zeros(e_gams.shape)

        # Pions
        npions = spec_helper(bfs['pi0 pi0'], self.dnde_pi0pi0)
        cpions = spec_helper(bfs['pi pi'], self.dnde_pipi)

        # Leptons
        muons = spec_helper(bfs['mu mu'], self.dnde_mumu)
        electrons = spec_helper(bfs['e e'], self.dnde_ee)

        # mediator
        mediator = spec_helper(bfs['s s'], self.dnde_ss)

        # Compute total spectrum
        total = muons + electrons + npions + cpions + mediator

        # Define dictionary for spectra
        specs = {'total': total,
                 'mu mu': muons,
                 'e e': electrons,
                 'pi0 pi0': npions,
                 'pi pi': cpions,
                 's s': mediator}

        return specs

    def spectrum_functions(self):
        """
        Returns a dictionary of all the avaiable spectrum functions for
        a pair of initial state fermions with mass `mx` annihilating into
        each available final state.

        Each argument of the spectrum functions in `eng_gams`, an array
        of the gamma ray energies to evaluate the spectra at and `e_cm`, the
        center of mass energy of the process.
        """
        return {'mu mu': lambda e_gams, e_cm: self.dnde_mumu(e_gams, e_cm),
                'e e': lambda e_gams, e_cm: self.dnde_ee(e_gams, e_cm),
                'pi0 pi0': lambda e_gams, e_cm:
                    self.dnde_pi0pi0(e_gams, e_cm),
                'pi pi': lambda e_gams, e_cm:
                    self.dnde_pipi(e_gams, e_cm),
                's s': lambda e_gams, e_cm: self.dnde_ss(e_gams, e_cm)}

    def gamma_ray_lines(self, e_cm):
        bf = self.annihilation_branching_fractions(e_cm)["g g"]

        return {"g g": {"energy": e_cm / 2.0, "bf": bf}}
