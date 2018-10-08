import numpy as np

from hazma.decay import muon
from hazma.decay import neutral_pion, charged_pion

from hazma.parameters import muon_mass as mmu
from hazma.parameters import electron_mass as me

from hazma.scalar_mediator.scalar_mediator_decay_spectrum \
    import dnde_decay_s, dnde_decay_s_pt


class ScalarMediatorSpectra:
    def dnde_ee(self, egams, cme, spectrum_type='All'):
        if spectrum_type == 'All':
            return (self.dnde_ee(egams, cme, 'FSR') +
                    self.dnde_ee(egams, cme, 'Decay'))
        elif spectrum_type == 'FSR':
            return self.dnde_xx_to_s_to_ffg(egams, cme, me)
        elif spectrum_type == 'Decay':
            return np.array([0.0 for _ in range(len(egams))])
        else:
            raise ValueError("Type {} is invalid. Use 'All', 'FSR' or \
                             'Decay'".format(spectrum_type))

    def dnde_mumu(self, egams, cme, spectrum_type='All'):
        if spectrum_type == 'All':
            return (self.dnde_mumu(egams, cme, 'FSR') +
                    self.dnde_mumu(egams, cme, 'Decay'))
        elif spectrum_type == 'FSR':
            return self.dnde_xx_to_s_to_ffg(egams, cme, mmu)
        elif spectrum_type == 'Decay':
            return 2. * muon(egams, cme / 2.0)
        else:
            raise ValueError("Type {} is invalid. Use 'All', 'FSR' or \
                             'Decay'".format(spectrum_type))

    def dnde_neutral_pion(self, egams, cme, spectrum_type='All'):
        if spectrum_type == 'All':
            return (self.dnde_neutral_pion(egams, cme, 'FSR') +
                    self.dnde_neutral_pion(egams, cme, 'Decay'))
        if spectrum_type == 'FSR':
            return np.array([0.0 for _ in range(len(egams))])
        if spectrum_type == 'Decay':
            return 2.0 * neutral_pion(egams, cme / 2.0)
        else:
            raise ValueError("Type {} is invalid. Use 'All', 'FSR' or \
                             'Decay'".format(spectrum_type))

    def dnde_charged_pion(self, egams, cme, spectrum_type='All'):
        if spectrum_type == 'All':
            return (self.dnde_charged_pion(egams, cme, 'FSR') +
                    self.dnde_charged_pion(egams, cme, 'Decay'))
        elif spectrum_type == 'FSR':
            return self.dnde_xx_to_s_to_pipig(egams, cme)
        elif spectrum_type == 'Decay':
            return 2. * charged_pion(egams, cme / 2.0)
        else:
            raise ValueError("Type {} is invalid. Use 'All', 'FSR' or \
                             'Decay'".format(spectrum_type))

    def dnde_ss(self, egams, Q, mode="total"):
        # Each scalar gets half the COM energy
        eng_s = Q / 2.

        ms = self.ms
        pws = self.partial_widths()
        pw_array = np.zeros(5, dtype=float)

        pw_array[0] = pws["e e"] / pws["total"]
        pw_array[1] = pws["mu mu"] / pws["total"]
        pw_array[2] = pws["pi0 pi0"] / pws["total"]
        pw_array[3] = pws["pi pi"] / pws["total"]
        pw_array[4] = pws["g g"] / pws["total"]

        if hasattr(egams, "__len__"):
            return 2. * dnde_decay_s(egams, eng_s, ms, pw_array, mode)
        return 2. * dnde_decay_s_pt(egams, eng_s, ms, pw_array, mode)

    def spectra(self, egams, cme, fsi=True):
        """
        Compute the total spectrum from two fermions annihilating through a
        scalar mediator to mesons and leptons.

        Parameters
        ----------
        cme : float
            Center of mass energy.
        egams : array-like, optional
            Gamma ray energies to evaluate the spectrum at.

        Returns
        -------
        specs : dictionary
            Dictionary of the spectra. The keys are 'total', 'mu mu', 'e e',
            'pi0 pi0', 'pi pi'
        """

        # Compute branching fractions
        bfs = self.annihilation_annihilation_branching_fractions(cme)

        # Only compute the spectrum if the channel's branching fraction is
        # nonzero
        def spec_helper(bf, specfn):
            if bf != 0:
                return bf * specfn(egams, cme)
            else:
                return np.zeros(egams.shape)

        # Pions
        npions = spec_helper(bfs['pi0 pi0'], self.dnde_neutral_pion)
        cpions = spec_helper(bfs['pi pi'], self.dnde_charged_pion)

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
        of the gamma ray energies to evaluate the spectra at and `cme`, the
        center of mass energy of the process.
        """
        return {'mu mu': lambda e_gams, cme: self.dnde_mumu(e_gams, cme),
                'e e': lambda e_gams, cme: self.dnde_ee(e_gams, cme),
                'pi0 pi0': lambda e_gams, cme:
                    self.dnde_neutral_pion(e_gams, cme),
                'pi pi': lambda e_gams, cme:
                    self.dnde_charged_pion(e_gams, cme),
                's s': lambda e_gams, cme: self.dnde_ss(e_gams, cme)}

    def gamma_ray_lines(self, cme):
        bf = self.annihilation_branching_fractions(cme)["g g"]

        return {"g g": {"energy": cme / 2.0, "bf": bf}}
