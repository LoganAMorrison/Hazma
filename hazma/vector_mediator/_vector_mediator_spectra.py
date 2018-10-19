import numpy as np

from hazma.decay import muon
from hazma.decay import neutral_pion, charged_pion

from hazma.parameters import neutral_pion_mass as mpi0

from hazma.vector_mediator.vector_mediator_decay_spectrum \
    import dnde_decay_v, dnde_decay_v_pt


class VectorMediatorSpectra:
    def dnde_ee(self, egams, cme, spectrum_type='All'):
        fsr = np.vectorize(self.dnde_xx_to_v_to_ffg)

        if spectrum_type == 'All':
            return (self.dnde_ee(egams, cme, "FSR") +
                    self.dnde_ee(egams, cme, "Decay"))
        elif spectrum_type == 'FSR':
            return fsr(egams, cme, "e")
        elif spectrum_type == 'Decay':
            return np.array([0.0 for _ in range(len(egams))])
        else:
            raise ValueError("Type {} is invalid. Use 'All', 'FSR' or \
                             'Decay'".format(spectrum_type))

    def dnde_mumu(self, egams, cme, spectrum_type='All'):
        fsr = np.vectorize(self.dnde_xx_to_v_to_ffg)  # todo: this line
        decay = np.vectorize(muon)

        if spectrum_type == 'All':
            return (self.dnde_mumu(egams, cme, "FSR") +
                    self.dnde_mumu(egams, cme, "Decay"))
        elif spectrum_type == 'FSR':
            return fsr(egams, cme, "mu")
        elif spectrum_type == 'Decay':
            return 2. * decay(egams, cme / 2.0)
        else:
            raise ValueError("Type {} is invalid. Use 'All', 'FSR' or \
                             'Decay'".format(spectrum_type))

    def dnde_pi0g(self, egams, cme, spectrum_type="All"):
        if spectrum_type == 'All':
            return (self.dnde_pi0g(egams, cme, "FSR") +
                    self.dnde_pi0g(egams, cme, "Decay"))
        elif spectrum_type == 'FSR':
            return np.array([0.0 for _ in range(len(egams))])
        elif spectrum_type == 'Decay':
            # Neutral pion's energy
            e_pi0 = (cme**2 + mpi0**2) / (2. * cme)

            return neutral_pion(egams, e_pi0)
        else:
            raise ValueError("Type {} is invalid. Use 'All', 'FSR' or \
                             'Decay'".format(spectrum_type))

    def dnde_pipi(self, egams, cme, spectrum_type="All"):
        if spectrum_type == 'All':
            return (self.dnde_pipi(egams, cme, "FSR") +
                    self.dnde_pipi(egams, cme, "Decay"))
        elif spectrum_type == 'FSR':
            return self.dnde_xx_to_v_to_pipig(egams, cme)
        elif spectrum_type == 'Decay':
            return 2. * charged_pion(egams, cme / 2.0)
        else:
            raise ValueError("Type {} is invalid. Use 'All', 'FSR' or \
                             'Decay'".format(spectrum_type))

    def dnde_vv(self, egams, eng_v, mode="total"):
        mv = self.mv
        pws = self.partial_widths()
        pw_array = np.zeros(5, dtype=float)

        pw_array[0] = pws["e e"] / pws["total"]
        pw_array[1] = pws["mu mu"] / pws["total"]
        pw_array[2] = pws["pi0 g"] / pws["total"]
        pw_array[3] = pws["pi pi"] / pws["total"]

        if hasattr(egams, "__len__"):
            return 2. * dnde_decay_v(egams, eng_v, mv, pw_array, mode)
        return 2. * dnde_decay_v_pt(egams, eng_v, mv, pw_array, mode)

    def spectra(self, egams, cme):
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
            'pi0 pi0', 'pi pi', 'k k', 'k0 k0'.
        """

        # Compute branching fractions
        bfs = self.annihilation_branching_fractions(cme)

        # Only compute the spectrum if the channel's branching fraction is
        # nonzero
        def spec_helper(bf, specfn):
            if bf != 0:
                return bf * specfn(egams, cme)
            else:
                return np.zeros(egams.shape)

        # Leptons
        muons = spec_helper(bfs['mu mu'], self.dnde_mumu)
        electrons = spec_helper(bfs['e e'], self.dnde_ee)

        # Pions
        pi0g = spec_helper(bfs["pi0 g"], self.dnde_pi0g)
        pipi = spec_helper(bfs["pi pi"], self.dnde_pipi)

        # mediator
        mediator = spec_helper(bfs['v v'], self.dnde_vv)

        # Compute total spectrum
        total = muons + electrons + pi0g + pipi + mediator

        # Define dictionary for spectra
        specs = {'total': total,
                 'mu mu': muons,
                 'e e': electrons,
                 "pi0 g": pi0g,
                 "pi pi": pipi,
                 "v v": mediator}

        return specs

    def spectrum_functions(self):
        """
        Returns a dictionary of all the avaiable spectrum functions for
        a pair of initial state fermions with mass `mx` annihilating into
        each available final state.

        Each argument of the spectrum functions in `eng_gams`, an array
        of the gamma ray energies to evaluate the spectra at and `cme`, the
        center of mass energy of the process.

        Note
        ----
        This does not return a function for computing the spectrum for the pi0
        pi pi final state since it always contributes orders of magnitude less
        than the pi pi and pi0 g final states.
        """
        return {'mu mu': lambda egams, cme: self.dnde_mumu(egams, cme),
                'e e': lambda egams, cme: self.dnde_ee(egams, cme),
                'pi pi': lambda egams, cme: self.dnde_pipi(egams, cme),
                'pi0 g': lambda egams, cme: self.dnde_pi0g(egams, cme)}

    def gamma_ray_lines(self, cme):
        bf = self.annihilation_branching_fractions(cme)["pi0 g"]

        return {"pi0 g": {"energy": (cme**2 - mpi0**2) / (2. * cme), "bf": bf}}
