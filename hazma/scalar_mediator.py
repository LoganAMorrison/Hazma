import numpy as np

from .decay import muon
from .decay import neutral_pion, charged_pion
from .decay import short_kaon, long_kaon, charged_kaon

from .fsr_helper_functions.scalar_mediator_fsr import dnde_xx_to_s_to_ffg
from .fsr_helper_functions.scalar_mediator_fsr import dnde_xx_to_s_to_pipig

from .cross_sections import scalar_mediator as SMCS

from .parameters import muon_mass as mmu
from .parameters import electron_mass as me
from .parameters import up_quark_mass as muq
from .parameters import down_quark_mass as mdq
from .parameters import strange_quark_mass as msq
from .parameters import fpi, b0, vh

trM = muq + mdq + msq


class ScalarMediator:
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
        self.mx = mx
        self.ms = ms
        self.gsxx = gsxx
        self.gsff = gsff
        self.gsGG = gsGG
        self.gsFF = gsFF
        self.vs = self.__compute_vs()

        self.CSEtaEta = np.vectorize(lambda cme: SMCS.sigma_xx_to_s_to_etaeta(
            cme, self.mx, 0., self.ms, self.gsxx, self.gsff, self.gsGG,
            self.gsFF, self.vs))

        self.CSff = np.vectorize(lambda cme, mf: SMCS.sigma_xx_to_s_to_ff(
            cme, self.mx, mf, self.ms, self.gsxx, self.gsff, self.gsGG,
            self.gsFF, self.vs))

        self.CSgg = np.vectorize(lambda cme: SMCS.sigma_xx_to_s_to_gg(
            cme, self.mx, 0., self.ms, self.gsxx, self.gsff, self.gsGG,
            self.gsFF, self.vs))

        self.CSk0k0 = np.vectorize(lambda cme: SMCS.sigma_xx_to_s_to_k0k0(
            cme, self.mx, 0., self.ms, self.gsxx, self.gsff, self.gsGG,
            self.gsFF, self.vs))

        self.CSkk = np.vectorize(lambda cme: SMCS.sigma_xx_to_s_to_kk(
            cme, self.mx, 0., self.ms, self.gsxx, self.gsff, self.gsGG,
            self.gsFF, self.vs))

        self.CSpi0pi0 = np.vectorize(lambda cme: SMCS.sigma_xx_to_s_to_pi0pi0(
            cme, self.mx, 0., self.ms, self.gsxx, self.gsff, self.gsGG,
            self.gsFF, self.vs))

        self.CSpipi = np.vectorize(lambda cme: SMCS.sigma_xx_to_s_to_pipi(
            cme, self.mx, 0., self.ms, self.gsxx, self.gsff, self.gsGG,
            self.gsFF, self.vs))

        self._2_body_final_states = ['eta eta', 'mu mu', 'e e', 'g g', 'K0 K0',
                                     'k k', 'pi0 pi0', 'pi pi']

    def list_2_body_final_states(self):
        """
        Return a list of the available final states.

        Returns
        -------
        fs : array-like
            Array of the available final states.
        """
        return self._2_body_final_states

    def total_cross_section(self, cme):
        """
        Compute the total cross section for two fermions annihilating through a
        scalar mediator to mesons and leptons.

        Parameters
        ----------
        cme : float
            Center of mass energy.

        Returns
        -------
        cs : float
            Total cross section.
        """
        eta_contr = self.CSEtaEta(cme)
        muon_contr = self.CSff(cme, mmu)
        electron_contr = self.CSff(cme, me)
        photon_contr = self.CSgg(cme)
        NKaon_contr = self.CSk0k0(cme)
        CKaon_contr = self.CSkk(cme)
        NPion_contr = self.CSpi0pi0(cme)
        CPion_contr = self.CSpipi(cme)

        return eta_contr + muon_contr + electron_contr + NKaon_contr + \
            CKaon_contr + NPion_contr + CPion_contr + photon_contr

    def branching_fractions(self, cme):
        """
        Compute the branching fractions for two fermions annihilating through a
        scalar mediator to mesons and leptons.

        Parameters
        ----------
        cme : float
            Center of mass energy.

        Returns
        -------
        bfs : dictionary
            Dictionary of the branching fractions. The keys are 'total',
            'mu mu', 'e e', 'pi0 pi0', 'pi pi', 'k k', 'k0 k0'.
        """
        CStot = self.total_cross_section(cme)

        bfs = {'eta eta': self.CSEtaEta(cme) / CStot,
               'mu mu': self.CSff(cme, mmu) / CStot,
               'e e': self.CSff(cme, me) / CStot,
               'g g': self.CSgg(cme) / CStot,
               'k0 k0': self.CSk0k0(cme) / CStot,
               'k k': self.CSkk(cme) / CStot,
               'pi0 pi0': self.CSpi0pi0(cme) / CStot,
               'pi pi': self.CSpipi(cme) / CStot}

        return bfs

    def total_spectrum(self, cme, egams=None):
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
        # Energies to compute spectrum
        if egams is None:
            egams = np.logspace(0.0, np.log10(cme), num=150)

        # Define analytic spectra
        dnde_ff = np.vectorize(dnde_xx_to_s_to_ffg)
        dnde_pipi = np.vectorize(dnde_xx_to_s_to_pipig)

        # Compute branching fractions
        bfs = self.branching_fractions(cme)

        # Decay Spectra
        nkaon_decay = 0.5 * long_kaon(egams, cme / 2.0) + \
            0.5 * short_kaon(egams, cme / 2.0)
        ckaon_decay = charged_kaon(egams, cme / 2.0)
        npion_decay = neutral_pion(egams, cme / 2.0)
        cpion_decay = charged_pion(egams, cme / 2.0)
        muon_decay = muon(egams, cme / 2.0)

        # FSR spectra
        muon_fsr = dnde_ff(egams, cme, mmu)
        elec_fsr = dnde_ff(egams, cme, me)

        # Pions
        npions = 2.0 * bfs['pi0 pi0'] * npion_decay
        cpions = bfs['pi pi'] * (dnde_pipi(egams, cme, self.mx, self.ms,
                                           self.gsxx, self.gsff,
                                           self.gsGG) + 2.0 * cpion_decay[:])

        # Leptons
        muons = bfs['mu mu'] * (muon_decay[:] + muon_fsr[:])
        electrons = bfs['e e'] * elec_fsr[:]

        # Kaons
        nkaons = bfs['k0 k0'] * 2.0 * nkaon_decay
        ckaons = bfs['k k'] * 2.0 * ckaon_decay

        # Comput total spectrum
        total = muons + electrons + npions + cpions + nkaons + ckaons

        # Define dictionary for spectra
        specs = {'total': total, 'mu mu': muons, 'e e': electrons,
                 'pi0 pi0': npions, 'pi pi': cpions, 'k0 k0': nkaons,
                 'k k': ckaons}

        return specs

    # ###############################
    # Function for finding scalar vev
    # ###############################

    def __alpha(self, fT, BT, msT):
        """
        Returns coefficent of linear term in the scalar potential before adding
        scalar vev.
        """
        return (BT * fT**2 * (self.gsff + (2 * self.gsGG) / 3.) * trM) / vh

    def __beta(self, fT, BT, msT):
        """
        Returns curvature of the scalar potential.
        """
        return msT**2 - (16 * BT * fT**2 * self.gsff * self.gsGG * trM) / \
            (9. * vh**2) + (32 * BT * fT**2 *
                            self.gsGG**2 * trM) / (81. * vh**2)

    def __vs_roots(self):
        """
        Returns the two possible values of the scalar potential.
        """
        root1 = (-3 * self.ms * np.sqrt(trM) * vh +
                 np.sqrt(4 * b0 * fpi**2 * (3 * self.gsff + 2 * self.gsGG)**2 +
                         9 * self.ms**2 * trM * vh**2)) / \
            (2. * (3 * self.gsff + 2 * self.gsGG) * self.ms * np.sqrt(trM))
        root2 = (-3 * self.ms * np.sqrt(trM) * vh -
                 np.sqrt(4 * b0 * fpi**2 * (3 * self.gsff + 2 * self.gsGG)**2 +
                         9 * self.ms**2 * trM * vh**2)) / \
            (2. * (3 * self.gsff + 2 * self.gsGG) * self.ms * np.sqrt(trM))

        return root1, root2

    def __fpiT(self, vs):
        """
        Returns the unphysical value of fpi.
        """
        return fpi / np.sqrt(1.0 + 4. * self.gsGG * vs / 9. / vh)

    def __kappa(self, fpiT):
        return fpi**2 / fpiT**2 - 1.

    def __BT(self, kappa):
        """
        Returns the unphysical value of B.
        """
        return b0 * (1 + kappa) / (1 + 6. * kappa *
                                   (1. + 3. * self.gsff / 2. / self.gsGG))

    def __msT(self, fpiT, BT):
        """
        Returns the unphysical mass of the scalar mediator.
        """
        gamma = BT * fpiT * trM / vh
        return np.sqrt(self.ms**2 +
                       16. * gamma * self.gsff * self.gsGG / 9. / vh -
                       32. * gamma * self.gsGG**2 / 81. / vh)

    def __compute_vs(self):
        """
        Returns the value of the scalar vev.
        """
        vs_roots = self.__vs_roots()
        fpiTs = [self.__fpiT(vs) for vs in vs_roots]
        kappas = [self.__kappa(fpiT) for fpiT in fpiTs]
        BTs = [self.__BT(kappa) for kappa in kappas]
        msTs = [self.__msT(fpiT, BT) for (fpiT, BT) in zip(fpiTs, BTs)]
        alphas = [self.__alpha(fpiT, BT, msT)
                  for (fpiT, BT, msT) in zip(fpiTs, BTs, msTs)]
        betas = [self.__beta(fpiT, BT, msT)
                 for (fpiT, BT, msT) in zip(fpiTs, BTs, msTs)]

        potvals = [- alpha * vs + 0.5 * beta * vs **
                   2 for (alpha, beta, vs) in zip(alphas, betas, vs_roots)]

        if potvals[0] < potvals[1]:
            return vs_roots[0]
        else:
            return vs_roots[1]
