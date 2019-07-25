import numpy as np

from hazma.decay import muon

from hazma.parameters import muon_mass as mmu
from hazma.parameters import electron_mass as me
from hazma.parameters import charged_pion_mass as mpi
from hazma.parameters import neutral_pion_mass as mpi0

from hazma.gamma_ray import gamma_ray_decay

# Stuff needed to compute fsr from x xbar -> P -> pip pim pi0
# from ..gamma_ray_decay import gamma_ray_fsr
# from .pseudo_scalar_mediator_mat_elem_sqrd_rambo import msqrd_xx_to_p_to_pm0g


class PseudoScalarMediatorSpectra:
    # TODO: pp spectrum. Gonna need Logan to do this since it
    # requires cython...
    def dnde_pp(self, egams, Q, mode="total"):
        # eng_p = Q / 2.
        pass

    def dnde_ee(self, egams, cme, spectrum_type="all"):
        """Computes spectrum from DM annihilation into electrons.
        """
        if spectrum_type == "all":
            return self.dnde_ee(egams, cme, "fsr") + self.dnde_ee(egams, cme, "decay")
        elif spectrum_type == "fsr":
            return self.dnde_xx_to_p_to_ffg(egams, cme, me)
        elif spectrum_type == "decay":
            return np.array([0.0 for _ in range(len(egams))])
        else:
            raise ValueError(
                "Type {} is invalid. Use 'all', 'fsr' or \
                             'decay'".format(
                    spectrum_type
                )
            )

    def dnde_mumu(self, egams, cme, spectrum_type="all"):
        """Computes spectrum from DM annihilation into muons.
        """
        if spectrum_type == "all":
            return self.dnde_mumu(egams, cme, "fsr") + self.dnde_mumu(
                egams, cme, "decay"
            )
        elif spectrum_type == "fsr":
            return self.dnde_xx_to_p_to_ffg(egams, cme, mmu)
        elif spectrum_type == "decay":
            return 2.0 * muon(egams, cme / 2.0)
        else:
            raise ValueError(
                "Type {} is invalid. Use 'all', 'fsr' or \
                             'decay'".format(
                    spectrum_type
                )
            )

    def dnde_pi0pipi(self, egams, cme, spectrum_type="all"):
        """Computes spectrum from DM annihilation into a neutral pion and two
        charged pions.

        Notes
        -----
        This function uses RAMBO to "convolve" the pions' spectra with the
        matrix element over the pi0 pi pi phase space.
        """
        if cme < 2.0 * mpi + mpi0:
            return np.array([0.0 for _ in range(len(egams))])

        if spectrum_type == "all":
            return self.dnde_pi0pipi(egams, cme, "fsr") + self.dnde_pi0pipi(
                egams, cme, "decay"
            )
        elif spectrum_type == "fsr":
            # Define the tree level and radiative matrix element squared for
            # RAMBO. These need to be of the form double(*func)(np.ndarray)
            # where
            # the np.ndarray is a list of 4-momenta. Note msqrd_xx_to_p_to_pm0
            # takes params as the second argument. The first and second FS
            # particles must be the charged pions and the third a neutral pion.

            # NOTE: I am removing this because it takes too long and need to
            # be extrapolated and evaluated at the correct egams

            """
            def msqrd_tree(momenta):
                return msqrd_xx_to_p_to_pm0(momenta)

            def msqrd_rad(momenta):
                return msqrd_xx_to_p_to_pm0g(momenta)

            isp_masses = np.array([params.mx, params.mx])
            fsp_masses = np.array([mpi, mpi, mpi0, 0.0])

            return gamma_ray_fsr(isp_masses, fsp_masses, cme,
                                   num_ps_pts=50000, num_bins=150,
                                   mat_elem_sqrd_tree=msqrd_tree,
                                   mat_elem_sqrd_rad=msqrd_rad)
            """

            return np.array([0.0 for _ in range(len(egams))])
        elif spectrum_type == "decay":
            # Define the matrix element squared for RAMBO. This needs to be
            # of the form double(*func)(np.ndarray) where the np.ndarray is
            # a list of 4-momenta. Note msqrd_xx_to_p_to_pm0 takes params as
            # the second argument. The first and second FS particles must be
            # the charged pions and the third a neutral pion.

            return gamma_ray_decay(
                ["charged_pion", "charged_pion", "neutral_pion"],
                cme,
                egams,
                num_ps_pts=1000,
                mat_elem_sqrd=self.msqrd_xx_to_p_to_pm0,
            )
        else:
            raise ValueError(
                "Type {} is invalid. Use 'all', 'fsr' or \
                             'decay'".format(
                    spectrum_type
                )
            )

    def dnde_pi0pi0pi0(self, egams, cme, spectrum_type="all"):
        """Return the gamma ray spectrum for dark matter annihilations into
        three neutral pions.

        Notes
        -----
        This function uses RAMBO to "convolve" the pions' spectra with the
        matrix element over the pi0 pi0 pi0 phase space.
        """
        if cme < 3.0 * mpi0:
            return np.array([0.0 for _ in range(len(egams))])

        if spectrum_type == "all":
            return self.dnde_pi0pi0pi0(egams, cme, "decay")
        elif spectrum_type == "fsr":
            return np.array([0.0 for _ in range(len(egams))])
        elif spectrum_type == "decay":
            # Define the matrix element squared for RAMBO. This needs to be
            # of the form double(*func)(np.ndarray) where the np.ndarray is
            # a list of 4-momenta. Note msqrd_xx_to_p_to_000 takes params as
            # the second argument.
            def msqrd_tree(momenta):
                return self.msqrd_xx_to_p_to_000(momenta)

            return gamma_ray_decay(
                ["neutral_pion", "neutral_pion", "neutral_pion"],
                cme,
                egams,
                num_ps_pts=1000,
                mat_elem_sqrd=msqrd_tree,
            )
        else:
            raise ValueError(
                "Type {} is invalid. Use 'all', 'fsr' or \
                             'decay'".format(
                    spectrum_type
                )
            )

    def spectrum_funcs(self):
        """
        Returns a dictionary of all the avaiable spectrum functions for
        a pair of initial state fermions with mass `mx` annihilating into
        each available final state.

        Each argument of the spectrum functions in `eng_gams`, an array
        of the gamma ray energies to evaluate the spectra at and `cme`, the
        center of mass energy of the process.
        """
        return {
            "mu mu": self.dnde_mumu,
            "e e": self.dnde_ee,
            "pi0 pi pi": self.dnde_pi0pipi,
            "pi0 pi0 pi0": self.dnde_pi0pi0pi0,
            "p p": self.dnde_pp,
        }

    def gamma_ray_lines(self, cme):
        bf = self.annihilation_branching_fractions(cme)["g g"]

        return {"g g": {"energy": cme / 2.0, "bf": bf}}
