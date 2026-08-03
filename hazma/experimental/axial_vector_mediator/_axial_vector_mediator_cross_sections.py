# from cmath import sqrt, pi, log
from hazma.parameters import charged_pion_mass as mpi
from hazma.parameters import neutral_pion_mass as mpi0

# from ..parameters import fpi
# from ..parameters import qe
from hazma.parameters import muon_mass as mmu
from hazma.parameters import electron_mass as me
from scipy.integrate import quad


class AxialVectorMediatorCrossSections:
    def sigma_xx_to_a_to_ff(self, Q, f):
        """
        Returns the cross section for xbar x to fbar f.

        Parameters
        ----------
        Q : float
            Center of mass energy.
        f : float
            Name of final state fermion: "e" or "mu".
        self : object
            Class containing the vector mediator parameters.

        Returns
        -------
        cross_section : float
            Cross section for xbar + x -> a -> fbar + f.
        """
        if f == "e":
            mf = me
            # gall = self.gaee
        elif f == "mu":
            mf = mmu
            # gall = self.gamumu
        mx = self.mx
        if Q >= 2.0 * mf and Q >= 2.0 * mx:
            # gaxx = self.gaxx
            # ma = self.ma
            # width_a = self.width_a
            ret_val = 0.0
            assert ret_val.imag == 0
            assert ret_val.real >= 0
            return ret_val.real
        else:
            return 0.0

    def dsigma_ds_xx_to_a_to_pi0pipi(self, s, Q):
        mx = self.mx

        if (
            Q > 2.0 * mpi + mpi0
            and Q > 2.0 * mx
            and s > 4.0 * mpi**2
            and s < (Q - mpi0) ** 2
        ):
            # gauu = self.gauu
            # gadd = self.gadd
            # gaxx = self.gaxx
            # ma = self.ma
            # width_a = self.width_a

            ret_val = 0.0

            assert ret_val.imag == 0.0

            return ret_val.real
        else:
            return 0.0

    def sigma_xx_to_a_to_pi0pipi(self, Q):
        if Q > 2.0 * mpi + mpi0 and Q > 2.0 * self.mx:
            s_min = 4.0 * mpi**2
            s_max = (Q - mpi0) ** 2

            ret_val = quad(self.dsigma_ds_xx_to_a_to_pi0pipi, s_min, s_max, args=(Q))[0]

            assert ret_val.imag == 0
            assert ret_val.real >= 0

            return ret_val.real
        else:
            return 0.0

    def sigma_xx_to_aa(self, Q):
        mx = self.mx
        ma = self.ma

        if Q >= 2.0 * ma and Q >= 2.0 * mx:
            # gaxx = self.gaxx

            ret_val = 0.0

            assert ret_val.imag == 0
            assert ret_val.real >= 0

            return ret_val.real
        else:
            return 0.0

    def annihilation_cross_sections(self, Q):
        """
        Compute the total cross section for two fermions annihilating through a
        vector mediator to mesons and leptons.

        Parameters
        ----------
        cme : float
            Center of mass energy.

        Returns
        -------
        cs : float
            Total cross section.
        """
        muon_contr = self.sigma_xx_to_a_to_ff(Q, "mu")
        electron_contr = self.sigma_xx_to_a_to_ff(Q, "e")
        pi0pipi_contr = self.sigma_xx_to_a_to_pi0pipi(Q)
        aa_contr = self.sigma_xx_to_aa(Q)

        total = muon_contr + electron_contr + pi0pipi_contr + aa_contr
        # pi0pipi_contr

        cross_secs = {
            "mu mu": muon_contr,
            "e e": electron_contr,
            "pi0 pi pi": pi0pipi_contr,
            "a a": aa_contr,
            "total": total,
        }

        return cross_secs

    def annihilation_branching_fractions(self, Q):
        """
        Compute the branching fractions for two fermions annihilating through
        an axial vector mediator to mesons and leptons.

        Parameters
        ----------
        Q : float
            Center of mass energy.

        Returns
        -------
        bfs : dictionary
            Dictionary of the branching fractions. The keys are 'total',
            'mu mu', 'e e', 'pi0 pi pi', 'a a'.
        """
        CSs = self.cross_sections(Q)

        if CSs["total"] == 0.0:
            return {"mu mu": 0.0, "e e": 0.0, "pi0 pi pi": 0.0, "a a": 0.0}
        else:
            return {
                "mu mu": CSs["mu mu"] / CSs["total"],
                "e e": CSs["e e"] / CSs["total"],
                "pi0 pi pi": CSs["pi0 pi pi"] / CSs["total"],
                "a a": CSs["a a"] / CSs["total"],
            }
