from cmath import sqrt, pi, atanh
from hazma.parameters import charged_pion_mass as mpi
from hazma.parameters import neutral_pion_mass as mpi0
from hazma.parameters import muon_mass as mmu
from hazma.parameters import electron_mass as me
from hazma.parameters import fpi, qe, sv_inv_MeV_to_cm3_per_s
from scipy.integrate import quad


class VectorMediatorCrossSections:
    def sigma_xx_to_v_to_ff(self, Q, f):
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
            Cross section for xbar + x -> v -> fbar + f.
        """
        if f == "e":
            mf = me
            gvll = self.gvee
        elif f == "mu":
            mf = mmu
            gvll = self.gvmumu

        mx = self.mx

        if Q >= 2. * mf and Q >= 2. * mx:
            gvxx = self.gvxx
            mv = self.mv
            width_v = self.width_v

            ret_val = (gvll**2 * gvxx**2 * sqrt(Q**2 - 4. * mf**2) *
                       (Q**2 + 2. * mf**2) *
                       (Q**2 + 2 * mx**2)) / \
                (12. * Q**2 * sqrt(Q**2 - 4. * mx**2) * pi *
                 (Q**4 - 2. * Q**2 * mv**2 + mv**4 + mv**2 * width_v**2))

            assert ret_val.imag == 0
            assert ret_val.real >= 0

            return ret_val.real * sv_inv_MeV_to_cm3_per_s
        else:
            return 0.

    def sigma_xx_to_v_to_pipi(self, Q):
        """
        Returns the cross section for xbar x to pi+ pi-.

        Parameters
        ----------
        Q : float
            Center of mass energy.
        self : object
            Class containing the vector mediator parameters.

        Returns
        -------
        cross_section : float
            Cross section for xbar + x -> v -> f + f.
        """
        mx = self.mx

        if Q >= 2. * mpi and Q >= 2. * mx:
            gvuu = self.gvuu
            gvdd = self.gvdd
            gvxx = self.gvxx
            mv = self.mv
            width_v = self.width_v

            ret_val = (((gvdd - gvuu)**2 * gvxx**2 *
                        (-4. * mpi**2 + Q**2)**1.5 *
                        (2. * mx**2 + Q**2)) /
                       (48. * pi * Q**2 * sqrt(-4. * mx**2 + Q**2) *
                        (mv**4 - 2. * mv**2 * Q**2 + Q**4 + mv**2 *
                         width_v**2)))

            assert ret_val.imag == 0
            assert ret_val.real >= 0

            return ret_val.real * sv_inv_MeV_to_cm3_per_s
        else:
            return 0.

    def sigma_xx_to_v_to_pi0g(self, Q):
        """
        Returns the cross section for xbar x to pi0 g.

        Parameters
        ----------
        Q : float
            Center of mass energy.
        self : object
            Class containing the vector mediator parameters.

        Returns
        -------
        cross_section : float
            Cross section for xbar + x -> v -> pi0 g
        """
        mx = self.mx

        if Q >= mpi0 and Q >= 2. * mx:
            gvuu = self.gvuu
            gvdd = self.gvdd
            gvxx = self.gvxx
            mv = self.mv
            width_v = self.width_v

            ret_val = 3. * ((gvdd + 2. * gvuu)**2. * gvxx**2 *
                            (-mpi0**2 + Q**2)**3 *
                            (2. * mx**2 + Q**2) * qe**2) / \
                (13824. * fpi**2 * pi**5 * Q**3 * sqrt(-4. * mx**2 + Q**2) *
                 (mv**4 - 2. * mv**2 * Q**2 + Q**4 + mv**2 * width_v**2))

            assert ret_val.imag == 0
            assert ret_val.real >= 0

            return ret_val.real * sv_inv_MeV_to_cm3_per_s
        else:
            return 0.

#    def sigma_xx_to_v_to_pi0v(self, Q):
#        """
#        Returns the cross section for xbar x to pi0 v.
#
#        Parameters
#        ----------
#        Q : float
#            Center of mass energy.
#        self : object
#            Class containing the vector mediator parameters.
#
#        Returns
#        -------
#        cross_section : float
#            Cross section for xbar + x -> v -> pi0 v
#        """
#        mx = self.mx
#        mv = self.mv
#
#        if Q >= mpi0 + mv and Q >= 2. * mx:
#            gvuu = self.gvuu
#            gvdd = self.gvdd
#            gvxx = self.gvxx
#            width_v = self.width_v
#
#            ret_val = ((gvdd - gvuu)**2 * (gvdd + gvuu)**2 * gvxx**2 *
#                       (2*mx**2 + Q**2) *
#                       ((mpi0**2 - mv**2)**2 -
#                        2*(mpi0**2 + mv**2)*Q**2 + Q**4)**1.5) / \
#                (1536.*fpi**2*pi**5*Q**3*sqrt(-4*mx**2 + Q**2) *
#                 (mv**4 - 2*mv**2*Q**2 + Q**4 + mv**2*width_v**2))
#
#            assert ret_val.imag == 0
#            assert ret_val.real >= 0
#
#            return ret_val.real * sv_inv_MeV_to_cm3_per_s
#        else:
#            return 0.

    def dsigma_ds_xx_to_v_to_pi0pipi(self, s, Q):
        mx = self.mx

        if (Q > 2. * mpi + mpi0 and Q > 2. * mx and s > 4. * mpi**2 and
                s < (Q - mpi0)**2):
            gvuu = self.gvuu
            gvdd = self.gvdd
            gvxx = self.gvxx
            mv = self.mv
            width_v = self.width_v

            ret_val = (3. * ((gvdd + gvuu)**2 * gvxx**2 *
                             sqrt(s * (-4. * mpi**2 + s)) *
                             sqrt(Q**4 + (mpi0**2 - s)**2 -
                                  2. * Q**2 * (mpi0**2 + s)) *
                             (-24. * mpi**6 * s + mpi**4 *
                              (-2. * mpi0**4 + 28. * mpi0**2 * s +
                               22. * s**2) +
                              2. * mpi**2 * (mpi0**6 - 4. * s**3) +
                              s * (-2. * mpi0**6 - 4. * mpi0**4 * s -
                                   mpi0**2 * s**2 + s**3) +
                              Q**4 * (-2. * mpi**4 +
                                      2. * mpi**2 * (mpi0**2 - s) +
                                      s * (-2. * mpi0**2 + s)) +
                              Q**2 * (4. * mpi**4 * (mpi0**2 + s) +
                                      s * (4. * mpi0**4 + 5. * mpi0**2 * s -
                                           2. * s**2) -
                                      4. * mpi**2 *
                                      (mpi0**4 +
                                       3. * mpi0**2 * s - s**2)))) /
                       (294912. * fpi**6 * pi**7 * sqrt(Q**2) *
                        sqrt(-4. * mx**2 + Q**2) * s**2 *
                        (mv**4 - 2. * mv**2 * Q**2 + Q**4 +
                         mv**2 * width_v**2)))

            assert ret_val.imag == 0.

            return ret_val.real * sv_inv_MeV_to_cm3_per_s
        else:
            return 0.

    def sigma_xx_to_v_to_pi0pipi(self, Q):
        if Q > 2. * mpi + mpi0 and Q > 2. * self.mx:
            s_min = 4. * mpi**2
            s_max = (Q - mpi0)**2

            ret_val = quad(self.dsigma_ds_xx_to_v_to_pi0pipi,
                           s_min, s_max, args=(Q))[0]

            assert ret_val.imag == 0
            assert ret_val.real >= 0

            return ret_val.real * sv_inv_MeV_to_cm3_per_s
        else:
            return 0.

    def sigma_xx_to_vv(self, Q):
        mx = self.mx
        mv = self.mv

        if Q >= 2. * mv and Q >= 2. * mx:
            gvxx = self.gvxx

            ret_val = ((gvxx**4 * sqrt(-4 * mv**2 + Q**2) *
                        (-2 - (2 * (mv**2 + 2 * mx**2)**2) /
                         (mv**4 - 4 * mv**2 * mx**2 + mx**2 * Q**2) +
                         (4 * (4 * mv**4 - 8 * mv**2 * mx**2 -
                               8 * mx**4 + 4 * mx**2 * Q**2 + Q**4) *
                            atanh((sqrt(4 * mv**2 - Q**2) *
                                   sqrt(4 * mx**2 - Q**2)) /
                                  (2 * mv**2 - Q**2))) /
                         ((2 * mv**2 - Q**2) * sqrt(4 * mv**2 - Q**2) *
                          sqrt(4 * mx**2 - Q**2)))) /
                       (8. * pi * Q**2 * sqrt(-4 * mx**2 + Q**2)))

            assert ret_val.imag == 0
            assert ret_val.real >= 0

            return ret_val.real * sv_inv_MeV_to_cm3_per_s
        else:
            return 0.0

    def sigma_xx_to_v_to_xx(self, Q):
        """
        Returns the DM annihilation cross section into DM.
        """
        rv = self.mv / Q
        rx = self.mx / Q
        gvxx = self.gvxx
        rwv = self.width_v / Q

        if Q > 2. * self.mx:
            def msqrd(z):
                return ((gvxx**4 *
                         (9 - 56 * rx**2 + 2 *
                          (9 * rv**2 * (-1 + rv**2 + rwv**2) +
                           4 * rv**2 *
                           (11 + 2 * rv**2 + 2 * rwv**2) * rx**2 +
                           8 * (7 + 6 * rv**2 *
                                (-4 + rv**2 + rwv**2)) * rx**4 +
                           64 * rx**6) -
                          2 * (-1 + 4 * rx**2) *
                          (3 * rv**2 * (-3 + 2 * rv**2 + 2 * rwv**2) +
                           4 *
                           (3 + 6 * rv**4 + 2 * rv**2 *
                            (-7 + 3 * rwv**2)) * rx**2 -
                           32 * (-2 + rv**2) * rx**4) * z +
                          2 * (3 * rv**2 - 4 * rx**2) *
                          (-1 + 4 * rx**2)**3 * z**3 +
                          (1 - 4 * rx**2)**4 * z**4 +
                          2 * (3 + 5 * rv**4 + 12 * rx**2 +
                               8 * rx**4 + rv**2 *
                               (-3 + 5 * rwv**2 - 20 * rx**2)) *
                          (z - 4 * rx**2 * z)**2)) /
                        ((1 + rv**4 + rv**2 * (-2 + rwv**2)) *
                         (4 * rv**4 + 4 * rv**2 *
                          (rwv**2 + (-1 + 4 * rx**2) * (-1 + z)) +
                            (1 - 4 * rx**2)**2 * (-1 + z)**2)))

            ret_val = (quad(msqrd, -1, 1)[0] / (32. * pi * Q))

            assert ret_val.imag == 0.
            assert ret_val.real >= 0.

            return ret_val.real * sv_inv_MeV_to_cm3_per_s
        else:
            return 0.

    def annihilation_cross_sections(self, Q):
        """
        Compute the total cross section for two fermions annihilating
        through a vector mediator to mesons and leptons.

        Parameters
        ----------
        cme : float
            Center of mass energy.

        Returns
        -------
        cs : float
            Total cross section.
        """
        muon_contr = self.sigma_xx_to_v_to_ff(Q, "mu")
        electron_contr = self.sigma_xx_to_v_to_ff(Q, "e")
        pipi_contr = self.sigma_xx_to_v_to_pipi(Q)
        pi0g_contr = self.sigma_xx_to_v_to_pi0g(Q)
        # pi0pipi_contr = sigma_xx_to_v_to_pi0pipi(Q)
        vv_contr = self.sigma_xx_to_vv(Q)

        total = (muon_contr + electron_contr + pipi_contr + pi0g_contr +
                 vv_contr)
        # pi0pipi_contr

        cross_secs = {'mu mu': muon_contr,
                      'e e': electron_contr,
                      'pi pi': pipi_contr,
                      'pi0 g': pi0g_contr,
                      # 'pi0 pi pi': pi0pipi_contr,
                      'v v': vv_contr,
                      'total': total}

        return cross_secs

    def annihilation_branching_fractions(self, Q):
        """
        Compute the branching fractions for two fermions annihilating
        through a vector mediator to mesons and leptons.

        Parameters
        ----------
        Q : float
            Center of mass energy.

        Returns
        -------
        bfs : dictionary
            Dictionary of the branching fractions. The keys are 'total',
            'mu mu', 'e e', 'pi0 g', 'pi pi'.
        """
        CSs = self.annihilation_cross_sections(Q)

        if CSs['total'] == 0.0:
            return {'mu mu': 0.0,
                    'e e': 0.0,
                    'pi pi': 0.0,
                    'pi0 g': 0.0,
                    # "pi0 pi pi": 0.0,
                    'v v': 0.0}
        else:
            return {'mu mu': CSs['mu mu'] / CSs['total'],
                    'e e': CSs['e e'] / CSs['total'],
                    'pi pi': CSs['pi pi'] / CSs['total'],
                    'pi0 g': CSs['pi0 g'] / CSs['total'],
                    # "pi0 pi pi": CSs["pi0 pi pi"] / CSs["total"],
                    'v v': CSs['v v'] / CSs['total']}
