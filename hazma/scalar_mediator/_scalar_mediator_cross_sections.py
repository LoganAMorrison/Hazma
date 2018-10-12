from cmath import sqrt, pi, atanh

from hazma.parameters import vh, b0, alpha_em, sv_inv_MeV_to_cm3_per_s
from hazma.parameters import charged_pion_mass as mpi
from hazma.parameters import neutral_pion_mass as mpi0
from hazma.parameters import up_quark_mass as muq
from hazma.parameters import down_quark_mass as mdq
from hazma.parameters import muon_mass as mmu
from hazma.parameters import electron_mass as me

from scipy.integrate import quad


class ScalarMediatorCrossSection:
    def sigma_xx_to_s_to_ff(self, Q, f):
        """Returns the spin-averaged, cross section for a pair of fermions,
        *x*, annihilating into a pair of fermions, *f* through a
        scalar mediator in the s-channel.

        Parameters
        ----------
        Q : float
            Center of mass energy.

        Returns
        -------
        sigma : float
            Cross section for x + x -> s* -> f + f.
        """
        mx = self.mx

        if f == 'e':
            mf = me
        elif f == 'mu':
            mf = mmu

        if Q > 2. * mf and Q >= 2. * mx:
            ms = self.ms
            gsff = self.gsff
            gsxx = self.gsxx
            width_s = self.width_s

            ret_val = (gsff**2 * gsxx**2 * mf**2 * (-2 * mx + Q) *
                       (2 * mx + Q) * (-4 * mf**2 + Q**2)**1.5) / \
                (16. * pi * Q**2 * sqrt(-4 * mx**2 + Q**2) * vh**2 *
                 (ms**4 - 2 * ms**2 * Q**2 + Q**4 + ms**2 * width_s**2))

            assert ret_val.imag == 0
            assert ret_val.real >= 0

            return ret_val.real * sv_inv_MeV_to_cm3_per_s
        else:
            return 0.

    def sigma_xx_to_s_to_gg(self, Q):
        """Returns the spin-averaged, cross section for a pair of fermions,
        *x*, annihilating into a pair of photons through a
        scalar mediator in the s-channel.

        Parameters
        ----------
        Q : float
            Center of mass energy.

        Returns
        -------
        sigma : float
            Cross section for x + x -> s* -> g + g.
        """
        mx = self.mx

        if Q >= 2. * mx:
            gsFF = self.gsFF
            gsxx = self.gsxx
            ms = self.ms
            widths = self.width_s
            rx = mx / Q
            Lam = self.lam

            ret_val = ((alpha_em**2 * gsFF**2 * gsxx**2 * Q**4 *
                        sqrt(1 - 4 * rx**2)) /
                       (64. * Lam**2 * pi**3 *
                        (ms**4 + Q**4 + ms**2 *
                         (-2 * Q**2 + widths**2))))

            assert ret_val.imag == 0
            assert ret_val.real >= 0

            return ret_val.real * sv_inv_MeV_to_cm3_per_s
        else:
            return 0.0

    def sigma_xx_to_s_to_pi0pi0(self, Q):
        """Returns the spin-averaged, cross section for a pair of fermions,
        *x*, annihilating into a pair of neutral pion through a
        scalar mediator in the s-channel.

        Parameters
        ----------
        Q : float
            Center of mass energy.

        Returns
        -------
        sigma : float
            Cross section for x + x -> s* -> pi0 + pi0.
        """
        mx = self.mx

        if Q > 2. * mpi0 and Q >= 2. * mx:
            gsxx = self.gsxx
            gsff = self.gsff
            gsGG = self.gsGG
            ms = self.ms
            vs = self.vs
            widths = self.width_s
            Lam = self.lam
            rpi0 = mpi0 / Q
            rx = mx / Q

            ret_val = ((gsxx**2 *
                        sqrt((-1 + 4 * rpi0**2) *
                             (-1 + 4 * rx**2)) *
                        (162 * gsGG * Lam**3 * Q**2 *
                         (-1 + 2 * rpi0**2) * vh**2 +
                         b0 * (mdq + muq) *
                         (9 * Lam + 4 * gsGG * vs) *
                         (27 * gsff**2 * Lam**2 * vs *
                          (3 * Lam + 4 * gsGG * vs) -
                          2 * gsGG * vh**2 *
                          (27 * Lam**2 -
                           30 * gsGG * Lam * vs + 8 * gsGG**2 * vs**2) +
                          gsff *
                          (-81 * Lam**3 * vh +
                           48 * gsGG**2 * Lam * vh * vs**2)))**2) /
                       (209952. * Lam**6 * pi * vh**4 *
                        (9 * Lam + 4 * gsGG * vs)**2 *
                        (ms**4 + Q**4 + ms**2 *
                         (-2 * Q**2 + widths**2))))

            assert ret_val.imag == 0
            assert ret_val.real >= 0

            return ret_val.real * sv_inv_MeV_to_cm3_per_s
        else:
            return 0.

    def sigma_xx_to_s_to_pipi(self, Q):
        """Returns the spin-averaged, cross section for a pair of fermions,
        *x*, annihilating into a pair of charged pions through a
        scalar mediator in the s-channel.

        Parameters
        ----------
        Q : float
            Center of mass energy.

        Returns
        -------
        sigma : float
            Cross section for x + x -> s* -> pi + pi.
        """
        mx = self.mx

        if Q > 2. * mpi and Q >= 2. * mx:
            gsxx = self.gsxx
            gsff = self.gsff
            gsGG = self.gsGG
            ms = self.ms
            vs = self.vs
            widths = self.width_s
            Lam = self.lam
            rpi = mpi / Q
            rx = mx / Q

            ret_val = ((gsxx**2 *
                        sqrt((-1 + 4 * rpi**2) *
                             (-1 + 4 * rx**2)) *
                        (162 * gsGG * Lam**3 * Q**2 *
                         (-1 + 2 * rpi**2) * vh**2 +
                         b0 * (mdq + muq) *
                         (9 * Lam + 4 * gsGG * vs) *
                         (27 * gsff**2 * Lam**2 * vs *
                          (3 * Lam + 4 * gsGG * vs) -
                          2 * gsGG * vh**2 *
                          (27 * Lam**2 -
                           30 * gsGG * Lam * vs + 8 * gsGG**2 * vs**2) +
                          gsff *
                          (-81 * Lam**3 * vh +
                           48 * gsGG**2 * Lam * vh * vs**2)))**2) /
                       (104976. * Lam**6 * pi * vh**4 *
                        (9 * Lam + 4 * gsGG * vs)**2 *
                        (ms**4 + Q**4 + ms**2 * (-2 * Q**2 + widths**2))))

            assert ret_val.imag == 0
            assert ret_val.real >= 0

            return ret_val.real * sv_inv_MeV_to_cm3_per_s
        else:
            return 0.

    def sigma_xx_to_ss(self, Q):
        ms = self.ms
        mx = self.mx

        if Q > 2. * ms and Q >= 2. * mx:
            gsxx = self.gsxx

            ret_val = ((-(gsxx**4 * sqrt(-4 * ms**2 + Q**2) *
                          sqrt(-4 * mx**2 + Q**2) *
                          (-2 / (4 * mx**2 - Q**2) - (ms**2 - 4 * mx**2)**2 /
                           ((4 * mx**2 - Q**2) *
                            (ms**4 - 4 * ms**2 * mx**2 + mx**2 * Q**2)) -
                           (2 * (6 * ms**4 - 32 * mx**4 +
                                 16 * mx**2 * Q**2 + Q**4 -
                                 4 * ms**2 * (4 * mx**2 + Q**2)) *
                            atanh((sqrt(-4 * ms**2 + Q**2) *
                                   sqrt(-4 * mx**2 + Q**2)) /
                                  (-2 * ms**2 + Q**2))) /
                           (sqrt(-4 * ms**2 + Q**2) * (-2 * ms**2 + Q**2) *
                            (-4 * mx**2 + Q**2)**1.5))) / (16. * pi * Q**2)))

            assert ret_val.imag == 0
            assert ret_val.real >= 0

            return ret_val.real * sv_inv_MeV_to_cm3_per_s
        else:
            return 0.

    def sigma_xx_to_s_to_xx(self, Q):
        """
        Returns the DM annihilation cross section into DM.
        """
        rs = self.ms / Q
        rx = self.mx / Q
        gsxx = self.gsxx
        rws = self.width_s / Q

        if Q > 2. * self.mx:
            def msqrd(z):
                return ((gsxx**4 *
                         (3 * (-1 + z)**2 +
                          256 * rx**8 * (-1 + z)**2 -
                          64 * rx**6 * (5 - 8 * z + 3 * z**2) +
                          32 * rx**4 * (5 - 6 * z + 3 * z**2) -
                          4 * rx**2 * (5 - 12 * z + 7 * z**2) +
                          rs**4 *
                          (3 + z**2 - 8 * rx**2 * (1 + z**2) +
                           16 * rx**4 * (3 + z**2)) +
                          rs**2 *
                          (3 - 3 * z**2 - 64 * rx**6 *
                           (3 - 4 * z + z**2) -
                           16 * rx**4 * (-9 + 12 * z + z**2) +
                           4 * rx**2 * (-17 + 8 * z + 5 * z**2) +
                           rws**2 * (3 + z**2 - 8 * rx**2 * (1 + z**2) +
                                     16 * rx**4 * (3 + z**2))))) /
                        ((1 + rs**4 + rs**2 * (-2 + rws**2)) *
                         (4 * rs**4 + 4 * rs**2 *
                          (rws**2 + (-1 + 4 * rx**2) * (-1 + z)) +
                          (1 - 4 * rx**2)**2 * (-1 + z)**2)))

            ret_val = (quad(msqrd, -1, 1)[0] / (32. * pi * Q))

            assert ret_val.imag == 0.
            assert ret_val.real >= 0.

            return ret_val.real * sv_inv_MeV_to_cm3_per_s
        else:
            return 0.

    def annihilation_cross_sections(self, Q):
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
        muon_contr = self.sigma_xx_to_s_to_ff(Q, 'mu')
        electron_contr = self.sigma_xx_to_s_to_ff(Q, 'e')
        photon_contr = self.sigma_xx_to_s_to_gg(Q)
        NPion_contr = self.sigma_xx_to_s_to_pi0pi0(Q)
        CPion_contr = self.sigma_xx_to_s_to_pipi(Q)
        ss_contr = self.sigma_xx_to_ss(Q)

        total = (muon_contr + electron_contr + NPion_contr + CPion_contr +
                 photon_contr + ss_contr)

        cross_secs = {'mu mu': muon_contr,
                      'e e': electron_contr,
                      'g g': photon_contr,
                      'pi0 pi0': NPion_contr,
                      'pi pi': CPion_contr,
                      's s': ss_contr,
                      'total': total}

        return cross_secs

    def annihilation_branching_fractions(self, Q):
        """
        Compute the branching fractions for two fermions annihilating
        through a scalar mediator to mesons and leptons.

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
        CSs = self.annihilation_cross_sections(Q)

        if CSs['total'] == 0.0:
            return {'mu mu': 0.0,
                    'e e': 0.0,
                    'g g': 0.0,
                    'pi0 pi0': 0.0,
                    'pi pi': 0.0,
                    's s': 0.0}
        else:
            return {'mu mu': CSs['mu mu'] / CSs['total'],
                    'e e': CSs['e e'] / CSs['total'],
                    'g g': CSs['g g'] / CSs['total'],
                    'pi0 pi0': CSs['pi0 pi0'] / CSs['total'],
                    'pi pi': CSs['pi pi'] / CSs['total'],
                    's s': CSs['s s'] / CSs['total']}
