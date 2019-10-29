from hazma.parameters import vh, b0, alpha_em
from hazma.parameters import charged_pion_mass as mpi
from hazma.parameters import neutral_pion_mass as mpi0
from hazma.parameters import up_quark_mass as muq
from hazma.parameters import down_quark_mass as mdq
from hazma.parameters import muon_mass as mmu
from hazma.parameters import electron_mass as me

from numpy.polynomial.legendre import leggauss
import warnings
import numpy as np


class ScalarMediatorCrossSection:
    def sigma_xx_to_s_to_ff(self, e_cm, f):
        """Returns the spin-averaged, cross section for dark matter
        annihilating into a pair of fermions, *f* through a scalar mediator in
        the s-channel.

        Parameters
        ----------
        e_cm : float or array-like
            Center of mass energy(ies).
        f: str
            String for the final state fermion: f = 'e' for electron and
            f = 'mu' for muon.

        Returns
        -------
        sigma : float or array-like
            Cross section for x + x -> s* -> f + f.
        """
        # Avoid the numpy warnings when e_cm = 0 and when e_cm = 2mx or when
        # e_cm < 2mf and e_cm < 2mx. We know the cross section is just zero for
        # these cases. NOTE: catching these causes about a ~20us slowdown.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', r'invalid value encountered in true_divide')
            warnings.filterwarnings(
                'ignore', r'divide by zero encountered in true_divide')
            warnings.filterwarnings(
                'ignore', r'invalid value encountered in multiply')
            warnings.filterwarnings(
                'ignore', r'invalid value encountered in sqrt')
            warnings.filterwarnings(
                'ignore', r'invalid value encountered in power')

            assert f == 'e' or f == 'mu'
            mf = me if f == 'e' else mmu

            e_cms = np.array(e_cm) if hasattr(e_cm, '__len__') else e_cm
            mask = (e_cms > 2.0 * mf) & (e_cms > 2.0 * self.mx)

            ret_val = mask * np.nan_to_num((
                self.gsff**2 * self.gsxx**2 * mf**2 *
                (-2 * self.mx + e_cms) * (2 * self.mx + e_cms) *
                (-4 * mf**2 + e_cms**2)**1.5) /
                (16.0 * np.pi * e_cms**2 *
                 np.sqrt(-4 * self.mx**2 + e_cms**2) *
                 vh**2 * (self.ms**4 - 2 * self.ms**2 * e_cms**2 + e_cms**4 +
                          self.ms**2 * self.width_s**2)))

        # need the .real for the case where the User passes a float.
        return ret_val.real

    def sigma_xx_to_s_to_gg(self, e_cm):
        """Returns the spin-averaged, cross section for dark matter
        annihilating into a pair of photons through a scalar mediator in the
        s-channel.

        Parameters
        ----------
        e_cm : float or array-like
            Center of mass energy(ies).

        Returns
        -------
        sigma : float or array-like
            Cross section for x + x -> s* -> g + g.
        """

        # see sigma_xx_to_s_to_ff for explaination of this context mangager
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', r'invalid value encountered in true_divide')
            warnings.filterwarnings(
                'ignore', r'divide by zero encountered in true_divide')
            warnings.filterwarnings(
                'ignore', r'invalid value encountered in multiply')
            warnings.filterwarnings(
                'ignore', r'invalid value encountered in sqrt')
            warnings.filterwarnings(
                'ignore', r'invalid value encountered in power')

            e_cms = np.array(e_cm) if hasattr(e_cm, '__len__') else e_cm
            mask = e_cms > 2.0 * self.mx

            rxs = self.mx / e_cms
            ret_val = mask * np.nan_to_num((
                alpha_em**2 * self.gsFF**2 * self.gsxx**2 * e_cms**4 *
                np.sqrt(1 - 4 * rxs**2)) /
                (64.0 * self.lam**2 * np.pi**3 *
                 (self.ms**4 + e_cms**4 + self.ms**2 *
                  (-2 * e_cms**2 + self.width_s**2))))

        return ret_val.real

    def sigma_xx_to_s_to_pi0pi0(self, e_cm):
        """Returns the spin-averaged, cross section for dark matter
        annihilating into a pair of neutral pion through a scalar mediator in
        the s-channel.

        Parameters
        ----------
        e_cm : float or array-like
            Center of mass energy(ies).

        Returns
        -------
        sigma : float or array-like
            Cross section for x + x -> s* -> pi0 + pi0.
        """
        # see sigma_xx_to_s_to_ff for explaination of this context mangager
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', r'invalid value encountered in true_divide')
            warnings.filterwarnings(
                'ignore', r'divide by zero encountered in true_divide')
            warnings.filterwarnings(
                'ignore', r'invalid value encountered in multiply')
            warnings.filterwarnings(
                'ignore', r'invalid value encountered in sqrt')
            warnings.filterwarnings(
                'ignore', r'invalid value encountered in power')

            e_cms = np.array(e_cm) if hasattr(e_cm, '__len__') else e_cm
            mask = (e_cms > 2.0 * mpi0) & (e_cms > 2.0 * self.mx)

            rpi0s = mpi0 / e_cms
            rxs = self.mx / e_cms

            ret_val = mask * np.nan_to_num((
                self.gsxx**2 *
                np.sqrt((-1 + 4 * rpi0s**2) * (-1 + 4 * rxs**2)) *
                (162 * self.gsGG * self.lam**3 * e_cms**2 *
                 (-1 + 2 * rpi0s**2) * vh**2 + b0 * (mdq + muq) *
                 (9 * self.lam + 4 * self.gsGG * self.vs) *
                 (27 * self.gsff**2 * self.lam**2 *
                  self.vs * (3 * self.lam + 4 * self.gsGG * self.vs) -
                  2 * self.gsGG * vh**2 *
                  (27 * self.lam**2 - 30 * self.gsGG * self.lam * self.vs +
                   8 * self.gsGG**2 * self.vs**2) + self.gsff *
                  (-81 * self.lam**3 * vh +
                   48 * self.gsGG**2 * self.lam * vh * self.vs**2)))**2) /
                (209952.0 * self.lam**6 * np.pi * vh**4 *
                 (9 * self.lam + 4 * self.gsGG * self.vs)**2 *
                 (self.ms**4 + e_cms**4 + self.ms**2 *
                  (-2 * e_cms**2 + self.width_s**2))))

        return ret_val.real

    def sigma_xx_to_s_to_pipi(self, e_cm):
        """Returns the spin-averaged, cross section for dark matter
        annihilating into a pair of charged pions through a scalar mediator in
        the s-channel.

        Parameters
        ----------
        e_cm : float or array-like
            Center of mass energy(ies).

        Returns
        -------
        sigma : float or array-like
            Cross section for x + x -> s* -> np.pi + np.pi.
        """
        # see sigma_xx_to_s_to_ff for explaination of this context mangager
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', r'invalid value encountered in true_divide')
            warnings.filterwarnings(
                'ignore', r'divide by zero encountered in true_divide')
            warnings.filterwarnings(
                'ignore', r'invalid value encountered in multiply')
            warnings.filterwarnings(
                'ignore', r'invalid value encountered in sqrt')
            warnings.filterwarnings(
                'ignore', r'invalid value encountered in power')

            e_cms = np.array(e_cm) if hasattr(e_cm, '__len__') else e_cm
            mask = (e_cms > 2.0 * mpi) & (e_cms > 2.0 * self.mx)

            rpis = mpi / e_cms
            rxs = self.mx / e_cms

            ret_val = mask * np.nan_to_num((
                self.gsxx**2 * np.sqrt((-1 + 4 * rpis**2) *
                                       (-1 + 4 * rxs**2)) *
                (162 * self.gsGG * self.lam**3 * e_cms**2 *
                 (-1 + 2 * rpis**2) * vh**2 +
                 b0 * (mdq + muq) * (9 * self.lam + 4 * self.gsGG * self.vs) *
                 (27 * self.gsff**2 * self.lam**2 * self.vs *
                  (3 * self.lam + 4 * self.gsGG * self.vs) -
                  2 * self.gsGG * vh**2 *
                  (27 * self.lam**2 - 30 * self.gsGG * self.lam * self.vs +
                   8 * self.gsGG**2 * self.vs**2) +
                  self.gsff *
                  (-81 * self.lam**3 * vh +
                   48 * self.gsGG**2 * self.lam * vh * self.vs**2)))**2) /
                (104976.0 * self.lam**6 * np.pi * vh**4 *
                 (9 * self.lam + 4 * self.gsGG * self.vs)**2 *
                 (self.ms**4 + e_cms**4 + self.ms**2 *
                  (-2 * e_cms**2 + self.width_s**2))))

        return ret_val.real

    def sigma_xx_to_ss(self, e_cm):
        """Returns the spin-averaged, cross section for dark matter
        annihilating into a pair of scalar mediator through the t and u
        channels.

        Parameters
        ----------
        e_cm : float or array-like
            Center of mass energy(ies).

        Returns
        -------
        sigma : float or array-like
            Cross section for x + x -> s + s.
        """
        # see sigma_xx_to_s_to_ff for explaination of this context mangager
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', r'invalid value encountered in true_divide')
            warnings.filterwarnings(
                'ignore', r'divide by zero encountered in true_divide')
            warnings.filterwarnings(
                'ignore', r'invalid value encountered in multiply')
            warnings.filterwarnings(
                'ignore', r'invalid value encountered in sqrt')
            warnings.filterwarnings(
                'ignore', r'invalid value encountered in power')

            e_cms = np.array(e_cm) if hasattr(e_cm, '__len__') else e_cm
            mask = (e_cms > 2.0 * self.ms) & (e_cms > 2.0 * self.mx)

            ret_val = mask * -np.nan_to_num((
                self.gsxx**4 * np.sqrt(-4 * self.ms**2 + e_cms**2) *
                np.sqrt(-4 * self.mx**2 + e_cms**2) *
                (-2 / (4 * self.mx**2 - e_cms**2) -
                 (self.ms**2 - 4 * self.mx**2)**2 /
                 ((4 * self.mx**2 - e_cms**2) *
                  (self.ms**4 - 4 * self.ms**2 * self.mx**2 +
                   self.mx**2 * e_cms**2)) -
                 (2 * (6 * self.ms**4 - 32 * self.mx**4 +
                       16 * self.mx**2 * e_cms**2 +
                       e_cms**4 - 4 * self.ms**2 *
                       (4 * self.mx**2 + e_cms**2)) *
                  np.arctanh((np.sqrt(-4 * self.ms**2 + e_cms**2) *
                              np.sqrt(-4 * self.mx**2 + e_cms**2)) /
                             (-2 * self.ms**2 + e_cms**2))) /
                 (np.sqrt(-4 * self.ms**2 + e_cms**2) *
                  (-2 * self.ms**2 + e_cms**2) *
                  (-4 * self.mx**2 + e_cms**2)**1.5))) /
                (16.0 * np.pi * e_cms**2))

        return ret_val.real

    def sigma_xx_to_s_to_xx(self, e_cm):
        """Returns the spin-averaged, self interaction cross section for dark
        matter.

        Parameters
        ----------
        e_cm : float or array-like
            Center of mass energy(ies).

        Returns
        -------
        sigma : float or array-like
            Cross section for x + x -> x + x
        """
        # see sigma_xx_to_s_to_ff for explaination of this context mangager
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', r'invalid value encountered in true_divide')
            warnings.filterwarnings(
                'ignore', r'divide by zero encountered in true_divide')
            warnings.filterwarnings(
                'ignore', r'invalid value encountered in multiply')
            warnings.filterwarnings(
                'ignore', r'invalid value encountered in sqrt')
            warnings.filterwarnings(
                'ignore', r'invalid value encountered in power')
            warnings.filterwarnings(
                'ignore', r'invalid value encountered in subtract')
            warnings.filterwarnings(
                'ignore', r'invalid value encountered in add')

            e_cms = np.array(e_cm) if hasattr(e_cm, '__len__') else e_cm
            mask = e_cms > 2.0 * self.mx

            rxs = self.mx / e_cms
            rss = self.ms / e_cms
            rwss = self.width_s / e_cms

            def msqrd(z):
                return ((self.gsxx**4 *
                         (3 * (-1 + z)**2 +
                          256 * rxs**8 * (-1 + z)**2 -
                          64 * rxs**6 * (5 - 8 * z + 3 * z**2) +
                          32 * rxs**4 * (5 - 6 * z + 3 * z**2) -
                          4 * rxs**2 * (5 - 12 * z + 7 * z**2) +
                          rss**4 *
                          (3 + z**2 - 8 * rxs**2 * (1 + z**2) +
                           16 * rxs**4 * (3 + z**2)) + rss**2 *
                          (3 - 3 * z**2 -
                           64 * rxs**6 * (3 - 4 * z + z**2) -
                           16 * rxs**4 * (-9 + 12 * z + z**2) +
                           4 * rxs**2 * (-17 + 8 * z + 5 * z**2) +
                           rwss**2 *
                           (3 + z**2 - 8 * rxs**2 * (1 + z**2) +
                            16 * rxs**4 * (3 + z**2))))) /
                        ((1 + rss**4 + rss**2 * (-2 + rwss**2)) *
                         (4 * rss**4 + 4 * rss**2 *
                          (rwss**2 + (-1 + 4 * rxs**2) * (-1 + z)) +
                          (1 - 4 * rxs**2)**2 * (-1 + z)**2))).real

            # Compute the nodes and weights for Legendre-Gauss quadrature
            nodes, weights = leggauss(10)
            nodes, weights = (np.reshape(nodes, (10, 1)),
                              np.reshape(weights, (10, 1)))
            ret_val = mask * \
                np.nan_to_num(np.sum(weights * msqrd(nodes),
                                     axis=0) / (32.0 * np.pi * e_cms))

        return ret_val.real

    def sigma_xpi_to_xpi(self, e_cm):
        """
        Returns the spin-averaged, elastic scattering cross section of DM off
        charged-pions for the scalar mediator model. Note only considers
        both a pi^+ or pi^- (i.e. sums over charges.)

        Parameters
        ----------
        e_cm : float
            Center of mass energy.

        Returns
        -------
        sigma : float
            Cross section for x + pi -> x + pi
        """
        gsGG = self.gsGG
        gsff = self.gsff
        gsxx = self.gsxx
        lam = self.lam
        ms = self.ms
        mx = self.mx
        widths = self.width_s
        vs = widths = self.vs

        e_cms = np.array(e_cm) if hasattr(e_cm, '__len__') else e_cm
        mask = e_cms > mx + mpi

        return 2.0 * mask * np.nan_to_num(
            (gsxx**2 *
             (2 *
              (b0**2 * (mdq + muq)**2 * (ms**2 - 4 * mx**2) *
               (9 * lam + 4 * gsGG * vs)**2 *
               (27 * gsff**2 * lam**2 * vs *
                (3 * lam + 4 * gsGG * vs) - 2 * gsGG * vh**2 *
                (27 * lam**2 - 30 * gsGG * lam * vs + 8 * gsGG**2 * vs**2) +
                gsff * (-81 * lam**3 * vh +
                        48 * gsGG**2 * lam * vh * vs**2))**2 +
               324 * b0 * gsGG * lam**3 * (mdq + muq) * vh**2 *
               (9 * lam + 4 * gsGG * vs) *
               (27 * gsff**2 * lam**2 * vs *
                (3 * lam + 4 * gsGG * vs) - 2 * gsGG * vh**2 *
                (27 * lam**2 - 30 * gsGG * lam * vs + 8 * gsGG**2 * vs**2) +
                gsff * (-81 * lam**3 * vh + 48 * gsGG**2 *
                        lam * vh * vs**2)) *
               (2 * mpi**2 * (ms**2 - 4 * mx**2) +
                ms**2 * (-ms**2 + 4 * mx**2 + widths**2)) +
               26244 * gsGG**2 * lam**6 * vh**4 *
               (ms**6 + 4 * mpi**4 * (ms**2 - 4 * mx**2) +
                4 * ms**2 * mx**2 * widths**2 - 4 * mpi**2 * ms**2 *
                (ms**2 - 4 * mx**2 - widths**2) -
                ms**4 * (4 * mx**2 + 3 * widths**2))) *
              np.arctan(ms / widths - e_cms**2 / (2. * ms * widths)) -
              2 * (b0**2 * (mdq + muq)**2 * (ms**2 - 4 * mx**2) *
                   (9 * lam + 4 * gsGG * vs)**2 *
                   (27 * gsff**2 * lam**2 * vs *
                    (3 * lam + 4 * gsGG * vs) - 2 * gsGG * vh**2 *
                    (27 * lam**2 - 30 * gsGG * lam * vs +
                     8 * gsGG**2 * vs**2) + gsff *
                    (-81 * lam**3 * vh + 48 * gsGG**2 * lam * vh * vs**2))**2 +
                   324 * b0 * gsGG * lam**3 * (mdq + muq) * vh**2 *
                   (9 * lam + 4 * gsGG * vs) *
                   (27 * gsff**2 * lam**2 * vs *
                    (3 * lam + 4 * gsGG * vs) - 2 * gsGG * vh**2 *
                    (27 * lam**2 - 30 * gsGG * lam * vs +
                     8 * gsGG**2 * vs**2) + gsff *
                    (-81 * lam**3 * vh + 48 * gsGG**2 * lam * vh * vs**2)) *
                   (2 * mpi**2 * (ms**2 - 4 * mx**2) +
                    ms**2 * (-ms**2 + 4 * mx**2 + widths**2)) +
                   26244 * gsGG**2 * lam**6 * vh**4 *
                   (ms**6 + 4 * mpi**4 * (ms**2 - 4 * mx**2) +
                    4 * ms**2 * mx**2 * widths**2 - 4 * mpi**2 * ms**2 *
                    (ms**2 - 4 * mx**2 - widths**2) - ms**4 *
                    (4 * mx**2 + 3 * widths**2))) *
              np.arctan((2 * ms**2 - 8 * mx**2 + e_cms**2) /
                        (2. * ms * widths)) + ms * widths *
              (-648 * gsGG * lam**3 * (4 * mx**2 - e_cms**2) * vh**2 *
               (162 * gsGG * lam**3 * (2 * mpi**2 - ms**2 + mx**2) * vh**2 +
                b0 * (mdq + muq) * (9 * lam + 4 * gsGG * vs) *
                (27 * gsff**2 * lam**2 * vs *
                 (3 * lam + 4 * gsGG * vs) - 2 * gsGG * vh**2 *
                 (27 * lam**2 - 30 * gsGG * lam * vs + 8 * gsGG**2 * vs**2) +
                 gsff * (-81 * lam**3 * vh + 48 * gsGG**2 *
                         lam * vh * vs**2))) -
               (648 * b0 * gsGG * lam**3 * (mdq + muq) *
                (mpi**2 - ms**2 + 2 * mx**2) * vh**2 *
                (9 * lam + 4 * gsGG * vs) *
                (27 * gsff**2 * lam**2 * vs *
                 (3 * lam + 4 * gsGG * vs) - 2 * gsGG * vh**2 *
                 (27 * lam**2 - 30 * gsGG * lam * vs + 8 * gsGG**2 * vs**2) +
                 gsff * (-81 * lam**3 * vh +
                         48 * gsGG**2 * lam * vh * vs**2)) +
                b0**2 * (mdq + muq)**2 * (9 * lam + 4 * gsGG * vs)**2 *
                (27 * gsff**2 * lam**2 * vs * (3 * lam + 4 * gsGG * vs) -
                 2 * gsGG * vh**2 *
                 (27 * lam**2 - 30 * gsGG * lam * vs + 8 * gsGG**2 * vs**2) +
                 gsff * (-81 * lam**3 * vh +
                         48 * gsGG**2 * lam * vh * vs**2))**2 +
                26244 * gsGG**2 * lam**6 * vh**4 *
                (4 * mpi**4 - 8 * mpi**2 * (ms**2 - 2 * mx**2) +
                 ms**2 * (3 * ms**2 - 8 * mx**2 - widths**2))) *
               np.log(4 * ms**4 + e_cms**4 - 4 * ms**2 *
                      (e_cms**2 - widths**2)) +
               (648 * b0 * gsGG * lam**3 * (mdq + muq) *
                (mpi**2 - ms**2 + 2 * mx**2) * vh**2 *
                (9 * lam + 4 * gsGG * vs) *
                (27 * gsff**2 * lam**2 * vs * (3 * lam + 4 * gsGG * vs) -
                 2 * gsGG * vh**2 *
                 (27 * lam**2 - 30 * gsGG * lam * vs +
                  8 * gsGG**2 * vs**2) + gsff *
                 (-81 * lam**3 * vh + 48 * gsGG**2 * lam * vh * vs**2)) +
                b0**2 * (mdq + muq)**2 * (9 * lam + 4 * gsGG * vs)**2 *
                (27 * gsff**2 * lam**2 * vs * (3 * lam + 4 * gsGG * vs) -
                 2 * gsGG * vh**2 *
                 (27 * lam**2 - 30 * gsGG * lam * vs +
                  8 * gsGG**2 * vs**2) + gsff *
                 (-81 * lam**3 * vh + 48 * gsGG**2 * lam * vh * vs**2))**2 +
                26244 * gsGG**2 * lam**6 * vh**4 *
                (4 * mpi**4 - 8 * mpi**2 * (ms**2 - 2 * mx**2) + ms**2 *
                 (3 * ms**2 - 8 * mx**2 - widths**2))) *
               np.log(4 * ms**4 + (-8 * mx**2 + e_cms**2)**2 +
                      4 * ms**2 * (-8 * mx**2 + e_cms**2 + widths**2))))) /
            (209952. * lam**6 * ms * np.pi * e_cms**2 *
             (-4 * mx**2 + e_cms**2) * vh**4 *
             (9 * lam + 4 * gsGG * vs)**2 * widths))

    def sigma_xpi0_to_xpi0(self, e_cm):
        """
        Returns the spin-averaged, elastic scattering cross section of DM off
        neutral pion for the scalar mediator model.

        Parameters
        ----------
        e_cm : float
            Center of mass energy.

        Returns
        -------
        sigma : float
            Cross section for x + pi^0 -> x + pi^0
        """
        gsGG = self.gsGG
        gsff = self.gsff
        gsxx = self.gsxx
        lam = self.lam
        ms = self.ms
        mx = self.mx
        widths = self.width_s
        vs = widths = self.vs

        e_cms = np.array(e_cm) if hasattr(e_cm, '__len__') else e_cm
        mask = e_cms > self.mx + mpi0

        return mask * (
            (gsxx**2 *
             (2 *
              (b0**2 * (mdq + muq)**2 *
               (ms**2 - 4 * mx**2) * (9 * lam + 4 * gsGG * vs)**2 *
               (27 * gsff**2 * lam**2 * vs * (3 * lam + 4 * gsGG * vs) -
                2 * gsGG * vh**2 *
                (27 * lam**2 - 30 * gsGG * lam * vs +
                 8 * gsGG**2 * vs**2) + gsff *
                (-81 * lam**3 * vh + 48 * gsGG**2 * lam * vh * vs**2))**2 +
               324 * b0 * gsGG * lam**3 * (mdq + muq) * vh**2 *
               (9 * lam + 4 * gsGG * vs) *
               (27 * gsff**2 * lam**2 * vs *
                (3 * lam + 4 * gsGG * vs) - 2 * gsGG * vh**2 *
                (27 * lam**2 - 30 * gsGG * lam * vs + 8 * gsGG**2 * vs**2) +
                gsff * (-81 * lam**3 * vh + 48 * gsGG**2 * lam * vh * vs**2)) *
               (2 * mpi0**2 * (ms**2 - 4 * mx**2) + ms**2 *
                (-ms**2 + 4 * mx**2 + widths**2)) +
               26244 * gsGG**2 * lam**6 * vh**4 *
               (ms**6 + 4 * mpi0**4 * (ms**2 - 4 * mx**2) +
                4 * ms**2 * mx**2 * widths**2 - 4 * mpi0**2 * ms**2 *
                (ms**2 - 4 * mx**2 - widths**2) -
                ms**4 * (4 * mx**2 + 3 * widths**2))) *
              np.arctan(ms / widths - e_cms**2 / (2. * ms * widths)) -
              2 * (b0**2 * (mdq + muq)**2 * (ms**2 - 4 * mx**2) *
                   (9 * lam + 4 * gsGG * vs)**2 *
                   (27 * gsff**2 * lam**2 * vs *
                    (3 * lam + 4 * gsGG * vs) - 2 * gsGG * vh**2 *
                    (27 * lam**2 - 30 * gsGG * lam * vs +
                     8 * gsGG**2 * vs**2) + gsff *
                    (-81 * lam**3 * vh + 48 * gsGG**2 * lam * vh * vs**2))**2 +
                   324 * b0 * gsGG * lam**3 * (mdq + muq) * vh**2 *
                   (9 * lam + 4 * gsGG * vs) *
                   (27 * gsff**2 * lam**2 * vs *
                    (3 * lam + 4 * gsGG * vs) -
                    2 * gsGG * vh**2 *
                    (27 * lam**2 - 30 * gsGG * lam * vs +
                     8 * gsGG**2 * vs**2) + gsff *
                    (-81 * lam**3 * vh + 48 * gsGG**2 * lam * vh * vs**2)) *
                   (2 * mpi0**2 * (ms**2 - 4 * mx**2) + ms**2 *
                    (-ms**2 + 4 * mx**2 + widths**2)) +
                   26244 * gsGG**2 * lam**6 * vh**4 *
                   (ms**6 + 4 * mpi0**4 *
                    (ms**2 - 4 * mx**2) +
                    4 * ms**2 * mx**2 * widths**2 -
                    4 * mpi0**2 * ms**2 *
                    (ms**2 - 4 * mx**2 - widths**2) -
                    ms**4 * (4 * mx**2 + 3 * widths**2))) *
              np.arctan((2 * ms**2 - 8 * mx**2 + e_cms**2) /
                        (2. * ms * widths)) + ms * widths *
              (-648 * gsGG * lam**3 * (4 * mx**2 - e_cms**2) * vh**2 *
               (162 * gsGG * lam**3 *
                (2 * mpi0**2 - ms**2 + mx**2) * vh**2 +
                b0 * (mdq + muq) * (9 * lam + 4 * gsGG * vs) *
                (27 * gsff**2 * lam**2 * vs *
                 (3 * lam + 4 * gsGG * vs) - 2 * gsGG * vh**2 *
                 (27 * lam**2 - 30 * gsGG * lam * vs +
                  8 * gsGG**2 * vs**2) + gsff *
                 (-81 * lam**3 * vh + 48 * gsGG**2 * lam * vh * vs**2))) -
               (648 * b0 * gsGG * lam**3 * (mdq + muq) *
                (mpi0**2 - ms**2 + 2 * mx**2) * vh**2 *
                (9 * lam + 4 * gsGG * vs) *
                (27 * gsff**2 * lam**2 * vs * (3 * lam + 4 * gsGG * vs) -
                 2 * gsGG * vh**2 *
                 (27 * lam**2 - 30 * gsGG * lam * vs +
                  8 * gsGG**2 * vs**2) + gsff *
                 (-81 * lam**3 * vh + 48 * gsGG**2 * lam * vh * vs**2)) +
                b0**2 * (mdq + muq)**2 * (9 * lam + 4 * gsGG * vs)**2 *
                (27 * gsff**2 * lam**2 * vs * (3 * lam + 4 * gsGG * vs) -
                 2 * gsGG * vh**2 *
                 (27 * lam**2 - 30 * gsGG * lam * vs +
                  8 * gsGG**2 * vs**2) + gsff *
                 (-81 * lam**3 * vh + 48 * gsGG**2 * lam * vh * vs**2))**2 +
                26244 * gsGG**2 * lam**6 * vh**4 *
                (4 * mpi0**4 - 8 * mpi0**2 * (ms**2 - 2 * mx**2) +
                 ms**2 * (3 * ms**2 - 8 * mx**2 - widths**2))) *
               np.log(4 * ms**4 + e_cms**4 - 4 * ms**2 *
                      (e_cms**2 - widths**2)) +
               (648 * b0 * gsGG * lam**3 *
                (mdq + muq) * (mpi0**2 - ms**2 + 2 * mx**2) * vh**2 *
                (9 * lam + 4 * gsGG * vs) *
                (27 * gsff**2 * lam**2 * vs *
                 (3 * lam + 4 * gsGG * vs) - 2 * gsGG * vh**2 *
                 (27 * lam**2 - 30 * gsGG * lam * vs + 8 * gsGG**2 * vs**2) +
                 gsff * (-81 * lam**3 * vh + 48 * gsGG**2 *
                         lam * vh * vs**2)) +
                b0**2 * (mdq + muq)**2 * (9 * lam + 4 * gsGG * vs)**2 *
                (27 * gsff**2 * lam**2 * vs * (3 * lam + 4 * gsGG * vs) -
                 2 * gsGG * vh**2 *
                 (27 * lam**2 - 30 * gsGG * lam * vs + 8 * gsGG**2 * vs**2) +
                 gsff * (-81 * lam**3 * vh +
                         48 * gsGG**2 * lam * vh * vs**2))**2 +
                26244 * gsGG**2 * lam**6 * vh**4 *
                (4 * mpi0**4 - 8 * mpi0**2 * (ms**2 - 2 * mx**2) +
                 ms**2 * (3 * ms**2 - 8 * mx**2 - widths**2))) *
               np.log(4 * ms**4 + (-8 * mx**2 + e_cms**2)**2 +
                      4 * ms**2 * (-8 * mx**2 + e_cms**2 + widths**2))))) /
            (209952. * lam**6 * ms * np.pi * e_cms**2 *
             (-4 * mx**2 + e_cms**2) * vh**4 *
             (9 * lam + 4 * gsGG * vs)**2 * widths))

    def sigma_xl_to_xl(self, e_cm, f):
        """
        Returns the spin-averaged, elastic scattering cross section of DM off
        leptons for the scalar mediator model. Note this considers
        both a l + or lbar (i.e. it sums over charges.)

        Parameters
        ----------
        e_cm : float
            Center of mass energy.
        f : string
            String labeling final state lepton: either 'e' or 'mu'.

        Returns
        -------
        sigma : float
            Cross section for x + l -> x + l
        """
        assert f == 'e' or f == 'mu'
        ml = me if f == 'e' else mmu

        gsll = self.gsff
        gsxx = self.gsxx
        ms = self.ms
        mx = self.mx
        widths = self.width_s

        e_cms = np.array(e_cm) if hasattr(e_cm, '__len__') else e_cm
        mask = e_cms > mx + ml

        return 2.0 * mask * np.nan_to_num(
            -(gsll**2 * gsxx**2 *
              ((4 * ml**2 * (ms**2 - 4 * mx**2) + ms**2 *
                (-ms**2 + 4 * mx**2 + widths**2)) *
               np.arctan(ms / widths - e_cms**2 /
                         (2. * ms * widths)) +
               (-4 * ml**2 * (ms**2 - 4 * mx**2) + ms**2 *
                (ms**2 - 4 * mx**2 - widths**2)) *
               np.arctan((2 * ms**2 - 8 * mx**2 + e_cms**2) /
                         (2. * ms * widths)) +
               ms * widths *
               (-4 * mx**2 + e_cms**2 +
                (-2 * ml**2 + ms**2 - 2 * mx**2) *
                np.log(4 * ms**4 + e_cms**4 - 4 * ms**2 *
                       (e_cms**2 - widths**2)) +
                (2 * ml**2 - ms**2 + 2 * mx**2) *
                np.log(4 * ms**4 + (-8 * mx**2 + e_cms**2)**2 +
                       4 * ms**2 * (-8 * mx**2 + e_cms**2 +
                                    widths**2))))) /
            (16. * ms * np.pi * e_cms**2 *
             (4 * mx**2 - e_cms**2) * widths))

    def sigma_xg_to_xg(self, e_cm):
        """
        Returns the spin-averaged, elastic scattering cross section of DM off
        photons for the scalar mediator model.

        Parameters
        ----------
        e_cm : float
            Center of mass energy.

        Returns
        -------
        sigma : float
            Cross section for x + g -> x + g
        """
        gsFF = self.gsFF
        gsxx = self.gsxx
        lam = self.lam
        ms = self.ms
        mx = self.mx
        widths = self.width_s

        e_cms = np.array(e_cm) if hasattr(e_cm, '__len__') else e_cm
        mask = e_cms > self.mx

        return mask * np.nan_to_num(
            (alpha_em**2 * gsFF**2 * gsxx**2 *
             (2 *
              (ms**5 + 4 * ms * mx**2 * widths**2 - ms**3 *
               (4 * mx**2 + 3 * widths**2)) *
              np.arctan(ms / widths - e_cms**2 /
                        (2. * ms * widths)) - 2 *
              (ms**5 + 4 * ms * mx**2 * widths**2 - ms**3 *
               (4 * mx**2 + 3 * widths**2)) *
              np.arctan((2 * ms**2 - 8 * mx**2 + e_cms**2) /
                        (2. * ms * widths)) +
              widths *
              (4 * (ms**2 - mx**2) *
               (4 * mx**2 - e_cms**2) + ms**2 *
               (-3 * ms**2 + 8 * mx**2 + widths**2) *
               np.log(4 * ms**4 + e_cms**4 - 4 * ms**2 *
                      (e_cms**2 - widths**2)) +
               ms**2 * (3 * ms**2 - 8 * mx**2 - widths**2) *
               np.log(4 * ms**4 + (-8 * mx**2 + e_cms**2)**2 +
                      4 * ms**2 * (-8 * mx**2 + e_cms**2 +
                                   widths**2))))) /
            (128. * lam**2 * np.pi**3 * e_cms**2 *
             (-4 * mx**2 + e_cms**2) * widths))

    def sigma_xs_to_xs(self, e_cm):
        """
        Returns the spin-averaged, elastic scattering cross section of DM off
        scalar mediators for the scalar mediator model.

        Parameters
        ----------
        e_cm : float
            Center of mass energy.

        Returns
        -------
        sigma : float
            Cross section for x + s -> x + s
        """
        gsxx = self.gsxx
        ms = self.ms
        mx = self.mx

        e_cms = np.array(e_cm) if hasattr(e_cm, '__len__') else e_cm
        mask = e_cms > mx + ms

        return mask * np.nan_to_num(
            (gsxx**4 *
             ((4 * mx**2 - e_cms**2) *
              (16 * ms**8 + 52 * mx**8 - 256 * mx**6 * e_cms**2 +
               223 * mx**4 * e_cms**4 + 84 * mx**2 * e_cms**6 +
               9 * e_cms**8 - 16 * ms**6 * (7 * mx**2 + 3 * e_cms**2) +
               ms**4 *
               (104 * mx**4 + 328 * mx**2 * e_cms**2 +
                87 * e_cms**4) + ms**2 *
               (24 * mx**6 - 216 * mx**4 * e_cms**2 -
                322 * mx**2 * e_cms**4 - 54 * e_cms**6)) +
              (mx**2 - e_cms**2) *
              (32 * ms**8 + 180 * mx**8 - 24 * mx**6 * e_cms**2 -
               345 * mx**4 * e_cms**4 - 38 * mx**2 * e_cms**6 +
               3 * e_cms**8 + 32 * ms**6 * (7 * mx**2 - e_cms**2) +
               ms**4 * (-520 * mx**4 - 512 * mx**2 * e_cms**2 +
                        22 * e_cms**4) + 16 * ms**2 *
               (3 * mx**6 + 49 * mx**4 * e_cms**2 +
                20 * mx**2 * e_cms**4 - e_cms**6)) *
              np.log(4 * ms**2 + 2 * mx**2 - 3 * e_cms**2) -
              (mx**2 - e_cms**2) *
              (32 * ms**8 + 180 * mx**8 -
               24 * mx**6 * e_cms**2 - 345 * mx**4 * e_cms**4 -
               38 * mx**2 * e_cms**6 + 3 * e_cms**8 +
               32 * ms**6 * (7 * mx**2 - e_cms**2) + ms**4 *
               (-520 * mx**4 - 512 * mx**2 * e_cms**2 + 22 * e_cms**4) +
               16 * ms**2 *
               (3 * mx**6 + 49 * mx**4 * e_cms**2 +
                20 * mx**2 * e_cms**4 - e_cms**6)) *
              np.log(4 * ms**2 - 6 * mx**2 - e_cms**2))) /
            (16. * np.pi * (mx - e_cms)**2 * e_cms**2 *
             (mx + e_cms)**2 * (4 * ms**2 + 2 * mx**2 - 3 * e_cms**2) *
             (4 * ms**2 - 6 * mx**2 - e_cms**2) * (4 * mx**2 - e_cms**2)))

    def annihilation_cross_section_funcs(self):
        return {
            "mu mu": lambda e_cm: self.sigma_xx_to_s_to_ff(e_cm, "mu"),
            "e e": lambda e_cm: self.sigma_xx_to_s_to_ff(e_cm, "e"),
            "g g": self.sigma_xx_to_s_to_gg,
            "pi0 pi0": self.sigma_xx_to_s_to_pi0pi0,
            "pi pi": self.sigma_xx_to_s_to_pipi,
            "s s": self.sigma_xx_to_ss,
        }

    def elastic_scattering_cross_sections(self, e_cm):
        return {
            "pi": self.sigma_xpi_to_xpi(e_cm),
            "pi0": self.sigma_xpi0_to_xpi0(e_cm),
            "e": self.sigma_xl_to_xl(e_cm, 'e'),
            "mu": self.sigma_xl_to_xl(e_cm, 'mu'),
            "g": self.sigma_xg_to_xg(e_cm),
            "s": self.sigma_xs_to_xs(e_cm)
        }
