from hazma.parameters import vh, b0, alpha_em
from hazma.parameters import charged_pion_mass as mpi
from hazma.parameters import neutral_pion_mass as mpi0
from hazma.parameters import up_quark_mass as muq
from hazma.parameters import down_quark_mass as mdq
from hazma.parameters import muon_mass as mmu
from hazma.parameters import electron_mass as me

from scipy.integrate import simps
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
                (-2 * self.mx + e_cm) * (2 * self.mx + e_cm) *
                (-4 * mf**2 + e_cm**2)**1.5) /
                (16.0 * np.pi * e_cm**2 *
                 np.sqrt(-4 * self.mx**2 + e_cm**2) *
                 vh**2 * (self.ms**4 - 2 * self.ms**2 * e_cm**2 + e_cm**4 +
                          self.ms**2 * self.width_s**2)))

        # need the .real for the case where the User passes a float.
        return ret_val.real

    def sigma_xx_to_s_to_gg(self, e_cm):
        """Returns the spin-averaged, cross section for dark matter
        annihilating into a pair of photons through a scalar mediator in the
        s-channel.

        Parameters
        ----------
        e_cm : float
            Center of mass energy.

        Returns
        -------
        sigma : float
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
        e_cm : float
            Center of mass energy.

        Returns
        -------
        sigma : float
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
            mask = (e_cms > 2.0 * mpi0) & (e_cms >= 2.0 * self.mx)

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
        e_cm : float
            Center of mass energy.

        Returns
        -------
        sigma : float
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
            mask = (e_cms > 2.0 * mpi) & (e_cms >= 2.0 * self.mx)

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
        e_cm : float
            Center of mass energy.

        Returns
        -------
        sigma : float
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
            mask = (e_cms > 2.0 * self.ms) & (e_cms >= 2.0 * self.mx)

            ret_val = mask * -np.nan_to_num((
                self.gsxx**4 * np.sqrt(-4 * self.ms**2 + e_cms**2) *
                np.sqrt(-4 * self.mx**2 + e_cms**2) *
                (-2 / (4 * self.mx**2 - e_cms**2) -
                 (self.ms**2 - 4 * self.mx**2)**2 /
                 ((4 * self.mx**2 - e_cms**2) *
                  (self.ms**4 - 4 * self.ms**2 * self.mx**2 +
                   self.mx**2 * e_cms**2)) -
                 (2 * (6 * self.ms**4 - 32 * self.mx**4 +
                       16 * self.mx**2 * e_cm**2 +
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
        e_cm : float
            Center of mass energy.

        Returns
        -------
        sigma : float
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

            zs = np.reshape(np.linspace(-1, 1, num=100), (100, 1))
            ret_val = mask * np.nan_to_num(simps(msqrd(zs), zs, axis=0) /
                                           (32.0 * np.pi * e_cms))

        return ret_val.real

    def annihilation_cross_section_funcs(self):
        return {
            "mu mu": lambda e_cm: self.sigma_xx_to_s_to_ff(e_cm, "mu"),
            "e e": lambda e_cm: self.sigma_xx_to_s_to_ff(e_cm, "e"),
            "g g": self.sigma_xx_to_s_to_gg,
            "pi0 pi0": self.sigma_xx_to_s_to_pi0pi0,
            "np.pi np.pi": self.sigma_xx_to_s_to_pipi,
            "s s": self.sigma_xx_to_ss,
        }
