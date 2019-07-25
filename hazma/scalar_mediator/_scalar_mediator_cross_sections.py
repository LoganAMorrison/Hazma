from cmath import sqrt, pi, atanh

from hazma.parameters import vh, b0, alpha_em
from hazma.parameters import charged_pion_mass as mpi
from hazma.parameters import neutral_pion_mass as mpi0
from hazma.parameters import up_quark_mass as muq
from hazma.parameters import down_quark_mass as mdq
from hazma.parameters import muon_mass as mmu
from hazma.parameters import electron_mass as me

from scipy.integrate import quad


class ScalarMediatorCrossSection:
    def sigma_xx_to_s_to_ff(self, e_cm, f):
        """Returns the spin-averaged, cross section for dark matter
        annihilating into a pair of fermions, *f* through a scalar mediator in
        the s-channel.

        Parameters
        ----------
        e_cm : float
            Center of mass energy.
        f: str
            String for the final state fermion: f = 'e' for electron and
            f = 'mu' for muon.

        Returns
        -------
        sigma : float
            Cross section for x + x -> s* -> f + f.
        """
        mx = self.mx

        if f == "e":
            mf = me
        elif f == "mu":
            mf = mmu

        if e_cm > 2.0 * mf and e_cm >= 2.0 * mx:
            ms = self.ms
            gsff = self.gsff
            gsxx = self.gsxx
            width_s = self.width_s

            ret_val = (
                gsff ** 2
                * gsxx ** 2
                * mf ** 2
                * (-2 * mx + e_cm)
                * (2 * mx + e_cm)
                * (-4 * mf ** 2 + e_cm ** 2) ** 1.5
            ) / (
                16.0
                * pi
                * e_cm ** 2
                * sqrt(-4 * mx ** 2 + e_cm ** 2)
                * vh ** 2
                * (
                    ms ** 4
                    - 2 * ms ** 2 * e_cm ** 2
                    + e_cm ** 4
                    + ms ** 2 * width_s ** 2
                )
            )

            assert ret_val.imag == 0
            assert ret_val.real >= 0

            return ret_val.real
        else:
            return 0.0

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
        mx = self.mx

        if e_cm >= 2.0 * mx:
            gsFF = self.gsFF
            gsxx = self.gsxx
            ms = self.ms
            widths = self.width_s
            rx = mx / e_cm
            Lam = self.lam

            ret_val = (
                alpha_em ** 2
                * gsFF ** 2
                * gsxx ** 2
                * e_cm ** 4
                * sqrt(1 - 4 * rx ** 2)
            ) / (
                64.0
                * Lam ** 2
                * pi ** 3
                * (ms ** 4 + e_cm ** 4 + ms ** 2 * (-2 * e_cm ** 2 + widths ** 2))
            )

            assert ret_val.imag == 0
            assert ret_val.real >= 0

            return ret_val.real
        else:
            return 0.0

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
        mx = self.mx

        if e_cm > 2.0 * mpi0 and e_cm >= 2.0 * mx:
            gsxx = self.gsxx
            gsff = self.gsff
            gsGG = self.gsGG
            ms = self.ms
            vs = self.vs
            widths = self.width_s
            Lam = self.lam
            rpi0 = mpi0 / e_cm
            rx = mx / e_cm

            ret_val = (
                gsxx ** 2
                * sqrt((-1 + 4 * rpi0 ** 2) * (-1 + 4 * rx ** 2))
                * (
                    162 * gsGG * Lam ** 3 * e_cm ** 2 * (-1 + 2 * rpi0 ** 2) * vh ** 2
                    + b0
                    * (mdq + muq)
                    * (9 * Lam + 4 * gsGG * vs)
                    * (
                        27 * gsff ** 2 * Lam ** 2 * vs * (3 * Lam + 4 * gsGG * vs)
                        - 2
                        * gsGG
                        * vh ** 2
                        * (
                            27 * Lam ** 2
                            - 30 * gsGG * Lam * vs
                            + 8 * gsGG ** 2 * vs ** 2
                        )
                        + gsff
                        * (-81 * Lam ** 3 * vh + 48 * gsGG ** 2 * Lam * vh * vs ** 2)
                    )
                )
                ** 2
            ) / (
                209952.0
                * Lam ** 6
                * pi
                * vh ** 4
                * (9 * Lam + 4 * gsGG * vs) ** 2
                * (ms ** 4 + e_cm ** 4 + ms ** 2 * (-2 * e_cm ** 2 + widths ** 2))
            )

            assert ret_val.imag == 0
            assert ret_val.real >= 0

            return ret_val.real
        else:
            return 0.0

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
            Cross section for x + x -> s* -> pi + pi.
        """
        mx = self.mx

        if e_cm > 2.0 * mpi and e_cm >= 2.0 * mx:
            gsxx = self.gsxx
            gsff = self.gsff
            gsGG = self.gsGG
            ms = self.ms
            vs = self.vs
            widths = self.width_s
            Lam = self.lam
            rpi = mpi / e_cm
            rx = mx / e_cm

            ret_val = (
                gsxx ** 2
                * sqrt((-1 + 4 * rpi ** 2) * (-1 + 4 * rx ** 2))
                * (
                    162 * gsGG * Lam ** 3 * e_cm ** 2 * (-1 + 2 * rpi ** 2) * vh ** 2
                    + b0
                    * (mdq + muq)
                    * (9 * Lam + 4 * gsGG * vs)
                    * (
                        27 * gsff ** 2 * Lam ** 2 * vs * (3 * Lam + 4 * gsGG * vs)
                        - 2
                        * gsGG
                        * vh ** 2
                        * (
                            27 * Lam ** 2
                            - 30 * gsGG * Lam * vs
                            + 8 * gsGG ** 2 * vs ** 2
                        )
                        + gsff
                        * (-81 * Lam ** 3 * vh + 48 * gsGG ** 2 * Lam * vh * vs ** 2)
                    )
                )
                ** 2
            ) / (
                104976.0
                * Lam ** 6
                * pi
                * vh ** 4
                * (9 * Lam + 4 * gsGG * vs) ** 2
                * (ms ** 4 + e_cm ** 4 + ms ** 2 * (-2 * e_cm ** 2 + widths ** 2))
            )

            assert ret_val.imag == 0
            assert ret_val.real >= 0

            return ret_val.real
        else:
            return 0.0

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
        ms = self.ms
        mx = self.mx

        if e_cm > 2.0 * ms and e_cm >= 2.0 * mx:
            gsxx = self.gsxx

            ret_val = -(
                gsxx ** 4
                * sqrt(-4 * ms ** 2 + e_cm ** 2)
                * sqrt(-4 * mx ** 2 + e_cm ** 2)
                * (
                    -2 / (4 * mx ** 2 - e_cm ** 2)
                    - (ms ** 2 - 4 * mx ** 2) ** 2
                    / (
                        (4 * mx ** 2 - e_cm ** 2)
                        * (ms ** 4 - 4 * ms ** 2 * mx ** 2 + mx ** 2 * e_cm ** 2)
                    )
                    - (
                        2
                        * (
                            6 * ms ** 4
                            - 32 * mx ** 4
                            + 16 * mx ** 2 * e_cm ** 2
                            + e_cm ** 4
                            - 4 * ms ** 2 * (4 * mx ** 2 + e_cm ** 2)
                        )
                        * atanh(
                            (
                                sqrt(-4 * ms ** 2 + e_cm ** 2)
                                * sqrt(-4 * mx ** 2 + e_cm ** 2)
                            )
                            / (-2 * ms ** 2 + e_cm ** 2)
                        )
                    )
                    / (
                        sqrt(-4 * ms ** 2 + e_cm ** 2)
                        * (-2 * ms ** 2 + e_cm ** 2)
                        * (-4 * mx ** 2 + e_cm ** 2) ** 1.5
                    )
                )
            ) / (16.0 * pi * e_cm ** 2)

            assert ret_val.imag == 0
            assert ret_val.real >= 0

            return ret_val.real
        else:
            return 0.0

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
        rs = self.ms / e_cm
        rx = self.mx / e_cm
        gsxx = self.gsxx
        rws = self.width_s / e_cm

        if e_cm > 2.0 * self.mx:

            def msqrd(z):
                return (
                    gsxx ** 4
                    * (
                        3 * (-1 + z) ** 2
                        + 256 * rx ** 8 * (-1 + z) ** 2
                        - 64 * rx ** 6 * (5 - 8 * z + 3 * z ** 2)
                        + 32 * rx ** 4 * (5 - 6 * z + 3 * z ** 2)
                        - 4 * rx ** 2 * (5 - 12 * z + 7 * z ** 2)
                        + rs ** 4
                        * (
                            3
                            + z ** 2
                            - 8 * rx ** 2 * (1 + z ** 2)
                            + 16 * rx ** 4 * (3 + z ** 2)
                        )
                        + rs ** 2
                        * (
                            3
                            - 3 * z ** 2
                            - 64 * rx ** 6 * (3 - 4 * z + z ** 2)
                            - 16 * rx ** 4 * (-9 + 12 * z + z ** 2)
                            + 4 * rx ** 2 * (-17 + 8 * z + 5 * z ** 2)
                            + rws ** 2
                            * (
                                3
                                + z ** 2
                                - 8 * rx ** 2 * (1 + z ** 2)
                                + 16 * rx ** 4 * (3 + z ** 2)
                            )
                        )
                    )
                ) / (
                    (1 + rs ** 4 + rs ** 2 * (-2 + rws ** 2))
                    * (
                        4 * rs ** 4
                        + 4 * rs ** 2 * (rws ** 2 + (-1 + 4 * rx ** 2) * (-1 + z))
                        + (1 - 4 * rx ** 2) ** 2 * (-1 + z) ** 2
                    )
                )

            ret_val = quad(msqrd, -1, 1)[0] / (32.0 * pi * e_cm)

            assert ret_val.imag == 0.0
            assert ret_val.real >= 0.0

            return ret_val.real
        else:
            return 0.0

    def annihilation_cross_section_funcs(self):
        return {
            "mu mu": lambda e_cm: self.sigma_xx_to_s_to_ff(e_cm, "mu"),
            "e e": lambda e_cm: self.sigma_xx_to_s_to_ff(e_cm, "e"),
            "g g": self.sigma_xx_to_s_to_gg,
            "pi0 pi0": self.sigma_xx_to_s_to_pi0pi0,
            "pi pi": self.sigma_xx_to_s_to_pipi,
            "s s": self.sigma_xx_to_ss,
        }
