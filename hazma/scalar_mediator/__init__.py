import numpy as np

from hazma.theory import TheoryAnn
from hazma.parameters import up_quark_mass as muq
from hazma.parameters import down_quark_mass as mdq
from hazma.parameters import strange_quark_mass as msq
from hazma.parameters import fpi, b0, vh


# Note that Theory must be inherited from AFTER all the other mixin classes,
# since they furnish definitions of the abstract methods in Theory.
class ScalarMediator(TheoryAnn):
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

    Parameters
    ----------
    mx : float
        Dark matter mass.
    ms : float
        Mass of the scalar mediator.
    gsxx : float
        Coupling constant for scalar interactions with dark matter.
    gsff : float
        Constant of proportionality for the scalar's couplings to Standard
        Model fermions.
    gsGG : float
        Coupling constant for scalar interactions with gluons.
    gsFF : float
        Coupling constant for scalar interactions with photons.
    lam : float
        Mass scale in scalar's interactions with photons and gluons.
    """

    from ._scalar_mediator_spectra import (
        dnde_ee,  # pylint: disable
        dnde_mumu,
        dnde_pi0pi0,
        dnde_pipi,
        dnde_ss,
        spectrum_funcs,
        gamma_ray_lines,
    )
    from ._scalar_mediator_widths import (
        width_s_to_gg,
        width_s_to_pi0pi0,
        width_s_to_pipi,
        width_s_to_xx,
        width_s_to_ff,
        partial_widths,
    )
    from ._scalar_mediator_positron_spectra import (
        dnde_pos_pipi,
        dnde_pos_mumu,
        dnde_pos_ss,
        positron_spectrum_funcs,
        positron_lines,
    )
    from ._scalar_mediator_fsr import (
        dnde_xx_to_s_to_ffg,
        dnde_xx_to_s_to_pipig,
    )
    from ._scalar_mediator_cross_sections import (
        sigma_xx_to_s_to_ff,
        sigma_xx_to_s_to_gg,
        sigma_xx_to_s_to_pi0pi0,
        sigma_xx_to_s_to_pipi,
        sigma_xx_to_ss,
        sigma_xx_to_s_to_xx,
        sigma_xpi_to_xpi,
        sigma_xpi0_to_xpi0,
        sigma_xl_to_xl,
        sigma_xg_to_xg,
        sigma_xs_to_xs,
        sigma_ss_to_xx,
        thermal_cross_section,
        annihilation_cross_section_funcs,
        elastic_scattering_cross_sections,
    )
    from ._scalar_mediator_constraints import (
        width_B_k_s,
        constraint_B_k_invis,
        constraint_B_k_mu_mu,
        constraint_B_k_e_e,
        width_k_pi_s,
        constraint_k_pi_invis,
        width_kl_pi0_s,
        constraint_kl_pi0_mu_mu,
        constraint_kl_pi0_e_e,
        width_B_xs_s,
        constrain_beam_dump,
        width_h_invis,
        constraint_higgs_invis,
        constraints,
    )

    def __init__(self, mx, ms, gsxx, gsff, gsGG, gsFF, lam):
        self._mx = mx
        self._ms = ms
        self._gsxx = gsxx
        self._gsff = gsff
        self._gsGG = gsGG
        self._gsFF = gsFF
        self._lam = lam
        self.width_s = 0.0
        self.vs = self.compute_vs()
        self.compute_width_s()  # vs MUST be computed first

    def __repr__(self):
        return (
            f"ScalarMediator(\n"
            f"\tmx={self.mx} MeV,\n"
            f"\tms={self.ms} MeV,\n"
            f"\tgsxx={self.gsxx},\n"
            f"\tgsff={self.gsff},\n"
            f"\tgsGG={self.gsGG},\n"
            f"\tgsFF={self.gsFF},\n"
            f"\tlam={self.lam} MeV,\n"
            ")"
        )

    @property
    def mx(self):
        return self._mx

    @mx.setter
    def mx(self, mx):
        self._mx = mx
        self.vs = self.compute_vs()
        self.compute_width_s()  # vs MUST be computed first

    @property
    def ms(self):
        return self._ms

    @ms.setter
    def ms(self, ms):
        self._ms = ms
        self.vs = self.compute_vs()
        self.compute_width_s()  # vs MUST be computed first

    @property
    def gsxx(self):
        return self._gsxx

    @gsxx.setter
    def gsxx(self, gsxx):
        self._gsxx = gsxx
        self.vs = self.compute_vs()
        self.compute_width_s()  # vs MUST be computed first

    @property
    def gsff(self):
        return self._gsff

    @gsff.setter
    def gsff(self, gsff):
        self._gsff = gsff
        self.vs = self.compute_vs()
        self.compute_width_s()  # vs MUST be computed first

    @property
    def gsGG(self):
        return self._gsGG

    @gsGG.setter
    def gsGG(self, gsGG):
        self._gsGG = gsGG
        self.vs = self.compute_vs()
        self.compute_width_s()  # vs MUST be computed first

    @property
    def gsFF(self):
        return self._gsFF

    @gsFF.setter
    def gsFF(self, gsFF):
        self._gsFF = gsFF
        self.vs = self.compute_vs()
        self.compute_width_s()  # vs MUST be computed first

    @property
    def lam(self):
        return self._lam

    @lam.setter
    def lam(self, lam):
        self._lam = lam
        self.vs = self.compute_vs()
        self.compute_width_s()  # vs MUST be computed first

    def compute_vs(self):
        """
        Compute the value of the scalar mediator vacuum expectation value.

        Notes
        -----
        Warning! Currently this function simply returns zero. This is a very
        good approximation in areas of parameters space which are allowed by
        experimental observations.

        Returns
        -------
        vs: float
            vacuum expectation value of scalar mediator.
        """
        if 3 * self.gsff + 2 * self.gsGG == 0:
            vs = 0.0
        else:
            # trM = muq + mdq + msq

            # ms = self.ms
            # gsff = self.gsff
            # gsGG = self.gsGG
            # Lam = self.lam

            # self.vs = (-3 * ms * vh +
            #           np.sqrt(4 * b0 * fpi**2 *
            #                   (3 * gsff + 2 * gsGG)**2 * trM +
            #                   9 * ms**2 * vh**2)) / (6 * gsff * ms +
            #                                          4 * gsGG * ms)
            vs = 0.0

        return vs

    def compute_width_s(self):
        """Updates the scalar's total width."""
        self.width_s = self.partial_widths()["total"]

    def __fpiT(self, vs):
        """Returns the Lagrangian parameter __fpiT."""
        return fpi / np.sqrt(1.0 + 4.0 * self._gsGG * vs / (9.0 * vh))

    def __b0T(self, vs, fpiT):
        """Returns the Lagrangian parameter __b0T."""
        return (
            b0
            * (fpi / fpiT) ** 2
            / (1.0 + vs / vh * (2.0 * self._gsGG / 3.0 + self._gsff))
        )

    def __msT(self, fpiT, b0T):
        """Returns the Lagrangian parameter __msT."""
        trM = muq + mdq + msq

        return np.sqrt(
            self._ms**2
            - 16.0
            * self._gsGG
            * b0T
            * fpiT**2
            / (81.0 * vh**2)
            * (2.0 * self._gsGG - 9.0 * self._gsff)
            * trM
        )

    @staticmethod
    def list_annihilation_final_states():
        """
        Return a list of the available final states.

        Returns
        -------
        fs : array-like
            Array of the available final states.
        """
        return ["mu mu", "e e", "g g", "pi0 pi0", "pi pi", "s s"]


class HiggsPortal(ScalarMediator):
    r"""Create a ``ScalarMediator`` object with Higgs Portal couplings.

    The couplings in the full scalar model are defined by::

        gsff = sin(theta)
        gsGG = 3 sin(theta)
        gsFF = -5/6 sin(theta)
        Lam = vh

    where theta is the mixing angle between the Standard Model Higgs
    and the scalar mediator.

    Parameters
    ----------
    mx : float
        Mass of the dark matter.
    ms : float
        Mass of the scalar mediator.
    gsxx : float
        Coupling of scalar mediator to dark matter.
    stheta : float
        Sine of the mixing angle between the Standard Model Higgs
        and the scalar mediator.
    """

    def __init__(self, mx, ms, gsxx, stheta):
        self._stheta = stheta

        super(HiggsPortal, self).__init__(
            mx, ms, gsxx, stheta, 3.0 * stheta, -5.0 * stheta / 6.0, vh
        )

    def __repr__(self):
        repr_ = "HiggsPortal("
        repr_ += f"mx={self.mx} [MeV], "
        repr_ += f"ms={self.ms} [MeV], "
        repr_ += f"gsxx={self.gsxx}, "
        repr_ += f"stheta={self.stheta}"
        repr_ += ")"
        return repr_

    @property
    def stheta(self):
        return self._stheta

    @stheta.setter
    def stheta(self, stheta):
        self._stheta = stheta
        self._gsff = stheta
        self._gsGG = 3.0 * stheta
        self._gsFF = -5.0 * stheta / 6.0
        self.compute_vs()
        self.compute_width_s()  # vs MUST be computed first

    # Hide underlying properties' setters
    @ScalarMediator.gsff.setter
    def gsff(self, _):
        raise AttributeError("Cannot set gsff")

    @ScalarMediator.gsGG.setter
    def gsGG(self, _):
        raise AttributeError("Cannot set gsGG")

    @ScalarMediator.gsFF.setter
    def gsFF(self, _):
        raise AttributeError("Cannot set gsFF")


class HeavyQuark(ScalarMediator):
    r"""
    Create a ScalarMediator object with heavy quark couplings.

    Creates an object for the scalar mediator model with the following
    specific coupling definitions:

        gsff = 0
        gsGG = gsQ
        gsFF = 0
        Lam = mQ

    where gsQ is the coupling of the heavy quark to the scalar mediator
    (-gsQ S Qbar Q) and mQ is the mass of the heavy quark.

    Parameters
    ----------
    mx : float
        Mass of the dark matter.
    ms : float
        Mass of the scalar mediator.
    gsxx : float
        Coupling of scalar mediator to dark matter.
    gsQ : float
        Coupling of the heavy quark to the scalar mediator.
    mQ : float
        Mass of the heavy quark.
    QQ : float
        Charge of the heavy quark.
    """

    def __init__(self, mx, ms, gsxx, gsQ, mQ, QQ):
        self._gsQ = gsQ
        self._mQ = mQ
        self._QQ = QQ

        super(HeavyQuark, self).__init__(mx, ms, gsxx, 0.0, gsQ, 2.0 * gsQ * QQ**2, mQ)

    def __repr__(self):
        repr_ = "HeavyQuark("
        repr_ += f"mx={self.mx} [MeV], "
        repr_ += f"ms={self.ms} [MeV], "
        repr_ += f"gsxx={self.gsxx}, "
        repr_ += f"gsQ={self.gsQ}, "
        repr_ += f"mQ={self.mQ} [MeV], "
        repr_ += f"QQ={self.QQ}"
        repr_ += ")"
        return repr_

    @staticmethod
    def list_annihilation_final_states():
        return ["g g", "pi0 pi0", "pi pi", "s s"]

    @property
    def gsQ(self):
        return self._gsQ

    @gsQ.setter
    def gsQ(self, gsQ):
        self._gsQ = gsQ
        self._gsGG = gsQ
        self._gsFF = 2.0 * gsQ * self._QQ**2
        self.compute_vs()
        self.compute_width_s()  # vs MUST be computed first

    @property
    def mQ(self):
        return self._mQ

    @mQ.setter
    def mQ(self, mQ):
        self._mQ = mQ
        self._lam = mQ
        self.compute_vs()
        self.compute_width_s()  # vs MUST be computed first

    @property
    def QQ(self):
        return self._QQ

    @QQ.setter
    def QQ(self, QQ):
        self._QQ = QQ
        self._gsFF = 2.0 * self._gsQ * QQ**2
        self.compute_vs()
        self.compute_width_s()  # vs MUST be computed first

    # Hide underlying properties' setters
    @ScalarMediator.gsff.setter
    def gsff(self, _):
        raise AttributeError("Cannot set gsff")

    @ScalarMediator.gsGG.setter
    def gsGG(self, _):
        raise AttributeError("Cannot set gsGG")

    @ScalarMediator.gsFF.setter
    def gsFF(self, _):
        raise AttributeError("Cannot set gsFF")
