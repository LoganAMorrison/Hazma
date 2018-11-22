from hazma.theory import Theory

from hazma.parameters import up_quark_mass as muq
from hazma.parameters import down_quark_mass as mdq
from hazma.parameters import strange_quark_mass as msq
from hazma.parameters import fpi, b0, vh

from hazma.scalar_mediator._scalar_mediator_constraints \
    import ScalarMediatorConstraints
from hazma.scalar_mediator._scalar_mediator_cross_sections \
    import ScalarMediatorCrossSection
from hazma.scalar_mediator._scalar_mediator_fsr \
    import ScalarMediatorFSR
from hazma.scalar_mediator._scalar_mediator_positron_spectra \
    import ScalarMediatorPositronSpectra
from hazma.scalar_mediator._scalar_mediator_spectra \
    import ScalarMediatorSpectra
from hazma.scalar_mediator._scalar_mediator_widths \
    import ScalarMediatorWidths

import numpy as np


# Note that Theory must be inherited from AFTER all the other mixin classes,
# since they furnish definitions of the abstract methods in Theory.
class ScalarMediator(ScalarMediatorConstraints,
                     ScalarMediatorCrossSection,
                     ScalarMediatorFSR,
                     ScalarMediatorPositronSpectra,
                     ScalarMediatorSpectra,
                     ScalarMediatorWidths,
                     Theory):
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

    def __init__(self, mx, ms, gsxx, gsff, gsGG, gsFF, lam):
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
        lam : float
            Mass scale associated with integrating out a heavy colored or
            charged fermion leading to SGG or SFF.
        """
        self._mx = mx
        self._ms = ms
        self._gsxx = gsxx
        self._gsff = gsff
        self._gsGG = gsGG
        self._gsFF = gsFF
        self._lam = lam
        self.compute_vs()
        self.compute_width_s()  # vs MUST be computed first

    def description(self):
        """
        Returns a string giving the details of the model.
        """
        return '''
        The UV complete models are: \n \n

        \t 1) Scalar mediator coupling to a new heavy quark. When the heavy \n
        \t    quark is integrated out of the theory, the scalar obtains an \n
        \t    effective coupling to gluons, leading to a coupling to pions \n
        \t    through a dialation current. \n \n

        \t 2) Scalar mediator mixing with the standard model higgs. The \n
        \t    scalar mediator obtains couplings to the massive standard \n
        \t    model states which will be `sin(theta) m / v_h` where theta \n
        \t    is the mixing angle between the higgs and the scalar, m is the \n
        \t    mass of the massive state and v_h is the higgs vev.  The \n
        \t    scalar mediator also gets an effective coupling to gluons when \n
        \t    the top quark is integrated out. \n

        Attributes \n
        ---------- \n
        mx : float \n
        \t Mass of the initial state fermion. \n
        ms : float \n
        \t Mass of the scalar mediator. \n
        gsxx : float \n
        \t Coupling of scalar mediator to initial state fermions. \n
        gsff : float \n
        \t Coupling of scalar mediator to standard model fermions. This is \n
        \t the sine of the mixing angle between scalar mediator and the
        \t higgs. \n
        gsGG : float \n
        \t Coupling of the scalar mediator to gluons. \n
        gsFF : float \n
        \t Coupling of the scalar mediator to photons. \n
        Lam : float \n
        \t Mass scale associated with integrating out a heavy colored or
        charged fermion leading to SGG or SFF. \n

        Methods \n
        ------- \n
        list_final_states : \n
        \t Return a list of the available final states. \n
        cross_sections : \n
        \t Computes the all the cross sections of the theory and returns \n
        \t a dictionary containing the cross sections. \n
        branching_fractions : \n
        \t Computes the all the branching fractions of the theory and \n
        \t returns a dictionary containing the branching fractions. \n
        spectra : \n
        \t Computes all spectra of the theory for a pair of initial \n
        \t state fermions annihilating into each available final state \n
        \t and returns a dictionary of arrays containing the spectra. \n
        spectrum_functions :
        \t Returns a dictionary of all the avaiable spectrum functions for \n
        \t a pair of initial state fermions with mass `mx` \n
        \t annihilating into each available final state. \n
        partial_widths : \n
        \t Returns a dictionary for the partial decay widths of the scalar \n
        \t mediator. \n
        '''

    @property
    def mx(self):
        return self._mx

    @mx.setter
    def mx(self, mx):
        self._mx = mx
        self.compute_vs()
        self.compute_width_s()  # vs MUST be computed first

    @property
    def ms(self):
        return self._ms

    @ms.setter
    def ms(self, ms):
        self._ms = ms
        self.compute_vs()
        self.compute_width_s()  # vs MUST be computed first

    @property
    def gsxx(self):
        return self._gsxx

    @gsxx.setter
    def gsxx(self, gsxx):
        self._gsxx = gsxx
        self.compute_vs()
        self.compute_width_s()  # vs MUST be computed first

    @property
    def gsff(self):
        return self._gsff

    @gsff.setter
    def gsff(self, gsff):
        self._gsff = gsff
        self.compute_vs()
        self.compute_width_s()  # vs MUST be computed first

    @property
    def gsGG(self):
        return self._gsGG

    @gsGG.setter
    def gsGG(self, gsGG):
        self._gsGG = gsGG
        self.compute_vs()
        self.compute_width_s()  # vs MUST be computed first

    @property
    def gsFF(self):
        return self._gsFF

    @gsFF.setter
    def gsFF(self, gsFF):
        self._gsFF = gsFF
        self.compute_vs()
        self.compute_width_s()  # vs MUST be computed first

    @property
    def lam(self):
        return self._lam

    @lam.setter
    def lam(self, lam):
        self._lam = lam
        self.compute_vs()
        self.compute_width_s()  # vs MUST be computed first

    def compute_vs(self):
        """Updates and returns the value of the scalar vev.
        """
        if 3 * self.gsff + 2 * self.gsGG == 0:
            self.vs = 0.
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
            self.vs = 0

            return self.vs

    def compute_width_s(self):
        """Updates the scalar's total width.
        """
        self.width_s = self.partial_widths()["total"]

    # #################### #
    """ HELPER FUNCTIONS """
    # #################### #

    def fpiT(self, vs):
        """Returns the Lagrangian parameter fpiT.
        """
        return fpi / np.sqrt(1. + 4. * self._gsGG * vs / (9. * vh))

    def b0T(self, vs, fpiT):
        """Returns the Lagrangian parameter b0T.
        """
        return (b0 * (fpi / fpiT)**2 /
                (1. + vs / vh * (2. * self._gsGG / 3. + self._gsff)))

    def msT(self, fpiT, b0T):
        """Returns the Lagrangian parameter msT.
        """
        trM = muq + mdq + msq

        return np.sqrt(self._ms**2 -
                       16. * self._gsGG * b0T * fpiT**2 / (81. * vh**2) *
                       (2. * self._gsGG - 9. * self._gsff) * trM)

    @classmethod
    def list_annihilation_final_states(cls):
        """
        Return a list of the available final states.

        Returns
        -------
        fs : array-like
            Array of the available final states.
        """
        return ['mu mu', 'e e', 'g g', 'pi0 pi0', 'pi pi', 's s']


class HiggsPortal(ScalarMediator):
    r"""
    Create a ScalarMediator object with Higgs Portal couplings.

    Creates an object for the scalar mediator model with the following
    specific coupling definitions:
        gsff = sin(theta)
        gsGG = 3 sin(theta)
        gsFF = -5/6 sin(theta)
        Lam = vh
    where theta is the mixing angle between the Standard Model Higgs
    and the scalar mediator.

    Attributes
    ----------
    mx : float
        Mass of the initial state fermion.
    ms : float
        Mass of the scalar mediator.
    gsxx : float
        Coupling of scalar mediator to initial state fermions.
    stheta : float
        Sine of the mixing angle between the Standard Model Higgs
        and the scalar mediator.
    """

    def __init__(self, mx, ms, gsxx, stheta):
        self._lam = vh
        self._stheta = stheta

        super(HiggsPortal, self).__init__(mx, ms, gsxx, stheta, 3.*stheta,
                                          -5.*stheta/6., vh)

    @property
    def stheta(self):
        return self._stheta

    @stheta.setter
    def stheta(self, stheta):
        self._stheta = stheta
        self._gsff = stheta
        self._gsGG = 3. * stheta
        self._gsFF = - 5. * stheta / 6.
        self.compute_vs()
        self.compute_width_s()  # vs MUST be computed first

    # Hide underlying properties' setters
    @ScalarMediator.gsff.setter
    def gsff(self, gsff):
        raise AttributeError("Cannot set gsff")

    @ScalarMediator.gsGG.setter
    def gsGG(self, gsGG):
        raise AttributeError("Cannot set gsGG")

    @ScalarMediator.gsFF.setter
    def gsFF(self, gsFF):
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

    Attributes
    ----------
    mx : float
        Mass of the initial state fermion.
    ms : float
        Mass of the scalar mediator.
    gsxx : float
        Coupling of scalar mediator to initial state fermions.
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

        super(HeavyQuark, self).__init__(mx, ms, gsxx, 0., gsQ,
                                         2.*gsQ*QQ**2, mQ)

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
    def gsff(self, gsff):
        raise AttributeError("Cannot set gsff")

    @ScalarMediator.gsGG.setter
    def gsGG(self, gsGG):
        raise AttributeError("Cannot set gsGG")

    @ScalarMediator.gsFF.setter
    def gsFF(self, gsFF):
        raise AttributeError("Cannot set gsFF")
