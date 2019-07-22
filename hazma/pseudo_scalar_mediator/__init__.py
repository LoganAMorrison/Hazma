from hazma.theory import Theory

from hazma.parameters import electron_mass as me
from hazma.parameters import muon_mass as mmu
from hazma.parameters import up_quark_mass as muq
from hazma.parameters import down_quark_mass as mdq
from hazma.parameters import strange_quark_mass as msq
from hazma.parameters import vh

from hazma.parameters import b0, fpi
from hazma.parameters import neutral_pion_mass as mpi0

from hazma.pseudo_scalar_mediator._pseudo_scalar_mediator_cross_sections import (
    PseudoScalarMediatorCrossSections,
)
from hazma.pseudo_scalar_mediator._pseudo_scalar_mediator_fsr import (
    PseudoScalarMediatorFSR,
)
from hazma.pseudo_scalar_mediator._pseudo_scalar_mediator_msqrd_rambo import (
    PseudoScalarMediatorMSqrdRambo,
)
from hazma.pseudo_scalar_mediator._pseudo_scalar_mediator_positron_spectra import (
    PseudoScalarMediatorPositronSpectra,
)
from hazma.pseudo_scalar_mediator._pseudo_scalar_mediator_spectra import (
    PseudoScalarMediatorSpectra,
)
from hazma.pseudo_scalar_mediator._pseudo_scalar_mediator_widths import (
    PseudoScalarMediatorWidths,
)

import warnings
from hazma.hazma_errors import PreAlphaWarning

import numpy as np


# Note that Theory must be inherited from AFTER all the other mixin classes,
# since they furnish definitions of the abstract methods in Theory.
class PseudoScalarMediator(
    PseudoScalarMediatorCrossSections,
    PseudoScalarMediatorFSR,
    PseudoScalarMediatorMSqrdRambo,
    PseudoScalarMediatorPositronSpectra,
    PseudoScalarMediatorSpectra,
    PseudoScalarMediatorWidths,
    Theory,
):
    r"""
    Create a pseudoscalar mediator model object.
    """

    def __init__(self, mx, mp, gpxx, gpuu, gpdd, gpss, gpee, gpmumu, gpGG, gpFF):
        self._mx = mx
        self._mp = mp
        self._gpxx = gpxx
        self._gpuu = gpuu
        self._gpdd = gpdd
        self._gpss = gpss
        self._gpee = gpee
        self._gpmumu = gpmumu
        self._gpGG = gpGG
        self._gpFF = gpFF

        self.determine_mixing()  # must be called BEFORE computing P's width!
        self.compute_width_p()

    @property
    def mx(self):
        return self._mx

    @mx.setter
    def mx(self, mx):
        self._mx = mx
        self.compute_width_p()

    @property
    def mp(self):
        return self._mp

    @mp.setter
    def mp(self, mp):
        self._mp = mp
        self.determine_mixing()
        self.compute_width_p()

    @property
    def gpxx(self):
        return self._gpxx

    @gpxx.setter
    def gpxx(self, gpxx):
        self._gpxx = gpxx
        self.compute_width_p()

    @property
    def gpee(self):
        return self._gpee

    @gpee.setter
    def gpee(self, gpee):
        self._gpee = gpee
        self.compute_width_p()

    @property
    def gpmumu(self):
        return self._gpmumu

    @gpmumu.setter
    def gpmumu(self, gpmumu):
        self._gpmumu = gpmumu
        self.compute_width_p()

    @property
    def gpuu(self):
        return self._gpuu

    @gpuu.setter
    def gpuu(self, gpuu):
        self._gpuu = gpuu
        self.determine_mixing()
        self.compute_width_p()

    @property
    def gpdd(self):
        return self._gpdd

    @gpdd.setter
    def gpdd(self, gpdd):
        self._gpdd = gpdd
        self.determine_mixing()
        self.compute_width_p()

    @property
    def gpss(self):
        return self._gpss

    @gpss.setter
    def gpss(self, gpss):
        self._gpss = gpss
        # This impacts P-eta mixing, but not P-pi0 mixing. The eta is above the
        # mass range we are interested in.
        self.compute_width_p()

    @property
    def gpGG(self):
        return self._gpGG

    @gpGG.setter
    def gpGG(self, gpGG):
        self._gpGG = gpGG
        self.determine_mixing()
        self.compute_width_p()

    @property
    def gpFF(self):
        return self._gpFF

    @gpFF.setter
    def gpFF(self, gpFF):
        self._gpFF = gpFF
        self.compute_width_p()

    def determine_mixing(self):
        """Recomputes the mixing between P and pi0 and computes the resulting
        shift in mpi0.

        Notes
        -----
        * Depends on gpuu, gpdd, gpGG and mp.
        * Must be called before calling compute_width_p().
        * Resets the attributes `beta` (the mixing angle) and `mpi0` (the
          shifted neutral pion mass). Warnings are printed if the neutral pion
          mass shift is larger than 10 MeV, with a specific warning if the mass
          is shifted to be imaginary.
        """
        eps = b0 * fpi * (self.gpuu - self.gpdd + (muq - mdq) / vh * self.gpGG)

        # Mixing angle between pi0 and p. Here I have assumed that the pi0 mass
        # is given by leading order chiPT.
        self.beta = eps / (self.mp ** 2 - mpi0 ** 2)

        # Shifted mass of neutral pion
        mpi0Sqrd = mpi0 ** 2 - eps * self.beta

        if mpi0Sqrd < 0:  # mixing is way too big if this fails
            print(
                "Warning: your choice of mp and/or couplings produced an"
                + " imaginary neutral pion mass. Undefined behavior."
            )

        self.mpi0 = np.sqrt(mpi0Sqrd)

        if abs(self.mpi0 - mpi0) > 10.0:
            print(
                "Warning: your choice of mp and/or couplings produced a "
                + "10 MeV or larger shift in m_pi0. Theory is invalid."
            )

    def compute_width_p(self):
        """Updates the pseudoscalar's total width.
        """
        self.width_p = self.partial_widths()["total"]

    def description(self):
        warnings.warn("", PreAlphaWarning)
        pass

    @classmethod
    def list_annihilation_final_states(cls):
        """
        Return a list of the available final states.

        Returns
        -------
        fs : array-like
            Array of the available final states.
        """
        return ["e e", "mu mu", "g g", "pi0 pi pi", "pi0 pi0 pi0", "p p"]

    def constraints(self):
        pass


class PseudoScalarMFV(PseudoScalarMediator):
    """MFV version of the pseudoscalar model.
    """

    def __init__(self, mx, mp, gpxx, gpup, gpdown, gpll):
        self._gpup = gpup
        self._gpdown = gpdown
        self._gpll = gpll

        yu = muq / vh
        yd = mdq / vh
        ys = msq / vh

        ye = me / vh
        ymu = mmu / vh

        gpGG = 2.0 * gpup + gpdown
        gpFF = gpll + (8.0 * gpup + gpdown) / 9.0

        super(PseudoScalarMFV, self).__init__(
            mx,
            mp,
            gpxx,
            gpup * yu,
            gpdown * yd,
            gpdown * ys,
            gpll * ye,
            gpll * ymu,
            gpGG,
            gpFF,
        )

    @property
    def gpll(self):
        return self._gpll

    @gpll.setter
    def gpll(self, gpll):
        self._gpll = gpll

        ye = me / vh
        ymu = mmu / vh

        self.gpee = gpll * ye
        self.gpmumu = gpll * ymu

        self.gpFF = gpll + (8.0 * self.gpup + self.gpdown) / 9.0

    @property
    def gpup(self):
        return self._gpup

    @gpup.setter
    def gpup(self, gpup):
        self._gpup = gpup

        yu = muq / vh

        self.gpuu = gpup * yu

        self.gpFF = self.gpll + (8.0 * gpup + self.gpdown) / 9.0
        self.gpGG = 2.0 * gpup + self.gpdown

    @property
    def gpdown(self):
        return self._gpdown

    @gpdown.setter
    def gpdown(self, gpdown):
        self._gpdown = gpdown

        yd = mdq / vh
        ys = msq / vh

        self.gpdd = gpdown * yd
        self.gpss = gpdown * ys

        self.gpFF = self.gpll + (8.0 * self.gpup + gpdown) / 9.0
        self.gpGG = 2.0 * self.gpup + gpdown

    def f_eff_g(self, x_kd=1e-4):
        # Need to override the default method since the pseudoscalar uses RAMBO
        # for its spectra.
        return self._f_eff_helper("g g", x_kd, "interp")

    def f_eff_ep(self, x_kd=1e-4):
        # Need to override the default method since the pseudoscalar uses RAMBO
        # for its spectra.
        return self._f_eff_helper("e e", x_kd, "interp")
