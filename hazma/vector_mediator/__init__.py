from typing import List

from hazma.parameters import Qd, Qe, Qu, qe
from hazma.theory import TheoryAnn
from hazma.vector_mediator._vector_mediator_cross_sections import (
    VectorMediatorCrossSections,
)
from hazma.vector_mediator._vector_mediator_fsr import VectorMediatorFSR
from hazma.vector_mediator._vector_mediator_positron_spectra import (
    VectorMediatorPositronSpectra,
)
from hazma.vector_mediator._vector_mediator_spectra import VectorMediatorSpectra
from hazma.vector_mediator._vector_mediator_widths import VectorMediatorWidths

# flake8: noqa: F401
from ._gev.model import BLGeV, KineticMixingGeV, VectorMediatorGeV  # pyright: ignore


# Note that Theory must be inherited from AFTER all the other mixin classes,
# since they furnish definitions of the abstract methods in Theory.
class VectorMediator(
    VectorMediatorCrossSections,
    VectorMediatorFSR,
    VectorMediatorPositronSpectra,
    VectorMediatorSpectra,
    VectorMediatorWidths,
    TheoryAnn,
):
    r"""
    Create a VectorMediator object with generic couplings.

    Parameters
    ----------
    mx : float
        Mass of the dark matter.
    mv : float
        Mass of the vector mediator.
    gvxx : float
        Coupling of vector mediator to dark matter.
    gvuu : float
        Coupling of vector mediator to the up quark.
    gvdd : float
        Coupling of vector mediator to the down quark.
    gvss : float
        Coupling of vector mediator to the strange quark.
    gvee : float
        Coupling of vector mediator to the electron.
    gvmumu : float
        Coupling of vector mediator to the muon.
    """

    def __init__(
        self,
        mx: float,
        mv: float,
        gvxx: float,
        gvuu: float,
        gvdd: float,
        gvss: float,
        gvee: float,
        gvmumu: float,
    ) -> None:
        self._mx: float = mx
        self._mv: float = mv
        self._gvxx: float = gvxx
        self._gvuu: float = gvuu
        self._gvdd: float = gvdd
        self._gvss: float = gvss
        self._gvee: float = gvee
        self._gvmumu: float = gvmumu
        self.width_v: float = 0.0
        self.compute_width_v()

    def __repr__(self) -> str:
        return (
            f"VectorMediator(\n"
            f"\tmx={self.mx} MeV,\n"
            f"\tmv={self.mv} MeV,\n"
            f"\tgvxx={self.gvxx},\n"
            f"\tgvuu={self.gvuu},\n"
            f"\tgvdd={self.gvdd},\n"
            f"\tgvss={self.gvss},\n"
            f"\tgvee={self.gvee},\n"
            f"\tgvmumu={self.gvmumu}\n"
            ")"
        )

    @property
    def mx(self) -> float:
        return self._mx

    @mx.setter
    def mx(self, mx: float) -> None:
        self._mx = mx
        self.compute_width_v()

    @property
    def mv(self) -> float:
        return self._mv

    @mv.setter
    def mv(self, mv: float) -> None:
        self._mv = mv
        self.compute_width_v()

    @property
    def gvxx(self) -> float:
        return self._gvxx

    @gvxx.setter
    def gvxx(self, gvxx: float) -> None:
        self._gvxx = gvxx
        self.compute_width_v()

    @property
    def gvuu(self) -> float:
        return self._gvuu

    @gvuu.setter
    def gvuu(self, gvuu: float) -> None:
        self._gvuu = gvuu
        self.compute_width_v()

    @property
    def gvdd(self) -> float:
        return self._gvdd

    @gvdd.setter
    def gvdd(self, gvdd: float) -> None:
        self._gvdd = gvdd
        self.compute_width_v()

    @property
    def gvss(self) -> float:
        return self._gvss

    @gvss.setter
    def gvss(self, gvss: float) -> None:
        self._gvss = gvss
        self.compute_width_v()

    @property
    def gvee(self) -> float:
        return self._gvee

    @gvee.setter
    def gvee(self, gvee: float) -> None:
        self._gvee = gvee
        self.compute_width_v()

    @property
    def gvmumu(self) -> float:
        return self._gvmumu

    @gvmumu.setter
    def gvmumu(self, gvmumu: float) -> None:
        self._gvmumu = gvmumu
        self.compute_width_v()

    def compute_width_v(self) -> None:
        """Recomputes the scalar's total width."""
        self.width_v = self.partial_widths()["total"]

    @staticmethod
    def list_annihilation_final_states() -> List[str]:
        """
        Return a list of the available final states.

        Returns
        -------
        fs : array-like
            Array of the available final states.
        """
        return ["mu mu", "e e", "pi pi", "pi0 g", "pi0 v", "v v"]

    def constraints(self):
        pass

    def constrain(self, p1, p1_vals, p2, p2_vals, ls_or_img="image"):
        pass


class KineticMixing(VectorMediator):
    r"""
    Create a ``VectorMediator`` object with kinetic mixing couplings.

    The couplings are defined as::

        gvuu = Qu qe eps
        gvdd = Qd qe eps
        gvss = Qd qe eps
        gvee = Qe qe eps
        gvmumu = Qe qe eps

    where Qu, Qd and Qe are the up-type quark, down-type quark and
    lepton electic charges in units of the electric charge, qe is the
    electric charge and eps is the kinetic mixing parameter.

    Parameters
    ----------
    mx : float
        Mass of the dark matter.
    mv : float
        Mass of the vector mediator.
    gvxx : float
        Coupling of vector mediator to dark matter.
    eps : float
        Kinetic mixing parameter.
    """

    def __init__(self, mx: float, mv: float, gvxx: float, eps: float) -> None:
        self._eps = eps

        super(KineticMixing, self).__init__(
            mx,
            mv,
            gvxx,
            -Qu * eps * qe,
            -Qd * eps * qe,
            -Qd * eps * qe,
            -Qe * eps * qe,
            -Qe * eps * qe,
        )

    def __repr__(self) -> str:
        repr_ = "KineticMixing("
        repr_ += f"mx={self.mx} [MeV], "
        repr_ += f"mv={self.mv} [MeV], "
        repr_ += f"gvxx={self.gvxx}, "
        repr_ += f"eps={self.eps}"
        repr_ += ")"
        return repr_

    @property
    def eps(self) -> float:
        return self._eps

    @eps.setter
    def eps(self, eps: float):
        self._eps = eps
        self._gvuu = -Qu * eps * qe
        self._gvdd = -Qd * eps * qe
        self._gvss = -Qd * eps * qe
        self._gvee = -Qe * eps * qe
        self._gvmumu = -Qe * eps * qe
        self.compute_width_v()

    # Hide underlying properties' setters
    @VectorMediator.gvuu.setter
    def gvuu(self, _: float):
        raise AttributeError("Cannot set gvuu")

    @VectorMediator.gvdd.setter
    def gvdd(self, _: float):
        raise AttributeError("Cannot set gvdd")

    @VectorMediator.gvss.setter
    def gvss(self, _: float):
        raise AttributeError("Cannot set gvss")

    @VectorMediator.gvee.setter
    def gvee(self, _: float):
        raise AttributeError("Cannot set gvee")

    @VectorMediator.gvmumu.setter
    def gvmumu(self, _: float):
        raise AttributeError("Cannot set gvmumu")


class QuarksOnly(VectorMediator):
    r"""
    Create a VectorMediator object with only quark couplings.

    Parameters
    ----------
    mx : float
        Mass of the dark matter.
    mv : float
        Mass of the vector mediator.
    gvxx : float
        Coupling of vector mediator to dark matter.
    gvuu : float
        Coupling of vector mediator to the up quark.
    gvdd : float
        Coupling of vector mediator to the down quark.
    gvss : float
        Coupling of vector mediator to the strange quark.
    """

    def __init__(
        self, mx: float, mv: float, gvxx: float, gvuu: float, gvdd: float, gvss: float
    ) -> None:
        super(QuarksOnly, self).__init__(mx, mv, gvxx, gvuu, gvdd, gvss, 0.0, 0.0)

    def __repr__(self) -> str:
        repr_ = "QuarksOnly("
        repr_ += f"mx={self.mx} [MeV], "
        repr_ += f"mv={self.mv} [MeV], "
        repr_ += f"gvxx={self.gvxx}, "
        repr_ += f"gvuu={self.gvuu}"
        repr_ += f"gvdd={self.gvdd}"
        repr_ += f"gvss={self.gvss}"
        repr_ += ")"

        return repr_

    @staticmethod
    def list_annihilation_final_states() -> List[str]:
        return ["pi pi", "pi0 g", "pi0 v", "v v"]

    # Hide underlying properties' setters
    @VectorMediator.gvee.setter
    def gvee(self, _: float) -> AttributeError:
        raise AttributeError("Cannot set gvee")

    @VectorMediator.gvmumu.setter
    def gvmumu(self, _: float) -> AttributeError:
        raise AttributeError("Cannot set gvmumu")
