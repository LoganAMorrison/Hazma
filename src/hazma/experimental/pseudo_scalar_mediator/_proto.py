from typing import Protocol


class PseudoScalarMediatorBase(Protocol):
    mx: float
    mp: float
    gpxx: float
    gpee: float
    gpmumu: float
    gpuu: float
    gpdd: float
    gpss: float
    gpFF: float
    gpGG: float

    beta: float
    width_p: float
    mpi0: float
