from typing import Protocol

from hazma.theory import TheoryAnn


class VectorMediatorProtocol(Protocol):
    @property
    def mx(self) -> float: ...

    @property
    def mv(self) -> float: ...

    @property
    def gvxx(self) -> float: ...

    @property
    def gvuu(self) -> float: ...

    @property
    def gvdd(self) -> float: ...

    @property
    def gvss(self) -> float: ...

    @property
    def gvee(self) -> float: ...

    @property
    def gvmumu(self) -> float: ...
