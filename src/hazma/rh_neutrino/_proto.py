from typing import Protocol
from enum import IntEnum


class Generation(IntEnum):
    Fst = 0
    Snd = 1
    Trd = 2

    def __str__(self):
        if self.value == 0:
            return "e"
        if self == 1:
            return "mu"
        return "tau"


class SingleRhNeutrinoModel(Protocol):
    mx: float
    theta: float
    gen: Generation
