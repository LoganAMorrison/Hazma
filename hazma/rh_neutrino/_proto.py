from typing import Protocol


class SingleRhNeutrinoModel(Protocol):
    mx: float
    theta: float
    flavor: str
