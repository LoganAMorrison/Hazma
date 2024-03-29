"""Types for the GeV vector mediator model."""

from typing import Literal, TypedDict

FINAL_STATES = (
    "e e",
    "mu mu",
    "ve ve",
    "vm vm",
    "vt vt",
    "x x",
    "pi pi",
    "k0 k0",
    "k k",
    "pi0 gamma",
    "eta gamma",
    "pi0 phi",
    "eta phi",
    "eta omega",
    "pi0 pi0 gamma",
    "pi pi pi0",
    "pi pi eta",
    "pi pi etap",
    "pi pi omega",
    "pi0 pi0 omega",
    "pi0 k0 k0",
    "pi0 k k",
    "pi k k0",
    "pi pi pi pi",
    "pi pi pi0 pi0",
)

DecayFinalStates = tuple[
    Literal["e e"],
    Literal["mu mu"],
    Literal["ve ve"],
    Literal["vm vm"],
    Literal["vt vt"],
    Literal["x x"],
    Literal["pi pi"],
    Literal["k0 k0"],
    Literal["k k"],
    Literal["pi0 gamma"],
    Literal["eta gamma"],
    Literal["pi0 phi"],
    Literal["eta phi"],
    Literal["eta omega"],
    Literal["pi0 pi0 gamma"],
    Literal["pi pi pi0"],
    Literal["pi pi eta"],
    Literal["pi pi etap"],
    Literal["pi pi omega"],
    Literal["pi0 pi0 omega"],
    Literal["pi0 k0 k0"],
    Literal["pi0 k k"],
    Literal["pi k k0"],
    Literal["pi pi pi pi"],
    Literal["pi pi pi0 pi0"],
]


PartialWidthsDict = TypedDict(
    "PartialWidthsDict",
    {
        "e e": float,
        "mu mu": float,
        "ve ve": float,
        "vm vm": float,
        "vt vt": float,
        "x x": float,
        "pi pi": float,
        "k0 k0": float,
        "k k": float,
        "pi0 gamma": float,
        "eta gamma": float,
        "pi0 phi": float,
        "eta phi": float,
        "eta omega": float,
        "pi0 pi0 gamma": float,
        "pi pi pi0": float,
        "pi pi eta": float,
        "pi pi etap": float,
        "pi pi omega": float,
        "pi0 pi0 omega": float,
        "pi0 k0 k0": float,
        "pi0 k k": float,
        "pi k k0": float,
        "pi pi pi pi": float,
        "pi pi pi0 pi0": float,
    },
)
