"""
The vector form factor library.

This module contains the implementation of 20 different vector form factors.

π⁺π⁻:     VectorFormFactorPiPi
π⁰π⁰:     VectorFormFactorPi0Pi0
K⁺K:      VectorFormFactorKK
K⁰K̅⁰:     VectorFormFactorK0K0
π⁰Ɣ:      VectorFormFactorPi0Gamma
π⁰ω:      VectorFormFactorPi0Omega
π⁰ɸ:      VectorFormFactorPi0Phi
ηƔ:       VectorFormFactorEtaGamma
ηω:       VectorFormFactorEtaOmega
ηɸ:       VectorFormFactorEtaPhi
π⁰K⁰K̅⁰:   VectorFormFactorPi0K0K0
π⁰K⁺K⁻:   VectorFormFactorPi0KpKm
π⁺K⁺K⁰:   VectorFormFactorPiKK0
ηπ⁺π⁻:    VectorFormFactorPiPiEta
η'π⁺π⁻:   VectorFormFactorPiPiEtaPrime
ωπ⁺π⁻:    VectorFormFactorPiPiOmega
ωπ⁰π⁰:    VectorFormFactorPi0Pi0Omega
π⁺π⁻π⁰:   VectorFormFactorPiPiPi0
π⁺π⁻π⁰π⁰: VectorFormFactorPiPiPi0Pi0
π⁺π⁻π⁺π⁻: VectorFormFactorPiPiPiPi

"""

from ._base import VectorFormFactorCouplings
from ._pi_pi import VectorFormFactorPiPi, VectorFormFactorPi0Pi0
from ._k_k import VectorFormFactorKK, VectorFormFactorK0K0
from ._pi_gamma import VectorFormFactorPi0Gamma
from ._pi_omega import VectorFormFactorPi0Omega
from ._pi_phi import VectorFormFactorPi0Phi
from ._eta_gamma import VectorFormFactorEtaGamma
from ._eta_omega import VectorFormFactorEtaOmega
from ._eta_phi import VectorFormFactorEtaPhi
from ._pi_k_k import (
    VectorFormFactorPi0K0K0,
    VectorFormFactorPi0KpKm,
    VectorFormFactorPiKK0,
)
from ._pi_pi_eta import VectorFormFactorPiPiEta
from ._pi_pi_etap import VectorFormFactorPiPiEtaPrime
from ._pi_pi_omega import VectorFormFactorPiPiOmega, VectorFormFactorPi0Pi0Omega
from ._pi_pi_pi0 import VectorFormFactorPiPiPi0
from ._pi_pi_pi_pi import VectorFormFactorPiPiPi0Pi0, VectorFormFactorPiPiPiPi

__all__ = [
    "VectorFormFactorPiPi",
    "VectorFormFactorPi0Pi0",
    "VectorFormFactorKK",
    "VectorFormFactorK0K0",
    "VectorFormFactorPi0Gamma",
    "VectorFormFactorPi0Omega",
    "VectorFormFactorPi0Phi",
    "VectorFormFactorEtaGamma",
    "VectorFormFactorEtaOmega",
    "VectorFormFactorEtaPhi",
    "VectorFormFactorPi0K0K0",
    "VectorFormFactorPi0KpKm",
    "VectorFormFactorPiKK0",
    "VectorFormFactorPiPiEta",
    "VectorFormFactorPiPiEtaPrime",
    "VectorFormFactorPiPiOmega",
    "VectorFormFactorPi0Pi0Omega",
    "VectorFormFactorPiPiPi0",
    "VectorFormFactorPiPiPi0Pi0",
    "VectorFormFactorPiPiPiPi",
    "VectorFormFactorCouplings",
]
