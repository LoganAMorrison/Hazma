from typing import overload
from numpy.typing import NDArray
import numpy as np

@overload
def neutral_rho_decay_spectrum(egam: float, emu: float) -> float: ...
@overload
def neutral_rho_decay_spectrum(
    egam: NDArray[np.float64], emu: float
) -> NDArray[np.float64]: ...
@overload
def charged_rho_decay_spectrum(egam: float, emu: float) -> float: ...
@overload
def charged_rho_decay_spectrum(
    egam: NDArray[np.float64], emu: float
) -> NDArray[np.float64]: ...
