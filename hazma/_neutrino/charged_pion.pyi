from typing import overload
from numpy.typing import NDArray
import numpy as np

@overload
def charged_pion_neutrino_spectrum(enu: float, epi: float) -> float: ...
@overload
def charged_pion_neutrino_spectrum(
    enu: NDArray[np.float64], epi: float
) -> NDArray[np.float64]: ...
