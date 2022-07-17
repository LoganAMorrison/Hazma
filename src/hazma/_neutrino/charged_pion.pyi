from typing import overload, Tuple
from numpy.typing import NDArray
import numpy as np

@overload
def charged_pion_neutrino_spectrum(
    enu: float, epi: float
) -> Tuple[float, float, float]: ...
@overload
def charged_pion_neutrino_spectrum(
    enu: NDArray[np.float64], epi: float
) -> NDArray[np.float64]: ...
