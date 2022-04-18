from typing import overload
from numpy.typing import NDArray
import numpy as np

@overload
def charged_pion_positron_spectrum(epos: float, emu: float) -> float: ...
@overload
def charged_pion_positron_spectrum(
    epos: NDArray[np.float64], emu: float
) -> NDArray[np.float64]: ...
