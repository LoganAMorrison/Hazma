from typing import overload
from numpy.typing import NDArray
import numpy as np

@overload
def muon_neutrino_spectrum(enu: float, emu: float) -> float: ...
@overload
def muon_neutrino_spectrum(
    enu: NDArray[np.float64], emu: float
) -> NDArray[np.float64]: ...
