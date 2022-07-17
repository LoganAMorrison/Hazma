from typing import overload
from numpy.typing import NDArray
import numpy as np

@overload
def muon_decay_spectrum(egam: float, emu: float) -> float: ...
@overload
def muon_decay_spectrum(
    egam: NDArray[np.float64], emu: float
) -> NDArray[np.float64]: ...
