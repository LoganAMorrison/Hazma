from typing import overload, Union
from numpy.typing import NDArray
import numpy as np

@overload
def neutral_pion_decay_spectrum(egam: float, epi: float) -> float: ...
@overload
def neutral_pion_decay_spectrum(
    egam: NDArray[np.float64], epi: float
) -> NDArray[np.float64]: ...
def neutral_pion_decay_spectrum(
    egam: Union[NDArray[np.float64], float], epi: float
) -> Union[NDArray[np.float64], float]: ...
