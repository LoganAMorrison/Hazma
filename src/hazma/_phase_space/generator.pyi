from typing import List, overload
from numpy.typing import NDArray
import numpy as np

@overload
def generate_point(
    masses: List[float], cme: float, num_fsp: int
) -> NDArray[np.float64]: ...
@overload
def generate_point(
    masses: NDArray[np.float64], cme: float, num_fsp: int
) -> NDArray[np.float64]: ...
@overload
def generate_space(
    num_ps_pts: int, masses: List[float], cme: float, num_fsp: int
) -> NDArray[np.float64]: ...
@overload
def generate_space(
    num_ps_pts: int, masses: NDArray[np.float64], cme: float, num_fsp: int
) -> NDArray[np.float64]: ...
