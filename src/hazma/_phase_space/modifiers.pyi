from typing import Callable
from numpy.typing import NDArray
import numpy as np

def apply_matrix_elem(
    pts: NDArray[np.float64],
    num_ps_pts: int,
    num_fsp: int,
    mat_elem_sqrd: Callable[[NDArray[np.float64]], float],
) -> NDArray[np.float64]: ...
