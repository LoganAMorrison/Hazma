from typing import Callable
import numpy as np
import numpy.typing as npt


def apply_matrix_elem(
    pts: npt.NDArray[np.float64], num_ps_pts: int, num_fsp: int, mat_elem_sqrd: Callable
):
    ...
