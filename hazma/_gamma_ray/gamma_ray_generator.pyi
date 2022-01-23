from typing import Callable
from numpy.typing import NDArray
import numpy as np

def gamma_point(
    particles: NDArray[np.str_],
    cme: float,
    eng_gam: float,
    mat_elem_sqrd: Callable[[NDArray[np.float64]], float] = ...,
    num_ps_pts: int = ...,
    num_bins: int = ...,
) -> float: ...
def gamma(
    particles: NDArray[np.str_],
    cme: float,
    eng_gams: NDArray[np.float64],
    mat_elem_sqrd: Callable[[NDArray[np.float64]], float] = ...,
    num_ps_pts: int = ...,
    num_bins: int = ...,
    verbose: bool = ...,
) -> NDArray[np.float64]: ...
