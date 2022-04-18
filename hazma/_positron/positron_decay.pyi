from typing import List, Callable, Union
from numpy.typing import NDArray
import numpy as np

def positron(
    particles: Union[List[str], NDArray[np.str_]],
    cme: float,
    eng_ps: Union[List[float], NDArray[np.float64]],
    mat_elem_sqrd: Callable[[NDArray[np.float64]], float] = ...,
    num_ps_pts: int = ...,
    num_bins: int = ...,
) -> NDArray[np.float64]: ...
