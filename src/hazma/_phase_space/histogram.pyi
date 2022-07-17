from numpy.typing import NDArray
import numpy as np

def space_to_energy_hist(
    pts: NDArray[np.float64],
    num_ps_pts: int,
    num_fsp: int,
    num_bins: int,
    density: bool = ...,
): ...
