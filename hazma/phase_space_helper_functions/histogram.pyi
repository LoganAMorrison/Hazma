import numpy as np
import numpy.typing as npt


def space_to_energy_hist(
    pts: npt.NDArray[np.float64],
    num_ps_pts: int,
    num_fsp: int,
    num_bins: int,
    density: bool = False,
) -> npt.NDArray[np.float64]:
    ...
