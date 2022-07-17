from typing import List, Callable, Union
from numpy.typing import NDArray
import numpy as np

def gamma_ray_fsr(
    photon_energies: Union[List[float], NDArray[np.float64]],
    cme: float,
    isp_masses: Union[List[float], NDArray[np.float64]],
    fsp_masses: Union[List[float], NDArray[np.float64]],
    non_rad: float,
    msqrd: Callable[[NDArray[np.float64]], float],
    nevents: int,
) -> NDArray[np.float64]: ...
