from ._decay_muon import Muon
from ._decay_muon import Muon
import numpy as np
cimport numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from libc.math cimport exp, log, M_PI, log10, sqrt
import cython
from functools import partial
include "parameters.pxd"
