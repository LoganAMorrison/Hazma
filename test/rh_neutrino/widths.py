"""
Test the widths of the RHNeutrino model.
"""

import numpy as np
import matplotlib.pyplot as plt
from hazma.rh_neutrino import RHNeutrino

if __name__ == "__main__":
    MASS = 10.0
    STHETA = 1e-3
    LEPTON = "e"

    NPTS = 300

    model = RHNeutrino(MASS, STHETA, LEPTON)

    mxs = np.logspace(np.log10(0.1), np.log10(1e3), NPTS)
    widths = {
        key: np.zeros(NPTS, dtype=np.float64) for key in model.decay_widths().keys()
    }

    for i, mx in enumerate(mxs):
        model.mx = mx
        for key, val in model.decay_widths().items():
            widths[key][i] = val

    plt.figure(dpi=150)

    for key, val in widths.items():
        plt.plot(mxs, val, label=key)

    plt.legend()
    plt.yscale("log")
    plt.xscale("log")
    plt.ylabel(r"$\Gamma_{N} \ (\mathrm{MeV})$", fontsize=16)
    plt.xlabel(r"$m_{N} \ (\mathrm{MeV})$", fontsize=16)
    plt.xlim([np.min(mxs), np.max(mxs)])
    plt.ylim([1e-40, 1e-13])

    plt.show()
