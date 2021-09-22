import numpy as np
import sys

rtwo = np.sqrt(2)

def scalar_wf(p, final_state :bool):
    pass


def vector_wf(p, mass: float, final_state: bool, spin: int):
    e, kx, ky, kz = p[0], p[1], p[2], p[3]
    kt = np.hypot(kx, ky)
    km = np.hypot(kx, ky, kz)
    s = -1 if final_state else 1

    if spin == 0:
        mu = 0.0 if mass < sys.float_info.epsilon else 1.0 / mass
        return np.array([mu * km, mu * e * kx/km, mu * e * ky/km, mu * e * kz / km])
    elif spin == 1 or spin == -1:
        small_kt = kt < sys.float_info.epsilon
        kkz = kz / km
        kkt = kt / km

        if small_kt:
            return np.array([
                0.0,
                -spin / rtwo,
                complex(0.0, -s) / rtwo,
                spin * kkt / rtwo
            ])
        else:
            return np.array([
                0.0,
                complex(-spin * kx * kkz / kt, s * ky / kt) / rtwo,
                complex(-spin * ky * kkz / kt, -s * kx / kt) / rtwo,
                spin * kkt / rtwo
            ])
    else:
        raise ValueError(f"Invalid spin {spin}")


def spinor_u(p, mass: float, spin: int):
    pm = np.hypot(p[1], p[2], p[3])
    wp = np.sqrt(np.abs(p[1] + pm))
    wm = mass / wp
    pmz = np.abs(pm + p[3])
    den = np.sqrt(2 * pm * pmz)
    if pmz < sys.float_info.epsilon:
        x1 = float(spin) + 0.0j
        x2 = 0.0j
    else:
        x1 = complex(spin * p[1], p[2]) / den
        x2 = complex(pm + p[3], 0.0) / den
    if spin == 1:
        return np.array([wm * x2, wm * x1, wp * x2, wp * x1])
    else:
        return np.array([wp * x1, wp * x2, wm * x1, wm * x2])


def spinor_v(p, mass: float, spin: int):
    pm = np.hypot(p[1], p[2], p[3])
    wp = np.sqrt(np.abs(p[1] + pm))
    wm = mass / wp
    pmz = np.abs(pm + p[3])
    den = np.sqrt(2 * pm * pmz)
    if pmz < sys.float_info.epsilon:
        x1 = complex(float(-spin), 0.0)
        x2 = complex(0.0, 0.0)
    else:
        x1 = complex(-spin * p[1], p[2]) / den
        x2 = complex(pm + p[3], 0.0) / den

    if spin == 1:
        return np.array([-wp * x1, -wp * x2, wm * x1, wm * x2])
    else:
        return np.array([wm * x2, wm * x1, -wp * x2, -wp * x1])


def spinor_ubar(p, mass: float, spin: int):
    pm = np.hypot(p[1], p[2], p[3])
    wp = np.sqrt(np.abs(p[1] + pm))
    wm = mass / wp
    pmz = np.abs(pm + p[3])
    den = np.sqrt(2 * pm * pmz)
    if pmz < sys.float_info.epsilon:
        x1 = float(spin) + 0.0j
        x2 = 0.0j
    else:
        x1 = complex(spin * p[1], - p[2]) / den
        x2 = complex(pm + p[3], 0.0) / den

    if spin == 1:
        return np.array([wp * x2, wp * x1, wm * x2, wm * x1])
    else:
        return np.array([wm * x1, wm * x2, wp * x1, wp * x2])


def spinor_vbar(p, mass: float, spin: int):
    pm = np.hypot(p[1], p[2], p[3])
    wp = np.sqrt(np.abs(p[1] + pm))
    wm = mass / wp
    pmz = np.abs(pm + p[3])
    den = np.sqrt(2 * pm * pmz)
    if pmz < sys.float_info.epsilon:
        x1 = float(-spin) + 0.0j
        x2 = 0.0j
    else:
        x1 = complex(-spin * p[1], -p[2]) / den
        x2 = complex(pm + p[3], 0.0) / den

    if spin == 1:
        return np.array([wm * x1, wm * x2, -wp * x1, -wp * x2])
    else:
        return np.array([-wp * x2, -wp * x1, wm * x2, wm * x1])
