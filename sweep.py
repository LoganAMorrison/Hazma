"""Usage: sweep.py {site,ref} out.json — evaluates <sigma v>(x) and Omega h^2(ms)."""
import json, sys, warnings
from itertools import pairwise
from multiprocessing import Pool
import numpy as np
from scipy.integrate import quad
from scipy.special import kn
warnings.filterwarnings("ignore")
from hazma.relic_density import relic_density
from hazma.relic_density._thermal_functions import thermal_cross_section, thermal_cross_section_integrand
from hazma.scalar_mediator import HiggsPortal

MODE = sys.argv[1] if __name__ == "__main__" else "site"
MX = 200.0
MODELS = {"narrow": dict(gsxx=1e-2, stheta=1e-3), "wide": dict(gsxx=1.0, stheta=1e-4)}
XS = np.geomspace(0.1, 300, 241)
MS = np.linspace(380.0, 700.0, 81)

class Generic:
    """A model with no thermal_cross_section of its own."""
    def __init__(self, inner):
        self.inner, self.mx = inner, inner.mx
    def annihilation_cross_sections(self, e_cm):
        return self.inner.annihilation_cross_sections(e_cm)
    def annihilation_resonances(self):
        return self.inner.annihilation_resonances()

def reference(x, inner):
    z_res, g = inner.ms / inner.mx, inner.width_s / inner.mx
    feats = [z_res + s * g * 2.0**k for s in (-1, 1) for k in range(80)] + [2 * z_res]
    decay = [2.0 + k / x for k in (0.0, 1.0, 4.0, 16.0, 50.0, 100.0, 200.0)]
    edges = sorted(decay + [z for z in feats if decay[0] + 1e-9 < z < decay[-1]])
    m = Generic(inner)
    val = sum(quad(thermal_cross_section_integrand, a, b, args=(x, m), epsabs=0.0, epsrel=1e-12, limit=200)[0] for a, b in pairwise(edges))
    return x / (2.0 * kn(2, x)) ** 2 * val

class Reference(Generic):
    def thermal_cross_section(self, x):
        return 0.0 if x > 300 else reference(x, self.inner)

def sv(args):
    name, x = args
    inner = HiggsPortal(mx=MX, ms=550.0, **MODELS[name])
    return reference(x, inner) if MODE == "ref" else thermal_cross_section(x, Generic(inner))

def omega(args):
    name, ms = args
    inner = HiggsPortal(mx=MX, ms=ms, **MODELS[name])
    m = Reference(inner) if MODE == "ref" else Generic(inner)
    return float(relic_density(m, semi_analytic=True))

if __name__ == "__main__":
    out = {"x": XS.tolist(), "ms": MS.tolist()}
    with Pool(8) as p:
        for name in MODELS:
            out[f"sv_{name}"] = p.map(sv, [(name, x) for x in XS])
            out[f"omega_{name}"] = p.map(omega, [(name, m) for m in MS])
    json.dump(out, open(sys.argv[2], "w"))
