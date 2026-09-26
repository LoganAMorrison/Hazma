import contextlib, time
import numpy as np
import hazma.relic_density._thermal_functions as tf
import hazma.vector_mediator._gev.thermal_cross_section as gev_site
from hazma.relic_density import relic_density
from hazma.scalar_mediator import HiggsPortal
from hazma.vector_mediator import VectorMediatorGeV

class NoThermalCrossSection:
    def __init__(self, inner):
        self._inner = inner; self.mx = inner.mx
    def annihilation_cross_sections(self, e_cm):
        return self._inner.annihilation_cross_sections(e_cm)

@contextlib.contextmanager
def old_limit():
    new = tf.thermal_cross_section_upper_limit
    old = lambda x: 50.0 / x
    tf.thermal_cross_section_upper_limit = old; gev_site.thermal_cross_section_upper_limit = old
    try: yield
    finally:
        tf.thermal_cross_section_upper_limit = new; gev_site.thermal_cross_section_upper_limit = new

def gev(mx):
    return VectorMediatorGeV(mx=mx, mv=2e3, gvxx=1.0, gvuu=3.0, gvdd=1.0, gvss=-1.0,
        gvee=0.0, gvmumu=0.0, gvveve=0.0, gvvmvm=0.0, gvvtvt=0.0)
