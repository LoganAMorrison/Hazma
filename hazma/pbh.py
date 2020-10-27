import numpy as np
import pandas as pd
from pkg_resources import resource_filename
from scipy.interpolate import interp1d

from hazma.parameters import g_to_MeV
from hazma.theory import TheoryDec


class PBH(TheoryDec):
    """
    A creative implementation of PBH dark matter as a `TheoryDec`.
    """

    def __init__(
        self, mx, f_pbh_dummy=1, spectrum_kind="secondary", bh_secondary=False
    ):
        """
        :param mx: PBH mass in MeV
        """
        self.f_pbh_dummy = f_pbh_dummy

        # Must call in this order
        self._spectrum_kind = spectrum_kind
        self._bh_secondary = bh_secondary
        self._load_spec_data()
        self.mx = mx

    def _load_spec_data(self):
        """
        Load spectrum data tables
        """
        if self.spectrum_kind == "primary":
            fname = resource_filename(__name__, "pbh_data/pbh_primary_spectra_bh.csv")
        elif self.spectrum_kind == "secondary" and self.bh_secondary:
            fname = resource_filename(__name__, "pbh_data/pbh_secondary_spectra_bh.csv")
        elif self.spectrum_kind == "secondary":
            fname = resource_filename(__name__, "pbh_data/pbh_secondary_spectra.csv")
        else:
            raise ValueError("invalid spectrum_kind")

        def to_float(s):
            try:
                return float(s)
            except ValueError:
                sig, exponent = s.split("e")
                return float(sig) * 10 ** float(exponent)

        df = pd.read_csv(fname)
        self._mxs = df.columns[2:].map(to_float).values * g_to_MeV
        self._e_gams = df.iloc[:, 1].values * 1e3  # GeV -> MeV
        self._d2n_dedt = df.iloc[:, 2:].values * 1e-3  # 1/GeV -> 1/MeV

    @property
    def bh_secondary(self):
        return self._bh_secondary

    @bh_secondary.setter
    def bh_secondary(self, bhs):
        raise RuntimeError("cannot set bh_secondary")

    @property
    def spectrum_kind(self):
        return self._spectrum_kind

    @spectrum_kind.setter
    def spectrum_kind(self, sk):
        raise RuntimeError("cannot set spectrum_kind")

    @property
    def mx(self):
        return self._mx

    @mx.setter
    def mx(self, mx):
        self._mx = mx

        idx = np.where(self._mxs == mx)[0][0]
        fn = interp1d(
            self._e_gams, self._d2n_dedt[:, idx], bounds_error=False, fill_value=0
        )

        self._spectrum_funcs = lambda: {"all": fn}

    @staticmethod
    def list_final_decay_states():
        return ["all"]

    def _decay_widths(self):
        return {"all": self.f_pbh_dummy}

    def _gamma_ray_line_energies(self):
        return {}
