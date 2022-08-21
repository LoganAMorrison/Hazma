import numpy as np
from pkg_resources import resource_filename
from scipy.interpolate import interp1d

from hazma.parameters import g_to_MeV, MeV_to_g
from hazma.theory import TheoryDec


class PBH(TheoryDec):
    """
    A creative implementation of PBH dark matter as a `TheoryDec`.
    """

    def __init__(
        self,
        mx: float,
        f_pbh_dummy=1,
        spectrum_kind: str = "secondary",
        bh_secondary: bool = False,
    ) -> None:
        """
        Create a PBH object used to constrain the fraction PBHs make of DM.

        Parameters
        ----------
        mx: float
            PBH mass in MeV
        f_pbh_dummy: float
            Fraction of PBH in DM ?
        spectrum_kind: str
            Type of radiation spectrum used for PBH evaporation. Options are
            'primary' or 'secondary'.
        bh_secondary: bool
            If true, BlackHawk v1 secondary is used.
        """
        self.f_pbh_dummy = f_pbh_dummy

        # Must call in this order
        self._spectrum_kind = spectrum_kind
        self._bh_secondary = bh_secondary
        self._load_spec_data()
        self.mx = mx

    def __repr__(self):
        return f"PBH(m={self.mx * MeV_to_g} g, f={self.f_pbh_dummy})"

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

        def parse_float(s):
            try:
                return float(s)
            except ValueError:
                sig, exponent = s.split("e")
                return float(sig) * 10 ** float(exponent)

        def parse_line(line: str):
            # Skip first value (it's the row index)
            return list(map(float, line.split(",")[1:]))

        with open(fname, "r") as f:
            # Header has the format: ,photon_energies,1e15.0,1e15.05,1e15.1,...
            # We don't need the first to entries after split on ',', then we
            # need to parser the weird floats.
            masses = np.array(list(map(parse_float, f.readline().split(",")[2:])))
            # Read the remaining lines, which have format: i,e1,e2,e3,... where
            # 'i' is the row index and the rest are the values we want.
            data = np.array(list(map(parse_line, f.readlines())))

        self._mxs = masses * g_to_MeV
        # GeV -> MeV
        self._e_gams = data[:, 0] * 1e3
        # 1/GeV -> 1/MeV
        self._d2n_dedt = data[:, 1:] * 1e-3  # type:ignore

    @property
    def bh_secondary(self):
        return self._bh_secondary

    @bh_secondary.setter
    def bh_secondary(self, _):
        raise RuntimeError("cannot set bh_secondary")

    @property
    def spectrum_kind(self):
        return self._spectrum_kind

    @spectrum_kind.setter
    def spectrum_kind(self, _):
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
