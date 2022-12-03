"""Utilities for testing form factors."""

# pylint: disable=too-few-public-methods,invalid-name,too-many-instance-attributes

import json
import pathlib
from dataclasses import dataclass, field, fields

from hazma.form_factors.vector import VectorFormFactorCouplings

DATA_DIR = pathlib.Path(__file__).parent.joinpath("testdata")


@dataclass
class FormFactorTestDataItem:
    """Deserialized test data."""

    gvxx: float = field(default=0.0)
    mx: float = field(default=0.0)
    mv: float = field(default=0.0)
    width_v: float = field(default=0.0)
    gvuu: float = field(default=0.0)
    gvdd: float = field(default=0.0)
    gvss: float = field(default=0.0)
    width: float = field(default=0.0)
    form_factor_re: float = field(default=0.0)
    form_factor_im: float = field(default=0.0)

    @property
    def couplings(self) -> VectorFormFactorCouplings:
        """Get the light quark couplings."""
        return VectorFormFactorCouplings(
            gvuu=self.gvuu,
            gvdd=self.gvdd,
            gvss=self.gvss,
        )


def load_test_data(filename: str) -> list[FormFactorTestDataItem]:
    """Load test data from the testdata directory."""
    filepath = DATA_DIR.joinpath(filename)
    assert filepath.exists(), f"Cannot find {filepath}"

    with open(filepath, "r", encoding="utf-8") as file:
        data = json.load(file)

    assert isinstance(data, list), "Invalid data format. Expected list."

    template = FormFactorTestDataItem()

    parsed_data: list[FormFactorTestDataItem] = []
    for datum in data:
        assert isinstance(datum, dict), "Invalid data format. Expected dict."

        data_entry = FormFactorTestDataItem()
        for fld in fields(template):
            fld_value = datum.get(fld.name)
            assert fld_value is not None, f"Expected field {fld.name}."
            setattr(data_entry, fld.name, fld_value)

        parsed_data.append(data_entry)

    return parsed_data
