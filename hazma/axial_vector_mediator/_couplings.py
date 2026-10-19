"""Lepton and neutrino couplings of the axial-vector mediator."""

from __future__ import annotations

from dataclasses import dataclass, fields

from hazma.form_factors.axial._sources import _finite_real


@dataclass(frozen=True, kw_only=True)
class LeptonCouplings:
    r"""Dimensionless mediator couplings of the charged leptons and neutrinos.

    Parameters
    ----------
    gvee, gaee, gvmumu, gamumu, gvtautau, gatautau : float
        Vector and axial couplings of each charged lepton in
        :math:`\mathcal L = Z'_\mu\bar\ell\gamma^\mu(g_V^\ell + g_A^\ell\gamma_5)\ell`.
    gnue, gnumu, gnutau : float
        Couplings of the left-handed neutrinos in
        :math:`\mathcal L = g_\nu Z'_\mu\bar\nu\gamma^\mu P_L\nu`, so that
        :math:`g_V^\nu = -g_A^\nu = g_\nu/2`.

    Notes
    -----
    All nine are required, so an omitted coupling is a declared zero.
    """

    gvee: float
    gaee: float
    gvmumu: float
    gamumu: float
    gvtautau: float
    gatautau: float
    gnue: float
    gnumu: float
    gnutau: float

    def __post_init__(self) -> None:
        for field in fields(self):
            value = _finite_real(getattr(self, field.name), field.name)
            object.__setattr__(self, field.name, value)
