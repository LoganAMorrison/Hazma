from typing import Dict, NamedTuple, Tuple, Protocol

# from typing_extensions import Protocol

import numpy as np

from hazma.parameters import standard_model_masses as sm_masses
from hazma.form_factors.vector import VectorFormFactorPiPi
from hazma.utils import RealOrRealArray

from ._proto import SingleRhNeutrinoModel, Generation
from . import _spectra as rhn_spectra
from . import _widths as rhn_widths
from ._utils import three_lepton_fs_generations

_LEP_STRS = ["e", "mu", "tau"]
_NU_STRS = ["ve", "vm", "vt"]


class DndeFn(Protocol):
    def __call__(
        self, product_energies: RealOrRealArray, product: str, **kwargs
    ) -> RealOrRealArray: ...


class WidthFn(Protocol):
    def __call__(self) -> float: ...


class RHNFinalStateAttrs(NamedTuple):
    dnde: DndeFn
    width: WidthFn
    self_conjugate: bool
    masses: Tuple[float, ...]


def configure(
    model: SingleRhNeutrinoModel, form_factor: VectorFormFactorPiPi
) -> Dict[str, RHNFinalStateAttrs]:

    ll = _LEP_STRS[model.gen]
    vv = _NU_STRS[model.gen]

    def mk_dnde_zero():
        def dnde(
            product_energies: RealOrRealArray, product: str, **_
        ) -> RealOrRealArray:
            scalar = np.isscalar(product_energies)
            e = np.atleast_1d(product_energies)
            if product == "neutrino":
                res = np.zeros((3, *e.shape))
            else:
                res = np.zeros_like(e)
            if scalar:
                return res[..., 0]
            return res

        return dnde

    def mk_dnde_2body(fn):
        def dnde(
            product_energies: RealOrRealArray, product: str, **_
        ) -> RealOrRealArray:
            return fn(model=model, product_energies=product_energies, product=product)

        return dnde

    def mk_dnde_3body(fn, **outer_kwargs):
        def dnde(product_energies, product, **kwargs):
            nbins = kwargs.get("nbins", 30)
            three_body_integrator = kwargs.get("three_body_integrator", "quad")
            return fn(
                model=model,
                product_energies=product_energies,
                product=product,
                nbins=nbins,
                three_body_integrator=three_body_integrator,
                **outer_kwargs,
            )

        return dnde

    def mk_width_fn(fn, **outer_kwargs):
        def newfn() -> float:
            return fn(model=model, **outer_kwargs)

        return newfn

    ff = form_factor
    fns = {}

    fns[f"{vv} a"] = RHNFinalStateAttrs(
        dnde=mk_dnde_zero(),
        width=mk_width_fn(rhn_widths.width_v_a),
        self_conjugate=True,
        masses=(0.0, 0.0),
    )

    fns[f"{vv} pi0"] = RHNFinalStateAttrs(
        dnde=mk_dnde_2body(rhn_spectra.dnde_v_pi0),
        width=mk_width_fn(rhn_widths.width_v_pi0),
        self_conjugate=True,
        masses=(0.0, sm_masses["pi0"]),
    )
    fns[f"{vv} eta"] = RHNFinalStateAttrs(
        dnde=mk_dnde_2body(rhn_spectra.dnde_v_eta),
        width=mk_width_fn(rhn_widths.width_v_eta),
        self_conjugate=True,
        masses=(0.0, sm_masses["eta"]),
    )
    fns[f"{vv} rho"] = RHNFinalStateAttrs(
        dnde=mk_dnde_2body(rhn_spectra.dnde_v_rho),
        width=mk_width_fn(rhn_widths.width_v_rho),
        self_conjugate=True,
        masses=(0.0, sm_masses["rho"]),
    )
    fns[f"{vv} omega"] = RHNFinalStateAttrs(
        dnde=mk_dnde_2body(rhn_spectra.dnde_v_omega),
        width=mk_width_fn(rhn_widths.width_v_omega),
        self_conjugate=True,
        masses=(0.0, sm_masses["omega"]),
    )
    fns[f"{vv} phi"] = RHNFinalStateAttrs(
        dnde=mk_dnde_2body(rhn_spectra.dnde_v_phi),
        width=mk_width_fn(rhn_widths.width_v_phi),
        self_conjugate=True,
        masses=(0.0, sm_masses["phi"]),
    )
    fns[f"{ll} k"] = RHNFinalStateAttrs(
        dnde=mk_dnde_2body(rhn_spectra.dnde_l_k),
        width=mk_width_fn(rhn_widths.width_l_k),
        self_conjugate=False,
        masses=(sm_masses[ll], sm_masses["k"]),
    )
    fns[f"{ll} pi"] = RHNFinalStateAttrs(
        dnde=mk_dnde_2body(rhn_spectra.dnde_l_pi),
        width=mk_width_fn(rhn_widths.width_l_pi),
        self_conjugate=False,
        masses=(sm_masses[ll], sm_masses["pi"]),
    )
    fns[f"{ll} rho"] = RHNFinalStateAttrs(
        dnde=mk_dnde_2body(rhn_spectra.dnde_l_rho),
        width=mk_width_fn(rhn_widths.width_l_rho),
        self_conjugate=False,
        masses=(sm_masses[ll], sm_masses["rho"]),
    )
    fns[f"{vv} pi pi"] = RHNFinalStateAttrs(
        dnde=mk_dnde_3body(rhn_spectra.dnde_v_pi_pi, form_factor=ff),
        width=mk_width_fn(
            rhn_widths.width_v_pi_pi, form_factor=ff, epsabs=0.0, epsrel=1e-3
        ),
        self_conjugate=True,
        masses=(0.0, sm_masses["pi"], sm_masses["pi"]),
    )
    fns[f"{ll} pi pi0"] = RHNFinalStateAttrs(
        dnde=mk_dnde_3body(rhn_spectra.dnde_l_pi0_pi, form_factor=ff),
        width=mk_width_fn(
            rhn_widths.width_l_pi0_pi, form_factor=ff, epsabs=0.0, epsrel=1e-3
        ),
        self_conjugate=False,
        masses=(sm_masses[ll], sm_masses["pi"], sm_masses["pi0"]),
    )

    # N -> v1 + l2 + lbar3
    gen_tups = three_lepton_fs_generations(model.gen)
    for gen_tup in gen_tups:
        g1, g2, g3 = gen_tup
        states = [_NU_STRS[g1], _LEP_STRS[g2], _LEP_STRS[g3]]
        key = " ".join(states)
        fns[key] = RHNFinalStateAttrs(
            dnde=mk_dnde_3body(rhn_spectra.dnde_v_l_l, genv=g1, genl1=g2, genl2=g3),
            width=mk_width_fn(rhn_widths.width_v_l_l, genv=g1, genl1=g2, genl2=g3),
            self_conjugate=g2 == g3,
            masses=tuple(sm_masses[s] for s in states),
        )

    # N -> v1 + v2 + v3
    for gen in [Generation.Fst, Generation.Snd, Generation.Trd]:
        states = [_NU_STRS[model.gen], _NU_STRS[gen], _NU_STRS[gen]]
        key = " ".join(states)
        fns[key] = RHNFinalStateAttrs(
            dnde=mk_dnde_3body(rhn_spectra.dnde_v_v_v, genv=gen),
            width=mk_width_fn(rhn_widths.width_v_v_v, genv=gen),
            self_conjugate=True,
            masses=(0.0, 0.0, 0.0),
        )

    return fns
