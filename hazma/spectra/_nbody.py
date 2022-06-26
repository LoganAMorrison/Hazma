from typing import List, Tuple, Sequence, Optional, Callable, Any, Union, Dict
import functools as ft
from collections import defaultdict

import numpy as np
import numpy.typing as npt

from hazma.phase_space import Rambo, ThreeBody, PhaseSpaceDistribution1D
from hazma.parameters import standard_model_masses as sm_masses
from . import _photon
from . import _neutrino
from . import _positron
from .altarelli_parisi import (
    dnde_photon_ap_scalar as _ap_scalar,
    dnde_photon_ap_fermion as _ap_fermion,
)
from hazma.utils import RealArray, RealOrRealArray


MSqrd = Union[Callable[[Any], Any], Callable[[Any, Any], Any]]


def _dnde_zero(product_energies, _):
    return np.zeros_like(product_energies)


def _dnde_zero_nu(product_energies, _, flavor: Optional[str] = None):
    if flavor is None:
        return np.zeros((3, *product_energies.shape), dtype=product_energies.dtype)
    else:
        return np.zeros_like(product_energies)


def _make_fsr(mass, charge, scalar):
    ap = _ap_scalar if scalar else _ap_fermion

    def fsr(energies, sqrts):
        return ap(energies, sqrts**2, mass=mass, charge=charge)

    return fsr


_dnde_photon_dict: Dict[str, Callable[[RealArray, float], RealArray]] = {
    "e": _dnde_zero,
    "ve": _dnde_zero,
    "vm": _dnde_zero,
    "vt": _dnde_zero,
    "mu": _photon.dnde_photon_muon,
    "pi": _photon.dnde_photon_charged_pion,
    "pi0": _photon.dnde_photon_neutral_pion,
    "k": _photon.dnde_photon_charged_kaon,
    "kl": _photon.dnde_photon_long_kaon,
    "ks": _photon.dnde_photon_short_kaon,
    "eta": _photon.dnde_photon_eta,
    "etap": _photon.dnde_photon_eta_prime,
    "rho": _photon.dnde_photon_charged_rho,
    "rho0": _photon.dnde_photon_neutral_rho,
    "omega": _photon.dnde_photon_omega,
    "phi": _photon.dnde_photon_phi,
}


_dnde_fsr_dict: Dict[str, Callable[[RealArray, float], RealArray]] = {
    "e": _make_fsr(mass=sm_masses["e"], charge=-1.0, scalar=False),
    "mu": _make_fsr(mass=sm_masses["mu"], charge=-1.0, scalar=False),
    "pi": _make_fsr(mass=sm_masses["pi"], charge=1.0, scalar=True),
    "k": _make_fsr(mass=sm_masses["k"], charge=1.0, scalar=True),
}


_dnde_positron_dict: Dict[str, Callable[[RealArray, float], RealArray]] = {
    "e": _dnde_zero,
    "ve": _dnde_zero,
    "vm": _dnde_zero,
    "vt": _dnde_zero,
    "mu": _positron.dnde_positron_muon,
    "pi": _positron.dnde_positron_charged_pion,
    "pi0": _dnde_zero,
    "k": _positron.dnde_positron_charged_kaon,
    "kl": _positron.dnde_positron_long_kaon,
    "ks": _positron.dnde_positron_short_kaon,
    "eta": _positron.dnde_positron_eta,
    "etap": _positron.dnde_positron_eta_prime,
    "rho": _positron.dnde_positron_charged_rho,
    "rho0": _dnde_zero,
    "omega": _positron.dnde_positron_omega,
    "phi": _positron.dnde_positron_phi,
}

_dnde_neutrino_dict = {
    "e": _dnde_zero_nu,
    "ve": _dnde_zero_nu,
    "vm": _dnde_zero_nu,
    "vt": _dnde_zero_nu,
    "mu": _neutrino.dnde_neutrino_muon,
    "pi": _neutrino.dnde_neutrino_charged_pion,
    "pi0": _dnde_zero_nu,
    "k": _neutrino.dnde_neutrino_charged_kaon,
    "kl": _neutrino.dnde_neutrino_long_kaon,
    "ks": _neutrino.dnde_neutrino_short_kaon,
    "eta": _neutrino.dnde_neutrino_eta,
    "etap": _neutrino.dnde_neutrino_eta_prime,
    "rho": _neutrino.dnde_neutrino_charged_rho,
    "rho0": _dnde_zero_nu,
    "omega": _neutrino.dnde_neutrino_omega,
    "phi": _neutrino.dnde_neutrino_phi,
}

_dnde_ve_dict: Dict[str, Callable[[RealArray, float], RealArray]] = {
    key: ft.partial(fn, flavor="e") for key, fn in _dnde_neutrino_dict.items()
}

_dnde_vm_dict: Dict[str, Callable[[RealArray, float], RealArray]] = {
    key: ft.partial(fn, flavor="mu") for key, fn in _dnde_neutrino_dict.items()
}

_dnde_vt_dict: Dict[str, Callable[[RealArray, float], RealArray]] = {
    key: ft.partial(fn, flavor="tau") for key, fn in _dnde_neutrino_dict.items()
}


_spectra_dict: Dict[str, Dict[str, Callable[[RealArray, float], RealArray]]] = {
    "photon": _dnde_photon_dict,
    "fsr": _dnde_fsr_dict,
    "positron": _dnde_positron_dict,
    "neutrino": _dnde_neutrino_dict,
    "ve": _dnde_ve_dict,
    "vm": _dnde_vm_dict,
    "vt": _dnde_vt_dict,
}


def _get_masses(final_states: Sequence[str]) -> List[float]:
    r"""Convert list of strings of final state particles to a list of their
    masses.

    Parameters
    ----------
    final_states: list[str]
        List of strings representing the final state particles.

    Returns
    -------
    masses: list[str]
        List containing the masses of the final-state particles.
    """
    masses: List[float] = []
    for s in final_states:
        m = sm_masses.get(s)
        if m is None:
            raise ValueError(f"Encountered unknown particle {s}.")
        masses.append(m)
    return masses


def _make_energy_distributions(
    cme: float,
    final_states: Sequence[str],
    msqrd,
    npts: int,
    nbins: int,
    three_body_integrator: str,
    msqrd_signature: Optional[str],
):
    r"""Use Monte-Carlo to generate energy distributions for each final-state particle.

    Parameters
    ----------
    cme: float
        Center-of-mass energies.
    final_states: list[str]
        List of strings representing the final state particles.
    msqrd: callable
        Function to compute the squared matrix element. Must except a numpy
        array of the momenta of the final state particles.
    npts: int
        Number of Monte-Carlo phase-space points used to generate distributions.
    nbins: int
        Number of energy bins used to construct distributions.

    Returns
    -------
    dists: List[PhaseSpaceDistribution1D]
        List containing the energy distributions of the final-state particles.
    """
    assert three_body_integrator in ["quad", "trapz", "simps", "rambo"], (
        f"Invalid value for 'three_body_integrator': {three_body_integrator}."
        "Use 'quad', 'trapz', 'simps', or 'rambo'."
    )

    masses = _get_masses(final_states)
    if len(final_states) == 3 and three_body_integrator in ["quad", "trapz", "simps"]:
        return ThreeBody(
            cme, masses, msqrd=msqrd, msqrd_signature=msqrd_signature
        ).energy_distributions(nbins=nbins)

    return Rambo(cme, masses, msqrd=msqrd).energy_distributions(n=npts, nbins=nbins)


def _make_invariant_mass_distributions(
    cme: float,
    final_states: Sequence[str],
    msqrd,
    npts: int,
    nbins: int,
    three_body_integrator: str,
    msqrd_signature: Optional[str],
):
    r"""Use Monte-Carlo to generate invariant-mass distributions for each pair
    of final-state particles.

    Parameters
    ----------
    cme: float
        Center-of-mass energies.
    final_states: list[str]
        List of strings representing the final state particles.
    msqrd: callable
        Function to compute the squared matrix element. Must except a numpy
        array of the momenta of the final state particles.
    npts: int
        Number of Monte-Carlo phase-space points used to generate distributions.
    nbins: int
        Number of bins used to construct distributions.

    Returns
    -------
    dists: dict[(int,int), PhaseSpaceDistribution1D]
        Dictionary containing the distributions. They keys represent the pair
        the distribution corresponds to.
    """
    assert three_body_integrator in ["quad", "trapz", "simps", "rambo"], (
        f"Invalid value for 'three_body_integrator': {three_body_integrator}."
        "Use 'quad', 'trapz', 'simps', or 'rambo'."
    )

    masses = _get_masses(final_states)

    if len(final_states) == 3 and three_body_integrator in ["quad", "trapz", "simps"]:
        return ThreeBody(
            cme, masses, msqrd=msqrd, msqrd_signature=msqrd_signature
        ).invariant_mass_distributions(nbins=nbins)

    return Rambo(cme, masses, msqrd=msqrd).invariant_mass_distributions(
        n=npts, nbins=nbins
    )


def _conv_dnde_dist(
    product_energies: RealArray,
    dist: PhaseSpaceDistribution1D,
    state: str,
    product: str,
) -> RealArray:
    r"""Convolve the spectrum with an energy distribution.

    Parameters
    ----------
    product_energies: np.ndarray
        Array of the energies of the product.
    dist: PhaseSpaceDistribution1D
        Distribution to convolve with.
    state: str
        State that will decay into the product.
    product: str
        The product produced. Should be 'photon' or 'positron' (neutrino is handeled)

    Returns
    -------
    dnde: np.ndarray
        The convolved spectrum.
    """
    # For the neutrino spectra, the 'dnde_neutrino_*' functions return an array
    # with shape (3,n). To accomidate this shape, we reshape the probabilities.
    # The result of generating the spectra for all parent energies will be of
    # shape (m, 3, n) with m = number energies, n=number neutrino energies.
    # We then reshape the probabilities to have shape (m, 1, 1), then integrate
    # over axis=0.
    # For the other cases, we just expand the probabilities to be shape (m, 1).

    assert product in _spectra_dict, f"Invalid product: {product}."
    assert state in _spectra_dict[product], f"Invalid final state: {state}."

    dnde_fn = _spectra_dict[product][state]

    # This will have shape (m, 3, n) for neutrinos and (m,n) otherwise.
    dndes = np.array([dnde_fn(product_energies, e) for e in dist.bin_centers])
    # Transform probs to (m, 1, 1) for neutrino, else (m, 1)
    expand = (1, 2) if product == "neutrino" else (1,)
    dndes = np.expand_dims(dist.probabilities, expand) * dndes
    # Integrate over leading axis. Result has shape (3, n) for neutrinos and
    # (n,) otherwise.
    dnde = np.trapz(dndes, dist.bin_centers, axis=0)

    return dnde


def _dnde_photon_fsr(
    photon_energies: RealArray,
    cme: float,
    final_states: Sequence[str],
    msqrd: Optional[MSqrd],
    three_body_integrator: str,
    msqrd_signature: Optional[str],
    npts: int,
    nbins: int,
    average_fsr: bool,
) -> RealArray:
    r"""Convolve the FSR spectra with the invariant-mass distributions.

    Parameters
    ----------
    photon_energies: np.ndarray
        Array of the photon energies.
    cme: float
        Center-of-mass energy.
    final_states: list[str]
        List of strings representing the final state particles.
    msqrd: callable
        Function to compute the squared matrix element. Must except a numpy
        array of the momenta of the final state particles.
    npts: int
        Number of Monte-Carlo phase-space points used to generate distributions.
    nbins: int
        Number of bins used to construct distributions.
    average_fsr: bool
        If True, the FSR is averaged over each pair. For example, if we have
        pairs (0, 1), (0, 2) and (1, 3), then the spectrum is computed twice for
        each particle, then is divided by 2. If False, then duplicates are
        skipped and the spectrum is computed once for each unique final-state.

    Returns
    -------
    fsr: np.ndarray
        The convolved FSR spectrum.
    """
    dnde = np.zeros_like(photon_energies)
    dnde_fns = _spectra_dict["fsr"]

    has_fsr = any([s in dnde_fns for s in final_states])
    if has_fsr:
        # Compute FSR convolved with invariant-mass distributions
        dists = _make_invariant_mass_distributions(
            cme=cme,
            final_states=final_states,
            msqrd=msqrd,
            npts=npts,
            nbins=nbins,
            three_body_integrator=three_body_integrator,
            msqrd_signature=msqrd_signature,
        )

        # Use a counter to determine how many times a spectrum has been
        # computed. We will average over spectra computed multiple times to
        # obtain a more accurate result. Since we're loop over all pairs or
        # particles, we might have keys like: (0,1), (0,2), (1,2). In this case,
        # we would be computing each spectrum twice. We thus need to average the
        # spectra. If 'average_fsr' is False, the we will skip terms computing
        # the same spectrum.
        counts = defaultdict(int)
        dndes = {
            key: np.zeros_like(photon_energies)
            for key in final_states
            if key in dnde_fns
        }

        for pair, dist in dists.items():
            for s in [final_states[p] for p in pair]:
                # Skip neutral particles
                if s not in dndes:
                    continue

                # If average_fsr is False and we've already encountered this
                # particle, skip it.
                if average_fsr or counts[s] == 0:
                    counts[s] += 1
                    dndes[s] = dndes[s] + _conv_dnde_dist(
                        photon_energies, dist, s, "fsr"
                    )

        for key in dndes.keys():
            dnde += dndes[key] / counts[key]

    return dnde


def _dnde_two_body(
    product_energies: RealArray,
    cme: float,
    states: Tuple[str, str],
    product: str,
    include_fsr: bool,
) -> RealArray:
    r"""Compute the differential energy spectrum of a product from the decays and FSR
    of a two-body final state.

    Parameters
    ----------
    product_energies: np.ndarray
        Array of the product energies.
    cme: float
        Center-of-mass energy.
    final_states: (str,str)
        The two final state particles.
    product: str
        Product to compute spectrum for.

    Returns
    -------
    dnde: np.ndarray
        The combined spectrum.
    """
    s1, s2 = states
    dnde_fn = _spectra_dict[product]
    dnde_fsr = _spectra_dict["fsr"]

    m1, m2 = _get_masses(states)
    e1 = (cme**2 + m1**2 - m2**2) / (2.0 * cme)
    e2 = (cme**2 - m1**2 + m2**2) / (2.0 * cme)

    shape = ((3,) if product == "neutrino" else tuple()) + product_energies.shape
    dnde = np.zeros(shape, dtype=product_energies.dtype)

    if s1 in dnde_fn:
        dnde += dnde_fn[s1](product_energies, e1)

    if s2 in dnde_fn:
        dnde += dnde_fn[s2](product_energies, e2)

    if product == "photon" and include_fsr:
        if s1 in dnde_fsr:
            dnde += dnde_fsr[s1](product_energies, cme**2)
        if s2 in dnde_fsr:
            dnde += dnde_fsr[s2](product_energies, cme**2)

    return dnde


def _dnde_multi_particle(
    product_energies: RealArray,
    cme: float,
    final_states: Sequence[str],
    product: str,
    msqrd: Optional[MSqrd],
    *,
    three_body_integrator: Optional[str],
    msqrd_signature: Optional[str],
    npts: int,
    nbins: int,
    include_fsr: bool,
    average_fsr: bool,
) -> RealArray:
    r"""Compute the differential energy spectrum of a product from the decays and FSR
    of a n-body final state for n >= 3.

    Parameters
    ----------
    product_energies: np.ndarray
        Array of the product energies.
    cme: float
        Center-of-mass energy.
    final_states: (str,str)
        The two final state particles.
    product: str
        Product to compute spectrum for.

    Returns
    -------
    dnde: np.ndarray
        The combined spectrum.
    """
    if three_body_integrator is None:
        tbi = "quad"
    else:
        tbi = three_body_integrator

    edists = _make_energy_distributions(
        cme=cme,
        final_states=final_states,
        msqrd=msqrd,
        three_body_integrator=tbi,
        msqrd_signature=msqrd_signature,
        npts=npts,
        nbins=nbins,
    )
    dnde = np.zeros_like(product_energies)
    for dist, state in zip(edists, final_states):
        dnde = dnde + _conv_dnde_dist(product_energies, dist, state, product)

        # if product is in final state, add it to spectrum
        cond = state == product
        cond = cond or (state == "e" and product == "positron")
        cond = cond or (state in ["ve", "vm", "vt"] and product == "neutrino")
        if cond:
            dnde += np.interp(
                product_energies,
                dist.bin_centers,
                dist.probabilities,
                right=0.0,
                left=0.0,
            )

    if product == "photon" and include_fsr:
        dnde = dnde + _dnde_photon_fsr(
            photon_energies=product_energies,
            cme=cme,
            final_states=final_states,
            three_body_integrator=tbi,
            msqrd_signature=msqrd_signature,
            msqrd=msqrd,
            npts=npts,
            nbins=nbins,
            average_fsr=average_fsr,
        )

    return dnde


def _dnde_nbody(
    product_energies: npt.ArrayLike,
    cme: float,
    final_states: Union[str, Sequence[str]],
    product: str,
    *,
    msqrd: Optional[MSqrd],
    three_body_integrator: Optional[str],
    msqrd_signature: Optional[str],
    npts: int,
    nbins: int,
    include_fsr: bool,
    average_fsr: bool,
) -> RealOrRealArray:

    # Check cme is large enough before we get into the weeds:
    if isinstance(final_states, str):
        msum = sm_masses[final_states]
    else:
        msum = sum(_get_masses(final_states))

    if msum > cme:
        raise ValueError(
            "Center of mass energy is too small."
            + f" cme={cme}, final state mass sum = {msum}."
        )

    scalar = np.isscalar(product_energies)
    es = np.atleast_1d(product_energies).astype(np.float64)
    dnde = np.zeros_like(es)

    if isinstance(final_states, str):
        dnde = _spectra_dict[product][final_states](es, cme)
    elif len(final_states) == 1:
        dnde = _spectra_dict[product][final_states[0]](es, cme)
    else:
        if len(final_states) == 2:
            s1, s2 = final_states[0], final_states[1]
            dnde = _dnde_two_body(
                product_energies=es,
                cme=cme,
                states=(s1, s2),
                product=product,
                include_fsr=include_fsr,
            )
        else:
            dnde = _dnde_multi_particle(
                product_energies=es,
                cme=cme,
                final_states=final_states,
                product=product,
                msqrd=msqrd,
                three_body_integrator=three_body_integrator,
                msqrd_signature=msqrd_signature,
                npts=npts,
                nbins=nbins,
                include_fsr=include_fsr,
                average_fsr=average_fsr,
            )

    if scalar:
        return np.take(dnde, 0, axis=-1)
    return dnde


def dnde_photon(
    photon_energies: npt.ArrayLike,
    cme: float,
    final_states: Union[str, Sequence[str]],
    *,
    msqrd: Optional[MSqrd] = None,
    three_body_integrator: Optional[str] = None,
    msqrd_signature: Optional[str] = None,
    npts: int = 1 << 14,
    nbins: int = 25,
    include_fsr: bool = True,
    average_fsr: bool = True,
):
    r"""Compute the differential photon energy spectrum from the decays and FSR
    of final state particles.

    To see all availible final states, use `dnde_photon.availible_final_states`.

    Parameters
    ----------
    photon_energies: float or array-like
        Photon energy(ies) where the spectrum should be computed.
    cme: float
        Center-of-mass energy.
    final_states: str or sequence[str]
        String or iterable of strings representing the final state particles.
    msqrd: callable, optional
        Function to compute the squared matrix element. Must except a single numpy
        array of the momenta of the final state particles. Only used if the
        number of final-state particles is greater than 2. Default is None.
    three_body_integrator: str, optional
        Type of integrator used for three body final-states. Can be:

            * 'quad': for `scipy.integrate.quad`,
            * 'trapz': for `scipy.integrate.trapz`,
            * 'simps': for `scipy.integrate.simps`,
            * 'rambo': for `hazma.phase_space.Rambo`.

        Default is 'quad'.
    msqrd_signature: str, optional
        Signature of the squared matrix element. Default is 'momenta' when
        `len(final_states) > 3` and 'st' when `len(final_states)==3`. See
        `hazma.phase_space.ThreeBody` for information.
    npts: int, optional
        Number of Monte-Carlo phase-space points used to generate
        energy/invariant mass distributions. Only used if the number of final
        state particles is greater than 2. Default is 2^14.
    nbins: int
        Number of bins used to construct energy/invariant-mass distributions of
        the final-state particles. Only used if the number of final-state
        particles is greater than 2. Default is 25.
    include_fsr: bool
        If True, the FSR is included using the Altarelli-Parisi splitting
        functions.
    average_fsr: bool
        If True, the FSR is averaged over each pair. For example, if we have
        pairs (0, 1), (0, 2) and (1, 3), then the spectrum is computed twice for
        each particle, then is divided by 2. If False, then duplicates are
        skipped and the spectrum is computed once for each unique final-state.
        Only used if at least one of the final states is charged.

    Returns
    -------
    dnde: np.ndarray
        The photon spectrum convolved with each final-state particle energy
        distribution.

    Notes
    -----
    The spectrum is computed using:

    .. math::

        \frac{dN}{dE} &=
        \sum_{f}\frac{dN_{\mathrm{dec.}}}{dE} +
        \sum_{f}\frac{dN_{\mathrm{FSR}}}{dE} \\
        \frac{dN_{\mathrm{dec}}}{dE} &= \sum_{f}\int{d\epsilon_f}
        P(\epsilon_f) \frac{dN_{f}}{dE}(E,\epsilon_f)\\
        \frac{dN}{dE} &= \sum_{i\neq j}\frac{1}{S_iS_j}\int{ds_{ij}}
        P(s_{ij}) \left(
        \frac{dN_{i,\mathrm{FSR}}}{dE}(E,s_{ij}) +
        \frac{dN_{j,\mathrm{FSR}}}{dE}(E,s_{ij})\right)

    where :math:`P(\epsilon_f)` is the probability distribution of particle
    :math:`f` having energy :math:`\epsilon_f`. The sum over `f` in the decay
    expression is over all final-state particles. For the FSR expression,
    :math:`s_{ij}` is the squared invariant-mass of particles :math:`i` and
    :math:`j`. The factors :math:`S_{i}` and :math:`S_{j}` are symmetry factors
    included to avoid overcounting.
    """
    return _dnde_nbody(
        product_energies=photon_energies,
        cme=cme,
        final_states=final_states,
        product="photon",
        msqrd=msqrd,
        three_body_integrator=three_body_integrator,
        msqrd_signature=msqrd_signature,
        npts=npts,
        nbins=nbins,
        include_fsr=include_fsr,
        average_fsr=average_fsr,
    )


dnde_photon.availible_final_states = {key for key in _spectra_dict["photon"].keys()}


def dnde_positron(
    positron_energies: npt.ArrayLike,
    cme: float,
    final_states: Union[str, Sequence[str]],
    msqrd: Optional[MSqrd] = None,
    *,
    three_body_integrator: Optional[str] = None,
    msqrd_signature: Optional[str] = None,
    npts: int = 1 << 14,
    nbins: int = 25,
):
    r"""Compute the differential positron energy spectrum from the decays of
    final state particles.

    To see all availible final states, use `dnde_positron.availible_final_states`.

    Parameters
    ----------
    positron_energies: float or array-like
        Positron energy(ies) where the spectrum should be computed.
    cme: float
        Center-of-mass energy in MeV.
    final_states: str or sequence[str]
        String or iterable of strings representing the final state particles.
    msqrd: callable, optional
        Function to compute the squared matrix element. Must except a single numpy
        array of the momenta of the final state particles. Only used if the
        number of final-state particles is greater than 2. Default is None.
    three_body_integrator: str, optional
        Type of integrator used for three body final-states. Can be:

            * 'quad': for `scipy.integrate.quad`,
            * 'trapz': for `scipy.integrate.trapz`,
            * 'simps': for `scipy.integrate.simps`,
            * 'rambo': for `hazma.phase_space.Rambo`.

        Default is 'quad'.
    msqrd_signature: str, optional
        Signature of the squared matrix element. Default is 'momenta' when
        `len(final_states) > 3` and 'st' when `len(final_states)==3`. See
        `hazma.phase_space.ThreeBody` for information.
    npts: int, optional
        Number of Monte-Carlo phase-space points used to generate
        energy/invariant mass distributions. Only used if the number of final
        state particles is greater than 2. Default is 2^14.
    nbins: int
        Number of bins used to construct energy/invariant-mass distributions of
        the final-state particles. Only used if the number of final-state
        particles is greater than 2. Default is 25.

    Returns
    -------
    dnde: np.ndarray
        The positron spectrum dN/dE convolved with each final-state particle
        energy distribution.

    Notes
    -----
    The spectrum is computed using:

    .. math::

        \frac{dN}{dE} = \sum_{f}\int{d\epsilon_f}
        P(\epsilon_f) \frac{dN_{f}}{dE}(E,\epsilon_f)

    where :math:`P(\epsilon_f)` is the probability distribution of particle
    :math:`f` having energy :math:`\epsilon_f`. The sum over `f` is over all
    final-state particles.
    """
    return _dnde_nbody(
        product_energies=positron_energies,
        cme=cme,
        final_states=final_states,
        product="positron",
        msqrd=msqrd,
        three_body_integrator=three_body_integrator,
        msqrd_signature=msqrd_signature,
        npts=npts,
        nbins=nbins,
        include_fsr=False,
        average_fsr=False,
    )


dnde_positron.availible_final_states = {key for key in _spectra_dict["positron"].keys()}


def dnde_neutrino(
    neutrino_energies: npt.ArrayLike,
    cme: float,
    final_states: Union[str, Sequence[str]],
    msqrd: Optional[MSqrd] = None,
    *,
    three_body_integrator: Optional[str] = None,
    msqrd_signature: Optional[str] = None,
    npts: int = 1 << 14,
    nbins: int = 25,
    flavor: Optional[str] = None,
):
    r"""Compute the differential neutrino energy spectrum from the decays of
    final state particles.

    To see all availible final states, use `dnde_positron.availible_final_states`.

    Parameters
    ----------
    neutrino_energies: float or array-like
        Neutrino energy(ies) where the spectrum should be computed.
    cme: float
        Center-of-mass energy.
    final_states: str or sequence[str]
        String or iterable of strings representing the final state particles.
    msqrd: callable, optional
        Function to compute the squared matrix element. Must except a single numpy
        array of the momenta of the final state particles. Only used if the
        number of final-state particles is greater than 2. Default is None.
    three_body_integrator: str, optional
        Type of integrator used for three body final-states. Can be:

            * 'quad': for `scipy.integrate.quad`,
            * 'trapz': for `scipy.integrate.trapz`,
            * 'simps': for `scipy.integrate.simps`,
            * 'rambo': for `hazma.phase_space.Rambo`.

        Default is 'quad'.
    msqrd_signature: str, optional
        Signature of the squared matrix element. Default is 'momenta' when
        `len(final_states) > 3` and 'st' when `len(final_states)==3`. See
        `hazma.phase_space.ThreeBody` for information.
    npts: int, optional
        Number of Monte-Carlo phase-space points used to generate
        energy/invariant mass distributions. Only used if the number of final
        state particles is greater than 2. Default is 2^14.
    nbins: int
        Number of bins used to construct energy/invariant-mass distributions of
        the final-state particles. Only used if the number of final-state
        particles is greater than 2. Default is 25.
    flavor: str, optional
        Flavor of neutrino. If None, all flavors are returned.

    Returns
    -------
    dnde: np.ndarray
        The neutrino spectrum dN/dE convolved with each final-state particle
        energy distribution. Note that the shape will be (3,
        len(neutrino_energies)) when `flavor` is None where the leading axis
        contains the spectra for the electron, muon and tau neutrino.

    Notes
    -----
    The spectrum is computed using:

    .. math::

        \frac{dN}{dE} = \sum_{f}\int{d\epsilon_f}
        P(\epsilon_f) \frac{dN_{f}}{dE}(E,\epsilon_f)

    where :math:`P(\epsilon_f)` is the probability distribution of particle
    :math:`f` having energy :math:`\epsilon_f`. The sum over `f` is over all
    final-state particles.
    """
    if flavor is None:
        product = "neutrino"
    elif flavor == "e":
        product = "ve"
    elif flavor == "mu":
        product = "vm"
    elif flavor == "tau":
        product = "vt"
    else:
        raise ValueError(f"Invalid flavor {flavor}. Use 'e', 'mu', or 'tau'.")

    return _dnde_nbody(
        product_energies=neutrino_energies,
        cme=cme,
        final_states=final_states,
        product=product,
        msqrd=msqrd,
        three_body_integrator=three_body_integrator,
        msqrd_signature=msqrd_signature,
        npts=npts,
        nbins=nbins,
        include_fsr=False,
        average_fsr=False,
    )


dnde_neutrino.availible_final_states = {key for key in _spectra_dict["neutrino"].keys()}
