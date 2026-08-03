"""
Module for working with Lorentz-invariant phase-space.

"""

# TODO: Code up specific functions for cross-section
#       functions for 2->2 processes.
# TODO: Code up specific functions for cross-section
#       functions for 2->3 processes.


import math
import multiprocessing as mp
import warnings
import logging
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
from scipy import integrate

from hazma._phase_space import generator, histogram
from hazma._phase_space.modifiers import apply_matrix_elem
from hazma.field_theory_helper_functions.common_functions import cross_section_prefactor
from hazma.hazma_errors import RamboCMETooSmall
from hazma.utils import RealArray, kallen_lambda, lnorm_sqr

from hazma.utils import warn_deprecated_module

warn_deprecated_module("hazma.rambo", alternative="hazma.phase_space")

MassList = Union[List[float], RealArray]
SquaredMatrixElement = Callable[[RealArray], float]


def __flat_squared_matrix_element(_: RealArray) -> float:
    return 1.0


def normalize_distribution(probabilities, edges):
    norm = np.sum([p * (edges[i + 1] - edges[i]) for i, p in enumerate(probabilities)])
    if norm <= 0.0:
        if np.min(probabilities) < 0.0:
            logging.warning(f"Negative probabilities encountered: {probabilities}")
            return np.ones_like(probabilities) * np.nan
        return probabilities
    return probabilities / norm


def generate_phase_space_point(masses: MassList, cme: float) -> RealArray:
    """
    Generate a phase space point given a set of final state particles and a
    given center of mass energy.

    Parameters
    ----------
    masses : numpy.ndarray
        List of masses of the final state particles.
    cme : double
        Center-of-mass-energy of the process.

    Returns
    -------
    phase_space_points : numpy.ndarray
        List of four momenta and a event weight. The returned numpy array is of
        the form::

            [ke1, kx1, ky1, kz1, ..., keN, kxN, kyN, kzN, weight]

    """
    if not hasattr(masses, "__len__"):
        masses = [masses]  # type: ignore

    masses = np.array(masses)
    return generator.generate_point(masses, cme, len(masses))


def generate_phase_space(
    masses: MassList,
    cme: float,
    num_ps_pts: int = 10000,
    mat_elem_sqrd: Optional[SquaredMatrixElement] = None,
    num_cpus: Optional[int] = None,
) -> RealArray:
    """
    Generate a specified number of phase space points given a set of
    final state particles and a given center of mass energy.
    Note that the weights are not normalized.

    Parameters
    ----------
    masses : numpy.ndarray
        List of masses of the final state particles.
    cme : double
        Center-of-mass-energy of the process.
    num_ps_pts : int {10000]
        Total number of phase space points to generate.
    mat_elem_sqrd : optional, Callable[[np.ndarray], float]
        Function for the matrix element squared.
    num_cpus : int {None]
        Number of cpus to use in parallel with rambo. If not specified, 75% of
        the cpus will be used.

    Returns
    -------
    phase_space_points : numpy.ndarray
        List of phase space points. The phase space points are in the form::

            [[ke11, kx11, ky11, kz11, ..., keN1, kxN1, kyN1, kzN1, weight1],
             ...
             [ke1N, kx1N, ky1N, kz1N, ..., keNN, kxNN, kyNN, kzNN, weightN]]

    Examples
    --------

    Generate 100000 phase space points for a 3 body final state::

        from hazma import rambo
        import numpy as np
        masses = np.array([100., 200., 0.0])
        cme = 10.0 * sum(masses)
        num_ps_pts = 100000
        rambo.generate_phase_space(masses, cme, num_ps_pts=num_ps_pts)

    """
    if not hasattr(masses, "__len__"):
        masses = [masses]  # type: ignore

    if cme < sum(masses):
        raise RamboCMETooSmall()

    num_fsp = len(masses)
    # If the user doesn't specify the number of cpus to use,
    # use 75% of them.
    if num_cpus is not None:
        if num_cpus > num_ps_pts:
            num_cpus = num_ps_pts
        if num_cpus > mp.cpu_count():
            num_cpus = int(np.floor(mp.cpu_count() * 0.75))
            warnings.warn(
                """You only have {} cpus.
                          Using {} cpus instead.
                          """.format(
                    mp.cpu_count(), num_cpus
                )
            )
    else:
        # Use 75% of the cpu power.
        num_cpus = int(np.floor(mp.cpu_count() * 0.75))
        # If user wants a number of phase space points which is less
        # than the number of cpus available, use num_ps_pts cpus instead.
        if num_cpus > num_ps_pts:
            num_cpus = num_ps_pts

    # Instantiate `num_cpus` number of workers and divide num_ps_pts among the
    # the workers to speed up phase space generation.
    pool = mp.Pool(num_cpus)
    num_ps_pts_per_cpu = int(num_ps_pts / num_cpus)
    # If num_ps_pts % num_cpus !=0, then we need to compute the actual number
    # of phase space points.
    actual_num_ps_pts = num_ps_pts_per_cpu * num_cpus
    # Create a container to store the results from the workers
    job_results = []
    # Run the jobs on 75% of the cpus.
    for _ in range(num_cpus):
        job_results.append(
            pool.apply_async(
                generator.generate_space,
                (num_ps_pts_per_cpu, masses, cme, num_fsp),
            )
        )
    # Close the pool and wait for results to finish
    pool.close()
    pool.join()
    # Put results in a numpy array.
    points = np.array([result.get() for result in job_results])
    # Flatten the outer axis to have a list of phase space points.
    points = points.reshape(actual_num_ps_pts, 4 * num_fsp + 1)
    # Resize the weights to have the correct cross section.
    if mat_elem_sqrd is None:
        msqrd = __flat_squared_matrix_element
    else:
        msqrd = mat_elem_sqrd
    points = apply_matrix_elem(points, actual_num_ps_pts, num_fsp, msqrd)

    return points


def generate_energy_histogram(
    masses: MassList,
    cme: float,
    num_ps_pts: int = 10000,
    mat_elem_sqrd: Optional[SquaredMatrixElement] = None,
    num_bins: int = 25,
    num_cpus: Optional[int] = None,
    density: bool = False,
):
    """
    Generate energy histograms for each of the final state particles.

    Parameters
    ----------
    num_ps_pts : int
        Total number of phase space points to generate.
    masses : numpy.ndarray
        List of masses of the final state particles.
    cme : double
        Center-of-mass-energy of the process.
    mat_elem_sqrd : optional, Callable[[np.ndarray], float]
        Function for the matrix element squared.
    num_bins : int
        Number of energy bins to use for each of the final state particles.
    num_cpus : int {None]
        Number of cpus to use in parallel with rambo. If not specified, 75% of
        the cpus will be used.
    density: Bool
        If true, the histograms will be normalized to have unit area underneath
        the curves, i.e. they will be probability density functions.

    Returns
    -------
    energy_histograms : numpy.ndarray
        List of energies and dsigma/dE's. The resulting array has the shape
        (num_fsp, 2, num_bins). The array is formatted as::

            [[[E11, E12, ....], [hist11, hist12, ...]],
             ...
             [[EN1, EN2, ....], [histM1, histN2, ...]]]

    Examples
    --------

    Making energy histograms for 4 final state particles and plotting their
    energy spectra::

        from hazma import rambo
        import numpy as np
        num_ps_pts = 100000
        masses = np.array([100., 100., 0.0, 0.0])
        cme = 1000.
        num_bins = 100
        eng_hist = rambo.generate_energy_histogram(masses, cme,
                                                   num_ps_pts=num_ps_pts
                                                   num_bins=num_bins)
        import matplotlib.pyplot as plt
        for i in range(len(masses)):
            plt.loglog(eng_hist[i, 0], eng_hist[i, 1])

    """
    if not hasattr(masses, "__len__"):
        masses = [masses]  # type: ignore

    if cme < sum(masses):
        raise RamboCMETooSmall()

    num_fsp = len(masses)

    if mat_elem_sqrd is None:
        msqrd = __flat_squared_matrix_element
    else:
        msqrd = mat_elem_sqrd

    pts = generate_phase_space(masses, cme, num_ps_pts, msqrd, num_cpus)

    actual_num_ps_pts = pts.shape[0]

    return histogram.space_to_energy_hist(
        pts, actual_num_ps_pts, num_fsp, num_bins, density=density
    )


def integrate_over_phase_space(
    fsp_masses: MassList,
    cme: float,
    num_ps_pts: int = 10000,
    mat_elem_sqrd: Optional[SquaredMatrixElement] = None,
    num_cpus: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Returns the integral over phase space given a squared matrix element, a
    set of final state particle masses and a given energy.

    Parameters
    ----------
    fsp_masses : numpy.ndarray
        List of masses of the final state particles.
    cme : double
        Center-of-mass-energy of the process.
    num_ps_pts : int {10000]
        Total number of phase space points to generate.
    mat_elem_sqrd : optional, Callable[[np.ndarray], float]
        Function for the matrix element squared.
    num_cpus : int {None]
        Number of cpus to use in parallel with rambo. If not specified, 75% of
        the cpus will be used.

    Returns
    -------
    integral : float
        The result of the integral over phase space.
    std : float
        The estimated error in the integral over phase space.

    """
    if not hasattr(fsp_masses, "__len__"):
        fsp_masses = np.array([fsp_masses])

    if cme < np.sum(fsp_masses):
        raise RamboCMETooSmall()

    num_fsp = len(fsp_masses)
    if mat_elem_sqrd is None:
        msqrd = __flat_squared_matrix_element
    else:
        msqrd = mat_elem_sqrd
    points = generate_phase_space(fsp_masses, cme, num_ps_pts, msqrd, num_cpus)
    actual_num_ps_pts = len(points[:, 4 * num_fsp])
    weights = points[:, 4 * num_fsp]
    integral = np.average(weights)
    std = np.std(weights) / np.sqrt(actual_num_ps_pts)

    return integral, std  # type: ignore


def compute_annihilation_cross_section(
    isp_masses: MassList,
    fsp_masses: MassList,
    cme: float,
    num_ps_pts: int = 10000,
    mat_elem_sqrd: Optional[SquaredMatrixElement] = None,
    num_cpus: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Computes the cross section for a given process.

    Parameters
    ----------
    isp_masses : numpy.ndarray
        List of masses of the initial state particles.
    fsp_masses : numpy.ndarray
        List of masses of the final state and particles.
    cme : double
        Center-of-mass-energy of the process.
    num_ps_pts : int
        Total number of phase space points to generate.
    mat_elem_sqrd : optional, Callable[[np.ndarray], float]
        Function for the matrix element squared.
    num_cpus : int {None]
        Number of cpus to use in parallel with rambo. If not specified, 75% of
        the cpus will be used.

    Returns
    -------
    cross_section : double
        Cross section for X -> final state particles(fsp), where the fsp have
        masses `masses` and the process X -> fsp has a squared matrix element
        of `mat_elem_sqrd`.
    std : double
        Estimated error in cross section.

    Examples
    --------

    Compute the cross section for electrons annihilating into muons through a
    photon. First, we construct a function for the matrix element::

        from hazma.parameters import electron_mass as me
        from hazma.parameters import muon_mass as mmu
        from hazma.parameters import qe
        MDot = lambda p1, p2: (p1[0] * p2[0] - p1[1] * p1[1] - p1[2] * p1[2] -
                               p1[3] * p1[3])
        def msqrd(momenta):
            Q = sum(momenta)[0] # center-of-mass energy
            # Momenta of the incoming electrons
            p1 = np.array([Q / 2., 0., 0., np.sqrt(Q**2 / 4. - me**2)])
            p2 = np.array([Q / 2., 0., 0., -np.sqrt(Q**2 / 4. - me**2)])
            # Momenta for the outgoing muons
            p3 = momenta[0]
            p4 = momenta[1]
            # Mandelstam variables
            s = MDot(p1 + p2, p1 + p2)
            t = MDot(p1 - p3, p1 - p3)
            u = MDot(p1 - p4, p1 - p4)
            return (2 * qe**4 * (t**2 + u**2 - 4 * (t + u)* me**2 +
                    6 * me**4 + 4 * s * mmu**2 + 2 * mmu**4)) / s**2

    Next we integrate over phase space using RAMBO::

        from hazma.rambo import compute_annihilation_cross_section
        import numpy as np
        isp_masses = np.array([me, me])
        fsp_masses = np.array([mmu, mmu])
        cme = 1000.
        compute_annihilation_cross_section(
            isp_masses, fsp_masses, cme, num_ps_pts=5000, mat_elem_sqrd=msqrd)

    """
    if mat_elem_sqrd is None:
        msqrd = __flat_squared_matrix_element
    else:
        msqrd = mat_elem_sqrd
    integral, std = integrate_over_phase_space(
        fsp_masses,
        cme,
        num_ps_pts=num_ps_pts,
        mat_elem_sqrd=msqrd,
        num_cpus=num_cpus,
    )

    m1 = isp_masses[0]
    m2 = isp_masses[1]

    cross_section = integral * cross_section_prefactor(m1, m2, cme)
    error = cross_section_prefactor(m1, m2, cme) * std

    return cross_section, error


def compute_decay_width(
    fsp_masses: MassList,
    cme: float,
    num_ps_pts: int = 10000,
    mat_elem_sqrd: Optional[SquaredMatrixElement] = None,
    num_cpus: Optional[int] = None,
) -> Tuple[float, float]:
    r"""
    Computes the decay width for a given process.

    Parameters
    ----------
    fsp_masses : numpy.ndarray
        List of masses of the final state particles.
    cme : double
        Center-of-mass-energy of the process.
    num_ps_pts : int
        Total number of phase space points to generate.
    mat_elem_sqrd : optional, Callable[[np.ndarray], float]
        Function for the matrix element squared.
    num_cpus : int {None]
        Number of cpus to use in parallel with rambo. If not specified, 75% of
        the cpus will be used.

    Returns
    -------
    cross_section : double
        Cross section for X -> final state particles (FSPs), where the FSPs have
        masses `masses` and the process X -> FSPs has a squared matrix element
        of `mat_elem_sqrd`.
    std : double
        Estimated error in cross section.

    Examples
    --------

    In this example we compute the partial decay width of the muon for
    :math:`\mu~\to~e\nu\nu`

    First, we declare the matrix element::

        from hazma.parameters import GF
        MDot = lambda p1, p2: (p1[0] * p2[0] - p1[1] * p1[1] - p1[2] * p1[2] -
                               p1[3] * p1[3])
        def msqrd(momenta):
            pe = momenta[0]
            pve = momenta[1]
            pvmu = momenta[2]
            pmu = sum(momenta)
            return 64. * GF**2 * MDot(pe, pvmu) * MDot(pmu, pve)

    Next, we compute the decay width::

        from hazma.rambo import compute_decay_width
        from hazma.parameters import electron_mass as me
        from hazma.parameters import muon_mass as mmu
        import numpy as np
        fsp_masses = np.array([me, 0.0, 0.0])
        cme = mmu
        compute_decay_width(fsp_masses, cme, mat_elem_sqrd=msqrd)
    """
    if mat_elem_sqrd is None:
        msqrd = __flat_squared_matrix_element
    else:
        msqrd = mat_elem_sqrd
    integral, std = integrate_over_phase_space(
        fsp_masses,
        cme,
        num_ps_pts=num_ps_pts,
        mat_elem_sqrd=msqrd,
        num_cpus=num_cpus,
    )

    width = integral / (2.0 * cme)
    error = std / (2.0 * cme)

    return width, error


class PhaseSpace:
    """
    Phase space generator and integrator.
    """

    def __init__(
        self,
        cme: float,
        masses: Union[np.ndarray, Sequence[float]],
        msqrd: Optional[SquaredMatrixElement] = None,
    ) -> None:
        """
        Parameters
        ----------
        cme: float
            Center-of-mass energy of the proccess.
        masses: array-like
            Array of the final state particle masses.
        msqrd: Callable[[ndarray], float], optional
            Squared matrix element of the proccess.
        """
        ms = np.array(masses)
        n = len(ms)

        self.__cme = cme
        self.__masses = ms.reshape((n, 1))
        self.__msqrd: Optional[SquaredMatrixElement] = msqrd

        self.__n = n
        self.__xi0 = np.sqrt(1.0 - (np.sum(ms) / cme) ** 2)
        self.__base_wgt = self._compute_base_weight()

        self.__rng = np.random.default_rng()

        self.__eps = 2 ** (-52)

    @property
    def cme(self) -> float:
        """
        Center-of-mass energy of the proccess.
        """
        return self.__cme

    @cme.setter
    def cme(self, val) -> None:
        self.__cme = val
        self.__base_wgt = self._compute_base_weight()

    @property
    def masses(self) -> np.ndarray:
        """
        Masses of the final state particles.
        """
        return self.__masses.squeeze()

    @masses.setter
    def masses(self, masses: npt.NDArray) -> None:
        self.__n = len(masses)
        self.__masses = np.array(masses).reshape((self.__n, 1))
        self.__base_wgt = self._compute_base_weight()

    @property
    def msqrd(self) -> Optional[SquaredMatrixElement]:
        """
        Squared matrix element of the proccess.
        """
        return self.__msqrd

    def _compute_base_weight(self) -> float:
        """
        Compute the starting weight for each phase space point.
        """
        cme = self.__cme
        n = self.__n

        fact_nm2 = math.factorial(n - 2)
        fact = 1.0 / ((n - 1) * fact_nm2**2)

        return (
            fact
            * (0.5 * np.pi) ** (n - 1)
            * cme ** (2 * n - 4)
            * (0.5 / np.pi) ** (3 * n - 4)
        )

    def _initialize_momenta(self, batch_size: int, dtype=np.float64) -> np.ndarray:
        """
        Initialize isotropic and massless momenta with energyies
        distributed according to E exp(-E).

        Parameters
        ----------
        batch_size: int
            Number of phase space points to generate.

        Returns
        -------
        momenta: ndarray
            Array containing the initialized four momenta with shape
            (4, # final state particles, batch_size). The 1st dimension
            contains the energy, x-, y- and z-components of the 3-momenta. The
            2nd dimension runs over all final state particles. The 3rd
            dimension runs over all events.
        """
        n = self.__n

        rho1 = self.__rng.random(size=(n, batch_size), dtype=dtype)
        rho2 = self.__rng.random(size=(n, batch_size), dtype=dtype)
        rho3 = self.__rng.random(size=(n, batch_size), dtype=dtype)
        rho4 = self.__rng.random(size=(n, batch_size), dtype=dtype)

        ctheta = 2 * rho1 - 1.0
        stheta = np.sqrt(1.0 - ctheta**2)
        phi = 2.0 * np.pi * rho2
        e = -np.log(rho3 * rho4)

        return np.array(
            [e, e * stheta * np.cos(phi), e * stheta * np.sin(phi), e * ctheta]
        )

    def _boost(self, ps: np.ndarray) -> np.ndarray:
        """
        Boost momenta into the center-of-mass frame. Input should be the
        result of `_initialize_momenta`.

        Parameters
        ----------
        ps: ndarray
            Momenta generated from `_initialize_momenta`.

        Returns
        -------
        momenta: ndarray
            Momenta in the center-of-mass frame. The shape is idential to
            input shape.
        """
        sum_ps = np.sum(ps, axis=1)
        inv_mass = np.sqrt(
            sum_ps[0] ** 2 - sum_ps[1] ** 2 - sum_ps[2] ** 2 - sum_ps[3] ** 2
        )
        inv_mass = 1.0 / inv_mass

        bx = -inv_mass * sum_ps[1]
        by = -inv_mass * sum_ps[2]
        bz = -inv_mass * sum_ps[3]

        x = self.__cme * inv_mass
        g = sum_ps[0] * inv_mass
        a = 1.0 / (1.0 + g)

        bdotp = bx * ps[1] + by * ps[2] + bz * ps[3]
        fact = a * bdotp + ps[0]

        ps[0] = x * (g * ps[0] + bdotp)
        ps[1] = x * (fact * bx + ps[1])
        ps[2] = x * (fact * by + ps[2])
        ps[3] = x * (fact * bz + ps[3])

        return ps

    def _compute_scale_factor(self, ps: np.ndarray, iterations=5) -> np.ndarray:
        """
        Compute the scale factor needed to correct the masses of the final
        state particle momenta.

        Parameters
        ----------
        ps: ndarray
            Momenta returned from `_boost`.

        Returns
        -------
        xi: ndarray
            Scale factors for each phase-space point. The shape is
            (batch_size,) where `batch_size` is the size of the last
            dimension of `ps`.
        """
        cme = self.__cme
        xi0 = self.__xi0

        shape = ps.shape[1:]
        e = ps[0]
        xi = xi0 * np.ones((shape[-1],))
        it = 0
        last = False

        while not last:
            deltaf = np.hypot(e * xi, self.__masses)
            f = np.sum(deltaf, axis=0) - cme
            df = np.sum(xi * e**2 / deltaf, axis=0)
            xi = xi - f / df

            # Newton iterations have quadratic convergence. If we are below sqrt(eps),
            # do only one more iteration
            if np.max(f) < np.sqrt(self.__eps) or it == iterations - 1:
                last = True

            it = it + 1

        return xi

    def _correct_masses(self, ps: np.ndarray) -> np.ndarray:
        """
        Correct the masses of the final state particles.

        Parameters
        ----------
        ps: ndarray
            Momenta returned from `_boost`.

        Returns
        -------
        ps: ndarray
            Momenta with corrected masses. The shape is identical to input.
        """
        xi = self._compute_scale_factor(ps)
        ps[0] = np.hypot(xi * ps[0], self.__masses)
        ps[1:] = xi * ps[1:]
        return ps

    def _compute_weights(self, ps) -> np.ndarray:
        """
        Compute the weights of phase space points.

        Parameters
        ----------
        ps: ndarray
            Momenta returned from `_correct_masses`.

        Returns
        -------
        weights: ndarray
            Weights of the phase space points. The shape is (batch_size, )
            where `batch_size` is the size of the last dimension of `ps`.
        """
        cme = self.__cme
        n = self.__n

        modsqr = np.sum(ps[1:] ** 2, axis=0)
        mod = np.sqrt(modsqr)

        t1 = np.sum(mod / cme, axis=0) ** (2 * n - 3)
        t2 = np.sum(modsqr / ps[0], axis=0)
        t3 = np.prod(mod / ps[0], axis=0)

        return t1 / t2 * t3 * self.__cme * self.__base_wgt

    def generator(
        self, n, batch_size: int, seed: Optional[int] = None, dtype=np.float64
    ) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        """
        Create a generator the yields four-momenta and weights distributed
        according to Lorentz-invariant phase space.

        Parameters
        ----------
        n: int
            Number of phase space points to generate.
        batch_size: int
            Number of phase space points to generate each step.
        seed: int, optional
            Seed used for numpy random number generator.
        dtype: DTypeLike, optional
            Type used for generation of momenta and weights.

        Yields
        -------
        momenta: ndarray
            Batch of momenta containing the initialized four momenta with shape
            (4, # final state particles, batch_size). The 1st dimension
            contains the energy, x-, y- and z-components of the 3-momenta. The
            2nd dimension runs over all final state particles. The 3rd
            dimension runs over all events.
        weights: ndarray
            Batch of weights of the phase space points. Shape is (batch_size,).

        Examples
        --------
        Integrating over phase-space in batches:

        >>> phase_space = PhaseSpace(cme=10.0, masses=[1.0, 2.0, 3.0])
        >>> integrals = []
        >>> for _, weights in phase_space.generator(n=100, batch_size=10, seed=1234):
        ...     integrals.append(np.nanmean(integrals))
        >>> np.average(integrals)
        0.003632349411041629
        """
        if dtype == np.float64:
            self.__eps = 2 ** (-52)
        elif dtype == np.float32:
            self.__eps = 2 ** (-23)

        self.__rng = np.random.default_rng(seed)

        niters = n // batch_size
        niters += 0 if batch_size * niters == n else 1

        for _ in range(niters):
            ps = self._initialize_momenta(batch_size, dtype=dtype)
            ps = self._boost(ps)
            ps = self._correct_masses(ps)
            ws = self._compute_weights(ps)

            if self.__msqrd is not None:
                ws *= self.__msqrd(ps)
            yield (ps, ws)

    def generate(
        self, n, seed: Optional[int] = None, dtype=np.float64
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate four-momenta and weights distributed according to
        Lorentz-invariant phase space.

        Parameters
        ----------
        n: int
            Number of phase space points to generate.
        seed: int, optional
            Seed used for numpy random number generator.
        dtype: DTypeLike, optional
            Type used for generation of momenta and weights.

        Returns
        -------
        momenta: ndarray
            Array containing the initialized four momenta with shape
            (4, # final state particles, n). The 1st dimension
            contains the energy, x-, y- and z-components of the 3-momenta. The
            2nd dimension runs over all final state particles. The 3rd
            dimension runs over all events.
        weights: ndarray
            Array containing weights of the phase space points. Shape is given by
            (n,).

        Examples
        --------
        Integrating over phase-space:

        >>> phase_space = PhaseSpace(cme=10.0, masses=[1.0, 2.0, 3.0])
        >>> integrals = []
        >>> momenta, weights = phase_space.generator(n=100, seed=1234)
        >>> np.average(weights)
        0.0037198111038192366
        """
        if dtype == np.float64:
            self.__eps = 2 ** (-52)
        elif dtype == np.float32:
            self.__eps = 2 ** (-23)

        self.__rng = np.random.default_rng(seed)

        ps = self._initialize_momenta(n, dtype=dtype)
        ps = self._boost(ps)
        ps = self._correct_masses(ps)
        ws = self._compute_weights(ps)

        if self.__msqrd is not None:
            ws *= self.__msqrd(ps)

        return ps, ws

    def _integrate_two_body(self) -> Tuple[float, float]:
        cme = self.cme
        m1, m2 = self.masses
        p = np.sqrt(kallen_lambda(cme**2, m1**2, m2**2)) / (2 * cme)
        e1 = np.hypot(m1, p)
        e2 = np.hypot(m2, p)
        ps = np.zeros((4, 2), dtype=np.float64)

        msqrd = __flat_squared_matrix_element if self.msqrd is None else self.msqrd

        def integrand(z):
            sin = np.sqrt(1 - z**2)
            ps[:, 0] = np.array([e1, sin * p, 0.0, z * p])
            ps[:, 1] = np.array([e2, -sin * p, 0.0, -z * p])
            return msqrd(ps)

        pre = 1.0 / (8.0 * np.pi) * p / cme

        integral, error = integrate.quad(integrand, -1.0, 1.0)

        return integral * pre, error * pre

    def integrate(
        self,
        n: int,
        batch_size: Optional[int] = None,
        seed: Optional[int] = None,
        dtype=np.float64,
    ) -> Tuple[float, float]:
        """
        Integrate over phase space.

        Parameters
        ----------
        n: int
            Number of phase space points used in integration.
        batch_size: int, optional
            If not None, the phase-space integration will be broken up into
            batches, processing `batch_size` points at a time. Default is None.
        seed: int, optional
            Seed used for numpy random number generator.
        dtype: DTypeLike, optional
            Type used for generation of momenta and weights.

        Returns
        -------
        integral: float
            Value of the phase space integration.
        error_estimate: float
            Estimation of the error.
        """
        if batch_size is not None and not batch_size == n:
            integrals = []
            errors = []
            for _, ws in self.generator(n, batch_size, seed, dtype):
                avg = np.nanmean(ws, dtype=float)
                std = np.nanstd(ws, dtype=float, ddof=1) / np.sqrt(batch_size)
                integrals.append(avg)
                errors.append(std)

            # Average of averages okay since all samples are same size
            integral: float = np.nanmean(integrals, dtype=float)
            # Combined error estimate using quadrature
            error: float = np.sqrt(np.nansum(np.square(errors))) / len(errors)

            return integral, error

        _, weights = self.generate(n, seed=seed, dtype=dtype)
        integral = np.nanmean(weights, dtype=float)
        error = np.nanstd(weights, dtype=np.float64, ddof=1) / np.sqrt(n)

        return integral, error

    def decay_width(
        self,
        n,
        batch_size: Optional[int] = None,
        seed: Optional[int] = None,
        dtype=np.float64,
    ) -> Tuple[float, float]:
        """
        Compute the decay width.

        Parameters
        ----------
        n: int
            Number of phase space points used in integration.
        seed: int, optional
            Seed used for numpy random number generator.
        dtype: DTypeLike, optional
            Type used for generation of momenta and weights.

        Returns
        -------
        width: float
            Estimated value of the decay width.
        error_estimate: float
            Estimation of the error.
        """
        integral, std = self.integrate(n, batch_size=batch_size, seed=seed, dtype=dtype)

        pre = 0.5 / self.__cme
        width = pre * integral
        error = pre * std

        return width, error

    def cross_section(
        self,
        m1,
        m2,
        n,
        batch_size: Optional[int] = None,
        seed: Optional[int] = None,
        dtype=np.float64,
    ) -> Tuple[float, float]:
        """
        Compute the cross section given input masses.

        Parameters
        ----------
        m1: float
            Mass of the 1st incoming particle.
        m2: float
            Mass of the 2nd incoming particle.
        n: int
            Number of phase space points used in integration.
        seed: int, optional
            Seed used for numpy random number generator.
        dtype: DTypeLike, optional
            Type used for generation of momenta and weights.

        Returns
        -------
        cross_section: float
            Estimated value of the cross-section.
        error_estimate: float
            Estimation of the error.
        """
        integral, std = self.integrate(n, batch_size=batch_size, seed=seed, dtype=dtype)

        pre = cross_section_prefactor(m1, m2, self.__cme)
        cross_section = pre * integral
        error = pre * std

        return cross_section, error

    def energy_distributions(
        self,
        n: int,
        nbins: int,
        batch_size: Optional[int] = None,
        seed: Optional[int] = None,
        dtype=np.float64,
        keep_edges: bool = False,
    ) -> List[Tuple[RealArray, RealArray]]:
        """
        Generate energy distributions of the final state particles.

        Parameters
        ----------
        n: int
            Number of phase space points used in generating the distributions.
        nbins: int
            Number of bins to use for the distributions.
        batch_size: int, optional
            If not None, the phase-space integration will be broken up into
            batches, processing `batch_size` points at a time. Default is None.
        seed: int, optional
            Seed used for numpy random number generator.
        dtype: DTypeLike, optional
            Type used for generation of momenta and weights.

        Returns
        -------
        distributions: List[Tuple[np.ndarray, np.ndarray]]
            The energy distributions. Each entry in the returned list is a
            tuple with the first item the probability distribution
            (shape=(nbins,)) and the second the energy bin edges
            (shape=(nbins+1,)).
        """
        cme = self.cme
        mass_sum = np.sum(self.masses)

        def bounds(m):
            msum = mass_sum - m
            emin = m
            emax = (cme**2 + m**2 - msum**2) / (2 * cme)
            return emin, emax

        ebounds = [bounds(m) for m in self.masses]
        bins = [np.linspace(emin, emax, nbins + 1) for emin, emax in ebounds]

        distributions = [
            np.histogram(np.zeros(nbins, dtype=np.float64), bins=b)[0] for b in bins
        ]

        if batch_size is not None and not batch_size == n:
            # Build up the distributions from batches
            for ps, ws in self.generator(n, batch_size, seed, dtype):
                for i in range(len(self.masses)):
                    d = np.histogram(ps[0, i], bins=bins[i], weights=ws)[0]
                    distributions[i] += d
        else:
            ps, ws = self.generate(n, seed=seed, dtype=dtype)
            for i in range(len(self.masses)):
                distributions[i] = np.histogram(ps[0, i], bins=bins[i], weights=ws)[0]

        for i in range(len(self.masses)):
            distributions[i] = normalize_distribution(distributions[i], bins[i])

        if keep_edges:
            return [(dpde, e) for dpde, e in zip(distributions, bins)]

        centers = [0.5 * (b[1:] + b[:-1]) for b in bins]
        return [(dpde, c) for dpde, c in zip(distributions, centers)]

    def invariant_mass_distributions(
        self,
        n: int,
        nbins: int,
        pairs: Optional[List[Tuple[int, int]]] = None,
        batch_size: Optional[int] = None,
        seed: Optional[int] = None,
        dtype=np.float64,
        keep_edges: bool = False,
    ) -> Dict[Tuple[int, int], Tuple[RealArray, RealArray]]:
        """
        Generate invariant mass distributions of the final state particles.

        The invariant masses are defined as sqrt((pi + pj)^2), where pi and pj
        are the four-momenta of the ith and jth particles.

        Parameters
        ----------
        n: int
            Number of phase space points used in generating the distributions.
        nbins: int
            Number of bins to use for the distributions.
        batch_size: int, optional
            If not None, the phase-space integration will be broken up into
            batches, processing `batch_size` points at a time. Default is None.
        seed: int, optional
            Seed used for numpy random number generator.
        dtype: DTypeLike, optional
            Type used for generation of momenta and weights.

        Returns
        -------
        distributions: Dict[Tuple[int,int], Tuple[np.ndarray, np.ndarray]]
            The invariant mass distributions. Each entry in the returned
            dictionary is a tuple with the first item the probability
            distribution (shape=(nbins,)) and the second the bin edges
            (shape=(nbins+1,)). The keys are the pairs of indices specifying
            which two particles the invariant mass-squared distribution
            corresponds to.
        """
        cme = self.cme
        masses = self.masses
        mass_sum = np.sum(masses)
        nfsp = len(masses)

        if pairs is None:
            pairs_: List[Tuple[int, int]] = []
            for i in range(nfsp):
                for j in range(i + 1, nfsp):
                    pairs_.append((i, j))
        else:
            pairs_ = pairs

        def bounds(m1, m2):
            msum = mass_sum - m1 - m2
            mmin = m1 + m2
            mmax = cme - msum
            return mmin, mmax

        def make_bins(pair):
            mmin, mmax = bounds(masses[pair[0]], masses[pair[1]])
            return np.linspace(mmin, mmax, nbins + 1)

        def inv_mass(ps, pair):
            i, j = pair
            return np.sqrt(np.abs(lnorm_sqr(ps[:, i] + ps[:, j])))

        bins = [make_bins(pair) for pair in pairs_]

        distributions = [
            np.histogram(np.zeros(nbins, dtype=np.float64), bins=b)[0] for b in bins
        ]

        if batch_size is not None and not batch_size == n:
            # Build up the distributions from batches
            for ps, ws in self.generator(n, batch_size, seed, dtype):
                for i, pair in enumerate(pairs_):
                    d = np.histogram(inv_mass(ps, pair), bins=bins[i], weights=ws)[0]
                    distributions[i] += d
        else:
            ps, ws = self.generate(n, seed=seed, dtype=dtype)
            for i, pair in enumerate(pairs_):
                distributions[i] = np.histogram(
                    inv_mass(ps, pair), bins=bins[i], weights=ws
                )[0]

        for i in range(len(pairs_)):
            distributions[i] = normalize_distribution(distributions[i], bins[i])

        if keep_edges:
            return {pair: (distributions[i], bins[i]) for i, pair in enumerate(pairs_)}

        centers = [0.5 * (b[1:] + b[:-1]) for b in bins]
        return {pair: (distributions[i], centers[i]) for i, pair in enumerate(pairs_)}
