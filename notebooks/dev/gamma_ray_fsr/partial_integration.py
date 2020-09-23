from hazma.rambo import generate_phase_space
from hazma.rh_neutrino import RHNeutrino
from hazma.parameters import (
    Vud,
    GF,
    fpi,
    qe,
    charged_pion_mass as mpi,
)
from hazma.field_theory_helper_functions.common_functions import (
    minkowski_dot as LDot,
)
from hazma.gamma_ray import gamma_ray_fsr
import numpy as np
import matplotlib.pyplot as plt

sw = np.sqrt(0.222)


def msqrd_n_to_pi_l_g(momenta, model):
    smix = model.stheta
    mx = model.mx
    ml = model.ml
    p1 = momenta[0]  # lepton
    p2 = momenta[1]  # pion
    p3 = momenta[2]  # photon
    P = p1 + p2 + p3
    s = LDot(P - p1, P - p1)
    t = LDot(P - p2, P - p2)

    return (
        (
            8
            * fpi ** 2
            * GF ** 2
            * qe ** 2
            * smix ** 2
            * (
                -2 * ml ** 8 * mpi ** 2
                + ml ** 6
                * (
                    mpi ** 4
                    + mx ** 2 * (6 * mpi ** 2 - 2 * s)
                    + 2 * mpi ** 2 * t
                    + s * (s + 2 * t)
                )
                + mx ** 2
                * t
                * (
                    -2 * mx ** 4 * (mpi ** 2 - s)
                    - (mpi ** 4 + s ** 2) * (mpi ** 2 - s - t)
                    + 2 * mx ** 2 * (mpi ** 4 - s * (s + t))
                )
                - ml ** 4
                * (
                    -3 * mpi ** 6
                    + mx ** 4 * (6 * mpi ** 2 - 4 * s)
                    - mpi ** 2 * s * (3 * s + 4 * t)
                    + mpi ** 4 * (5 * s + 4 * t)
                    + s * (s ** 2 + 4 * s * t + 2 * t ** 2)
                    + mx ** 2
                    * (
                        -5 * mpi ** 4
                        - s * (s - 2 * t)
                        + mpi ** 2 * (4 * s + 6 * t)
                    )
                )
                + ml ** 2
                * (
                    2 * mx ** 6 * (mpi ** 2 - s)
                    - (mpi ** 4 + s ** 2) * (mpi ** 2 - s - t) * t
                    + mx ** 4
                    * (-4 * mpi ** 4 - 2 * s * t + mpi ** 2 * (4 * s + 6 * t))
                    + mx ** 2
                    * (
                        3 * mpi ** 6
                        + mpi ** 2 * s * (3 * s - 4 * t)
                        - mpi ** 4 * (5 * s + 2 * t)
                        + s * (-(s ** 2) + 2 * s * t + 4 * t ** 2)
                    )
                )
            )
            * Vud ** 2
        )
        / ((mpi ** 2 - s) ** 2 * (ml ** 2 - t) ** 2)
        / 2.0
    )


def width_nu_l_l(model):
    """
    Compute the width for right-handed neutrino to an active neutrino and
    two charged leptons.

    Returns
    -------
    width: float
        Partial width for N -> nu + l + l.
    """
    mx = model.mx
    ml = model.ml

    if mx < 2.0 * ml:
        return 0.0

    smix = model.stheta

    return (
        -(GF ** 2)
        * smix ** 2
        * (
            mx
            * np.sqrt(-4 * ml ** 2 + mx ** 2)
            * (
                -(mx ** 6 * (1 + 4 * sw ** 2 + 8 * sw ** 4))
                + 12 * ml ** 6 * (1 + 12 * sw ** 2 + 24 * sw ** 4)
                + 2 * ml ** 2 * mx ** 4 * (7 + 20 * sw ** 2 + 40 * sw ** 4)
                - 2 * ml ** 4 * mx ** 2 * (-1 + 36 * sw ** 2 + 72 * sw ** 4)
            )
            - 48
            * (
                -(ml ** 4 * mx ** 4)
                - 8 * ml ** 6 * mx ** 2 * sw ** 2 * (1 + 2 * sw ** 2)
                + ml ** 8 * (1 + 12 * sw ** 2 + 24 * sw ** 4)
            )
            * np.log((2 * ml) / (mx + np.sqrt(-4 * ml ** 2 + mx ** 2)))
        )
    ) / (192.0 * mx ** 3 * np.pi ** 3)


def msqrd_n_to_nu_l_l(momenta, model):
    """
    Compute the squared matrix-element for a RH neutrino decaying into a
    neutrino and two charged leptons at leading order in the Fermi constant.
    Momenta are ordered as follows: {nu,l+,l-}.

    Parameters
    ----------
    momenta: List
        List of NumPy arrays storing the four-momenta of the final state
        particles.

    Returns
    -------
    msqrd: float
        The matrix element for N -> nu + l + l for the given model and
        four-momenta.
    """
    pnu = momenta[0]
    plp = momenta[1]
    plm = momenta[2]
    P = pnu + plp + plm
    s = LDot(P - pnu, P - pnu)
    t = LDot(P - plp, P - plp)

    smix = model.stheta
    ml = model.ml
    mx = model.mx
    return (
        8
        * GF ** 2
        * smix ** 2
        * (-1 + smix ** 2)
        * (
            2 * ml ** 4 * (1 + 4 * sw ** 2 + 8 * sw ** 4)
            + 2
            * ml ** 2
            * (mx ** 2 - s - 2 * (1 + 4 * sw ** 2 + 8 * sw ** 4) * t)
            + (1 + 4 * sw ** 2 + 8 * sw ** 4)
            * (s ** 2 + 2 * s * t + 2 * t ** 2 - mx ** 2 * (s + 2 * t))
        )
    )


def msqrd_n_to_nu_l_l_g(momenta, model):
    k1 = momenta[0]
    k2 = momenta[1]
    k3 = momenta[2]
    k4 = momenta[3]

    smix = model.stheta
    ml = model.ml

    return (
        16
        * GF ** 2
        * qe ** 2
        * smix ** 2
        * (-1 + smix ** 2)
        * (
            2
            * (1 + 4 * sw ** 2 + 8 * sw ** 4)
            * LDot(k1, k3) ** 2
            * LDot(k2, k4) ** 2
            * LDot(k3, k4)
            + 2
            * (1 + 4 * sw ** 2 + 8 * sw ** 4)
            * LDot(k1, k2) ** 2
            * LDot(k2, k4)
            * LDot(k3, k4) ** 2
            + LDot(k1, k4)
            * (
                (1 + 4 * sw ** 2 + 8 * sw ** 4)
                * LDot(k2, k4) ** 3
                * (ml ** 2 - LDot(k3, k4))
                + ml ** 2
                * LDot(k3, k4) ** 2
                * (
                    (ml + 4 * ml * sw ** 2) ** 2
                    + (1 + 4 * sw ** 2 + 8 * sw ** 4) * LDot(k2, k3)
                    + (1 + 4 * sw ** 2 + 8 * sw ** 4) * LDot(k3, k4)
                )
                + LDot(k2, k4) ** 2
                * (
                    (1 + 4 * sw ** 2 + 8 * sw ** 4)
                    * LDot(k2, k3)
                    * (ml ** 2 - 2 * LDot(k3, k4))
                    + ml ** 2
                    * (
                        (ml + 4 * ml * sw ** 2) ** 2
                        + (-1 + 4 * sw ** 2 + 8 * sw ** 4) * LDot(k3, k4)
                    )
                )
                - LDot(k2, k4)
                * LDot(k3, k4)
                * (
                    -8 * ml ** 2 * sw ** 2 * (1 + 2 * sw ** 2) * LDot(k1, k4)
                    + 2 * (1 + 4 * sw ** 2 + 8 * sw ** 4) * LDot(k2, k3) ** 2
                    + 2
                    * LDot(k2, k3)
                    * (
                        (ml + 4 * ml * sw ** 2) ** 2
                        + (1 + 4 * sw ** 2 + 8 * sw ** 4) * LDot(k3, k4)
                    )
                    + LDot(k3, k4)
                    * (
                        ml ** 2 * (1 - 4 * sw ** 2 - 8 * sw ** 4)
                        + (1 + 4 * sw ** 2 + 8 * sw ** 4) * LDot(k3, k4)
                    )
                )
            )
            + LDot(k1, k3)
            * (
                (1 + 4 * sw ** 2 + 8 * sw ** 4)
                * LDot(k2, k4) ** 3
                * (ml ** 2 - LDot(k3, k4))
                + ml ** 2
                * LDot(k3, k4) ** 2
                * (
                    (ml + 4 * ml * sw ** 2) ** 2
                    + 2 * (1 + 4 * sw ** 2 + 8 * sw ** 4) * LDot(k1, k4)
                    + LDot(k2, k3)
                    + LDot(k3, k4)
                    + 4
                    * sw ** 2
                    * (1 + 2 * sw ** 2)
                    * (LDot(k2, k3) + LDot(k3, k4))
                )
                + LDot(k2, k4) ** 2
                * (
                    ml ** 4 * (1 + 4 * sw ** 2) ** 2
                    + (1 + 4 * sw ** 2 + 8 * sw ** 4)
                    * (
                        LDot(k2, k3) * (ml ** 2 - 2 * LDot(k3, k4))
                        + ml ** 2 * LDot(k3, k4)
                    )
                )
                - LDot(k2, k4)
                * LDot(k3, k4)
                * (
                    2 * (1 + 4 * sw ** 2 + 8 * sw ** 4) * LDot(k2, k3) ** 2
                    + (1 + 4 * sw ** 2 + 8 * sw ** 4)
                    * LDot(k3, k4)
                    * (-(ml ** 2) + 2 * LDot(k1, k4) + LDot(k3, k4))
                    + 2
                    * LDot(k2, k3)
                    * (
                        (ml + 4 * ml * sw ** 2) ** 2
                        + (1 + 4 * sw ** 2 + 8 * sw ** 4) * LDot(k1, k4)
                        + (1 + 4 * sw ** 2 + 8 * sw ** 4) * LDot(k3, k4)
                    )
                )
            )
            + LDot(k1, k2)
            * (
                (1 + 4 * sw ** 2 + 8 * sw ** 4)
                * LDot(k2, k4) ** 3
                * (ml ** 2 - LDot(k3, k4))
                + LDot(k2, k4)
                * LDot(k3, k4)
                * (
                    -2
                    * LDot(k2, k3)
                    * (
                        (ml + 4 * ml * sw ** 2) ** 2
                        + 2 * (1 + 4 * sw ** 2 + 8 * sw ** 4) * LDot(k1, k3)
                        + LDot(k1, k4)
                        + LDot(k2, k3)
                        + 4
                        * sw ** 2
                        * (1 + 2 * sw ** 2)
                        * (LDot(k1, k4) + LDot(k2, k3))
                    )
                    + (1 + 4 * sw ** 2 + 8 * sw ** 4)
                    * (ml ** 2 - 2 * LDot(k1, k3) - 2 * LDot(k2, k3))
                    * LDot(k3, k4)
                    - (1 + 4 * sw ** 2 + 8 * sw ** 4) * LDot(k3, k4) ** 2
                )
                + ml ** 2
                * LDot(k3, k4) ** 2
                * (
                    (ml + 4 * ml * sw ** 2) ** 2
                    + 2 * (1 + 4 * sw ** 2 + 8 * sw ** 4) * LDot(k1, k3)
                    + LDot(k2, k3)
                    + LDot(k3, k4)
                    + 4
                    * sw ** 2
                    * (1 + 2 * sw ** 2)
                    * (LDot(k2, k3) + LDot(k3, k4))
                )
                + LDot(k2, k4) ** 2
                * (
                    ml ** 4 * (1 + 4 * sw ** 2) ** 2
                    + (1 + 4 * sw ** 2 + 8 * sw ** 4)
                    * (
                        ml ** 2
                        * (2 * (LDot(k1, k3) + LDot(k1, k4)) + LDot(k2, k3))
                        + (
                            ml ** 2
                            - 2 * LDot(k1, k3)
                            - 2 * LDot(k1, k4)
                            - 2 * LDot(k2, k3)
                        )
                        * LDot(k3, k4)
                    )
                )
            )
        )
    ) / (LDot(k2, k4) ** 2 * LDot(k3, k4) ** 2)


def __dnde_rambo(photon_energy, mx, masses, msqrd, width, nevents=1000):
    if mx * (mx - 2 * photon_energy) < np.sum(masses) ** 2:
        return (0.0, 0.0)

    # Energy of the photon in the rest frame where final state particles
    # (excluding the photon)
    e_gamma = (photon_energy * mx) / np.sqrt(mx * (-2 * photon_energy + mx))
    # Total energy of the final state particles (excluding the photon) in their
    # rest frame
    cme = np.sqrt(mx * (-2 * photon_energy + mx))
    # Number of final state particles
    nfsp = len(masses)
    # Generate events for the final state particles in their rest frame
    events = generate_phase_space(masses, cme, nevents)

    # Photon momenta in N + photon rest frame
    phis = np.random.rand(nevents) * 2.0 * np.pi
    cts = 2.0 * np.random.rand(nevents) - 1.0
    g_momenta = [
        np.array(
            [
                e_gamma,
                e_gamma * np.cos(phi) * np.sqrt(1 - ct ** 2),
                e_gamma * np.sin(phi) * np.sqrt(1 - ct ** 2),
                e_gamma * ct,
            ]
        )
        for phi, ct in zip(phis, cts)
    ]

    # momenta in the rest frame of N + photon
    fsp_momenta = [
        np.append(event[:-1], pg) for event, pg in zip(events, g_momenta)
    ]

    weights = [event[-1] for event in events]

    terms = [
        weight * msqrd(ps_fps.reshape((nfsp + 1, 4)))
        for ps_fps, weight in zip(fsp_momenta, weights)
    ]
    res = np.average(terms)
    std = np.std(terms) / np.sqrt(nevents)
    pre = (
        1.0
        / (2.0 * mx)
        / width
        * photon_energy
        / (16 * np.pi ** 3)
        * (4.0 * np.pi)
    )

    return pre * res, pre * std


def dnde_rambo(photon_energies, mx, masses, msqrd, width, nevents=1000):
    if hasattr(photon_energies, "__len__"):
        return np.array(
            [
                __dnde_rambo(e, mx, masses, msqrd, width, nevents=nevents)
                for e in photon_energies
            ]
        )
    return __dnde_rambo(
        photon_energies, mx, masses, msqrd, width, nevents=nevents
    )


def test_n_to_pi_e_g():
    model = RHNeutrino(500.0, 1e-3, "e")
    es = np.logspace(-2, np.log10(model.mx), 200)

    dndes = model.dnde_pi_l_fsr(es)
    msqrd = lambda moms: msqrd_n_to_pi_l_g(moms, model)
    dndesr = dnde_rambo(
        es, model.mx, [model.ml, mpi], msqrd, model.width_pi_l()
    )

    plt.plot(es, [dnde[0] for dnde in dndesr])
    plt.fill_between(
        es,
        [dnde[0] + dnde[1] for dnde in dndesr],
        [dnde[0] - dnde[1] for dnde in dndesr],
        alpha=0.7,
        color="mediumorchid",
    )
    plt.plot(es, dndes, ls="--", c="k")
    plt.yscale("log")
    plt.xscale("log")
    plt.xlim([np.min(es), np.max(es)])
    plt.show()


def test_n_to_pi_mu_g():
    model = RHNeutrino(500.0, 1e-3, "mu")
    es = np.logspace(-2, np.log10(model.mx), 200)

    dndes = model.dnde_pi_l_fsr(es)
    msqrd = lambda moms: msqrd_n_to_pi_l_g(moms, model)
    dndesr = dnde_rambo(
        es, model.mx, [model.ml, mpi], msqrd, model.width_pi_l()
    )

    plt.plot(es, [dnde[0] for dnde in dndesr])
    plt.fill_between(
        es,
        [dnde[0] + dnde[1] for dnde in dndesr],
        [dnde[0] - dnde[1] for dnde in dndesr],
        alpha=0.7,
        color="mediumorchid",
    )
    plt.plot(es, dndes, ls="--", c="k")
    plt.yscale("log")
    plt.xscale("log")
    plt.xlim([np.min(es), np.max(es)])
    plt.show()


def test_n_to_nu_e_e_g():
    model = RHNeutrino(500.0, 1e-3, "e")
    es = np.logspace(-2, np.log10(model.mx), 100)

    msqrd = lambda moms: msqrd_n_to_nu_l_l_g(moms, model)
    msqrd_tree = lambda moms: msqrd_n_to_nu_l_l(moms, model)
    dndes = gamma_ray_fsr(
        [model.mx],
        [0.0, model.ml, model.ml, 0.0],
        model.mx,
        mat_elem_sqrd_tree=msqrd_tree,
        mat_elem_sqrd_rad=msqrd,
        num_ps_pts=500000,
        num_bins=50,
    )
    dndesr = dnde_rambo(
        es, model.mx, [0.0, model.ml, model.ml], msqrd, width_nu_l_l(model)
    )
    print([dnde[0] for dnde in dndesr])
    plt.plot(es, [dnde[0] for dnde in dndesr])
    plt.fill_between(
        es,
        [dnde[0] + dnde[1] for dnde in dndesr],
        [dnde[0] - dnde[1] for dnde in dndesr],
        alpha=0.7,
        color="mediumorchid",
    )
    plt.plot(dndes[0], dndes[1])
    plt.yscale("log")
    plt.xscale("log")
    plt.xlim([np.min(es), np.max(es)])
    plt.show()


def test_n_to_nu_mu_mu_g():
    model = RHNeutrino(500.0, 1e-3, "mu")
    es = np.logspace(-2, np.log10(model.mx), 100)

    msqrd = lambda moms: msqrd_n_to_nu_l_l_g(moms, model)
    msqrd_tree = lambda moms: msqrd_n_to_nu_l_l(moms, model)
    dndes = gamma_ray_fsr(
        [model.mx],
        [0.0, model.ml, model.ml, 0.0],
        model.mx,
        mat_elem_sqrd_tree=msqrd_tree,
        mat_elem_sqrd_rad=msqrd,
        num_ps_pts=500000,
        num_bins=50,
    )
    dndesr = dnde_rambo(
        es, model.mx, [0.0, model.ml, model.ml], msqrd, width_nu_l_l(model)
    )
    print([dnde[0] for dnde in dndesr])
    plt.plot(es, [dnde[0] for dnde in dndesr])
    plt.fill_between(
        es,
        [dnde[0] + dnde[1] for dnde in dndesr],
        [dnde[0] - dnde[1] for dnde in dndesr],
        alpha=0.7,
        color="mediumorchid",
    )
    plt.plot(dndes[0], dndes[1])
    plt.yscale("log")
    plt.xscale("log")
    plt.xlim([np.min(es), np.max(es)])
    plt.show()


if __name__ == "__main__":
    # test_n_to_pi_e_g()
    # test_n_to_pi_mu_g()
    # test_n_to_nu_e_e_g()
    test_n_to_nu_mu_mu_g()

