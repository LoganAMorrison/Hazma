from hazma.scalar_mediator import ScalarMediator
from hazma.parameters import vh
from hazma.parameters import electron_mass as me
import numpy as np
import os
from os import path


def e_cm_mw(mx, vrel=1e-3):
    """Computes DM COM energy, assuming its velocity is much less than c.
    """
    return 2 * mx * (1 + 0.5 * vrel**2)


def save_data(params_list, Models):
    """Generates and saves data for a set of scalar mediator models.
    """
    # Make data directory
    data_dir = path.join(path.dirname(__file__), "data")
    if not path.exists(path.join(data_dir)):
        os.makedirs(data_dir)

    for i, (params, Model) in enumerate(zip(params_list, Models)):
        # Make directory for model
        cur_dir = path.join(data_dir, "sm_{}".format(i + 1))
        print("writing tests to {}".format(cur_dir))
        if not path.exists(cur_dir):
            os.makedirs(cur_dir)

        SM = ScalarMediator(**params)
        np.save(path.join(cur_dir, "params.npy"), params)

        # Particle physics quantities
        e_cm = e_cm_mw(SM.mx)
        np.save(path.join(cur_dir, "e_cm.npy"), e_cm)
        np.save(path.join(cur_dir, "ann_cross_sections.npy"), SM.annihilation_cross_sections(e_cm))
        np.save(path.join(cur_dir, "ann_branching_fractions.npy"), SM.annihilation_branching_fractions(e_cm))
        np.save(path.join(cur_dir, "partial_widths.npy"), SM.partial_widths())
        np.save(path.join(cur_dir, "vs.npy"), SM.compute_vs())

        # Gamma-ray spectra
        e_gams = np.geomspace(1.0, e_cm, 10)
        np.save(path.join(cur_dir, "e_gams.npy"), e_gams)
        np.save(path.join(cur_dir, "spectra.npy"), SM.spectra(e_gams, e_cm))
        np.save(path.join(cur_dir, "gamma_ray_lines.npy"), SM.gamma_ray_lines(e_cm))

        # Positron spectra
        e_ps = np.geomspace(me, e_cm, 10)
        np.save(path.join(cur_dir, "e_ps.npy"), e_ps)
        np.save(path.join(cur_dir, "positron_spectra.npy"), SM.positron_spectra(e_ps, e_cm))
        np.save(path.join(cur_dir, "positron_lines.npy"), SM.positron_lines(e_cm))

def generate_test_data():
    params = []
    # Higgs-portal couplings
    stheta = 1e-3

    params.append({
        "mx": 250.0,
        "ms": 125.0,
        "gsxx": 1.0,
        "gsff": stheta,
        "gsGG": 3 * stheta,
        "gsFF": -5 / 6 * stheta,
        "lam": vh,
    })
    params.append({
        "mx": 250.0,
        "ms": 550.0,
        "gsxx": 1.0,
        "gsff": stheta,
        "gsGG": 3 * stheta,
        "gsFF": -5 / 6 * stheta,
        "lam": vh,
    })

    save_data(params, 2 * [ScalarMediator])


if __name__ == "__main__":
    generate_test_data()
