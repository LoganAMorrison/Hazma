from hazma.vector_mediator import VectorMediator, KineticMixing
from hazma.parameters import vh
from hazma.parameters import electron_mass as me
import numpy as np
import os
from os import path


def e_cm_mw(mx, vrel=1e-3):
    """Computes DM COM energy, assuming its velocity is much less than c.
    """
    return 2 * mx * (1 + 0.5 * vrel ** 2)


def save_data(params_list, Models):
    """Generates and saves data for a set of scalar mediator models.
    """
    # Make data directory
    data_dir = path.join(path.dirname(__file__), "data")
    if not path.exists(path.join(data_dir)):
        os.makedirs(data_dir)

    for i, (params, Model) in enumerate(zip(params_list, Models)):
        # Make directory for model
        cur_dir = path.join(data_dir, "vm_{}".format(i + 1))
        print("writing tests to {}".format(cur_dir))
        if not path.exists(cur_dir):
            os.makedirs(cur_dir)

        model = Model(**params)
        np.save(path.join(cur_dir, "params.npy"), params)

        # Particle physics quantities
        e_cm = e_cm_mw(model.mx)
        np.save(path.join(cur_dir, "e_cm.npy"), e_cm)
        np.save(
            path.join(cur_dir, "ann_cross_sections.npy"),
            model.annihilation_cross_sections(e_cm),
        )
        np.save(
            path.join(cur_dir, "ann_branching_fractions.npy"),
            model.annihilation_branching_fractions(e_cm),
        )
        np.save(path.join(cur_dir, "partial_widths.npy"), model.partial_widths())

        # Gamma-ray spectra
        e_gams = np.geomspace(1.0, e_cm, 10)
        np.save(path.join(cur_dir, "e_gams.npy"), e_gams)
        np.save(path.join(cur_dir, "spectra.npy"), model.spectra(e_gams, e_cm))
        np.save(path.join(cur_dir, "gamma_ray_lines.npy"), model.gamma_ray_lines(e_cm))

        # Positron spectra
        e_ps = np.geomspace(me, e_cm, 10)
        np.save(path.join(cur_dir, "e_ps.npy"), e_ps)
        np.save(
            path.join(cur_dir, "positron_spectra.npy"),
            model.positron_spectra(e_ps, e_cm),
        )
        np.save(path.join(cur_dir, "positron_lines.npy"), model.positron_lines(e_cm))


def generate_test_data():
    params = []

    mx = 250.0
    eps = 0.1
    gvxx = 1.0
    mvs = 2 * [125.0, 550.0]
    gvuus = 4 * [1.0]
    gvdds = 2 * [1.0, -1.0]
    gvsss = 4 * [1.0]
    gvees = 4 * [1.0]
    gvmumus = 4 * [1.0]

    for mv in [125.0, 550.0]:
        params.append({"mx": mx, "mv": mv, "gvxx": gvxx, "eps": eps})

    for mv, gvuu, gvdd, gvss, gvee, gvmumu in zip(
        mvs, gvuus, gvdds, gvsss, gvees, gvmumus
    ):
        params.append(
            {
                "mx": mx,
                "mv": mv,
                "gvxx": gvxx,
                "gvuu": gvuu,
                "gvdd": gvdd,
                "gvss": gvss,
                "gvee": gvee,
                "gvmumu": gvmumu,
            }
        )

    save_data(params, 2 * [KineticMixing] + 4 * [VectorMediator])


if __name__ == "__main__":
    generate_test_data()
