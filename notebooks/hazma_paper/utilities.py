from hazma.scalar_mediator import ScalarMediator
from hazma.vector_mediator import VectorMediator

import matplotlib.pyplot as plt

# latex_text_width_1col = 5.75113  # inches. UCSC thesis class.
latex_text_width_1col = 7.05826  # inches. Article class.

# Useful for setting text/curve colors
colors = [c["color"] for c in list(plt.rcParams["axes.prop_cycle"])]


def get_tex_label(fs):
    """Makes labels look nice.

    Parameters
    ----------
    fs : string
        An annihilation final state for one of the models defined in hazma.

    Returns
    -------
    label : string
        The LaTeX string to be used for labeling plots with the final state.
    """
    tex_label = r"$" + fs
    tex_label = tex_label.replace("pi0", r"\pi^0")
    tex_label = tex_label.replace("pi pi", r"\pi^+ \pi^-")
    tex_label = tex_label.replace("mu mu", r"\mu^+ \mu^-")
    tex_label = tex_label.replace("g", r"\gamma")
    tex_label = tex_label.replace("e e", r"e^+ e^-")
    tex_label = tex_label.replace("x x", r"\bar{\chi} \chi")
    tex_label = tex_label.replace("s s", r"S S")
    tex_label = tex_label.replace("v v", r"V V")
    return tex_label + r"$"


def get_color(fs):
    """Ensures that colors for different final states are standardized across
    all files.

    Parameters
    ----------
    fs : string
        An annihilation final state for one of the models defined in hazma.

    Returns
    -------
    The color to be used when plotting that final state.
    """
    fss = set(ScalarMediator.list_annihilation_final_states() +
              VectorMediator.list_annihilation_final_states()) - \
        set(['s s', 'v v'])

    fss = sorted(list(fss)) + ['s s', 'v v', 'x x']

    return (2*list(plt.rcParams["axes.prop_cycle"]))[fss.index(fs)]["color"]
