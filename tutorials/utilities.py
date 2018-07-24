from hazma.scalar_mediator import ScalarMediator
from hazma.pseudo_scalar_mediator import PseudoScalarMediator
from hazma.vector_mediator import VectorMediator

import matplotlib.pyplot as plt

latex_text_width_1col = 5.75113  # inches


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
    tex_label = tex_label.replace("pi0", "\pi^0")
    tex_label = tex_label.replace("pi pi", "\pi^+ \pi^-")
    tex_label = tex_label.replace("mu mu", "\mu^+ \mu^-")
    tex_label = tex_label.replace("g", "\gamma")
    tex_label = tex_label.replace("e e", "e^+ e^-")
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
    fss = sorted(list(set(ScalarMediator.list_final_states() +
                          VectorMediator.list_final_states() +
                          PseudoScalarMediator.list_final_states()) -
                      set(['s s', 'p p', 'v v']))) + ['s s', 'p p', 'v v']
    return (2*list(plt.rcParams["axes.prop_cycle"]))[fss.index(fs)]["color"]
