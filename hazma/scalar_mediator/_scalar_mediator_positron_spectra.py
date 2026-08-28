import numpy as np

from hazma import spectra

# Served by the Rust extension since cython-to-rust Task 6.3 deleted
# `hazma/scalar_mediator/scalar_mediator_positron_spec.pyx`. Import
# paths, call signatures and returned values are unchanged; the `_core`
# names are spelled out because the vector twin's Cython names collide
# with the photon pair Task 6.2 registered. See
# `rust/src/kernels/mediator_decay_positron.rs` for the port.
#
# Both names are bound here rather than imported directly, because
# `dnde_decay_s_pt` is a re-export: it was a public `def` of the deleted
# extension and this module is what stands in for it, but nothing below
# calls it.
from hazma._core import scalar_mediator as _core

dnde_decay_s = _core.dnde_positron_decay_s
dnde_decay_s_pt = _core.dnde_positron_decay_s_pt


def dnde_pos_pipi(_, positron_energies, cme):
    """
    Positron/electron spectrum from dark matter annihilating into charged
    pions.

    Parameters
    ----------
    positron_energies: float or np.array
        Energies of the positrons/electrons
    cme: float
        Center of mass energy

    Returns
    -------
    dnde: float or np.array
        Positron spectrum evaluated at the `positron_energies`.
    """
    return spectra.dnde_positron_charged_pion(positron_energies, cme / 2.0)


def dnde_pos_mumu(_, positron_energies, cme):
    """
    Positron/electron spectrum from dark matter annihilating into muons

    Parameters
    ----------
    positron_energies: float or np.array
        Energies of the positrons/electrons
    cme: float
        Center of mass energy

    Returns
    -------
    dnde: float or np.array
        Positron spectrum evaluated at the `positron_energies`.
    """
    return spectra.dnde_positron_muon(positron_energies, cme / 2.0)


def dnde_pos_ss(self, positron_energies, cme, fs="total"):
    """
    Positron/electron spectrum from dark matter annihilating into muons

    Parameters
    ----------
    positron_energies: float or np.array
        Energies of the positrons/electrons
    cme: float
        Center of mass energy
    fs: str {'total'}
        String for which final states to consider when computing scalar
        mediator decay spectrum. Options are 'total', 'pi pi' or 'mu mu'.

    Returns
    -------
    dnde: float or np.array
        Positron spectrum evaluated at the `positron_energies`.
    """
    scalar_energy = cme / 2.0
    pws = self.partial_widths()

    if pws["total"] != 0:
        pw_array = np.array([pws["e e"], pws["mu mu"], pws["pi pi"]])
        pw_array /= pws["total"]

        # Factor of 2 since S is self-conjugate
        return 2.0 * dnde_decay_s(
            positron_energies, scalar_energy, self.ms, pw_array, fs
        )
    else:
        return np.zeros_like(positron_energies)


def positron_spectrum_funcs(self):
    return {
        "mu mu": self.dnde_pos_mumu,
        "pi pi": self.dnde_pos_pipi,
        "s s": self.dnde_pos_ss,
    }


def positron_lines(self, cme):
    """
    Positron/electron lines from dark matter annihilating into
    electrons/positrons.

    Parameters
    ----------
    cme: float
        Center of mass energy

    Returns
    -------
    dnde: dict
        Dictionary of dictionaries. Each sub-dictionary contains the
        location of the line and the branching fraction for the
        corresponding dark matter annihilation process.
    """
    bf = self.annihilation_branching_fractions(cme)["e e"]

    return {"e e": {"energy": cme / 2.0, "bf": bf}}
