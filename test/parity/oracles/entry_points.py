r"""Where the *Cython* value of a corpus case comes from, today.

The parity corpus (``test/parity/data``) was captured from pre-port
Cython, and its manifest records the entry point each case was captured
through. Those entry points no longer resolve to Cython: the
``cython-to-rust`` port repointed most of them at ``hazma._core`` and
deleted the ``def``\ s (Tasks 4.1-4.5) and then, by Task 6.4, every
``.pyx`` that carried one. Nothing here resolves against a module in the
working tree any more; each case names the source and the revision a
capture would have to restore first.

This module maps each case name this project needs back to a callable
with the *pre-port* signature, so `test/parity/cases.py`'s own `Block`
objects — the same grids, the same call shapes — can drive it unchanged.
Three resolution kinds, in ascending order of how much they had to be
recovered:

``live``
    The ``.pyx`` still carries the ``def``. **No case is ``live`` any
    more.** All seven ``mediator_spectra.*`` cases were, until
    ``cython-to-rust`` Task 6.2 deleted the two decay modules and Task
    6.3 the two positron ones. The kind is kept because it is what the
    committed ``data/manifest.json`` records for the captures taken while
    those modules were alive, and because a future capture of a
    not-yet-ported entry point would use it again.

``capsule``
    The ``.pyx`` survives but its ``def`` is gone; the ``cdef`` is exported
    as a ``PyCapsule`` in ``__pyx_capi__``. `capsule_entry_point` rebuilds
    the deleted ``def``: scalar in, scalar out, looping over an array
    argument exactly as the deleted ``*_array`` ``cdef`` did. Same shim
    ``test/test_core_boost.py`` uses, and the same ``ctypes.PYFUNCTYPE``
    caveat applies — see that module's docstring.

``restored``
    The ``.pyx`` itself is gone and was resurrected from git history for
    the capture; see `test/parity/oracles/README.md`. The entry point is
    the original ``def``, in a module built from the original source, so
    resolution is ordinary ``importlib``.

Whether a shim reproduces the deleted ``def`` is not assumed. The capture's
baseline pass drives every case here through its *unpatched* build and
requires bit-for-bit equality with the stored corpus array; nothing is
captured until that passes.
"""

from __future__ import annotations

import ctypes
import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

#: The C signature every ``*_point`` ``cdef`` this module drives carries.
#: The capsule *name* is the signature, so `capsule_entry_point` checks it
#: rather than trusting that the argument list has not changed —
#: ``test/test_core_boost.py::TestOracle`` draws the same line.
POINT_SIGNATURE = b"double (double, double)"

Kind = Literal["live", "capsule", "restored"]


@dataclass(frozen=True)
class Source:
    """How one corpus case's Cython value is reached.

    Parameters
    ----------
    module : str
        Importable module holding the entry point.
    function : str
        ``def`` name for ``live`` / ``restored``; ``__pyx_capi__`` key for
        ``capsule``.
    kind : str
        ``"live"``, ``"capsule"`` or ``"restored"`` — see the module
        docstring.
    note : str
        Why this case needs this kind. Recorded in the oracle manifest, so
        a later reader does not have to re-derive the port's state at
        capture time.
    """

    module: str
    function: str
    kind: Kind
    note: str


def capsule_entry_point(module_name: str, capsule_name: str) -> Callable[..., Any]:
    r"""Rebuild a deleted ``def`` over a surviving ``cdef``'s capsule.

    Parameters
    ----------
    module_name : str
        Module whose ``__pyx_capi__`` holds the capsule.
    capsule_name : str
        Key in that mapping. Its capsule name must be
        :data:`POINT_SIGNATURE`.

    Returns
    -------
    callable
        ``(energies, parent_energy)`` accepting a scalar or a 1-D array,
        mirroring the ``def``\ s the port deleted: array in, array out.

    Raises
    ------
    KeyError
        If the module exports no such capsule.
    AssertionError
        If the capsule's signature is not :data:`POINT_SIGNATURE` — the
        argument list changed and the ``ctypes`` prototype below would be
        reading the wrong stack.
    """
    module = importlib.import_module(module_name)
    capsule = module.__pyx_capi__[capsule_name]

    get_name = ctypes.pythonapi.PyCapsule_GetName
    get_name.restype = ctypes.c_char_p
    get_name.argtypes = [ctypes.py_object]
    signature = get_name(capsule)
    assert signature == POINT_SIGNATURE, (
        f"{module_name}.{capsule_name} is {signature!r}, "
        f"not {POINT_SIGNATURE!r}; the shim below would misread its arguments"
    )

    get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    get_pointer.restype = ctypes.c_void_p
    get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    address = get_pointer(capsule, signature)

    # PYFUNCTYPE, not CFUNCTYPE: the latter releases the GIL, and these
    # cdefs reach Python (scipy's `quad`, numpy allocation), so a
    # CFUNCTYPE call segfaults. test/test_core_boost.py's module
    # docstring is where that was established.
    point = ctypes.PYFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)(
        address
    )

    def entry_point(
        energies: float | np.ndarray, parent_energy: float
    ) -> float | np.ndarray:
        if hasattr(energies, "__len__"):
            grid = np.asarray(energies, dtype=np.float64)
            assert grid.ndim == 1, "energies must be 0- or 1-dimensional"
            return np.array(
                [point(float(x), float(parent_energy)) for x in grid],
                dtype=np.float64,
            )
        return point(float(energies), float(parent_energy))

    return entry_point


def resolve(source: Source) -> Callable[..., Any]:
    """Return the callable `source` names, whatever kind it is."""
    if source.kind == "capsule":
        return capsule_entry_point(source.module, source.function)
    return getattr(importlib.import_module(source.module), source.function)


_MEDIATOR_DECAY_RESTORED = (
    "deleted by cython-to-rust Task 6.2; RESTORED_SOURCES in defects.py "
    "carries the revision"
)
_MEDIATOR_POSITRON_RESTORED = (
    "deleted by cython-to-rust Task 6.3; RESTORED_SOURCES in defects.py "
    "carries the revision"
)
#: The four capi survivors, deleted by cython-to-rust Task 6.4. Until then
#: these four cases were ``capsule``: the ``.pyx`` was still built, its
#: ``def`` gone and its ``cdef`` reachable through ``__pyx_capi__``. With
#: the file deleted there is no capsule to read, so they join the
#: ``restored`` majority.
_SURVIVORS_RESTORED = (
    "def deleted in Phase 04, the file itself by cython-to-rust Task 6.4; "
    "RESTORED_SOURCES in defects.py carries the revision, and lists the "
    "headers and cimported twins a restore has to compile against"
)
_TABLES_RESTORED = (
    "deleted by cython-to-rust Task 4.2 (0954e5a); restored from git for " "the capture"
)
_RHO_RESTORED = (
    "deleted by cython-to-rust Task 4.5 (b5f7f90); restored from git for " "the capture"
)

#: Corpus case name -> how to reach its Cython value. Covers the twenty
#: cases `projects/parity-pinned-defect-repair/references/defect-blast-radius.md`
#: puts inside some Group A defect's radius, and nothing else.
SOURCES: dict[str, Source] = {
    # -- A1's consumers: the seven tabulated photon spectra ----------------
    "spectra.photon.eta": Source(
        "hazma.spectra._photon._eta", "dnde_photon_eta", "restored", _TABLES_RESTORED
    ),
    "spectra.photon.eta_prime": Source(
        "hazma.spectra._photon._eta_prime",
        "dnde_photon_eta_prime",
        "restored",
        _TABLES_RESTORED,
    ),
    "spectra.photon.omega": Source(
        "hazma.spectra._photon._omega",
        "dnde_photon_omega",
        "restored",
        _TABLES_RESTORED,
    ),
    "spectra.photon.phi": Source(
        "hazma.spectra._photon._phi", "dnde_photon_phi", "restored", _TABLES_RESTORED
    ),
    "spectra.photon.charged_kaon": Source(
        "hazma.spectra._photon._kaon",
        "dnde_photon_charged_kaon",
        "restored",
        _TABLES_RESTORED,
    ),
    "spectra.photon.long_kaon": Source(
        "hazma.spectra._photon._kaon",
        "dnde_photon_long_kaon",
        "restored",
        _TABLES_RESTORED,
    ),
    "spectra.photon.short_kaon": Source(
        "hazma.spectra._photon._kaon",
        "dnde_photon_short_kaon",
        "restored",
        _TABLES_RESTORED,
    ),
    # -- A2 / A3: the muon and charged-pion photon chains ------------------
    "spectra.photon.muon": Source(
        "hazma.spectra._photon._muon",
        "dnde_photon",
        "restored",
        _SURVIVORS_RESTORED,
    ),
    "spectra.photon.charged_pion": Source(
        "hazma.spectra._photon._pion",
        "dnde_photon_charged_pion",
        "restored",
        _SURVIVORS_RESTORED,
    ),
    "spectra.photon.charged_rho": Source(
        "hazma.spectra._photon._rho",
        "dnde_photon_charged_rho",
        "restored",
        _RHO_RESTORED,
    ),
    "spectra.photon.neutral_rho": Source(
        "hazma.spectra._photon._rho",
        "dnde_photon_neutral_rho",
        "restored",
        _RHO_RESTORED,
    ),
    "mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum": Source(
        "hazma.scalar_mediator.scalar_mediator_decay_spectrum",
        "scalar_mediator_decay_spectrum",
        "restored",
        _MEDIATOR_DECAY_RESTORED,
    ),
    "mediator_spectra.vector.photon.dnde_decay_v": Source(
        "hazma.vector_mediator.vector_mediator_decay_spectrum",
        "dnde_decay_v",
        "restored",
        _MEDIATOR_DECAY_RESTORED,
    ),
    "mediator_spectra.vector.photon.dnde_decay_v_pt": Source(
        "hazma.vector_mediator.vector_mediator_decay_spectrum",
        "dnde_decay_v_pt",
        "restored",
        _MEDIATOR_DECAY_RESTORED,
    ),
    # -- A4: the positron chain -------------------------------------------
    "spectra.positron.muon": Source(
        "hazma.spectra._positron._muon",
        "dnde_positron_muon",
        "restored",
        _SURVIVORS_RESTORED,
    ),
    "spectra.positron.charged_pion": Source(
        "hazma.spectra._positron._pion",
        "dnde_positron_charged_pion",
        "restored",
        # The committed `data/manifest.json` records this note as it read
        # when Task 2 captured the arrays: "def deleted by cython-to-rust
        # Task 4.4; the .pyx itself dies at Task 4.6, the earliest of the
        # deletion waves". Both halves were prospective and both turned
        # out wrong -- Task 4.6 deleted the `def`, not Task 4.4, and it
        # kept the file, because both mediator positron spectrum modules
        # cimported `dnde_positron_charged_pion_array`. Task 6.3 deleted
        # those two and Task 6.4 the file. The capture is unaffected: the
        # arrays were taken while the capsule was live, and this Source
        # only says how to reach that value again.
        _SURVIVORS_RESTORED,
    ),
    "mediator_spectra.scalar.positron.dnde_decay_s": Source(
        "hazma.scalar_mediator.scalar_mediator_positron_spec",
        "dnde_decay_s",
        "restored",
        _MEDIATOR_POSITRON_RESTORED,
    ),
    "mediator_spectra.scalar.positron.dnde_decay_s_pt": Source(
        "hazma.scalar_mediator.scalar_mediator_positron_spec",
        "dnde_decay_s_pt",
        "restored",
        _MEDIATOR_POSITRON_RESTORED,
    ),
    "mediator_spectra.vector.positron.dnde_decay_v": Source(
        "hazma.vector_mediator.vector_mediator_positron_spec",
        "dnde_decay_v",
        "restored",
        _MEDIATOR_POSITRON_RESTORED,
    ),
    "mediator_spectra.vector.positron.dnde_decay_v_pt": Source(
        "hazma.vector_mediator.vector_mediator_positron_spec",
        "dnde_decay_v_pt",
        "restored",
        _MEDIATOR_POSITRON_RESTORED,
    ),
}
