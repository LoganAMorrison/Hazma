r"""The four Group A defects, the file each patches, and what it reaches.

Group A is the half of `projects/parity-pinned-defect-repair`'s roster
that still has a live Cython twin, so its corrected values can be captured
from an implementation that predates the Rust port. The roster itself, and
which corpus cases each defect's radius covers, is
`projects/parity-pinned-defect-repair/references/defect-blast-radius.md`;
`cases` below quotes that file's per-defect enumeration and is checked
against the corpus manifest by `test/parity/test_oracles.py`.

The case lists here are what the capture *drives*, not what the repair
will end up declaring. That file says so of itself: it is a prediction
from the composition graph, and each repair task re-derives its own row.
Capturing a case the repair turns out not to move costs one array; not
capturing one it does move costs the oracle, permanently.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Cases reached through the muon photon kernel. A3 shares all but the
#: muon case itself, which is why the two are spelled as one tuple plus a
#: difference rather than as two hand-written lists that can drift.
_MUON_PHOTON_CHAIN = (
    "spectra.photon.charged_pion",
    "spectra.photon.charged_rho",
    "spectra.photon.neutral_rho",
    "mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum",
    "mediator_spectra.vector.photon.dnde_decay_v",
    "mediator_spectra.vector.photon.dnde_decay_v_pt",
)


@dataclass(frozen=True)
class Defect:
    """One Group A defect, as the capture needs it.

    Parameters
    ----------
    label : str
        ``A1`` .. ``A4``, the labels `PLAN.md` and the blast-radius
        reference use.
    summary : str
        One line, in the shape of the follow-up's title.
    follow_up : str
        Repo-relative path to the follow-up that filed it.
    repair_task : str
        Which `PLAN.md` task consumes this oracle.
    patch : str
        Repo-relative path to the unified diff applied for the capture,
        under ``patches/``. Exactly one ``.pyx`` per defect, so a captured
        array carries one repair and not a combination of them.
    source : str
        The ``.pyx`` that patch edits.
    cases : tuple of str
        Corpus case names to capture, in blast-radius order.
    """

    label: str
    summary: str
    follow_up: str
    repair_task: str
    patch: str
    source: str
    cases: tuple[str, ...]


DEFECTS: dict[str, Defect] = {
    "A1": Defect(
        label="A1",
        summary="the boost integral mis-covers its window at both ends",
        follow_up="docs/followups/todo/boost-integral-drops-last-interior-cell.md",
        repair_task="Task 4",
        patch="test/parity/oracles/patches/A1-boost-integral-window.patch",
        source="hazma/_utils/boost.pyx",
        cases=(
            "spectra.photon.eta",
            "spectra.photon.eta_prime",
            "spectra.photon.omega",
            "spectra.photon.phi",
            "spectra.photon.charged_kaon",
            "spectra.photon.long_kaon",
            "spectra.photon.short_kaon",
        ),
    ),
    "A2": Defect(
        label="A2",
        summary="the muon photon rest-frame branch stops short of the endpoint",
        follow_up=(
            "docs/followups/todo/"
            "photon-muon-rest-frame-endpoint-uses-the-wrong-power-of-r.md"
        ),
        repair_task="Task 7",
        patch="test/parity/oracles/patches/A2-muon-rest-frame-endpoint.patch",
        source="hazma/spectra/_photon/_muon.pyx",
        cases=("spectra.photon.muon", *_MUON_PHOTON_CHAIN),
    ),
    "A3": Defect(
        label="A3",
        summary="the charged-pion photon spectrum returns zero in the forward cone",
        follow_up=(
            "docs/followups/todo/charged-pion-photon-spectrum-misses-the-forward-cone.md"
        ),
        repair_task="Task 8",
        patch="test/parity/oracles/patches/A3-charged-pion-forward-cone.patch",
        source="hazma/spectra/_photon/_pion.pyx",
        cases=_MUON_PHOTON_CHAIN,
    ),
    "A4": Defect(
        label="A4",
        summary="the muon positron spectrum divides by its normalization",
        follow_up="docs/followups/todo/positron-muon-spectrum-normalization-inverted.md",
        repair_task="Task 10",
        patch="test/parity/oracles/patches/A4-positron-muon-normalization.patch",
        source="hazma/spectra/_positron/_muon.pyx",
        cases=(
            "spectra.positron.muon",
            "spectra.positron.charged_pion",
            "mediator_spectra.scalar.positron.dnde_decay_s",
            "mediator_spectra.scalar.positron.dnde_decay_s_pt",
            "mediator_spectra.vector.positron.dnde_decay_v",
            "mediator_spectra.vector.positron.dnde_decay_v_pt",
        ),
    ),
}

#: Sources deleted by the port that the composition chains above need, and
#: the revision each is recovered from. `README.md` carries the restore
#: command; the capture records a digest of the recovered bytes so a later
#: reader can prove which source produced the arrays.
RESTORED_SOURCES: dict[str, str] = {
    "hazma/spectra/_photon/_eta.pyx": "0954e5a^",
    "hazma/spectra/_photon/_eta.pxd": "0954e5a^",
    "hazma/spectra/_photon/_eta_prime.pyx": "0954e5a^",
    "hazma/spectra/_photon/_eta_prime.pxd": "0954e5a^",
    "hazma/spectra/_photon/_kaon.pyx": "0954e5a^",
    "hazma/spectra/_photon/_kaon.pxd": "0954e5a^",
    "hazma/spectra/_photon/_omega.pyx": "0954e5a^",
    "hazma/spectra/_photon/_omega.pxd": "0954e5a^",
    "hazma/spectra/_photon/_phi.pyx": "0954e5a^",
    "hazma/spectra/_photon/_phi.pxd": "0954e5a^",
    "hazma/spectra/_photon/path.py": "0954e5a^",
    "hazma/spectra/_photon/_rho.pyx": "b5f7f90^",
    "hazma/spectra/_photon/_rho.pxd": "b5f7f90^",
    # cython-to-rust Task 6.2 deleted
    # `hazma/{scalar,vector}_mediator/*_mediator_decay_spectrum.pyx`, which
    # the three `mediator_spectra.*.photon.*` cases run through. They are
    # deliberately *not* listed here: every revision above is a literal
    # SHA so a restore is reproducible, and that task could not know the
    # SHA of its own commit. A re-capture must add both files with
    # `git log -1 --format=%h --diff-filter=D -- <path>` suffixed `^`.
    # Tracked as `docs/followups/todo/oracle-restore-revisions-for-the-
    # mediator-decay-pyx.md`.
}
