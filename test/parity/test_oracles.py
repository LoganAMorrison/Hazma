r"""The gate on ``test/parity/oracles`` — the corrected-value captures.

`test/parity/test_parity.py` asks "does this build still reproduce what
2.1.0 shipped?". This module asks a narrower and cheaper question about a
different set of arrays: is the committed capture from the Cython twins
intact, does it describe the corpus it claims to, and was it taken
somewhere the corpus's own numbers mean anything?

None of these evaluate a kernel. The capture cannot be re-run from a
repaired tree by construction — it needs a *patched Cython* build, and
`cython-to-rust` Task 6.4 deletes the last of those sources — so what is
checkable afterwards is the record, not the measurement. That is the same
line `python test/parity/generate.py --check` draws around the corpus.

The one substantive claim here is the platform one. An oracle captured on
a different libm is a measurement of a different libm, and asserting it
globally is `docs/agents/lessons.md`
``[platform-scoped-oracle-asserted-globally]``. Both platforms are read
out of their own manifests and compared through
`tolerances._libm_identity`, which is the repo's existing answer to "same
libm?" and already knows an OS point release is not a platform change.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "oracles"))

import capture as oracles  # noqa: E402  (needs the sys.path entries above)
import defects as roster  # noqa: E402
import entry_points  # noqa: E402
import generate as corpus_generate  # noqa: E402
import tolerances  # noqa: E402

CORPUS_MANIFEST = corpus_generate.load_manifest()

pytestmark = pytest.mark.skipif(
    not oracles.MANIFEST_PATH.exists(),
    reason="no oracle capture committed",
)


@pytest.fixture(scope="module")
def manifest() -> dict[str, Any]:
    """The committed oracle manifest."""
    return oracles.load_manifest()


def test_the_committed_arrays_hash_to_the_manifest() -> None:
    """`capture.py --check`, as a test.

    Same integrity check, same code path, so the command a person runs by
    hand and the one CI runs cannot answer differently.
    """
    assert oracles.check() == 0


def test_the_capture_platform_matches_the_corpus_platform(
    manifest: dict[str, Any],
) -> None:
    """Every defect was captured where the corpus's own numbers mean something.

    A diff between an oracle and a corpus array taken on two libms
    measures the two libms as much as it measures the defect. The
    comparison is `tolerances._libm_identity` rather than the whole
    `platform.platform()` string, for the reason that function documents:
    the capturing machine's OS point release has already moved once, and
    calling that a platform change would be a false alarm.
    """
    want = tolerances._libm_identity(CORPUS_MANIFEST["environment"])
    for label, defect in manifest["defects"].items():
        got = tolerances._libm_identity(defect["environment"])
        assert got == want, (
            f"oracle {label} was captured on {got}, the corpus on {want}; "
            "the recorded per-defect diffs would carry libm noise"
        )


def test_the_capture_toolchain_matches_the_corpus_toolchain(
    manifest: dict[str, Any],
) -> None:
    """The versions that can move a value agree with the corpus manifest's.

    `platform` is excluded — `_libm_identity` above is what governs it,
    and the point release is allowed to differ. `hazma` is excluded for
    the reason `tolerances._NUMERICS_ENVIRONMENT_KEYS` excludes it: the
    version bumps without touching a number.
    """
    keys = ("python", "numpy", "scipy", "cython", "machine")
    want = {key: CORPUS_MANIFEST["environment"][key] for key in keys}
    for label, defect in manifest["defects"].items():
        got = {key: defect["environment"][key] for key in keys}
        assert got == want, f"oracle {label} toolchain {got} != corpus {want}"


def test_the_corpus_has_not_moved_under_the_capture(
    manifest: dict[str, Any],
) -> None:
    """The corpus manifest still hashes to what the capture measured against.

    Rule 1 forbids rewriting the corpus arrays, so this should never fire.
    If it does, every recorded diff below is against something that no
    longer exists and the capture has to be re-read before it is trusted.
    """
    assert (
        oracles.hash_file(oracles.CORPUS_MANIFEST) == manifest["corpus_manifest_sha256"]
    )


@pytest.mark.parametrize("label", sorted(roster.DEFECTS))
def test_every_defect_has_a_capture(label: str, manifest: dict[str, Any]) -> None:
    """The roster and the committed capture name the same four defects.

    A defect whose oracle went missing is the failure this project exists
    to prevent, and it must not be discoverable only by a repair task
    finding nothing to compare against.
    """
    assert label in manifest["defects"], f"{label} has no committed oracle"
    captured = set(manifest["defects"][label]["cases"])
    assert captured == set(roster.DEFECTS[label].cases), (
        f"{label} captured {sorted(captured)}, roster names "
        f"{sorted(roster.DEFECTS[label].cases)}"
    )


def test_every_case_the_roster_names_is_a_corpus_case() -> None:
    """`defects.py` quotes the blast-radius reference; the corpus decides.

    A typo in a case name would otherwise show up as a silently missing
    oracle rather than as a red test.
    """
    named = {name for defect in roster.DEFECTS.values() for name in defect.cases}
    unknown = sorted(named - set(CORPUS_MANIFEST["cases"]))
    assert not unknown, f"not corpus cases: {unknown}"
    assert named == set(entry_points.SOURCES), (
        "entry_points.SOURCES and the defect roster disagree: "
        f"{sorted(named ^ set(entry_points.SOURCES))}"
    )


def test_every_captured_array_matches_a_corpus_array(
    manifest: dict[str, Any],
) -> None:
    """Shapes and dtypes line up with the corpus arrays being corrected.

    The capture drives `test/parity/cases.py`'s own `Block` objects, so
    this should hold by construction — which is exactly why it is worth
    asserting: if the specification changes shape under a committed
    capture, the arrays stop being comparable and nothing else would say
    so.
    """
    for label, defect in manifest["defects"].items():
        for name, case in defect["cases"].items():
            corpus_blocks = CORPUS_MANIFEST["cases"][name]["blocks"]
            assert len(case["blocks"]) == len(corpus_blocks), (
                f"{label}/{name}: {len(case['blocks'])} blocks captured, "
                f"{len(corpus_blocks)} in the corpus"
            )
            for captured, stored in zip(case["blocks"], corpus_blocks, strict=True):
                assert captured["label"] == stored["label"]
                for suffix, entry in captured["arrays"].items():
                    assert suffix in stored["arrays"], (
                        f"{label}/{name}[{captured['label']}]: no {suffix} in "
                        "the corpus"
                    )
                    for field in ("shape", "dtype"):
                        assert entry[field] == stored["arrays"][suffix][field], (
                            f"{label}/{name}[{captured['label']}].{suffix}: "
                            f"{field} {entry[field]} != corpus "
                            f"{stored['arrays'][suffix][field]}"
                        )


@pytest.mark.parametrize("label", sorted(roster.DEFECTS))
def test_every_defect_moved_something(label: str, manifest: dict[str, Any]) -> None:
    """A capture that describes no change is a hole in the evidence.

    The same reasoning as the delta layer's rule that a declaration
    describing no change fails
    (`projects/parity-pinned-defect-repair/references/corpus-repinning.md`):
    an oracle identical to the corpus array it corrects would let a repair
    task "reproduce the oracle" without repairing anything.

    Scoped per defect, not per case. A defect legitimately reaches some
    cases and not others — that is what the recorded diff is for, and the
    blast-radius reference says outright it is a prediction. What no
    defect may do is move nothing at all.
    """
    diffs = manifest["defects"][label]["diff_against_corpus"]
    moved = {name: d["values_moved"] for name, d in diffs.items() if d["values_moved"]}
    assert moved, (
        f"{label}'s oracle is identical to the corpus everywhere; either the "
        f"patch {roster.DEFECTS[label].patch} did not apply or it is inert"
    )


def test_the_recorded_diffs_agree_with_the_stored_arrays(
    manifest: dict[str, Any],
) -> None:
    """Re-derive one case's moved-value count instead of trusting the manifest.

    `docs/agents/lessons.md` ``[derived-count-not-rederived]``: a count
    written into a manifest and then quoted in a task note has been
    measured once and repeated twice. This recomputes it from the
    committed bytes, which is the only copy that cannot go stale.

    One case per defect keeps the test cheap; the arrays it does not
    re-derive are covered by the hash check above, which is what says the
    bytes are the ones the count was taken from.
    """
    for label, defect in manifest["defects"].items():
        name = next(iter(defect["cases"]))
        recorded = defect["diff_against_corpus"][name]
        corpus_arrays = oracles.load_corpus_arrays(name)
        moved = total = 0
        with np.load(oracles.DATA_DIR / defect["file"]) as stored:
            for block in defect["cases"][name]["blocks"]:
                shipped = corpus_arrays[block["label"]]
                for suffix, entry in block["arrays"].items():
                    got = stored[entry["key"]]
                    want = shipped[suffix]
                    both_nan = np.isnan(got) & np.isnan(want)
                    moved += int(np.sum(~(both_nan | (got == want))))
                    total += got.size
        assert (moved, total) == (
            recorded["values_moved"],
            recorded["values_total"],
        ), f"{label}/{name}: re-derived ({moved}, {total}), manifest {recorded}"


def test_the_patches_are_the_ones_the_manifest_recorded(
    manifest: dict[str, Any],
) -> None:
    """Each committed patch still hashes to what its capture was taken from.

    The patch *is* the statement of what "corrected" means for that
    defect, and the repair tasks are told to match it. An edited patch
    with the arrays left alone would leave that statement and those
    numbers describing different repairs.
    """
    for label, defect in manifest["defects"].items():
        patch = oracles.REPO_ROOT / roster.DEFECTS[label].patch
        assert patch.exists(), f"{label}: {patch} is missing"
        assert (
            oracles.hash_file(patch) == defect["patched_source"]["patch_sha256"]
        ), f"{label}: {roster.DEFECTS[label].patch} has changed since the capture"


def test_the_restored_sources_are_still_recoverable(
    manifest: dict[str, Any],
) -> None:
    """Every resurrected source still resolves at the revision recorded.

    Nine of the twenty cases were captured through `.pyx` the port had
    already deleted. What makes that auditable is that the bytes are
    recoverable — `git show <rev>:<path>` — and hashed here against what
    the capture recorded. A rewritten history would break this, which is
    the point: the arrays would no longer have a provenance anyone could
    check.

    Needs the history to be present, which is not everywhere.
    ``actions/checkout`` clones at ``fetch-depth: 1`` by default, so on CI
    the recorded revisions are simply absent and `git show` exits 128 —
    that is a fact about the checkout, not about the capture, so it skips
    rather than fails. The revision *resolving* and then hashing wrong is
    a real failure and still is one.
    """
    checked = 0
    for label, defect in manifest["defects"].items():
        for source in defect["restored_sources"]:
            revision = source["revision"]
            present = subprocess.run(
                [
                    "git",
                    "-C",
                    str(oracles.REPO_ROOT),
                    "cat-file",
                    "-e",
                    f"{revision}^{{commit}}",
                ],
                capture_output=True,
                check=False,
            )
            if present.returncode != 0:
                pytest.skip(
                    f"{revision} is not in this clone — a shallow checkout "
                    "cannot answer whether the restored sources are still "
                    "recoverable"
                )
            blob = subprocess.run(
                [
                    "git",
                    "-C",
                    str(oracles.REPO_ROOT),
                    "show",
                    f"{revision}:{source['path']}",
                ],
                check=True,
                capture_output=True,
            ).stdout
            assert hashlib.sha256(blob).hexdigest() == source["sha256"], (
                f"{label}: {revision}:{source['path']} no longer "
                "hashes to what the capture recorded"
            )
            checked += 1
    assert checked == sum(
        len(d["restored_sources"]) for d in manifest["defects"].values()
    )


def test_the_capture_left_no_library_behavior_behind() -> None:
    """The patched sources are not what the tree ships.

    Task 2 ships no library behavior: the patches exist only inside the
    capture, and the tree the repository ships is the unpatched one. The
    check is ``git apply --reverse --check``, which succeeds exactly when
    a patch *is* applied — so this requires it to fail, for every defect.

    Deliberately not a search for the patch's added lines in the source.
    That was the first form of this test and it failed on A1: its added
    block re-uses lines the surrounding partial-cell arithmetic already
    contains verbatim (``x2 = x[ilow]``, ``b = y1 - m * x1``), so
    per-line containment reports a leak on a provably clean tree. A
    reverse-apply asks the exact question instead of a proxy for it.
    """
    for label, defect in roster.DEFECTS.items():
        applied = subprocess.run(
            [
                "git",
                "-C",
                str(oracles.REPO_ROOT),
                "apply",
                "--reverse",
                "--check",
                defect.patch,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert applied.returncode != 0, (
            f"{label}: {defect.patch} reverse-applies cleanly, so "
            f"{defect.source} is still carrying the capture's patch"
        )
