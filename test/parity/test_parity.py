"""The parity gate: re-evaluate every pinned entry point and compare.

One test per corpus block (`cases.Block`) — 623 of them across the 41
consumed compiled entry points. Each one re-derives the block from
`cases`, re-evaluates the live implementation through the *generator's
own* evaluation path, and compares against the arrays committed under
``data/``.

Why the generator's path
------------------------
`generate.evaluate_block` is what produced the stored numbers: it
suppresses scipy's `IntegrationWarning` on the deliberately pathological
grid points, and when a batched call raises it recovers the surviving
points one at a time and records what raised. Re-implementing any of that
here would let a difference in the harness masquerade as a difference in
the implementation. Calling the same function leaves the kernel as the
only variable.

What is compared
----------------
For every block:

1. **The abscissae the specification produces** — the swept grid and the
   scalar probe drawn from it — against the ones stored in the corpus.
   `cases` is re-evaluated live, so a change to grid construction would
   otherwise silently move every sample point and leave the value
   comparison comparing two differently-sampled functions. Held to
   `tolerances.abscissa_budget`: bit-exact on the capturing tree, one
   ulp elsewhere. Task 1.2 made this exact in *both* modes on the premise
   that grids are arithmetic on constants; `numpy.geomspace` reaches the
   platform libm, so Task 1.3's first Linux CI run failed all 623 blocks
   by exactly 1 ulp. The bound is still ten orders of magnitude tighter
   than any redesigned grid, which is what the check is actually for.
2. **The values**, array path and scalar path, within the budget
   `tolerances.effective_budget` selects — bit-equality on the capturing
   tree, the declared per-function budget anywhere else, and
   `tolerances.PLATFORM_EXACT_RTOL` for the ``EXACT`` class once the libm
   itself has changed. Positions `stability` marks unpinnable are dropped
   from this comparison first: four scalar elastic cross sections cancel
   every significant bit out of a difference of two ``atan``s, and what
   the corpus stored there is one platform's rounding, not a number any
   reimplementation reproduces. Where the corpus stored an exact ``0.0``
   the comparison is absolute instead of relative, against
   `tolerances.zero_floor`.
3. **The raises**, replayed rather than skipped. Three blocks record a
   `TypeError` at a kinematic edge (`sigma_xx_to_v_to_pipi` and
   `sigma_xx_to_v_to_pi0v` at ``e_cm = 2 mx``). The stored value there is
   `nan`, so a runner that only compared arrays would pass against an
   implementation that quietly returned a number. Comparing the *records*
   catches the reverse too — a new raise where the corpus has none.

Reading a failure
-----------------
`numpy.testing.assert_allclose` prints the max relative difference it
saw, which is the measurement a later phase needs in order to tighten a
budget: set the budget to the value you want and read what the run
reports.

Runtime
-------
Around five minutes of single-core work, nearly all of it nested
adaptive quadrature in the rho and mediator-spectrum kernels. That is
the cost of the gate, not overhead — it is the same work `generate.py`
does. The pytest-xdist `addopts` in `pyproject.toml` spread it across
workers, so the wall-clock is that cost divided by the machine.
"""

from __future__ import annotations

import importlib
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cases as corpus  # (imported after the sys.path entry above)
import generate as corpus_generate
import stability
import tolerances

MANIFEST = corpus_generate.load_manifest()

#: Which stored positions assert nothing, from `stability`. Loaded once:
#: it is a small JSON and every block consults it.
UNPINNABLE = stability.load_mask()

#: The live specification. Built once at import: `pytest.mark.parametrize`
#: needs the block list at collection time, and constructing the mediator
#: models (which solve for `vs` and integrate partial widths) is not free.
CASES = corpus.build_cases()

#: Which tree we are running on, and therefore which budget applies.
TREE = tolerances.provenance(MANIFEST)

#: What the `stored_arrays` fixture hands back: case name -> npz contents.
ArrayLoader = Callable[[str], dict[str, np.ndarray]]

#: Block arrays that record *where* an entry point was sampled rather than
#: what it returned. Compared against `tolerances.abscissa_budget`, not
#: against the case's value budget.
ABSCISSAE = frozenset({"grid", "scalar_grid"})

#: How many stored positions `stability` currently declines to pin. Held
#: as a literal so that regenerating the mask has to show up in a diff --
#: see `test_the_mask_is_a_small_fraction_of_the_corpus`.
EXPECTED_MASKED_POSITIONS = 494

#: How many stored zeros get an absolute floor instead of exact equality.
#: Held as a literal for the same reason, and a stricter one: the first
#: version of that exemption covered 66,840 positions, so its size is the
#: number worth making somebody defend in a diff (PR #71 review round 1).
EXPECTED_PORTABILITY_ZEROS = 4


def _drop_unpinnable(
    live: np.ndarray,
    pinned: np.ndarray,
    case_name: str,
    block_label: str,
    suffix: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove the positions `stability` says are rounding residue.

    Returns both arrays with the same positions deleted, so the caller
    compares like with like. For an unaffected case -- every case outside
    `stability.AFFECTED_CASES` -- this is the identity.

    Deleting rather than masking keeps `numpy.testing.assert_allclose`'s
    "max relative difference" line meaningful: a masked entry would still
    print in the mismatch summary and read as a compared point.
    """
    skip = stability.unpinnable_indices(UNPINNABLE, case_name, block_label, suffix)
    if not skip:
        return live, pinned
    keep = np.setdiff1d(np.arange(pinned.shape[-1]), sorted(skip))
    return live[..., keep], pinned[..., keep]


def _portability_zero_mask(
    pinned: np.ndarray, case_name: str, block_label: str, suffix: str
) -> np.ndarray:
    """Boolean mask of the declared portability zeros in one block array.

    All-``False`` for every array but the four
    `stability.PORTABILITY_ZEROS` names, so the ordinary comparison —
    including `atol = 0` against every other stored zero — is what runs
    almost everywhere.
    """
    mask = np.zeros(pinned.shape, dtype=bool)
    declared = stability.portability_zeros(case_name, block_label, suffix)
    if declared:
        mask[..., sorted(declared)] = True
    return mask


def _blocks() -> list[Any]:
    """One `pytest.param` per corpus block, ided by case and block label."""
    return [
        pytest.param(name, index, id=f"{name}[{block['label']}]")
        for name, case in MANIFEST["cases"].items()
        for index, block in enumerate(case["blocks"])
    ]


@pytest.fixture(scope="session")
def stored_arrays() -> ArrayLoader:
    """Load a case's ``.npz`` once and hand out its arrays by key.

    The whole corpus is ~180k float64 values, so holding every case in
    memory for the session is cheaper than reopening an npz 623 times.
    """
    cache: dict[str, dict[str, np.ndarray]] = {}

    def load(case_name: str) -> dict[str, np.ndarray]:
        if case_name not in cache:
            path = corpus_generate.DATA_DIR / MANIFEST["cases"][case_name]["file"]
            with np.load(path) as npz:
                cache[case_name] = {key: npz[key] for key in npz.files}
        return cache[case_name]

    return load


@pytest.mark.parametrize(("case_name", "block_index"), _blocks())
def test_entry_point_matches_corpus(
    case_name: str, block_index: int, stored_arrays: ArrayLoader
) -> None:
    """One corpus block: same grid, same raises, same values."""
    manifest_block = MANIFEST["cases"][case_name]["blocks"][block_index]
    case = CASES[case_name]
    block = case.blocks[block_index]
    arrays = stored_arrays(case_name)
    budget = tolerances.effective_budget(case_name, TREE)
    grid_budget = tolerances.abscissa_budget(TREE)

    assert block.label == manifest_block["label"], (
        f"{case_name}: block {block_index} is {block.label!r} in cases.py but "
        f"{manifest_block['label']!r} in the manifest — the specification and "
        "the corpus have diverged; regenerate or revert."
    )

    # The specification must still produce the abscissae the values were
    # captured at, or the comparison below is between two different
    # functions sampled differently.
    np.testing.assert_allclose(
        block.grid,
        arrays[manifest_block["arrays"]["grid"]["key"]],
        rtol=grid_budget.rtol,
        atol=grid_budget.atol,
        err_msg=f"{case_name}[{block.label}]: cases.py no longer produces the "
        f"grid the corpus was captured on ({grid_budget.why})",
    )

    actual, raised = corpus_generate.evaluate_block(case.resolve(), block)

    # Replay, not skip: where the corpus says the entry point raised, the
    # live one must raise the same type at the same argument, and where it
    # says nothing raised, nothing may.
    assert raised == manifest_block.get("raises", {}), (
        f"{case_name}[{block.label}]: exceptions changed.\n"
        f"  corpus: {manifest_block.get('raises', {})}\n"
        f"  live:   {raised}"
    )

    # Both directions: a block that lost its scalar branch would otherwise
    # fail as a KeyError below, and one that gained a branch the corpus
    # never captured would pass without anyone checking it.
    assert set(actual) == set(manifest_block["arrays"]), (
        f"{case_name}[{block.label}]: the block produced "
        f"{sorted(actual)} where the corpus holds "
        f"{sorted(manifest_block['arrays'])}"
    )

    for suffix, entry in manifest_block["arrays"].items():
        expected = arrays[entry["key"]]
        where = f"{case_name}[{block.label}].{suffix}"
        if suffix in ABSCISSAE:
            # Where the values were sampled, not what came back, so this
            # gets the abscissa budget rather than the case's: bit-exact
            # on the capturing tree, one-ulp-tolerant elsewhere because
            # geomspace reaches the platform libm. Comparing at genuinely
            # drifted abscissae would compare two different functions,
            # which is what the tight bound still forbids.
            np.testing.assert_allclose(
                actual[suffix],
                expected,
                rtol=grid_budget.rtol,
                atol=grid_budget.atol,
                err_msg=f"{where} is not the pinned grid ({grid_budget.why})",
            )
            continue
        live, pinned = _drop_unpinnable(
            actual[suffix], expected, case_name, block.label, suffix
        )
        # The four positions `stability` declares are compared against an
        # absolute floor; everything else -- including all 66,836 other
        # stored zeros, which `atol = 0` holds to exact equality -- goes
        # through the ordinary budget.
        floored = _portability_zero_mask(pinned, case_name, block.label, suffix)
        np.testing.assert_allclose(
            live[~floored],
            pinned[~floored],
            rtol=budget.rtol,
            atol=budget.atol,
            equal_nan=True,
            err_msg=f"{where} moved beyond its budget ({budget.why})",
        )
        if floored.any():
            floor = tolerances.zero_floor(pinned)
            np.testing.assert_array_less(
                np.abs(live[floored]),
                np.nextafter(floor, np.inf),
                err_msg=f"{where}: a declared portability zero "
                f"(stability.PORTABILITY_ZEROS) came back larger than "
                f"{floor:.3e}, which is "
                f"{tolerances.ZERO_FLOOR_FRACTION:.0e} of the array's median "
                "non-zero magnitude",
            )


def test_running_on_the_capturing_tree() -> None:
    """Assert the gate is in bit-equality mode, or say why it is not.

    The mode switch in `tolerances.effective_budget` is load-bearing — it
    is the difference between catching a one-ulp regression and tolerating
    1e-6 — so it is reported rather than left to be inferred. A skip here
    (visible with ``-rs``) means the declared budgets are what is being
    enforced, and its reason names what differs from the capture.
    """
    if not TREE.exact:
        pytest.skip(f"declared budgets in force: {TREE.detail}")


def test_every_corpus_case_has_a_budget() -> None:
    """No entry point may run through the gate without a declared budget.

    `tolerances.budget_for` raises on a missing case, but only for a case
    something actually evaluates. This closes the other direction too: a
    budget for a case the corpus no longer has is a leftover, and reading
    it as coverage would overstate the gate.
    """
    corpus_cases = set(MANIFEST["cases"])
    declared = set(tolerances.BUDGETS)
    assert declared == corpus_cases, (
        "tolerances.BUDGETS and the corpus manifest disagree.\n"
        f"  cases with no budget: {sorted(corpus_cases - declared)}\n"
        f"  budgets with no case: {sorted(declared - corpus_cases)}"
    )


def test_an_os_point_release_is_not_a_platform_change() -> None:
    """The `EXACT` class does not quietly relax when macOS updates.

    `tolerances._libm_identity` reads the OS family and the CPU
    architecture out of the environment mapping, not the whole
    `platform.platform()` string. The distinction is load-bearing and
    invisible: the capturing machine has already moved from macOS 26.5.2
    to 26.6.1, and a whole-string comparison would have taken the corpus
    off bit-equality on the very host it was captured on -- passing, and
    six decades weaker, with nothing in the output saying so.
    """
    capture = MANIFEST["environment"]
    point_release = {**capture, "platform": "macOS-99.9.9-arm64-arm-64bit"}
    other_arch = {**capture, "machine": "x86_64"}
    other_os = {**capture, "platform": "Linux-6.8.0-x86_64-with-glibc2.39"}

    identity = tolerances._libm_identity
    assert identity(point_release) == identity(capture)
    assert identity(other_arch) != identity(capture)
    assert identity(other_os) != identity(capture)


def test_the_platform_branch_only_moves_the_exact_class() -> None:
    """Off the capturing libm, only `EXACT` cases change budget.

    Every other class is already >= 1e-13 and was written for a
    replacement implementation, not for a replacement libm, so widening
    them here would be a second, undeclared relaxation riding along with
    the first.
    """
    off_platform = tolerances.Provenance(
        exact=False, same_platform=False, detail="synthetic"
    )
    on_platform = tolerances.Provenance(
        exact=False, same_platform=True, detail="synthetic"
    )
    moved = {
        name
        for name in tolerances.BUDGETS
        if tolerances.effective_budget(name, off_platform).rtol
        != tolerances.effective_budget(name, on_platform).rtol
    }
    exact_class = {
        name
        for name, budget in tolerances.BUDGETS.items()
        if budget.rtol == tolerances.EXACT_RTOL
    }
    assert moved == exact_class
    assert (
        tolerances.effective_budget(next(iter(exact_class)), off_platform).rtol
        == tolerances.PLATFORM_EXACT_RTOL
    )


def test_every_portability_zero_is_a_boundary_zero(
    stored_arrays: ArrayLoader,
) -> None:
    """`stability.PORTABILITY_ZEROS` may only name support boundaries.

    The exemption exists for one mechanism — a quadrature whose integrand
    sits at *its own* threshold, so the weighted sum lands on ``0.0`` or
    on a rounding residue depending on the libm. That can only happen at
    the last zero before the support starts. Deeper below threshold the
    integrand is identically zero at every node and the sum is exactly
    zero everywhere, so a floor there would weaken the gate for nothing.

    Asserting the shape rather than the list keeps the registry from
    drifting into interior positions, which is the failure mode that
    turned the first version of this fix into a floor over 66,840 stored
    zeros (PR #71 review round 1).
    """
    for (
        case_name,
        block_label,
        suffix,
    ), positions in stability.PORTABILITY_ZEROS.items():
        manifest_case = MANIFEST["cases"][case_name]
        manifest_block = next(
            block for block in manifest_case["blocks"] if block["label"] == block_label
        )
        stored = stored_arrays(case_name)[manifest_block["arrays"][suffix]["key"]]
        for position in positions:
            where = f"{case_name}[{block_label}].{suffix}[{position}]"
            assert np.all(stored[..., position] == 0.0), (
                f"{where} is not a stored zero, so it has no business in "
                "PORTABILITY_ZEROS"
            )
            assert position + 1 < stored.shape[-1], f"{where} is the last position"
            assert np.any(stored[..., position + 1] != 0.0), (
                f"{where} is not the boundary: position {position + 1} is also "
                "zero, so the integrand is identically zero here and the sum "
                "is exact on every platform"
            )


def test_the_portability_floor_covers_only_four_positions() -> None:
    """The exemption stays an allowlist rather than becoming a rule.

    Pinned as a total for the same reason
    `test_the_mask_is_a_small_fraction_of_the_corpus` is: widening it has
    to show up in a diff and be argued for. A fifth platform disagreement
    is expected to arrive as a *failure* somebody measures and declares,
    not as something a general rule silently absorbs.
    """
    assert (
        sum(len(v) for v in stability.PORTABILITY_ZEROS.values())
        == EXPECTED_PORTABILITY_ZEROS
    )


def test_only_the_declared_cases_are_masked() -> None:
    """`stability`'s mask covers exactly the entry points it names.

    The mask removes points from the gate, so the set it may remove them
    from is a declaration rather than a consequence. A case that acquired
    a mask without a row in `stability.AFFECTED_CASES` would be one
    somebody quarantined without saying which mechanism made it
    unpinnable.
    """
    masked = {name for name, blocks in UNPINNABLE["cases"].items() if blocks}
    assert masked <= set(stability.AFFECTED_CASES), (
        "masked cases with no AFFECTED_CASES row: "
        f"{sorted(masked - set(stability.AFFECTED_CASES))}"
    )
    assert masked, "the mask is empty; regenerate it from the corpus"


def test_the_mask_was_built_from_this_corpus() -> None:
    """A mask built against other reference arrays would mask elsewhere.

    The indices are positions in the stored arrays, so they only mean
    anything against the corpus they were derived from. The manifest's
    kernel digest is what identifies it.
    """
    assert UNPINNABLE["kernel_digest"] == MANIFEST["kernel_digest"]["sha256"], (
        "data/unpinnable.json was built against kernel digest "
        f"{UNPINNABLE['kernel_digest'][:12]} but the corpus manifest records "
        f"{MANIFEST['kernel_digest']['sha256'][:12]}; regenerate with "
        "`python test/parity/stability.py --regenerate`"
    )


def test_every_masked_index_addresses_a_real_stored_value(
    stored_arrays: ArrayLoader,
) -> None:
    """No masked position is out of range, and none is a whole block.

    Out-of-range would silently drop nothing (``setdiff1d`` ignores it),
    which is the failure mode where the gate looks narrowed but is not.
    A block masked in full is the opposite failure: it would pass
    unconditionally, so it must be dropped from the corpus rather than
    emptied in place.
    """
    for case_name, blocks in UNPINNABLE["cases"].items():
        arrays = stored_arrays(case_name)
        manifest_case = MANIFEST["cases"][case_name]
        for block_label, masked in blocks.items():
            manifest_block = next(
                block
                for block in manifest_case["blocks"]
                if block["label"] == block_label
            )
            for suffix, indices in masked.items():
                size = arrays[manifest_block["arrays"][suffix]["key"]].shape[-1]
                where = f"{case_name}[{block_label}].{suffix}"
                assert indices, f"{where}: an empty index list, not a mask"
                assert min(indices) >= 0 and max(indices) < size, (
                    f"{where}: masked indices {min(indices)}..{max(indices)} "
                    f"outside the stored array of length {size}"
                )
                assert len(set(indices)) == len(indices), f"{where}: duplicates"
                assert len(indices) < size, (
                    f"{where}: every one of its {size} positions is masked, so "
                    "the block asserts nothing. Remove the block from cases.py "
                    "instead of emptying it here."
                )


def test_the_mask_is_a_small_fraction_of_the_corpus() -> None:
    """The quarantine stays a quarantine.

    494 of the corpus's ~180k stored values, in 12 of the 15 blocks the
    four `stability.AFFECTED_CASES` entry points have. Pinned as a total
    rather than as a bound so that regenerating the mask has to show up
    in a diff and be argued for, which is what
    ``projects/cython-to-rust/rules.md`` rule 2 asks of anything that
    loosens the gate.
    """
    assert stability.total_masked(UNPINNABLE) == EXPECTED_MASKED_POSITIONS


def test_every_budget_states_a_reason() -> None:
    """rules.md rule 2: a tolerance nobody justified cannot be argued with."""
    unjustified = sorted(
        name for name, budget in tolerances.BUDGETS.items() if not budget.why.strip()
    )
    assert not unjustified, f"budgets with no justification: {unjustified}"


# ---------------------------------------------------------------------------
# The "has the port started?" predicate.
#
# From cython-to-rust Phase 02 the `hazma._core` extension exists in every
# build while every value still comes from Cython. Keying the mode switch on
# its mere importability would have taken the gate out of bit-equality mode
# for the whole of Phases 02-03 — the stretch where a one-ulp regression is
# most worth catching and least expected. These pin the distinction.
# ---------------------------------------------------------------------------


def _fake_core_submodule(name: str, **members: object) -> types.ModuleType:
    """A stand-in submodule, named as if it lived under `hazma._core`."""
    module = types.ModuleType(name)
    for attribute, value in members.items():
        setattr(module, attribute, value)
    return module


@pytest.mark.skipif(
    not corpus.rust_core_available(), reason="hazma._core is not built in this tree"
)
def test_the_served_roster_is_exactly_the_ported_entry_points() -> None:
    """`hazma._core` serves the swapped kernels and nothing else.

    Until cython-to-rust Task 4.1 this asserted the roster was *empty* —
    `roundtrip` is a plumbing probe with no caller in `hazma/`, the five
    per-domain submodules were unfilled, and `special` (Task 3.2), `quad`
    (3.3), `interp` and `boost` (3.4) and `dispatch` (3.5) are test-only
    shims exempted wholesale by `cases._CORE_TEST_ONLY_MODULES`. Phase 04
    fills the per-domain submodules one kernel at a time, so the check
    that survives is the roster's *agreement with the corpus*: exactly
    one served kernel per `cases.PORTED_ENTRY_POINTS` row, matched by
    name.

    Compared on the leaf function name rather than the fully-qualified
    one so the assertion says what it means — that the extension serves
    the ported entry points — without also pinning which submodule each
    lives in, which `hazma/spectra/**/__init__.py` already fixes by
    importing it.

    The names come from the *cases*, not from the `PORTED_ENTRY_POINTS`
    values: those values record the `.pyx` origin, whose ``def`` need not
    be named after the public entry point. Task 4.1's and 4.2's happened
    to be; Task 4.3's is `hazma.spectra._photon._muon:dnde_photon`,
    serving `dnde_photon_muon`.

    If this fails, either a swap landed without its `PORTED_ENTRY_POINTS`
    row (add it; the corpus case must move to the wrapper in the same
    change) or something non-kernel became public on the extension and
    needs adding to `cases._CORE_SCAFFOLD_NAMES` (one name) or
    `cases._CORE_TEST_ONLY_MODULES` (a whole submodule, and then also to
    `test_test_only_core_submodules_have_no_importer`'s guarantee).
    """
    served = corpus.rust_core_kernels()
    ported = {CASES[name].function for name in corpus.PORTED_ENTRY_POINTS}

    assert {name.rpartition(".")[2] for name in served} == ported
    assert len(served) == len(corpus.PORTED_ENTRY_POINTS), (
        f"hazma._core serves {served}, which is not one callable per "
        f"ported entry point ({sorted(corpus.PORTED_ENTRY_POINTS)})"
    )

    # Regeneration is closed from the first swap: rules.md rule 2 allows
    # corpus data to come only from pre-port Cython.
    with pytest.raises(RuntimeError, match=r"serves \d+ kernel"):
        corpus.assert_no_rust_core()


def test_test_only_core_submodules_have_no_importer() -> None:
    """The exemption in `cases._CORE_TEST_ONLY_MODULES` stays honest.

    Exempting a submodule from the served-kernel walk is exactly the move
    that could disable the corpus's bit-equality mode by hand, so the
    exemption is conditional on a property of the tree rather than on
    intent: nothing under `hazma/` may import an exempted submodule. The
    day a wrapper does, it is a served kernel and this fails.

    Text scan rather than an import graph on purpose — it covers the
    `.pyx`/`.pxd` sources too, which no Python-level walk would see.
    """
    package = corpus.REPO_ROOT / "hazma"
    sources = [
        path
        for suffix in ("*.py", "*.pyx", "*.pxd", "*.pyi")
        for path in package.rglob(suffix)
    ]
    assert sources, "found no hazma sources to scan"

    offenders = {
        # `hazma/_core.pyi` documents why the module is unstubbed; a
        # comment is not an import, so match the module path only where
        # it could be one.
        f"{path.relative_to(corpus.REPO_ROOT)}:{number}": line.strip()
        for module in sorted(corpus._CORE_TEST_ONLY_MODULES)
        for path in sources
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        )
        if f"import {module}" in line
        or f"from {module}" in line
        or f"from {module.rpartition('.')[0]} import {module.rpartition('.')[2]}"
        in line
    }
    assert not offenders, (
        f"a test-only hazma._core submodule is imported by the library: "
        f"{offenders}. Either the import is a mistake, or that submodule "
        f"now serves a kernel and must come out of "
        f"cases._CORE_TEST_ONLY_MODULES."
    )


@pytest.mark.skipif(
    not corpus.rust_core_available(), reason="hazma._core is not built in this tree"
)
def test_a_served_kernel_is_found_and_blocks_regeneration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One more public callable under `hazma._core` is one more kernel.

    Measured as a *delta* on whatever the tree already serves, so the
    test keeps working as Phase 04-06 fill the real submodules: the fake
    kernel must appear on top of the live roster, not instead of it.

    The fake is attached under a name no domain will ever take, rather
    than shadowing `photon` or `positron`. Shadowing a real submodule
    subtracts its kernels from the roster while adding one, so the delta
    stops being a delta the moment that submodule is filled — which is
    exactly what happened when Task 4.2 put seven kernels behind
    `photon`.
    """
    core = importlib.import_module("hazma._core")
    baseline = corpus.rust_core_kernels()
    monkeypatch.setattr(
        core,
        "not_a_real_domain",
        _fake_core_submodule(
            "hazma._core.not_a_real_domain", dnde_photon_muon=lambda e, m: 0.0
        ),
        raising=False,
    )

    assert corpus.rust_core_kernels() == sorted(
        [*baseline, "hazma._core.not_a_real_domain.dnde_photon_muon"]
    )
    with pytest.raises(RuntimeError, match=r"serves \d+ kernel"):
        corpus.assert_no_rust_core()
    assert not tolerances.provenance(MANIFEST).exact


@pytest.mark.skipif(
    not corpus.rust_core_available(), reason="hazma._core is not built in this tree"
)
def test_an_imported_third_party_module_is_not_a_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The walk stays inside `hazma._core`.

    A submodule that does `import numpy` exposes `numpy` as a public
    attribute — and `numpy` is full of callables. Counting them would make
    the very first kernel module look like hundreds of ported kernels, and
    (worse) would fire before any port at all if the scaffold ever grew an
    import.

    Asserted against the live roster rather than against the empty list,
    for the reason the test above gives: from Task 4.1 on, the roster is
    never empty.
    """
    core = importlib.import_module("hazma._core")
    baseline = corpus.rust_core_kernels()
    monkeypatch.setattr(
        core,
        "not_a_real_domain",
        _fake_core_submodule("hazma._core.not_a_real_domain", np=np),
        raising=False,
    )

    assert corpus.rust_core_kernels() == baseline
