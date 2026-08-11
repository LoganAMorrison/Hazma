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
   tree, the declared per-function budget anywhere else.
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
Around five minutes, nearly all of it nested adaptive quadrature in the
rho and mediator-spectrum kernels. That is the cost of the gate, not
overhead — it is the same work `generate.py` does.
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
import tolerances

MANIFEST = corpus_generate.load_manifest()

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
        np.testing.assert_allclose(
            actual[suffix],
            expected,
            rtol=budget.rtol,
            atol=budget.atol,
            equal_nan=True,
            err_msg=f"{where} moved beyond its budget ({budget.why})",
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
def test_scaffolded_core_serves_no_kernels() -> None:
    """The pre-Phase-04 extension must not read as a started port.

    `roundtrip` is a plumbing probe with no caller in `hazma/`; the five
    per-domain submodules are empty until Phase 04; and `special`
    (Phase 03 Task 3.2) and `quad` (Task 3.3) are test-only shims
    exempted wholesale by
    `cases._CORE_TEST_ONLY_MODULES`. If this fails, either a kernel
    landed (in which case the corpus really should leave exact mode) or
    something non-kernel became public on the extension and needs adding
    to `cases._CORE_SCAFFOLD_NAMES` (one name) or
    `cases._CORE_TEST_ONLY_MODULES` (a whole submodule, and then also to
    `test_test_only_core_submodules_have_no_importer`'s guarantee).
    """
    assert corpus.rust_core_kernels() == []
    corpus.assert_no_rust_core()  # must not raise


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
    """One public callable under `hazma._core` is a started port."""
    core = importlib.import_module("hazma._core")
    monkeypatch.setattr(
        core,
        "photon",
        _fake_core_submodule("hazma._core.photon", dnde_photon_muon=lambda e, m: 0.0),
        raising=False,
    )

    assert corpus.rust_core_kernels() == ["hazma._core.photon.dnde_photon_muon"]
    with pytest.raises(RuntimeError, match="serves 1 kernel"):
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
    """
    core = importlib.import_module("hazma._core")
    monkeypatch.setattr(
        core,
        "positron",
        _fake_core_submodule("hazma._core.positron", np=np),
        raising=False,
    )

    assert corpus.rust_core_kernels() == []
