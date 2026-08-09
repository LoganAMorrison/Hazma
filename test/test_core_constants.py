"""Bit-equality of ``rust/src/constants.rs`` against the Cython it ports.

The Rust crate carries roughly 220 physical constants transcribed from
three kinds of Cython source: the two ``.pxd`` tables that
``hazma/spectra/**`` and the mediator extensions ``include``, and the
module-local ``DEF`` entries a handful of ``.pyx`` files declare for
themselves. Transcription at that volume is exactly the sort of thing a
reader signs off on and a typo survives, so nothing here trusts the
transcription: every value on both sides is re-derived from source and
compared **bit for bit**.

What this module proves, and what it does not
---------------------------------------------
It parses text. Both parsers hand their expressions to the same
restricted evaluator, so the comparison is between what the two source
files *denote*, under IEEE-754 correctly-rounded decimal-to-``f64``
conversion — which CPython's ``float()`` and rustc's literal parsing
both guarantee. That is the right question for a constants table, and it
runs on every platform in milliseconds without a built extension.

It says nothing about the compiled crate. ``cargo test`` covers that
side: ``rust/src/constants.rs``'s own unit tests re-check the derived
values against runtime arithmetic and pin the divergences between the
two tables. The two halves are complementary and neither replaces the
other.

Lifetime
--------
This module reads the Cython, so it dies with it. When Phases 04-06
delete a ``.pyx`` named in :data:`DERIVED_SOURCES`, delete its row; when
the last ``.pxd`` goes in Phase 06, delete the module. The parity corpus
under ``test/parity/`` is what pins the numbers after that. Each test
below fails with that instruction rather than a ``FileNotFoundError``.

See ``projects/cython-to-rust/rules.md`` rule 4 (Constants 1) for why
the two tables are kept apart in the first place.
"""

from __future__ import annotations

import ast
import math
import re
import struct
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

#: One namespace's constants; and the mapping from namespace to those.
Table = dict[str, float]
Namespaces = dict[str, Table]

REPO_ROOT = Path(__file__).resolve().parents[1]
RUST_CONSTANTS = REPO_ROOT / "rust" / "src" / "constants.rs"

#: ``DEF NAME = expr`` — the compile-time constants of ``constants.pxd``
#: and of every ``.pyx`` that declares its own.
DEF_RE = re.compile(r"^DEF\s+(\w+)\s*=\s*([^#]+?)\s*(?:#.*)?$")
#: ``cdef double NAME = expr`` — how ``legacy_parameters.pxd`` spells the
#: same thing.
CDEF_RE = re.compile(r"^cdef\s+double\s+(\w+)\s*=\s*([^#]+?)\s*(?:#.*)?$")

_RUST_CONST_RE = re.compile(r"^pub const (\w+)\s*:\s*f64\s*=\s*(.+?);\s*(?://.*)?$")
_RUST_MOD_RE = re.compile(r"^pub mod (\w+)\s*\{$")
# The lookbehind keeps the `e` of `1.230e-4` from reading as an identifier.
_RUST_PATH_RE = re.compile(r"(?<![\w.])[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*")

#: The two ``.pxd`` tables, keyed by the Rust module that ports each.
CYTHON_TABLES = {
    "pdg": (Path("hazma/_utils/constants.pxd"), DEF_RE),
    "legacy": (Path("hazma/_utils/legacy_parameters.pxd"), CDEF_RE),
}

#: The ``.pyx`` files that declare module-local ``DEF`` entries, keyed by the
#: ``derived::`` submodule that ports each. Every value in this mapping
#: is checked against a fresh scan of the tree by
#: :func:`test_no_pyx_declares_constants_this_module_ignores`, so a file
#: cannot quietly appear or disappear from the list.
DERIVED_SOURCES = {
    "derived::photon_pion": Path("hazma/spectra/_photon/_pion.pyx"),
    "derived::photon_rho": Path("hazma/spectra/_photon/_rho.pyx"),
    "derived::positron_muon": Path("hazma/spectra/_positron/_muon.pyx"),
    "derived::positron_pion": Path("hazma/spectra/_positron/_pion.pyx"),
    "derived::neutrino_muon": Path("hazma/spectra/_neutrino/_muon.pyx"),
}

#: Which ``.pxd`` each of those ``.pyx`` ``include``-s, and therefore
#: which table its *computed* ``DEF`` entries resolve names against. All five
#: take ``constants.pxd``; the mediator extensions that take the legacy
#: header declare no ``DEF`` entries of their own.
DERIVED_INCLUDES = dict.fromkeys(DERIVED_SOURCES, "pdg")

#: Loose floors for the sanity check that the parsers matched *something*.
#: Deliberately well under the real counts (151 / 48 / 25) so that editing
#: a table is not also editing this test.
FLOOR_PDG = 100
FLOOR_LEGACY = 30
FLOOR_DERIVED = 20
#: Decay widths in ``constants.pxd``. The legacy table has none, on
#: purpose -- see :func:`test_the_legacy_widths_table_is_still_empty`.
PDG_WIDTH_COUNT = 13


# ---------------------------------------------------------------------
# ---- Source parsing -------------------------------------------------
# ---------------------------------------------------------------------


class _Evaluator(ast.NodeVisitor):
    """Evaluate a float expression against a name → value mapping.

    Deliberately tiny: literals, names, unary minus, and the five binary
    operators the two languages spell the same way. ``**`` is included
    because Cython uses it and Python evaluates it identically; the Rust
    side never produces one, since Rust has no such operator.
    """

    _BINOPS: ClassVar[dict[type[ast.operator], Callable[[float, float], float]]] = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.Pow: lambda a, b: a**b,
    }

    def __init__(self, scope: Table) -> None:
        self.scope = scope

    def visit_Constant(self, node: ast.Constant) -> float:
        if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
            raise ValueError(f"not a number: {node.value!r}")
        return float(node.value)

    def visit_Name(self, node: ast.Name) -> float:
        try:
            return self.scope[node.id]
        except KeyError:
            raise ValueError(f"undefined constant {node.id!r}") from None

    def visit_UnaryOp(self, node: ast.UnaryOp) -> float:
        value = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return +value
        raise ValueError(f"unsupported unary operator {node.op!r}")

    def visit_BinOp(self, node: ast.BinOp) -> float:
        try:
            op = self._BINOPS[type(node.op)]
        except KeyError:
            raise ValueError(f"unsupported operator {node.op!r}") from None
        return op(self.visit(node.left), self.visit(node.right))

    def generic_visit(self, node: ast.AST) -> float:
        raise ValueError(f"unsupported syntax {type(node).__name__}")


def _evaluate(expression: str, scope: Table) -> float:
    """Evaluate `expression` in `scope`, rejecting anything exotic."""
    tree = ast.parse(expression.strip(), mode="eval")
    return _Evaluator(scope).visit(tree.body)


def parse_cython(path: Path, pattern: re.Pattern[str], base: Table) -> Table:
    """Return the constants `path` declares, in declaration order.

    `base` seeds the scope with the table the file ``include``-s, so a
    ``.pyx`` whose ``DEF`` entries reference ``MASS_MU`` resolves it. Later
    declarations see earlier ones, matching Cython.
    """
    scope = dict(base)
    values: Table = {}
    for line in path.read_text().splitlines():
        match = pattern.match(line.strip())
        if match is None:
            continue
        name, expression = match.group(1), match.group(2)
        value = _evaluate(expression, scope)
        scope[name] = value
        values[name] = value
    return values


def parse_rust(path: Path) -> Namespaces:
    """Return ``{module path: {NAME: value}}`` for a Rust constants file.

    Modules are tracked by brace depth, so ``derived::photon_pion`` comes
    out fully qualified. Name resolution walks outward from the current
    module exactly as Rust's does, which is what lets the source write
    ``pdg::MASS_E`` from inside ``derived::positron_muon`` — and what
    lets this parser stay ignorant of ``use`` statements.
    """
    modules: Namespaces = {}
    flat: Table = {}
    stack: list[str] = []
    depth_of_module: list[int] = []
    depth = 0

    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line.startswith("#[cfg(test)]"):
            # The unit tests below carry braces inside format strings, and
            # nothing past here declares a constant. Stopping is cheaper
            # than teaching the brace counter about string literals.
            break
        if match := _RUST_MOD_RE.match(line):
            stack.append(match.group(1))
            depth_of_module.append(depth)
            modules.setdefault("::".join(stack), {})
            depth += 1
            continue
        if match := _RUST_CONST_RE.match(line):
            module = "::".join(stack)
            name, expression = match.group(1), match.group(2)
            value = _evaluate(*_rustify(expression, module, flat))
            modules[module][name] = value
            flat[f"{module}::{name}" if module else name] = value
            continue
        depth += line.count("{") - line.count("}")
        while depth_of_module and depth <= depth_of_module[-1]:
            stack.pop()
            depth_of_module.pop()
    return modules


def _rustify(expression: str, module: str, flat: Table) -> tuple[str, Table]:
    """Rewrite Rust paths in `expression` into plain Python identifiers.

    Returns the rewritten expression and the scope it needs. Each
    ``a::b::C`` is resolved by trying it under the current module, then
    each enclosing module, then the crate root — Rust's own rule — and
    replaced by a mangled identifier.
    """
    parts = module.split("::") if module else []
    scope: Table = {}

    def replace(match: re.Match[str]) -> str:
        path = match.group(0)
        for i in range(len(parts), -1, -1):
            candidate = "::".join([*parts[:i], path])
            if candidate in flat:
                mangled = candidate.replace("::", "__")
                scope[mangled] = flat[candidate]
                return mangled
        raise AssertionError(
            f"{path!r} in module {module!r} resolves to no constant declared "
            f"earlier in {RUST_CONSTANTS.name}"
        )

    return _RUST_PATH_RE.sub(replace, expression), scope


def bits(value: float) -> str:
    """Hex of the IEEE-754 payload — the only equality this module uses.

    ``==`` would call ``0.0`` and ``-0.0`` equal and every NaN unequal.
    Neither case arises in a constants table today, but a table is
    exactly where a sign-of-zero change would hide.
    """
    return struct.pack("<d", value).hex()


def require(path: Path) -> Path:
    """Return `path`, or fail with what to do now that it is gone."""
    if not path.exists():
        pytest.fail(
            f"{path} no longer exists. Phases 04-06 of the cython-to-rust "
            f"project delete the Cython this module reads; when they do, "
            f"drop the corresponding row from CYTHON_TABLES or "
            f"DERIVED_SOURCES (and delete this module once the last one "
            f"goes). test/parity/ is what pins the numbers afterwards."
        )
    return path


# ---------------------------------------------------------------------
# ---- Fixtures -------------------------------------------------------
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def rust() -> Namespaces:
    return parse_rust(require(RUST_CONSTANTS))


@pytest.fixture(scope="module")
def tables() -> Namespaces:
    return {
        name: parse_cython(require(REPO_ROOT / path), pattern, {})
        for name, (path, pattern) in CYTHON_TABLES.items()
    }


# ---------------------------------------------------------------------
# ---- The parsers themselves -----------------------------------------
# ---------------------------------------------------------------------


class TestParsers:
    """A silent parser would turn every check below into a tautology."""

    def test_the_rust_file_yields_the_expected_modules(self, rust: Namespaces) -> None:
        assert set(rust) == {"pdg", "legacy", "derived", *DERIVED_SOURCES}
        assert rust["derived"] == {}, "derived holds submodules, not constants"

    def test_each_table_is_substantial(
        self, rust: Namespaces, tables: Namespaces
    ) -> None:
        # Guards against a regex that matches nothing. The real counts
        # are 151 / 48 in the two .pxd and 25 module-local DEFs across
        # five .pyx; the floors are loose so ordinary edits to the tables
        # do not have to touch this test.
        assert len(tables["pdg"]) > FLOOR_PDG
        assert len(tables["legacy"]) > FLOOR_LEGACY
        assert sum(len(rust[m]) for m in DERIVED_SOURCES) > FLOOR_DERIVED

    def test_an_unsupported_expression_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unsupported syntax"):
            _evaluate("sqrt(2.0)", {})
        with pytest.raises(ValueError, match="undefined constant"):
            _evaluate("NOT_A_CONSTANT", {})

    def test_uppercasing_a_cython_name_cannot_collide(
        self, tables: Namespaces, derived: Namespaces
    ) -> None:
        # The Rust side is SCREAMING_SNAKE by convention, so the name
        # comparison below upper-cases the Cython. That is only sound
        # while the mapping stays injective -- `etap_BR_pi0_pi0_eta` in
        # the legacy table and the lowercase `DEF`s of
        # `_positron/_pion.pyx` are the names it has to fold.
        for label, table in {**tables, **derived}.items():
            folded = [name.upper() for name in table]
            assert len(set(folded)) == len(folded), label


# ---------------------------------------------------------------------
# ---- The two .pxd tables --------------------------------------------
# ---------------------------------------------------------------------


@pytest.mark.parametrize("table", sorted(CYTHON_TABLES))
class TestTables:
    def test_names_match_exactly(
        self, rust: Namespaces, tables: Namespaces, table: str
    ) -> None:
        assert {name.upper() for name in tables[table]} == set(rust[table])

    def test_values_are_bit_equal(
        self, rust: Namespaces, tables: Namespaces, table: str
    ) -> None:
        mismatched = {
            name: (bits(value), bits(rust[table][name.upper()]))
            for name, value in tables[table].items()
            if bits(value) != bits(rust[table][name.upper()])
        }
        assert not mismatched


# ---------------------------------------------------------------------
# ---- Module-local DEFs ----------------------------------------------
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def derived(tables: Namespaces) -> Namespaces:
    return {
        module: parse_cython(
            require(REPO_ROOT / path), DEF_RE, tables[DERIVED_INCLUDES[module]]
        )
        for module, path in DERIVED_SOURCES.items()
    }


@pytest.mark.parametrize("module", sorted(DERIVED_SOURCES))
class TestDerived:
    def test_names_match_exactly(
        self, rust: Namespaces, derived: Namespaces, module: str
    ) -> None:
        assert {name.upper() for name in derived[module]} == set(rust[module])

    def test_values_are_bit_equal(
        self, rust: Namespaces, derived: Namespaces, module: str
    ) -> None:
        mismatched = {
            name: (bits(value), bits(rust[module][name.upper()]))
            for name, value in derived[module].items()
            if bits(value) != bits(rust[module][name.upper()])
        }
        assert not mismatched


def test_no_pyx_declares_constants_this_module_ignores() -> None:
    """`DERIVED_SOURCES` is complete against the tree, not a snapshot."""
    found = {
        path.relative_to(REPO_ROOT)
        for path in (REPO_ROOT / "hazma").rglob("*.pyx")
        if any(DEF_RE.match(line.strip()) for line in path.read_text().splitlines())
    }
    assert found == set(DERIVED_SOURCES.values())


def test_every_derived_pyx_includes_the_table_it_is_scored_against() -> None:
    """The scope a computed ``DEF`` resolves in is the ``include``, not a guess."""
    headers = {"pdg": "constants.pxd", "legacy": "legacy_parameters.pxd"}
    for module, path in DERIVED_SOURCES.items():
        text = require(REPO_ROOT / path).read_text()
        included = [
            table
            for table, header in headers.items()
            if re.search(rf'^include\s+"[^"]*{re.escape(header)}"', text, re.M)
        ]
        assert included == [DERIVED_INCLUDES[module]], path


# ---------------------------------------------------------------------
# ---- Provenance of the frozen literals ------------------------------
# ---------------------------------------------------------------------
#
# Seven DEFs are hard-coded digit strings rather than expressions -- someone
# evaluated a formula once and pasted the result. Their comments name the
# formula but not the mass table, and the two tables give different
# answers. Bit-equality against the .pyx (above) would be satisfied by any
# digits at all, so these tests reconstruct each literal and pin which
# table it came from. That is the difference between "the port copied the
# Cython" and "the port knows what the Cython means".


def kinematics(table: dict[str, float]) -> dict[str, float]:
    """Pion/muon decay kinematics from one mass table, in MeV.

    ``eng_mu`` and ``eng_gam_max_murf`` are the muon energy and the
    maximum photon energy in the pion and muon rest frames respectively;
    ``eng_gam_max_pirg`` boosts the latter into the pion rest frame.
    """
    mpi, mmu, me = table["MASS_PI"], table["MASS_MU"], table["MASS_E"]
    eng_mu = 0.5 * (mpi * mpi + mmu * mmu) / mpi
    gamma = eng_mu / mmu
    beta = math.sqrt(1 - 1 / (gamma * gamma))
    eng_gam_max_murf = (mmu * mmu - me * me) / (2 * mmu)
    return {
        "ENG_MU_PIRF": eng_mu,
        "GAMMA_MU_PIRF": gamma,
        "BETA_MU_PIRF": beta,
        "ENG_GAM_MAX_MURF": eng_gam_max_murf,
        "ENG_GAM_MAX_PIRG": eng_gam_max_murf * gamma * (1 + beta),
    }


def test_photon_pion_literals_come_from_the_legacy_table(
    rust: Namespaces, tables: Namespaces
) -> None:
    """All five reproduce from ``legacy``, and none of them from ``pdg``.

    ``hazma/spectra/_photon/_pion.pyx`` ``include``-s ``constants.pxd``,
    so this is a genuine mix inside one module: its three mass aliases are
    PDG values while its five kinematic literals are frozen against the
    older table. Recomputing them from the header the file actually
    includes moves ``ENG_MU_PIRF`` by 4.7e-5 MeV and every charged-pion
    photon spectrum with it, which is why rule 4 forbids the tidy-up.
    """
    frozen = rust["derived::photon_pion"]
    from_legacy = kinematics(tables["legacy"])
    from_pdg = kinematics(tables["pdg"])

    assert {name: bits(value) for name, value in from_legacy.items()} == {
        name: bits(frozen[name]) for name in from_legacy
    }
    assert all(bits(from_pdg[name]) != bits(frozen[name]) for name in from_pdg)


def test_photon_pion_mass_aliases_come_from_the_pdg_table(
    rust: Namespaces, tables: Namespaces
) -> None:
    """The other half of the mix, so the test above is not half a story."""
    frozen = rust["derived::photon_pion"]
    for alias, canonical in (("MPI", "MASS_PI"), ("ME", "MASS_E"), ("MMU", "MASS_MU")):
        assert bits(frozen[alias]) == bits(tables["pdg"][canonical])
        assert bits(frozen[alias]) != bits(tables["legacy"][canonical])


def test_r_factor_is_the_michel_normalization_over_the_pdg_ratio(
    rust: Namespaces,
) -> None:
    """``R_FACTOR`` reproduces, and its ``.pyx`` comment has a typo.

    The comment above the literal in both muon kernels reads
    ``1 / (1 - 8 r^2 + 8 r^6 - r^8 - 12 r^2 ln(r^2))``. The log term's
    exponent is wrong: only ``r^4`` reproduces the digits, and the
    difference is a factor of ``r^2 ~ 2.3e-5`` on that term, so the
    published value settles it. Pinned here so a port that trusts the
    comment over the number fails loudly.
    """
    r = rust["derived::positron_muon"]["R"]
    r2, r4 = r * r, r * r * r * r
    r6, r8 = r4 * r2, r4 * r4
    log = math.log(r2)

    correct = 1 / (1 - 8 * r2 + 8 * r6 - r8 - 12 * r4 * log)
    as_commented = 1 / (1 - 8 * r2 + 8 * r6 - r8 - 12 * r2 * log)

    assert bits(rust["derived::positron_muon"]["R_FACTOR"]) == bits(correct)
    assert bits(rust["derived::neutrino_muon"]["R_FACTOR"]) == bits(correct)
    assert bits(as_commented) != bits(correct)


# ---------------------------------------------------------------------
# ---- The divergence itself ------------------------------------------
# ---------------------------------------------------------------------

#: Every name the two ``.pxd`` share. Ten masses plus alpha and the mass
#: ratio diverge; the seven form factors and decay constants agree. Kept
#: as a literal roster because the whole content of rule 4 is that this
#: partition does not move — a computed one would accept any partition.
SHARED_AND_DIVERGENT = {
    "MASS_E",
    "MASS_MU",
    "MASS_PI0",
    "MASS_PI",
    "MASS_K0",
    "MASS_K",
    "MASS_ETA",
    "MASS_ETAP",
    "MASS_RHO",
    "MASS_OMEGA",
    "ALPHA_EM",
    "RATIO_E_MU_MASS_SQ",
}
SHARED_AND_EQUAL = {
    "F_A_PI",
    "F_V_PI",
    "F_V_PI_SLOPE",
    "F_A_K",
    "F_V_K",
    "DECAY_CONST_PI",
    "DECAY_CONST_K",
}


def test_the_two_tables_diverge_on_exactly_the_recorded_names(rust: Namespaces) -> None:
    """rules.md rule 4, as an assertion.

    A consolidation that quietly adopted one table's masses everywhere
    would pass every bit-equality test above — those compare Rust to
    Cython file by file — and fail only here.
    """
    pdg, legacy = rust["pdg"], rust["legacy"]
    shared = set(pdg) & set(legacy)
    assert shared == SHARED_AND_DIVERGENT | SHARED_AND_EQUAL

    diverged = {name for name in shared if bits(pdg[name]) != bits(legacy[name])}
    assert diverged == SHARED_AND_DIVERGENT


def test_the_legacy_widths_table_is_still_empty(rust: Namespaces) -> None:
    """Two malformed width entries were deleted on 2026-08-05.

    ``docs/followups/done/legacy-parameters-width-exponent-bug.md``
    records why: ``3.3406**-13.`` is exponentiation, not a decimal
    exponent, and nothing referenced either name. If a ``WIDTH_*``
    reappears in the legacy namespace it was resurrected, not ported.
    """
    assert not [name for name in rust["legacy"] if name.startswith("WIDTH_")]
    widths = [name for name in rust["pdg"] if name.startswith("WIDTH_")]
    assert len(widths) == PDG_WIDTH_COUNT
