"""The tree carries no Cython, and the build declares none.

cython-to-rust Phase 06 Task 6.4 deleted the last ``.pyx`` and ``.pxd``:
the four capi-survivor spectra extensions, whose ``def``s had gone in
Phase 04 but whose ``cdef``s outlived them as ``__pyx_capi__``
capsules, and the four ``hazma/_utils/`` headers they ``cimport``ed and
``include``d. Every one of the 41 consumed entry points is served by
``hazma._core``.

Each swap task asserted its own twin's absence in its own test module,
which is the right scope while twins remain. This module is the
tree-wide statement those add up to, and it is what keeps the property
after the project closes: a ``.pyx`` reintroduced anywhere, or a build
that quietly regrows a Cython step, fails here rather than in a release.

The assertions are on **source files and build declarations**, never on
an ``ImportError``. A built ``.so`` and its generated ``.c`` sit beside a
deleted ``.pyx``, are gitignored, and survive ``git rm`` and ``git
checkout`` alike (``docs/agents/environment.md``), so an import check
passes on a stale tree and says nothing.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Suffixes no file in the repository may carry. ``.pyx.bak`` is here
#: because one such backup was tracked until Task 6.4 removed it, so the
#: glob that would have caught it is worth keeping.
CYTHON_GLOBS = ("*.pyx", "*.pxd", "*.pyx.bak", "*.pxi")

#: Directories that carry no tracked source and would only slow the walk.
#: The crate's own ``rust/`` is *not* skipped -- a ``.pyx`` there would be
#: as wrong as one anywhere else -- but its ``target/`` is, being large and
#: rebuilt constantly.
SKIPPED_DIRS = frozenset({".git", "build", "dist", "target", "__pycache__"})


def tracked_cython_sources() -> list[Path]:
    """Every Cython source in the repository, repo-relative."""
    found: list[Path] = []
    for glob in CYTHON_GLOBS:
        for path in REPO_ROOT.rglob(glob):
            if SKIPPED_DIRS.isdisjoint(path.parts):
                found.append(path.relative_to(REPO_ROOT))
    return sorted(found)


def test_no_cython_source_remains_anywhere() -> None:
    assert tracked_cython_sources() == []


def test_setup_py_declares_only_the_rust_extension() -> None:
    """No ``cythonize``, no ``Extension``, no ``ext_modules``.

    Parsed rather than imported, because importing ``setup.py`` runs
    ``setup()``; and parsed rather than grepped, because the file's own
    docstring says the word "Cython" while declaring none of it. What is
    scanned is the identifiers and keywords the module actually uses.
    """
    tree = ast.parse((REPO_ROOT / "setup.py").read_text())
    used = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    used |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    used |= {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    used |= {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    used |= {
        keyword.arg
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg
    }
    for banned in ("cythonize", "Cython", "Extension", "ext_modules", "numpy"):
        assert banned not in used, banned
    assert "RustExtension" in used
    assert "rust_extensions" in used


def test_the_build_requirements_name_no_cython_toolchain() -> None:
    """``numpy``, ``cython`` and ``scipy`` were there for the ``.pyx`` alone.

    ``numpy`` supplied the headers every ``Extension`` compiled against,
    ``cython`` the compiler, and ``scipy`` the ``cython_special.pxd``
    that ``hazma/spectra/_photon/_muon.pyx`` ``cimport``ed. None of the
    three has a build-time reader now. They remain *runtime* dependencies
    and are deliberately not asserted against here.
    """
    text = (REPO_ROOT / "pyproject.toml").read_text()
    # Parsed with a regex rather than with `tomllib`, which is 3.11+ while
    # this project supports 3.10 (`requires-python`), and rather than with
    # a third-party TOML reader, which the test group does not carry. The
    # block is a single flat array of strings, so this is sufficient.
    block = re.search(
        r"^\[build-system\].*?^requires\s*=\s*\[(.*?)\]",
        text,
        re.S | re.M,
    )
    assert block is not None, "no [build-system] requires array in pyproject.toml"
    requires = {
        name.split(">")[0].split("=")[0].split("<")[0].strip().lower()
        for name in re.findall(r'"([^"]+)"', block.group(1))
    }
    assert requires == {"setuptools", "setuptools-rust"}


def test_the_sdist_manifest_sweeps_up_no_transpiler_output() -> None:
    """``global-include *.c`` matched only gitignored build output.

    It never matched a tracked file -- the repository has never committed
    generated C (``AGENTS.md``) -- so its only effect was to ship a local
    build's artifacts in an sdist made from a dirty tree.
    """
    manifest = (REPO_ROOT / "MANIFEST.in").read_text()
    sweep = next(
        line for line in manifest.splitlines() if line.startswith("global-include")
    )
    for pattern in ("*.pyx", "*.pxd", "*.c"):
        assert pattern not in sweep, pattern
