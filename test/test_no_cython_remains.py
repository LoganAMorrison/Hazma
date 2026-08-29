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

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _toml_array(table: str, key: str) -> str:
    """The raw text inside ``[table] key = [...]`` in ``pyproject.toml``.

    Read with a regex rather than with ``tomllib``, which is 3.11+ while
    this project supports 3.10 (``requires-python``), and rather than with
    a third-party TOML reader, which the test group does not carry. Both
    arrays this module reads are flat lists of strings, so this is
    sufficient.
    """
    text = (REPO_ROOT / "pyproject.toml").read_text()
    match = re.search(
        rf"^\[{re.escape(table)}\].*?^{re.escape(key)}\s*=\s*\[(.*?)\]",
        text,
        re.S | re.M,
    )
    assert match is not None, f"no [{table}] {key} array in pyproject.toml"
    return match.group(1)


#: Suffixes no file in the repository may carry. ``.pyx.bak`` is here
#: because one such backup was tracked until Task 6.4 removed it, so the
#: glob that would have caught it is worth keeping.
CYTHON_GLOBS = ("*.pyx", "*.pxd", "*.pyx.bak", "*.pxi")

#: Directories that carry no tracked source and would only slow the walk.
#: The crate's own ``rust/`` is *not* skipped -- a ``.pyx`` there would be
#: as wrong as one anywhere else -- but its ``target/`` is, being large and
#: rebuilt constantly.
#:
#: ``site-packages`` is the load-bearing one. The documented dev loop
#: builds its virtualenv inside the checkout (``uv venv``,
#: ``docs/agents/environment.md``), and numpy ships 26 ``.pxd`` headers, so
#: without this the walk reports a working environment as a Cython
#: regression. Skipping by that name rather than by ``.venv`` catches every
#: virtualenv layout: the headers are always under ``site-packages``,
#: whatever the environment directory is called.
SKIPPED_DIRS = frozenset(
    {".git", "build", "dist", "target", "__pycache__", "site-packages"}
)


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


def test_no_setuptools_build_script_remains() -> None:
    """``setup.py`` was the last place a Cython step could be declared.

    It is gone with the setuptools backend (cython-to-rust Task 7.1), so
    the assertion is on its absence rather than on its contents. A
    ``setup.py`` reappearing beside a maturin backend would be dead at
    best and a second, unnoticed build path at worst.
    """
    assert not (REPO_ROOT / "setup.py").exists()
    assert not (REPO_ROOT / "setup.cfg").read_text().count("cythonize")


def test_the_build_requirements_name_no_cython_toolchain() -> None:
    """One backend, and it is not a compiler of ``.pyx``.

    ``numpy`` supplied the headers every ``Extension`` compiled against,
    ``cython`` the compiler, and ``scipy`` the ``cython_special.pxd``
    that ``hazma/spectra/_photon/_muon.pyx`` ``cimport``ed. None of the
    three has a build-time reader now, and neither do ``setuptools`` and
    ``setuptools-rust``, which Task 7.1 replaced with ``maturin``. All
    five remain unasserted as *runtime* dependencies where they are one.
    """
    requires = {
        name.split(">")[0].split("=")[0].split("<")[0].strip().lower()
        for name in re.findall(r'"([^"]+)"', _toml_array("build-system", "requires"))
    }
    assert requires == {"maturin"}


def test_the_distribution_sweep_ships_no_editor_leftovers() -> None:
    """The one class of package cruft no ignore rule can catch.

    ``MANIFEST.in`` carried a broader version of this claim until Task 7.1
    deleted it with the setuptools backend: its ``global-include`` was a
    repo-wide filesystem sweep that shipped even ``.gitignore``d build
    output. maturin honors ``.gitignore``, so the ``*.so``, ``*.c`` and
    ``__pycache__`` a built tree accumulates need no help staying out.

    What ``.gitignore`` cannot reach is a file that is *tracked* and still
    does not belong in a release, which is what the four editor leftovers
    under ``hazma/`` are. ``[tool.maturin] exclude`` is the only thing
    keeping them out, so it is worth asserting; a ``*.so`` entry would not
    be, having been measured to change nothing.
    """
    assert not (REPO_ROOT / "MANIFEST.in").exists()
    excluded = set(re.findall(r'"([^"]+)"', _toml_array("tool.maturin", "exclude")))
    assert {"hazma/**/*.bak", "hazma/**/*.org"} <= excluded
