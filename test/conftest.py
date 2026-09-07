from __future__ import annotations

import importlib

import pytest

# test/decay/ was removed with hazma/_decay/ (cython-to-rust Task 0.3), and
# test_gamma_ray.py with hazma/gamma_ray.py (Task 0.2, ADR-0003). The last
# entry was the repo's setup.py, which the maturin cutover deleted
# (Task 7.1). Nothing is skipped at collection any more, and this empty
# list is the statement of that -- a `collect_ignore` naming a path that no
# longer exists silently protects nothing.
collect_ignore: list[str] = []

#: The `hazma._core` submodules that exist only so this suite can reach
#: the crate's foundation layers from Python. They are compiled in by the
#: crate's `test-probes` feature, which `rust/Cargo.toml` deliberately
#: leaves out of `default` so that neither the wheel nor the sdist carries
#: them. `test/parity/cases.py` names the same six as
#: `_CORE_TEST_ONLY_MODULES`, where they serve the unrelated purpose of
#: being excused from the served-kernel walk.
TEST_PROBE_SUBMODULES = (
    "special",
    "quad",
    "interp",
    "boost",
    "dispatch",
    "mediator_tables",
)

#: The one command that turns a probe-less tree into one this suite can
#: run against. `docs/agents/environment.md` carries the same line as the
#: documented development install.
REBUILD_COMMAND = (
    'pip install -e . --config-settings build-args="--features test-probes"'
)


def pytest_configure(config: pytest.Config) -> None:
    """Refuse to run against an extension built without the probes.

    Seven ``test_core_*.py`` modules import a probe submodule at module
    scope, so a default-feature build fails collection seven times over
    with an ``ImportError`` that names a missing attribute rather than the
    reason for it. Answering once, before collection, is what keeps that
    failure from reading as a broken checkout.

    Deliberately an error and not a skip: a suite that skips the modules
    covering the QUADPACK port, the boost integrals and the dispatch
    layer's error text is a gate that passes having checked none of them
    (``docs/agents/lessons.md``, ``[gate-disabled-stays-green]``).

    A tree with no extension at all is left alone. That is a different
    problem, with its own failure, and answering it with this message
    would send the reader after the wrong fix.
    """
    try:
        core = importlib.import_module("hazma._core")
    except ImportError:
        return

    missing = [name for name in TEST_PROBE_SUBMODULES if not hasattr(core, name)]
    if missing:
        raise pytest.UsageError(
            f"hazma._core was built without its test probes "
            f"({', '.join(missing)}), which this suite's test_core_* modules "
            f"import at module scope. Rebuild with: {REBUILD_COMMAND}"
        )
