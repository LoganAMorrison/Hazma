# Record the restore revisions for the deleted mediator decay `.pyx`

- **Added:** 2026-08-23
- **Source:** cython-to-rust Task 6.2 (`projects/cython-to-rust/task-notes/phase-06/task-6.2-decay-spectra.md`)
- **Scope:** commit
- **Status:** open
- **Triggers / blockers:** ripens the moment Task 6.2's PR is merged and
  its commit has a SHA. Must land before any re-capture of
  `test/parity/oracles/data/*.npz`, and is moot once `cython-to-rust`
  Task 6.4 closes re-capture for good.

## Why

`test/parity/oracles/defects.py`'s `RESTORED_SOURCES` maps each `.pyx`
the port deleted to the literal revision a re-capture must restore it
from — `"0954e5a^"` for the five tabulated photon modules,
`"b5f7f90^"` for `_rho`. Every entry is a literal SHA on purpose: a
re-capture that restored the wrong bytes would produce a corrected-value
array that is silently not the repair the committed patch describes, and
`capture.py` guards against that by comparing the restored file's bytes
against the recorded revision.

cython-to-rust Task 6.2 deleted
`hazma/{scalar,vector}_mediator/*_mediator_decay_spectrum.pyx`, which the
three `mediator_spectra.*.photon.*` corpus cases run through. It could
not add its own rows, because the revision it needs is the parent of the
commit that carries the deletion, and that SHA does not exist while the
task is authoring the file. Leaving the rows out is the safe failure —
`capture.py --check`, the gate that runs in `pytest`, does not read
`RESTORED_SOURCES` at all — but it does mean a re-capture would restore
nothing for those three cases and resolve them through `importlib`
against modules that are not there.

## What

Add two rows to `RESTORED_SOURCES` in
`test/parity/oracles/defects.py`, resolving the revision from git rather
than by hand:

```bash
git log -1 --format=%h --diff-filter=D -- \
  hazma/scalar_mediator/scalar_mediator_decay_spectrum.pyx
```

The row's value is that SHA suffixed with `^`, matching the existing
entries' shape. Neither file has a `.pxd`, so it is two rows and not
four. Then drop the comment Task 6.2 left in that dict and the pointer
this file is named in from `entry_points.py`'s
`_MEDIATOR_DECAY_RESTORED` string.

Task 6.3 deletes `*_positron_spec.pyx` and inherits exactly the same
problem for the four `mediator_spectra.*.positron.*` cases, so the
cheapest resolution is to do both in one pass after 6.3 merges — at which
point the roster is complete and `RESTORED_SOURCES` covers every case the
oracles need.

## Entry points

- `test/parity/oracles/defects.py:132-146` — `RESTORED_SOURCES`
- `test/parity/oracles/entry_points.py` — `_MEDIATOR_DECAY_RESTORED`
- `test/parity/oracles/README.md` — the "Recapturing" recipe, step 1
- Related project: `projects/parity-pinned-defect-repair/`
- Related project: `projects/cython-to-rust/` (Tasks 6.2, 6.3, 6.4)
