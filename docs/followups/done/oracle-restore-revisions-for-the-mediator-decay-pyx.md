# Record the restore revisions for the deleted mediator spectrum `.pyx`

- **Added:** 2026-08-23
- **Source:** cython-to-rust Task 6.2 (`projects/cython-to-rust/task-notes/phase-06/task-6.2-decay-spectra.md`);
  widened to the positron pair by Task 6.3
  (`.../task-6.3-positron-spectra.md`)
- **Scope:** commit
- **Status:** done (cython-to-rust Task 6.4, 2026-08-27)
- **Triggers / blockers:** none left. Discharged by `cython-to-rust`
  Task 6.4, which was the first task that could resolve the two merged
  SHAs and also the task that deleted the remaining sources. Rather than
  let re-capture become impossible, that task completed the roster for
  every file a chain compiles from — see `## Resolution` at the end.

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

Task 6.3 then deleted `hazma/{scalar,vector}_mediator/*_positron_spec.pyx`,
which the four `mediator_spectra.*.positron.*` cases run through, and
inherited the identical problem. Seven of the corpus's cases are now in
it.

## What

Add **four** rows to `RESTORED_SOURCES` in
`test/parity/oracles/defects.py` — one per deleted `.pyx` — resolving
each revision from git rather than by hand:

```bash
git log -1 --format=%h --diff-filter=D -- \
  hazma/scalar_mediator/scalar_mediator_decay_spectrum.pyx
```

and the same for `vector_mediator_decay_spectrum.pyx` (Task 6.2) and
`hazma/{scalar,vector}_mediator/*_mediator_positron_spec.pyx` (Task 6.3).
The row's value is that SHA suffixed with `^`, matching the existing
entries' shape. None of the four has a `.pxd`, so it is four rows and not
eight. Then drop the comment Task 6.2 left in that dict and the pointer
this file is named in from `entry_points.py`'s
`_MEDIATOR_DECAY_RESTORED` and `_MEDIATOR_POSITRON_RESTORED` strings.

Task 6.3 inherited exactly this problem for the four
`mediator_spectra.*.positron.*` cases and could not discharge it for the
same reason 6.2 could not: the revision is the parent of the commit
carrying the deletion, and that SHA does not exist while the task is
authoring the file. So the whole thing lands in one pass after 6.3
merges, at which point the roster is complete and `RESTORED_SOURCES`
covers every case the oracles need.

While there, note that Task 6.3 corrected an attribution bug 6.2 left in
`entry_points.py`: the two `mediator_spectra.vector.positron.*` rows were
flipped from `live` to `restored` with 6.2's note, alongside the three
photon rows that task really was deleting. All four positron rows now
carry `_MEDIATOR_POSITRON_RESTORED`.

## Entry points

- `test/parity/oracles/defects.py:132-146` — `RESTORED_SOURCES`
- `test/parity/oracles/entry_points.py` — `_MEDIATOR_DECAY_RESTORED`
  and `_MEDIATOR_POSITRON_RESTORED`
- `test/parity/oracles/README.md` — the "Recapturing" recipe, step 1
- Related project: `projects/parity-pinned-defect-repair/`
- Related project: `projects/cython-to-rust/` (Tasks 6.2, 6.3, 6.4)

## Resolution

`cython-to-rust` Task 6.4 (2026-08-27) completed
`test/parity/oracles/defects.py`'s `RESTORED_SOURCES`, taking it from 13
entries to 29, and every corpus case a defect chain reaches is now
covered.

The four mediator modules resolve as this file specified, from the
parents of the two deleting commits, both merged by then:

- `7594761^` — Task 6.2's decay pair.
- `c384aff^` — Task 6.3's positron pair.

The twelve files Task 6.4 itself deleted — the four capi survivors with
their `.pxd`, `hazma/_utils/boost.{pyx,pxd}`, `constants.pxd` and
`legacy_parameters.pxd` — could not use that spelling, for exactly the
reason 6.2 and 6.3 could not: a task cannot know the SHA of its own
commit. The recursion was broken by noticing that `capture.py` runs
`git show <rev>:<path>`, which does not care whether `<rev>` is a plain
SHA or a `^` expression. They are therefore pinned to `1b022d4`, the
`origin/master` Task 6.4 branched from, where all twelve are present in
their final form — a revision that already exists is strictly more
robust than one computed from a later commit's parent.

All 29 entries were verified to resolve with `git show`. Two things
beyond the original scope were needed to make a re-capture actually
possible rather than nominally recorded:

- The roster now lists the **headers and cimported twins** a restore has
  to compile against, not only the patched sources. `_pion` cimports its
  `_muon` twin in both families, all four `include` the pdg header, and
  all four cimport `boost`.
- `test/parity/oracles/README.md`'s recipe now says that Task 6.4 also
  stripped `setup.py` to the Rust extension and dropped `cython`,
  `numpy` and `scipy` from `[build-system] requires`, so both must be
  restored before anything compiles.

The note in `RESTORED_SOURCES` and the pointers in `entry_points.py`'s
`_MEDIATOR_DECAY_RESTORED` / `_MEDIATOR_POSITRON_RESTORED` were replaced
by references to the roster itself.
