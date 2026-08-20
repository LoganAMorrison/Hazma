# Corpus re-pinning without losing the evidence

**Audience:** Tasks 1, 2, 3, and every repair task (4–10).
**Nature:** Spec — the delta-declaration mechanism, the oracle capture
protocol, and the proof obligations each repair inherits.

## The problem in one paragraph

`test/parity/data/*.npz` holds 179,695 values captured from pre-port
Cython at kernel digest `f5e6e269be47`. Seven of the numbers in there are
wrong, and the corpus is the gate that keeps the Rust port faithful to
them. The obvious move — regenerate — is barred three ways: by
`projects/cython-to-rust/rules.md` rule 2, by
`test/parity/cases.py`'s `assert_no_rust_core` (which already refuses,
since Phase 04 serves the photon family), and by arithmetic — after
Phase 06 Task 6.4 there is no non-Rust implementation left to generate
from. The less obvious move — re-pin the affected arrays in place — is
worse: it destroys the record of what 2.1.0 shipped, which is the only
thing that lets anyone later ask "did this repair move what it claimed?"

## The mechanism

**The committed arrays are never rewritten.** A repair is expressed as a
*declaration* that named positions of named arrays now differ from what
is stored, in a named way.

### Declaration schema

`test/parity/deltas.py`, keyed the same way
`test/parity/stability.py`'s `PORTABILITY_ZEROS` is keyed — by
`(case_name, block_label, array_suffix)`, so the key needs no mapping to
be usable by the runner:

```python
DECLARED_DELTAS: dict[tuple[str, str, str], Delta] = {
    ("spectra.photon.charged_rho", "rest", "values"): Delta(
        repair="B3",                       # which task moved it
        positions=ALL,                     # explicit tuple, or ALL
        relation=Ratio(...),               # see "Relations" below
        measured="ratio is E_gamma exactly at all 100 positions",
        evidence="projects/.../task-9-rho-rest-frame.md",
    ),
}
```

Every field is load-bearing. `repair` is what makes an aggregate
possible at close time without re-reading seven task notes; `measured`
and `evidence` are what stop the table from becoming a list of
assertions nobody re-derived.

### Relations

A declaration says how the repaired value relates to the stored one, not
merely that it differs. In descending order of strength — prefer the
strongest the physics supports:

| Relation | Use when | Example |
| --- | --- | --- |
| `Exact(f)` | the delta is a closed-form transform of the stored array | rho `rest`: repaired == stored × `E_γ` |
| `Oracle(path)` | a Task 2 capture holds the corrected value | all four Group A repairs |
| `Additive(term)` | the delta is a computable additive term | η′: `+ BR · boost_delta_function(M/2, …)` |
| `Bounded(lo, hi, sign)` | only a magnitude and a sign are known | fallback; requires a written justification |

`Bounded` is the escape hatch and should be rare. A repair that can only
say "it got bigger" has not been characterized, and
`docs/agents/lessons.md` `[exemption-wider-than-its-mechanism]` is about
exactly this: a carve-out written wider than the mechanism that earned it
only ever loosens, so nothing turns red when it is wrong.

### How the runner uses it

`test/parity/test_parity.py`, after it has selected a budget and applied
the `stability.py` masks:

1. Position is **not** declared → compare against the original stored
   value under the existing `test/parity/tolerances.py` budget.
   Unchanged behavior, and this is the bulk of the corpus.
2. Position **is** declared → compare against the declared relation.
3. A declared position whose repaired value equals the stored value →
   **fail**. A declaration that no longer describes a real change is
   stale, and a stale declaration is a hole in the gate.

Rule 3 is what makes the layer self-cleaning. Without it, a reverted
repair passes.

### Shape tests (Task 1)

The declaration table is data, so it needs its own gate:

- Every key resolves to a case, block and array the corpus contains.
- Every position index is in range for that array.
- No two declarations claim the same position (two repairs touching one
  array declare disjoint position sets, or one composite declaration —
  never two overlapping ones; see the rho case in `PLAN.md` Task 9).
- `repair` values are drawn from the closed set of task ids.

## The oracle capture protocol (Task 2)

The four Group A twins are `cdef`-only — no top-level `def` — so they are
reachable from Python solely through `__pyx_capi__` capsules.
`test/test_core_boost.py` already drives `hazma._utils.boost` that way
against the Rust port; reuse that harness rather than inventing a second
one.

The protocol, per defect:

1. **Snapshot outside the tree.** `git checkout --` cannot revert a file
   git has never seen, and a mutation harness that does not verify its
   own restore accumulates edits while reporting independent
   measurements — `[mutation-harness-poisons-its-own-baseline]`, hit
   twice in `cython-to-rust`. Snapshot, `cmp` before each step, verify
   the restore.
2. **Patch the `.pyx`, rebuild with `pip install -e .`.** `cargo build`
   is not a rebuild for anything Python imports, and neither is an
   un-rebuilt Cython edit — `AGENTS.md` says both. Confirm with
   `python -c "import hazma; print(hazma.__file__)"` landing inside the
   worktree.
3. **Capture on the corpus's capturing platform.** Read it from
   `test/parity/data/manifest.json`; do not probe for it
   (`[platform-scoped-oracle-asserted-globally]`). A capture taken
   somewhere else is a measurement of a different libm.
4. **Capture the whole composition chain**, not just the defective
   function — see `defect-blast-radius.md`. The chain is what dies at
   Tasks 4.6 and 6.2/6.3, earlier than the twin itself.
5. **Record provenance the way the corpus does**: a digest over the
   *patched* sources, the platform, the resolved `hazma` package path,
   and the numpy/scipy/cython versions. Recording the resolved path is
   what keeps a past capture auditable rather than only a future one
   guarded (`[measured-tree-vs-imported-module]`).
6. **Revert, rebuild, verify `git diff -- hazma` is empty** on the final
   tree. Task 2 ships no library behavior.

### Why this is not circular

The corrected value comes from a source tree and a compiler that predate
the Rust port, driven through an FFI boundary the port does not use. It
is the same class of evidence the corpus itself is: an independent
implementation, pinned. What it is *not* is a proof that the physics is
right — that is what the analytic checks in each repair task's gate are
for. Two independent wrong implementations agreeing would still be
wrong, which is why every repair task's gate names a physics invariant
(a yield, a unit, an endpoint, a normalization) alongside the oracle
comparison.

## Proof obligations every repair inherits

A repair task is not done until all five hold.

1. **The declared delta is exhaustive.** Every position that moved is
   declared; every undeclared position still matches the original stored
   array under its existing budget. This is the "moved only what it
   intended" proof, and it is a property of the gate rather than of a
   claim in a task note.
2. **The declaration is minimal.** Reverting the repair fails the
   declared-delta assertion (schema rule 3 above). Widening the
   declaration by one position fails a shape test. Run both as
   mutations; do not argue them.
3. **An independent oracle agrees.** Group A: the Task 2 Cython capture.
   Group B: the closed form, and where it is analytic an `mpmath`
   reference in the shape of `test/parity/reference.py`.
4. **A physics invariant holds.** Named per defect in `PLAN.md`'s task
   gates — a yield in photons per decay, a unit, an endpoint, a
   normalization integral. This is what a corpus comparison cannot give
   you.
5. **The second gate stays green.** `test/test_theory_aggregation.py`
   pins the pure-Python aggregation as identities (`total` is the channel
   sum, a spectrum is `bf × kernel`). A repair that moves a kernel must
   not move an identity.

## Whole-corpus accounting

At close, one number: how many of the corpus's 179,695 pinned values are
under a declaration. Derive it with a command and paste the command
(`[derived-count-not-rederived]`), broken out per repair so the parts
sum (`[measurement-taken-before-the-task-ended]`). If the total is
larger than the sum of the seven per-repair figures, two declarations
overlap and the shape test missed it.
