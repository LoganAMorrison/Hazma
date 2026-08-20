# parity-pinned-defect-repair — Cross-Cutting Rules

Rules every task in this project must follow. Repo-wide invariants
(the preflight gate, PR conventions, versioning) are not restated here —
see `AGENTS.md` and `docs/agents/`. The `cython-to-rust` project's own
rules (`projects/cython-to-rust/rules.md`) still bind everything this
project touches; nothing below relaxes them.

## Rules

One flat scheme, cited as "rule N" from `PLAN.md`, the references
and the task notes. There is no per-section numbering to reconcile.

1. **The committed corpus arrays are never rewritten.**
   `test/parity/data/*.npz` and `test/parity/data/manifest.json` are the
   record of what 2.1.0 shipped. A repair declares a delta against them
   (see `references/corpus-repinning.md`); it does not regenerate,
   re-pin, or edit them. `git diff --stat -- test/parity/data` is empty
   in every PR this project ships.

2. **No tolerance is widened.** If a repaired value does not match its
   declared relation, the declaration is wrong or the repair is — not
   the budget. `projects/cython-to-rust/task-notes/README.md` records
   two rounds of a `cython-to-rust` PR learning this the expensive way:
   widening a budget until it passes is how a gate becomes vacuous.

3. **Every repair carries an independent oracle.** Group A: the Task 2
   Cython capture. Group B: the closed form, with an `mpmath` reference
   in the shape of `test/parity/reference.py` where the form is
   analytic. A repair verified only against the implementation being
   repaired is not verified.

4. **Every repair carries a physics invariant too.** A yield in photons
   per decay, a unit, an endpoint, a normalization integral — something
   a corpus comparison cannot give you. Two independent implementations
   agreeing on a wrong number is the failure mode this rule exists for.

5. **A declaration is an allowlist, not a rule over a shape.** Name the
   positions the mechanism actually reaches, with the measurement beside
   the row. A carve-out written wider than its mechanism only ever
   loosens, so nothing turns red when it is wrong —
   `docs/agents/lessons.md` `[exemption-wider-than-its-mechanism]`.

6. **A declaration that describes no change fails.** Reverting a repair
   must turn the gate red. A stale declaration is a hole in the gate,
   not a harmless leftover.

7. **Declarations do not overlap.** Where two repairs move the same
   array (A3 and B3 on both rho cases; A1 and B1/B2 on `eta_prime` and
   `phi`), either the position sets are provably disjoint or the two
   collapse into one composite declaration. A shape test enforces this.
   This rule first read "A2 and A3 on both rho cases"; Task 2 measured
   A2's radius at one case and neither rho is in it, so A2 overlaps
   nothing — see `references/defect-blast-radius.md`.

8. **Task 2 before the port's next deletion.** The oracle capture is the
   only step with a hard external deadline —
   `references/defect-blast-radius.md` has the schedule. If a window
   closes before the capture, say which oracle was lost and what the
   affected repair is now verified against instead. Do not proceed as
   if nothing changed.

9. **Primitives before their consumers.** A1 before B1/B2 (they share
   the tabulated photon arrays); A2 before A3 before B3 (the rho quads
   over the pion, which quads over the muon). A4 is a separate branch
   and may run in parallel. Repairing a consumer first means measuring
   its delta twice and reconciling two numbers that were both correct
   when taken.

10. **Every repair updates "Numerical impact so far" in
    `task-notes/README.md` in its own PR**, with the function, the grid,
    and the max shift — `projects/cython-to-rust/rules.md` rule 3, and
    the same reason: Task 12 aggregates that section rather than
    reconstructing it from memory at close time.

11. **Re-derive every count after the last edit, and quote the command
    next to the number.** Position counts, case counts, test counts. A
    count measured correctly still goes stale if the task's own later
    output changes what the command measures —
    `[measurement-taken-before-the-task-ended]`.
