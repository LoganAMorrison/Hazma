# Move the relic-density Boltzmann solve into Rust, if it is ever justified

- **Added:** 2026-08-29
- **Source:** `projects/cython-to-rust/learnings/project-retrospective.md` §5
- **Scope:** cross-cutting
- **Status:** open
- **Triggers / blockers:** **nothing today**, and that is the point of the
  entry. It ripens only if someone produces a profile showing the ODE
  path dominating a real workload. Blocked behind
  [`thermal-cross-section-quadrature-never-converges`](thermal-cross-section-quadrature-never-converges.md)
  regardless: porting a solver whose integrand is 0.5%–5% wrong buys
  speed on a wrong answer.

## Why

`hazma/relic_density/` is one of the numeric subsystems the cython-to-rust
project deliberately left on NumPy and SciPy (`PLAN.md` §Scope), alongside
`hazma/phase_space/`, the form factors and the spline utilities. That call
was right on the evidence the project had, and it stays the default. The
entry exists so the question is answered from a measurement rather than
re-litigated from intuition every time someone notices that the ODE path
is the slowest thing in a relic-density calculation.

Two facts from the port bound the answer. Task 5.3 measured
`relic_density(semi_analytic=True)` — the path with no adaptive solver —
reproducing the pre-port tree to 4.11e-16, and the whole relic-density
calculation through a real mediator came out **1.46×–1.93× faster** once
⟨σv⟩ stopped re-entering Python per quadrature node. That speedup came
from moving the *integrand*, which was the expensive part. The remaining
cost is `scipy.integrate.solve_ivp`'s own stepping, which is already
compiled.

The second fact is a warning. `relic_density(semi_analytic=False)` runs
at the caller's default `rtol=1e-5`, and at that tolerance a last-bit
change in the integrand flips a step-acceptance decision and moves the
answer by 3.8e-5 — larger than any drift the port introduced anywhere.
Tightening the solver collapses it (2.75e-7 at `rtol=1e-8`, 3.84e-9 at
`rtol=1e-10`), which is the tell that the number is the solver's error
rather than the physics. Any reimplementation of the stepping will move
the default-tolerance answer by about that much, and that is a declared
numerical change even though the physics does not move.

## What

If a profile ever justifies it:

- Port the Boltzmann right-hand side and the stepping to Rust behind
  `hazma._core`, keeping `relic_density`'s signature and default
  tolerances.
- Declare the shift at the default `rtol` and pin the result at a
  *tightened* solver tolerance, the way
  `test/test_relic_density.py::TestMediatorRelicDensity` already does —
  its Boltzmann pins are taken at `rtol=1e-10` rather than the default,
  because a pin at the default is not portable across libm. That
  precedent is the model to follow, not something to rediscover.
- Do the ⟨σv⟩ quadrature convergence fix first, so the speedup is not
  measured against a wrong integrand.

## Entry points

- `hazma/relic_density/` — the ODE path
- `test/test_relic_density.py::TestMediatorRelicDensity` — the pins, and
  the reason they sit at `rtol=1e-10`
- `projects/cython-to-rust/task-notes/numerical-impact.md` — Task 5.3, for
  the measured figures quoted above
- `projects/cython-to-rust/PLAN.md` §Scope — the original exclusion
- [`thermal-cross-section-quadrature-never-converges`](thermal-cross-section-quadrature-never-converges.md)

## Risks / open questions

- **The default answer moves and the physics does not.** Explaining that
  distinction in a CHANGELOG is harder than making the change. Do not
  start without a plan for how to state it.
- **The likely payoff is small.** The expensive Python-level work was
  already moved in Phase 05; what remains is a compiled SciPy solver. A
  profile, not a hunch, is the entry condition.
