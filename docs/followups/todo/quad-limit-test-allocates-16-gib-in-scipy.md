# The INT_MAX `limit` test allocates 16 GiB inside scipy

- **Added:** 2026-09-24
- **Source:** `projects/parity-pinned-defect-repair/task-notes/task-12-close.md`,
  Verification (the preflight run of the closing task)
- **Scope:** commit
- **Status:** open
- **Triggers / blockers:** none. Any host with less than about 16 GiB
  free fails the bare `pytest` gate on a tree that changes nothing near
  it.

## Why

`test/test_core_quad.py::TestErrorBehavior::test_the_largest_accepted_limit_is_not_rejected`
checks `hazma._core`'s `quad` at `limit=2**31 - 1`, and then calls
`scipy.integrate.quad(math.exp, 0.0, 1.0, limit=2**31 - 1)` as the
comparison. The hazma half grows its workspaces on demand and costs
nothing. The scipy half does not. With scipy 1.17.1 on a 15 GiB Linux
container, `_quadpack._qagse` raises
`numpy._core._exceptions._ArrayMemoryError: Unable to allocate 16.0 GiB
for an array with shape (2147483647,) and data type float64`.

It was measured on 2026-09-24 with
`python -m pytest -n 0 -q test/test_core_quad.py -k largest_accepted`,
and it failed the same way inside the full suite: `1 failed, 2318 passed,
17 skipped`. The test file is identical to `origin/master` at `e360126`,
so the failure comes from the host and has nothing to do with the diff
that was being tested.

## What

Take the scipy call out of the INT_MAX test. The test is about hazma's
guard, and scipy's behavior at that limit is not what it asserts.
Compare against scipy at an ordinary `limit` in a separate assertion, or
against the known value `e - 1`.

## Entry points

- `test/test_core_quad.py` — `test_the_largest_accepted_limit_is_not_rejected`
