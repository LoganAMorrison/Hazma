# PR Title and Description Guidelines

Commits and PR titles in this repo follow
[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).

## Title format

```text
type(scope): subject
```

**Rules:**

| Constraint        | Rule                                                        |
| ----------------- | ----------------------------------------------------------- |
| Max header length | 69 characters total, counting the `type(scope):` prefix and the space after it. |
| Type              | Valid Conventional Commits type: `feat`, `fix`, `chore`, `ci`, `docs`, `test`, `refactor`, `perf`, `style`, `build`, `revert`. |
| Scope             | Required. Lowercase alphanumeric with optional hyphens (`^[a-z0-9-]+$`), max 10 chars, no leading or trailing hyphen. Must not be a type name. |
| Subject           | Start with an alphanumeric character. No trailing `.` or space. Prefer a lowercase first word. |

Validate a header deterministically rather than counting by hand:

```bash
scripts/agents/check_pr_title.py "feat(spectra): add eta-prime photon channel"
```

**This convention is not enforced by CI.** No workflow rejects a malformed
title, so a green CI run says nothing about the title. The checker above and
review are what uphold it — run the checker before opening a PR rather than
assuming something downstream will catch a mistake.

## Scopes for hazma

Use the most specific scope that applies. Common scopes (non-exhaustive):

| Scope       | Area                                                         |
| ----------- | ------------------------------------------------------------ |
| `spectra`   | `hazma/spectra/` — photon / positron / neutrino spectra      |
| `phase`     | `hazma/phase_space/` (RAMBO, N-body)                         |
| `theory`    | `hazma/theory/` — the model interface and its mixins         |
| `models`    | `scalar_mediator/`, `vector_mediator/`, `rh_neutrino/`, …    |
| `limits`    | `hazma/limits/`, gamma-ray limit machinery                   |
| `cmb`       | `hazma/cmb.py`, CMB constraints                              |
| `relic`     | `hazma/relic_density/`                                       |
| `form`      | `hazma/form_factors/`                                        |
| `params`    | `hazma/parameters.py`, `gamma_ray_parameters.py`             |
| `utils`     | `hazma/utils.py`, `hazma/_utils/` Cython helpers             |
| `packaging` | `pyproject.toml`, `rust/Cargo.toml`, maturin/wheel wiring     |
| `actions`   | `.github/workflows/` — CI and release automation             |
| `sphinx`    | `docs/source/` — the published docs and their build          |
| `readme`    | `README.md` and other top-level prose                        |
| `agents`    | `AGENTS.md`, `docs/agents/`, and the agent skills            |
| `deps`      | dependency updates                                           |
| `suite`     | `test/` — suite-wide test infrastructure                     |

**No scope may be a type name.** The Title format rule above forbids
`feat`, `fix`, `chore`, `ci`, `docs`, `test`, `refactor`, `perf`,
`style`, `build`, and `revert` as scopes, and `check_pr_title.py` rejects
them. That is why the rows above read `actions`, `packaging`, `sphinx`,
and `suite` rather than the type names they would otherwise collide
with — `ci(ci):` and `test(test):` are hard failures, not warnings.

The scope names the *area*, so a `docs` or `test` change usually scopes
to the area it touches rather than to a generic bucket:
`docs(spectra): document the endpoint convention`,
`test(phase): pin the rambo weight regression`. Reach for `sphinx`,
`readme`, or `suite` when the change is to the docs or test machinery
itself.

If none fit, pick the closest module name truncated to 10 chars. Avoid
inventing a scope that will only appear once.

## Title examples

**Good:**

- `feat(spectra): add eta-prime photon channel`
- `fix(decay): correct charged-kaon endpoint interpolation`
- `perf(phase): vectorize rambo momentum generation`
- `docs(workflow): add project scaffolding guide`
- `chore(deps): bump black to 24.x`
- `ci(actions): publish to pypi via trusted publishing`
- `test(suite): make monte-carlo tests deterministic`

**Bad:**

- `feat(spectra): Add the eta-prime photon channel spectrum` — uppercase
  first subject word. A convention the checker does not enforce, so this
  one is caught in review, not by a red exit code.
- `fix(Decay): correct endpoint` — uppercase in scope.
- `feat: add neutrino spectra` — missing scope.
- `feat(spectra): add eta-prime channel.` — trailing `.`.
- `feat(phase-space): vectorize rambo` — scope is 11 chars, rejected on
  length; use `phase`.
- `ci(ci): publish to pypi via trusted publishing` — scope is a type
  name; use `ci(actions)`.

## Description format

Use this structure for PR descriptions:

```markdown
## Summary
<1-3 bullets describing what changed and why>

## Test plan
- [ ] <how you verified the change, with real command output>
```

### For project work, add a Project section

```markdown
## Project
`projects/<slug>/` — Task N: <title>.

See `projects/<slug>/task-notes/task-N-<slug>.md` for implementation
detail, decisions, and verification.
```

### When the numbers move

If the change alters any value the library returns — a spectrum, a limit,
a cross section, a width — say so explicitly in the Summary, even when
the test suite stays green because a tolerance absorbed it. State the old
value, the new value, and which is right. Silent numerical drift is the
failure mode this section exists to prevent.

Put task identifiers, links to related issues, and extra context in the
body (Summary or Project section), not the title.

For reverts, include a `Refs:` trailer linking to the original PR.
