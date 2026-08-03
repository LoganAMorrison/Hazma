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
| Max header length | 69 characters total (including `type(scope): `).            |
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

| Scope      | Area                                                        |
| ---------- | ----------------------------------------------------------- |
| `spectra`  | `hazma/spectra/` — photon / positron / neutrino spectra      |
| `decay`    | `hazma/_decay/` Cython decay spectra and interpolation data  |
| `phase`    | `hazma/phase_space/`, `hazma/_phase_space/` (RAMBO, N-body)  |
| `theory`   | `hazma/theory/` — the model interface and its mixins         |
| `models`   | `scalar_mediator/`, `vector_mediator/`, `rh_neutrino/`, …    |
| `limits`   | `hazma/limits/`, `gamma_ray.py`, gamma-ray limit machinery   |
| `cmb`      | `hazma/cmb.py`, CMB constraints                              |
| `relic`    | `hazma/relic_density/`                                       |
| `form`     | `hazma/form_factors/`                                        |
| `params`   | `hazma/parameters.py`, `gamma_ray_parameters.py`             |
| `utils`    | `hazma/utils.py`, `hazma/_utils/` Cython helpers             |
| `build`    | `_build.py`, Cython build wiring, packaging                  |
| `ci`       | CI workflows and automation                                  |
| `docs`     | Sphinx docs, README, `docs/`                                 |
| `deps`     | dependency updates                                           |
| `test`     | test-only changes                                            |

If none fit, pick the closest module name truncated to 10 chars. Avoid
inventing a scope that will only appear once.

## Title examples

**Good:**

- `feat(spectra): add eta-prime photon channel`
- `fix(decay): correct charged-kaon endpoint interpolation`
- `perf(phase): vectorize rambo momentum generation`
- `docs(workflow): add project scaffolding guide`
- `chore(deps): bump black to 24.x`

**Bad:**

- `feat(spectra): Add the eta-prime photon channel spectrum` — uppercase
  first subject word (convention), likely over 69 chars.
- `fix(Decay): correct endpoint` — uppercase in scope.
- `feat: add neutrino spectra` — missing scope.
- `feat(spectra): add eta-prime channel.` — trailing `.`.
- `feat(phase-space): vectorize rambo` — scope is 11 chars, rejected on
  length; use `phase`.

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
