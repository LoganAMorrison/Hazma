# Add a repo markdownlint config so template-shaped docs lint clean

- **Added:** 2026-08-03
- **Source:** conversation (cython-to-rust project scaffolding PR)
- **Scope:** cross-cutting
- **Status:** open
- **Triggers / blockers:** none — ripe whenever; touches only lint
  config and docs.

## Why

Scaffolding `projects/cython-to-rust/` surfaced a conflict between the
preflight markdownlint gate (which runs with **no config**, so all
defaults apply) and the repo's own canonical shapes:

- The phase-file schema (`projects/_template/phases/_template.md`) puts
  `title:` in frontmatter *and* an H1 in the body — MD025 flags every
  phase file, including the template itself.
- The `_template.md` reference files use `<angle-bracket>` placeholders —
  MD033 (inline HTML) flags all of them.
- Grounded-fact tables (dead-code maps, call-site tables) are
  legitimately wider than MD013's 80-column default; the repo already
  tolerates this in `docs/PR_GUIDELINES.md` (10 standing MD013 table
  errors). Under the installed markdownlint, link-bearing lines are
  exempt from MD013 but plain wide table rows are not — an accidental
  and surprising distinction.

The scaffolding PR worked around this with inline
`markdownlint-disable` pragmas, which is fine once but should not
become the per-project ritual.

## What

Add a committed `.markdownlint.jsonc` (or `.yaml`) encoding the de
facto conventions, roughly: `MD013: {tables: false}`, `MD025:
{front_matter_title: ""}`, `MD033: {allowed_elements: []}` relaxed for
the placeholder idiom or scoped via `.markdownlintignore` for
`**/_template.md`. Then remove the now-redundant inline pragmas from
`projects/cython-to-rust/` and re-run `markdownlint --dot` over
`docs/` and `projects/` to confirm the net error count only goes down.
Update `docs/agents/preflight.md`'s markdownlint section to mention
the config.

## Entry points

- `scripts/agents/preflight.sh` (markdownlint gate, ~line 214)
- `docs/agents/preflight.md` (gate 6)
- `projects/_template/` (the shapes that must lint clean)
- `projects/cython-to-rust/phases/*.md` (inline pragmas to remove)
- `projects/cython-to-rust/references/*.md` (inline pragmas to remove)

## Risks / open questions

Loosening MD013 for tables repo-wide also stops flagging genuinely
sloppy wide prose in tables; acceptable given the alternative is
pragma noise. Keep prose (non-table) lines at 80.
