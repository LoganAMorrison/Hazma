# Add a repo markdownlint config so template-shaped docs lint clean

- **Added:** 2026-08-03
- **Source:** conversation (cython-to-rust project scaffolding PR)
- **Scope:** cross-cutting
- **Status:** done — resolved 2026-08-05 by `.markdownlint.jsonc` (see
  [Resolution](#resolution)).
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
- `docs/agents/preflight.md` (the markdownlint gate)
- `projects/_template/` (the shapes that must lint clean)
- `projects/cython-to-rust/phases/*.md` (inline pragmas to remove)
- `projects/cython-to-rust/references/*.md` (inline pragmas to remove)

## Risks / open questions

Loosening MD013 for tables repo-wide also stops flagging genuinely
sloppy wide prose in tables; acceptable given the alternative is
pragma noise. Keep prose (non-table) lines at 80.

## Resolution

Committed [`.markdownlint.jsonc`](../../../.markdownlint.jsonc) at the
repo root — auto-discovered by `markdownlint` from the current
directory, so preflight (which `cd`s to the repo root) picks it up with
no `--config` flag. Relaxations, each annotated in the file itself:
`MD013 {tables: false, code_blocks: false}`, `MD025
{front_matter_title: ""}`, `MD041` accepting frontmatter `name:` as a
title, and `MD033` / `MD049` / `MD060` off.

Two rules went off rather than being narrowed, because narrowing was
not expressible:

- **MD033** — `allowed_elements` takes a fixed element list, but the
  placeholder idiom generates a new "element" per placeholder word
  (`<phase title>`, `<slug>`, `<YYYY-MM-DD>`, …). 33 distinct ones were
  already in the tree.
- **MD060** (table pipe alignment, a rule newer than this follow-up) —
  an aligned template becomes a misaligned copy the moment a project
  fills its placeholder cells in, and nothing formats Markdown tables
  in this repo.

`.markdownlintignore` was considered for `**/_template.md` and
rejected: it would leave the canonical shapes unlinted entirely, and
`markdownlint` exits **0** printing its usage banner when every named
file is ignored — a preflight `--md` list of only-ignored files would
pass vacuously.

Error counts over `docs/` + `projects/`, with the pragmas removed:
**132 → 0**. Repo-wide: **203 → 35** (the rest is `README.md`,
`CODE_OF_CONDUCT.md`, `notebooks/README.md`, and two `SKILL.md` files —
untouched by this change). The 12 inline pragmas in
`projects/cython-to-rust/` and 6 in
[`done/black-pin-divergence-pyproject-vs-ci.md`](black-pin-divergence-pyproject-vs-ci.md)
are gone. A handful of residual errors were content fixes, not config:
a double blank line and an unwrappable placeholder line in the
templates, a `` `type(scope): ` `` code span with a trailing space in
`PR_GUIDELINES.md`, and one prose line a removed file-region pragma had
been covering.

One bug fell out of verifying the gate: `markdownlint` treats its
arguments as globs, so a path matching nothing prints the usage banner
and exits **0** — a typo'd `--md` reported `PASS markdownlint` having
linted nothing. The markdownlint gate in
[`preflight.sh`](../../../scripts/agents/preflight.sh) now checks each
`--md` path exists before trusting the exit code, the same
false-pass guard the pytest gate already had.

The last 5 errors needed a template edit, because config could not
reach them: `MD013 {tables: false}` exempts *parsed* tables, and
nothing inside an HTML comment is parsed, so the commented-out phased
tables in `projects/_template/` stayed red — and they cannot fit in 80
columns without dropping a column. Rather than pragma them (a pragma in
`projects/_template/` is copied into every project that follows):

- `task-notes/README.md` — the phased Phases table moved from an HTML
  comment into a ```` ```markdown ```` fence. The instructions stay in
  the comment; the table is now visible, copy-pasteable, and exempt.
  "Delete the phased block" became "unfence the block below".
- `PLAN.md` — its second copy of that same table is gone, replaced by a
  pointer to the canonical one in `task-notes/README.md`. One copy per
  invariant, per [`docs/agents/README.md`](../../agents/README.md).
