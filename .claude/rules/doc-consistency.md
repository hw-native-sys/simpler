# Doc and Comment Consistency

Keep code, comments, and docs in sync on every change. The most expensive
kind of documentation is the kind that silently lies — a comment or a doc
paragraph that was true once but now disagrees with the code will mislead
future readers for years.

This rule is a *consistency* obligation, not a *volume* obligation. It does
not override the system-prompt defaults ("default to writing no comments",
"do not create new documentation files unless explicitly requested"). It
sharpens them: when you *do* modify something that is documented, update
the documentation in the same change.

## 1. Audit references before and after the change

Before finishing any non-trivial code edit, grep the repo for references to
the identifiers, flags, file paths, and behaviors you modified. Anything
stale that results from your edit is part of your edit:

```bash
# Symbol rename, removal, or signature change
rg '\bold_name\b' docs/ .claude/rules/ src/ python/ examples/ tests/

# Flag / CLI option added, renamed, or removed
rg -- '--old-flag' docs/ .github/workflows/ .claude/skills/ README.md
```

Each match is either already correct, or a doc/comment you must update in
the same commit. Never leave rename-churn for "someone else."

## 2. Prefer updating over creating

- If a relevant doc already exists (e.g. `docs/testing.md`, `docs/ci.md`,
  a `README.md`), extend it rather than writing a new file.
- If code you changed is referenced only by a comment that is now wrong,
  fix the comment — do not write a new doc.
- Only create a new doc file when the user asks for one, or when the
  change is genuinely unclassifiable under any existing doc.

## 3. Delete docs and comments when their referent is gone

If you remove a function, flag, config knob, or workflow step, remove the
doc section and comments that described it. A "deprecated — see …" line is
only appropriate when external callers still rely on the old name; inside
this repo, remove the stale text outright.

## 4. Update the *same* commit, not a follow-up

The doc/comment fix lands in the commit that changes the code. Splitting
them invites the doc update to get dropped on review, rebase, or context
switch. If the doc change is large enough to warrant its own commit, it is
large enough to flag to the user before shipping either half.

## 5. Comments describe *why*, docs describe *contracts*

- **Comments**: non-obvious invariants, workarounds, hidden constraints.
  Not file-level summaries, not WHAT-it-does narration (see the
  codestyle rule and the system prompt default).
- **Docs**: public contracts, pipeline shapes, decision tables, command
  recipes, architectural invariants. Anything a new contributor would
  need to operate the repo without reading every file.

A rule of thumb: if your change breaks nothing but a reader's mental model
of the system, the fix is in docs. If your change breaks a specific
subtle behavior that future edits could regress, the fix is in a code
comment on the load-bearing line.

## 6. Keep docs maintainable — length and structure

Line length is already enforced by `markdownlint-cli2` (MD013, 80 cols) via
pre-commit. The rules below are the file-level counterparts that tooling
cannot enforce for you.

- **Soft size target.** Past roughly **300 lines** or **5 H2 sections**,
  the reader has to scroll to orient. Treat that as a trigger to split or
  restructure, not as a hard cap — a 400-line reference table is fine;
  a 400-line prose page is not.
- **Landing doc plus focused subdocs beats a mega-doc.** When a topic
  grows past the soft target, create focused children (`docs/topic.md`
  → `docs/topic/<subtopic>.md`) and leave a short landing page that
  links to them. One page per reader intent (tutorial / how-to /
  reference / explanation) is the cleanest split.
- **Restructure in the commit that triggers the overflow.** If your edit
  pushes a file past the soft target, reorganize in the same commit —
  do not leave "TODO: split this" for the next contributor. If the split
  is too large to bundle, flag it to the user before shipping the code
  change.
- **A doc edit is also a structural review.** Every time you touch a
  doc, ask whether any section now belongs under a different heading,
  or whether two sections have merged into one topic. Move or merge
  sections in the same diff; do not accumulate structural debt.
