---
name: git-commit
description: Create a clean git commit after reviewing changes, running appropriate verification, staging intentionally, and writing a project-style commit message.
---

# Git Commit

Use this when the user asks to commit or when another workflow reaches its commit step.

## Pre-Commit Review

Verify the repository-local Git identity before staging. Use the identity
configured for this checkout unless the user gives an explicit override:

```bash
git config --local --get user.name
git config --local --get user.email
```

Do not commit with an unknown or accidental identity.

Inspect:

```bash
git status --short
git diff --name-only
git diff --cached --name-only
```

Run verification based on changed files:

- Python or tests: focused `pytest`, then broader relevant tests.
- Tooling or hooks: `pre-commit run --all-files`.
- Docs only: targeted pre-commit over changed docs is usually enough.
- Public API/docs contract changes: tests and docs must move together.

## Stage

Stage only relevant files:

```bash
git add <paths>
git diff --staged
```

Never stage local caches, build outputs, ignored workspaces, or unrelated user changes.

## Message Format

Use:

```text
type: concise imperative summary
```

Recommended lowercase types: `feat`, `fix`, `docs`, `refactor`, `test`, `ci`, `chore`.
Keep the subject under 72 characters and omit the period.

Add a body when the change touches multiple areas or the reason is not obvious. Explain what changed
and why. Reference issues with `Fixes #N` only when intended.

Never add AI co-author trailers. Preserve human coauthors only when squashing commits from multiple
human authors.

## Verify

After committing:

```bash
git log -1 --oneline
git show --stat --oneline HEAD
```

Do not amend unless explicitly requested or fixing an unpushed commit in the active workflow.
