---
name: clean-branches
description: Remove stale local and fork remote git branches after work has merged. Use when cleaning merged branches, pruning stale refs, or tidying branch state.
---

# Clean Merged Git Branches

Identify and remove branches whose work is already on `main`. Detect both regular merges and
squash merges. Never delete upstream branches.

## Workflow

1. Run `git remote -v` and identify the fork/push remote, usually `origin`.
2. Fetch the fork remote and current base: `git fetch <fork>` and `git fetch origin main`.
3. Gather local branches, fork remote branches, `git branch --merged main`, and
   `git remote prune <fork> --dry-run`.
4. For branches not reported by `git --merged`, query GitHub:
   `gh pr list --head "<branch>" --state merged --json number,title,headRefOid --limit 1`.
5. Compare branch tip SHA to `headRefOid`; if they differ, treat the branch as possibly reused and
   do not mark it safe.
6. Present a table of safe branches and unfinished branches.
7. Ask for explicit approval before deletion.
8. Delete approved local branches with `git branch -D`.
9. Delete approved fork remote branches with `git push <fork> --delete <branch>`.
10. Run `git remote prune <fork>`.

## Constraints

- Never delete `main` or `HEAD`.
- Never delete branches on an upstream remote.
- Warn if the current branch is a deletion candidate; do not delete the checked-out branch.
- If `gh` is unavailable, say squash-merge detection was skipped.
