---
name: address-pr-comments
description: Analyze and address GitHub PR review comments intelligently, distinguishing between actionable feedback and comments that can be resolved without changes. Use when addressing PR feedback or review comments.
---

# Address PR Comments Workflow

Intelligently triage PR review comments, address actionable feedback, and resolve informational comments.

## Input

Accept PR number (`123`, `#123`), branch name (`feature-branch`), or no input (auto-detect).

## Setup

1. [Setup](../../lib/github/setup.md) — authenticate and detect context (role, remotes, state)

## Step 1: Match Input to PR

Use [lookup-pr](../../lib/github/lookup-pr.md) to find the PR.

- If PR number or branch name provided: use "By PR number" or "By branch name" lookup
- If no input: auto-detect from current branch, or list open PRs for user selection

Validate PR state: OPEN (continue), CLOSED (warn), MERGED (exit).

## Step 2: Detect Permission

Run [detect-permission](../../lib/github/detect-permission.md) to determine push access.

## Step 3: Fetch Unresolved Comments

Run [fetch-comments](../../lib/github/fetch-comments.md).

If no unresolved comments exist, inform user and exit.

## Step 4: Classify Comments

| Category | Description | Examples |
| -------- | ----------- | -------- |
| **A: Actionable** | Code changes required | Bugs, missing validation, race conditions, incorrect logic |
| **B: Discussable** | May skip if follows `.claude/rules/` | Style preferences, premature optimizations |
| **C: Informational** | Resolve without changes | Acknowledgments, "optional" suggestions |

Present summary showing category, file:line, and issue for each comment. For Category B, explain why code may already comply with `.claude/rules/`.

## Step 5: Get User Confirmation

**Always let the user decide which comments to address and which to skip.** Present ALL unresolved comments (A, B, and C) in a numbered list with their classification and brief summary.

Ask the user to specify which comments to address, skip, or discuss:

- Recommend addressing Category A items
- Mark Category B with rationale for skipping or addressing
- Mark Category C as skippable by default

**User choices per comment:** Address (make changes) / Skip (resolve as-is) / Discuss (need clarification)

Only proceed with the comments the user explicitly selects. Do NOT auto-resolve any comment without user consent.

## Step 6: Work Location Setup

Work directly on the PR branch. If not already on `$HEAD_BRANCH`, checkout and pull:

```bash
git checkout $HEAD_BRANCH
git pull "$PUSH_REMOTE" "$HEAD_BRANCH"
```

If in a worktree on different branch, offer to create new worktree or switch to main repo.

## Step 7: Address Comments

For each "Address" comment:

1. Read files with Read tool
2. Make changes with Edit tool
3. After all changes, commit using `/git-commit` skill

Then run [commit-and-push](../../lib/github/commit-and-push.md):
1. Rebase onto `$BASE_REF`
2. Ensure single valid commit (squash with original PR commit)
3. Push (update push with `--force-with-lease` to `$PUSH_REMOTE`)

## Step 8: Reply and Resolve

Use [reply-and-resolve](../../lib/github/reply-and-resolve.md) for each comment.

## Checklist

- [ ] PR matched, permission determined, comments fetched and classified
- [ ] Unresolved comments fetched and classified
- [ ] ALL comments presented to user for selection
- [ ] Code changes made and committed (use `/git-commit`)
- [ ] Changes pushed (single valid commit, squashed with original PR commit)
- [ ] All selected comments replied to and resolved
- [ ] Summary presented

## Remember

**Not all comments require code changes.** Evaluate against `.claude/rules/` first. When in doubt, consult user.
