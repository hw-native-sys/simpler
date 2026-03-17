---
name: review-pr
description: Review a GitHub PR by analyzing the correct diff (merge-base to HEAD), summarizing changes, and providing feedback. Use when the user asks to review a PR, analyze PR changes, or give feedback on a pull request.
---

# Review PR Workflow

Analyze and review PR changes using the correct merge-base diff.

## Input

Accept PR number (`123`, `#123`), or no input (review current branch against its base).

## Step 1: Setup

1. [Setup](../../lib/github/setup.md) — authenticate and detect context (role, remotes, state)

## Step 2: Determine Diff Base

**This is the critical step.** Never diff against a stale local branch.

```bash
# Ensure upstream is fresh
git fetch upstream "$DEFAULT_BRANCH" --quiet

# Find the true fork point (merge-base)
MERGE_BASE=$(git merge-base "upstream/$DEFAULT_BRANCH" HEAD)
echo "Merge base: $MERGE_BASE"
```

If reviewing a PR by number (not the current branch):

```bash
# Fetch PR metadata
PR_DATA=$(gh pr view $PR_NUMBER --repo "$PR_REPO_OWNER/$PR_REPO_NAME" \
  --json headRefName,baseRefName,commits)
BASE_BRANCH=$(echo "$PR_DATA" | jq -r '.baseRefName')

# Fetch and compute merge-base against the PR's actual base branch
git fetch upstream "$BASE_BRANCH" --quiet
MERGE_BASE=$(git merge-base "upstream/$BASE_BRANCH" HEAD)
```

**Validation:** Verify the merge-base is NOT a commit on the PR branch itself:

```bash
# List PR-only commits
PR_COMMITS=$(git log --oneline "$MERGE_BASE"..HEAD)
echo "$PR_COMMITS"

# Sanity: merge-base should not appear in PR commits
if git log --oneline "$MERGE_BASE"..HEAD | grep -q "$(git rev-parse --short $MERGE_BASE)"; then
  echo "ERROR: merge-base is on the PR branch — check upstream fetch"
fi
```

## Step 3: Gather Diff

All diffs use the three-dot syntax against the merge-base:

```bash
# File list and stats
git diff "$MERGE_BASE"...HEAD --stat
git diff "$MERGE_BASE"...HEAD --name-only

# Full diff (for reading)
git diff "$MERGE_BASE"...HEAD
```

For large PRs, read diffs per-category (e.g., by directory or file type) to avoid overwhelming context.

## Step 4: Gather PR Context

```bash
# PR description and metadata
gh pr view $PR_NUMBER --repo "$PR_REPO_OWNER/$PR_REPO_NAME"

# Commit messages
git log --oneline "$MERGE_BASE"..HEAD
```

## Step 5: Analyze Changes

For each changed file or logical group:

1. **Read the diff** — understand what changed
2. **Read surrounding context** if the diff alone is unclear (use `Read` tool on the full file)
3. **Categorize** the change: new feature, bugfix, refactor, test, docs, config, etc.

## Step 6: Write Review

Structure the review as:

### Summary
- What this PR does (1-3 sentences)
- Number of files changed, lines added/removed

### Change Breakdown
Group changes by logical area. For each group:
- What changed and why
- Any concerns (correctness, performance, compatibility, style)

### Issues Found
Categorize by severity:
- **Must fix**: Bugs, correctness issues, security problems
- **Should fix**: Style violations per `.claude/rules/`, missing error handling
- **Consider**: Suggestions, optional improvements

### Verdict
Overall assessment: approve / request changes / needs discussion

## Common Pitfalls

1. **Stale local main** — Always `git fetch upstream` before computing merge-base. Never use local `main` directly.
2. **Squash-merged upstream commits** — If upstream squash-merges, the merge-base may include commits that look like they are on the PR branch. Verify with `git log --oneline MERGE_BASE..HEAD`.
3. **Cross-fork PRs** — The PR head may be on a different remote. Use `/checkout-pr` first if not already on the PR branch.
4. **Large PRs** — Read diffs in chunks by directory. Do not try to read the entire diff in one tool call if it exceeds ~2000 lines.
