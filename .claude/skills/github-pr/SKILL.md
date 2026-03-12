---
name: github-pr
description: Create or update a GitHub pull request after committing, rebasing, and pushing changes. Use when the user asks to create a PR, submit changes for review, or open a pull request.
---

# GitHub Pull Request Workflow

## Input

Accept optional PR number (`123`, `#123`) to update a specific PR, or no input (auto-detect from current branch).

## Setup

1. [Setup](../../lib/github/setup.md) — authenticate and detect context (role, remotes, state)
2. **Auto-detect cross-fork PR context**: If no PR number provided and on a `pr-*-work` branch, check upstream tracking to detect cross-fork push target:
   ```bash
   if [ -z "$PR_NUMBER" ]; then
     UPSTREAM=$(git rev-parse --abbrev-ref "@{upstream}" 2>/dev/null || echo "")
     if [ -n "$UPSTREAM" ]; then
       UPSTREAM_REMOTE=$(echo "$UPSTREAM" | cut -d'/' -f1)
       # Check if upstream remote is NOT origin/upstream (i.e. it's a fork remote)
       if [ "$UPSTREAM_REMOTE" != "origin" ] && [ "$UPSTREAM_REMOTE" != "upstream" ]; then
         PUSH_REMOTE="$UPSTREAM_REMOTE"
         HEAD_BRANCH=$(echo "$UPSTREAM" | cut -d'/' -f2-)
         BRANCH_NAME="$BRANCH_NAME:$HEAD_BRANCH"
       fi
     fi
   fi
   ```
3. [Lookup PR](../../lib/github/lookup-pr.md) by PR number (if provided) or by upstream head branch (`$HEAD_BRANCH` with `--head "$UPSTREAM_REMOTE:$HEAD_BRANCH"` for cross-fork) or by local branch name to check for existing PR
4. **If PR number provided:** Run [detect-permission](../../lib/github/detect-permission.md) to setup cross-fork push target

## Route

| Existing PR? | Commits ahead? | Uncommitted? | Route |
| ------------ | -------------- | ------------ | ----- |
| No | * | * | **Path A** (create new PR) |
| Yes | > 0 | * | **Path B** (update existing PR) |
| Yes | 0 | Yes | **Path B** (commit + update) |
| Yes | 0 | No | Already up to date — exit |

**Validation**: If no existing PR, `COMMITS_AHEAD == 0`, and no uncommitted changes — error. Nothing to PR.

---

## Path A: Create New PR

### A1. Prepare Branch

If on `$DEFAULT_BRANCH`, create new branch. If on unrelated feature branch, stash and create new branch from `$BASE_REF`.

```bash
# On default branch
git checkout -b <branch-name>

# On unrelated branch
git stash
git checkout -b <branch-name> "$BASE_REF"
git stash pop
```

Generate branch name using [branch-naming](../../lib/github/branch-naming.md). Do NOT ask the user.

### A2. Commit Changes

If there are uncommitted changes, delegate to `/git-commit`.

If already committed, skip.

### A3. Commit and Push

Run [commit-and-push](../../lib/github/commit-and-push.md):
1. Rebase onto `$BASE_REF`
2. Ensure single valid commit (squash if needed)
3. Push (first push with `--set-upstream`)

If rebase introduces changes, re-validate the commit.

### A4. Create PR

```bash
gh pr create \
  --repo "$PR_REPO_OWNER/$PR_REPO_NAME" \
  --base "$DEFAULT_BRANCH" \
  --head "${PR_HEAD_PREFIX}${BRANCH_NAME}" \
  --title "Brief description" \
  --body "$(cat <<'EOF'
## Summary
- Key change 1
- Key change 2

## Testing
- [ ] Simulation tests pass
- [ ] Hardware tests pass (if applicable)

Fixes #ISSUE_NUMBER (if applicable)
EOF
)"
```

Auto-generate title and body from the commit message. Keep title under 72 characters. Do NOT add AI co-author footers.

### Checklist A

- [ ] `gh auth status` passes
- [ ] Branch created (if on default branch)
- [ ] Changes committed via `/git-commit`
- [ ] Exactly 1 valid commit ahead of base
- [ ] Rebased onto `$BASE_REF`
- [ ] Pushed to `origin`
- [ ] PR created with clear title and summary

---

## Path B: Update Existing PR

Display the existing PR with `gh pr view`.

### B0. Setup Work Branch (if updating cross-fork PR)

If PR number was provided and permission is "maintainer" (cross-fork with maintainerCanModify), run [checkout-fork-branch](../../lib/github/checkout-fork-branch.md) to create/switch to the local working branch and set the push refspec.

For owner/write permission, work directly on the existing branch.

### B1. Commit Changes

If there are uncommitted changes, delegate to `/git-commit`.

If already committed, skip.

### B2. Commit and Push

Run [commit-and-push](../../lib/github/commit-and-push.md):
1. Rebase onto `$BASE_REF`
2. Ensure single valid commit (squash if needed)
3. Push (update push with `--force-with-lease`)

If rebase introduces changes, re-validate the commit.

### B3. Update PR Title/Body

If the commit message changed, update the PR to match:

```bash
gh pr edit --title "Updated title" --body "Updated body"
```

### Checklist B

- [ ] `gh auth status` passes
- [ ] Changes committed via `/git-commit`
- [ ] Exactly 1 valid commit ahead of base
- [ ] Rebased onto `$BASE_REF`
- [ ] Force-pushed to `$PUSH_REMOTE`
- [ ] PR title/body updated (if commit changed)

---

## Post-Merge Cleanup (Repo Owner Only)

After the PR is merged:

```bash
git checkout "$DEFAULT_BRANCH"
git pull origin "$DEFAULT_BRANCH"
git branch -d "$BRANCH_NAME"
git push origin --delete "$BRANCH_NAME"
```

Fork contributors do not need remote cleanup.
