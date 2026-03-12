# Lookup PR

Find PR by number, branch name, or list all open PRs. Always use `$PR_REPO_OWNER/$PR_REPO_NAME` (the canonical repo).

## By PR Number

```bash
gh pr view $PR_NUMBER --repo "$PR_REPO_OWNER/$PR_REPO_NAME" \
  --json number,title,headRefName,state
```

## By Branch Name

For owner (origin IS canonical): search directly by branch name.

```bash
gh pr list --repo "$PR_REPO_OWNER/$PR_REPO_NAME" --head "$BRANCH_NAME" \
  --json number,title,state
```

For fork contributor: prefix with fork owner so GitHub matches the correct head.

```bash
gh pr list --repo "$PR_REPO_OWNER/$PR_REPO_NAME" \
  --head "$PR_HEAD_PREFIX$BRANCH_NAME" \
  --json number,title,state
```

`$PR_HEAD_PREFIX` is `""` for owner, `"myuser:"` for fork (set by Setup).

## By Upstream Tracking (cross-fork auto-detect)

When on a `pr-*-work` branch with upstream tracking to a fork remote:

```bash
gh pr list --repo "$PR_REPO_OWNER/$PR_REPO_NAME" \
  --head "$UPSTREAM_REMOTE:$HEAD_BRANCH" \
  --json number,title,state
```

## List Open PRs

For user selection:

```bash
gh pr list --repo "$PR_REPO_OWNER/$PR_REPO_NAME" --state open \
  --json number,title,headRefName,author
```
