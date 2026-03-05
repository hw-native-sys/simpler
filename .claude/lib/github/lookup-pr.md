# Lookup PR

Find PR by number, branch name, or list all open PRs.

## By PR Number

```bash
gh pr view $PR_NUMBER --repo "$PR_REPO_OWNER/$PR_REPO_NAME" \
  --json number,title,headRefName,state
```

## By Branch Name

```bash
gh pr list --repo "$PR_REPO_OWNER/$PR_REPO_NAME" --head "$BRANCH_NAME" \
  --json number,title,state
```

## List Open PRs

For user selection:

```bash
gh pr list --repo "$PR_REPO_OWNER/$PR_REPO_NAME" --state open \
  --json number,title,headRefName,author
```
