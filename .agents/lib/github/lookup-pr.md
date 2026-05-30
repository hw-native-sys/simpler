# Lookup PR

Find a pull request in `$PR_REPO_OWNER/$PR_REPO_NAME`.

## By Number

```bash
gh pr view "$PR_NUMBER" --repo "$PR_REPO_OWNER/$PR_REPO_NAME" \
  --json number,title,headRefName,baseRefName,state,author
```

## By Branch

For owner branches:

```bash
gh pr list --repo "$PR_REPO_OWNER/$PR_REPO_NAME" --head "$BRANCH_NAME" \
  --json number,title,state,headRefName
```

For fork branches:

```bash
gh pr list --repo "$PR_REPO_OWNER/$PR_REPO_NAME" \
  --head "$PR_HEAD_PREFIX$BRANCH_NAME" \
  --json number,title,state,headRefName
```

## By Upstream Tracking

When a local branch tracks a contributor remote:

```bash
UPSTREAM=$(git rev-parse --abbrev-ref '@{upstream}' 2>/dev/null || echo "")
UPSTREAM_REMOTE=$(echo "$UPSTREAM" | cut -d/ -f1)
HEAD_BRANCH=$(echo "$UPSTREAM" | cut -d/ -f2-)
gh pr list --repo "$PR_REPO_OWNER/$PR_REPO_NAME" \
  --head "$UPSTREAM_REMOTE:$HEAD_BRANCH" \
  --json number,title,state,headRefName
```

If no match is found, list open PRs and ask the user to choose.
