---
name: github-pr
description: Create or update a GitHub pull request after changes are committed, rebased, pushed, and summarized for review.
---

# GitHub PR

Use this to create a new PR or update an existing PR.

## Workflow

1. Read [setup](../../lib/github/setup.md).
2. Use [lookup-pr](../../lib/github/lookup-pr.md) to detect an existing PR.
3. If no PR exists and there are no commits ahead plus no uncommitted changes, stop.
4. If on the default branch, create a feature branch from `BASE_REF`.
5. If changes are uncommitted, use the `git-commit` skill.
6. Use [commit-and-push](../../lib/github/commit-and-push.md).
7. Create or update the PR.

## Create PR

```bash
gh pr create \
  --repo "$PR_REPO_OWNER/$PR_REPO_NAME" \
  --base "$DEFAULT_BRANCH" \
  --head "${PR_HEAD_PREFIX}${BRANCH_NAME}" \
  --title "Brief title" \
  --body "$(cat <<'EOF'
## Summary
- ...

## Testing
- ...
EOF
)"
```

Generate the title and body from the final commit and actual verification. Keep the PR description
current when additional commits are pushed.

## Update Existing PR

Show the current PR, push the branch, and update title/body if the commit scope changed:

```bash
gh pr view "$PR_NUMBER" --repo "$PR_REPO_OWNER/$PR_REPO_NAME"
gh pr edit "$PR_NUMBER" --repo "$PR_REPO_OWNER/$PR_REPO_NAME" --title "..." --body "..."
```

## Constraints

- Do not describe the PR as ready or mergeable until relevant local verification is recorded.
- Do not push directly to `main`.
- Use draft PRs when work is still under active development.
