---
name: checkout-pr
description: Check out another user's GitHub PR locally by adding/fetching its head remote and creating a pr-N-work branch. Use when working on a PR branch locally.
---

# Checkout PR

Use this to work on another user's PR branch without guessing remotes.

## Workflow

1. Read [setup](../../lib/github/setup.md).
2. Fetch PR metadata:

```bash
gh pr view "$PR_NUMBER" --repo "$PR_REPO_OWNER/$PR_REPO_NAME" --json \
  number,title,headRefName,headRepository,headRepositoryOwner,baseRefName,state,maintainerCanModify,author
```

3. Stop if the PR is merged. Warn before continuing if it is closed.
4. If the head repo is canonical, use the existing base remote. Otherwise add/fetch a remote named
   after the PR head owner:

```bash
git remote add "$HEAD_REPO_OWNER" "git@github.com:$HEAD_REPO_OWNER/$HEAD_REPO_NAME.git"
git fetch "$HEAD_REPO_OWNER" "$HEAD_BRANCH"
```

5. Follow [checkout-fork-branch](../../lib/github/checkout-fork-branch.md).
6. Report the local branch, remote, and push refspec.

The `github-pr` and `fix-pr` skills can use the upstream tracking set by this workflow.
