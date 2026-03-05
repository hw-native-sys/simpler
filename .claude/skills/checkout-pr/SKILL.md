---
name: checkout-pr
description: Add a remote for a given PR's head repository and checkout a local pr-xxx-work branch. Use when the user wants to work on someone else's PR locally.
---

# Checkout PR Workflow

Add a remote for a PR's head repository (if not already added) and checkout a local working branch.

## Input

Accept PR number (`123`, `#123`). Required.

## Setup

1. [Setup](../../lib/github/setup.md) — authenticate and detect context (role, remotes, state)

## Step 1: Fetch PR Metadata

```bash
PR_DATA=$(gh pr view $PR_NUMBER --repo "$PR_REPO_OWNER/$PR_REPO_NAME" --json \
  number,title,headRefName,headRepository,headRepositoryOwner,\
  baseRefName,state,maintainerCanModify,author)

HEAD_BRANCH=$(echo "$PR_DATA" | jq -r '.headRefName')
HEAD_REPO_OWNER=$(echo "$PR_DATA" | jq -r '.headRepositoryOwner.login')
HEAD_REPO_NAME=$(echo "$PR_DATA" | jq -r '.headRepository.name')
PR_AUTHOR=$(echo "$PR_DATA" | jq -r '.author.login')
MAINTAINER_CAN_MODIFY=$(echo "$PR_DATA" | jq -r '.maintainerCanModify')
PR_STATE=$(echo "$PR_DATA" | jq -r '.state')
```

Validate PR state: OPEN (continue), CLOSED (warn user), MERGED (exit).

## Step 2: Add Remote (if needed)

```bash
FORK_REMOTE="pr-$PR_NUMBER"

if git remote | grep -q "^${FORK_REMOTE}$"; then
  echo "Remote '$FORK_REMOTE' already exists, fetching latest..."
else
  git remote add "$FORK_REMOTE" \
    "git@github.com:$HEAD_REPO_OWNER/$HEAD_REPO_NAME.git"
fi

git fetch "$FORK_REMOTE" "$HEAD_BRANCH"
```

## Step 3: Checkout Work Branch

Run [checkout-fork-branch](../../lib/github/checkout-fork-branch.md) with `PUSH_REMOTE=$FORK_REMOTE` and `HEAD_BRANCH` from Step 1.

## Step 4: Report

Print summary:

```
Checked out PR #$PR_NUMBER ($PR_AUTHOR)
  Remote: $FORK_REMOTE -> git@github.com:$HEAD_REPO_OWNER/$HEAD_REPO_NAME.git
  Branch: $LOCAL_BRANCH -> $FORK_REMOTE/$HEAD_BRANCH
  Push:   PUSH_REMOTE=$FORK_REMOTE  BRANCH_NAME=$LOCAL_BRANCH:$HEAD_BRANCH
```

Remind user that `/github-pr $PR_NUMBER` and `/address-pr-comments $PR_NUMBER` will pick up the correct push target automatically.
