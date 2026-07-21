# Detect Permission

Used by `fix-pr` when working on someone else's PR. Determines push access and overrides `PUSH_REMOTE` if needed.

## Fetch PR Metadata

```bash
PR_DATA=$(gh pr view $PR_NUMBER --repo "$PR_REPO_OWNER/$PR_REPO_NAME" --json \
  number,title,headRefName,headRepository,headRepositoryOwner,\
  baseRefName,state,maintainerCanModify,author)

HEAD_BRANCH=$(echo "$PR_DATA" | jq -r '.headRefName')
HEAD_REPO_OWNER=$(echo "$PR_DATA" | jq -r '.headRepositoryOwner.login')
HEAD_REPO_NAME=$(echo "$PR_DATA" | jq -r '.headRepository.name')
PR_AUTHOR=$(echo "$PR_DATA" | jq -r '.author.login')
MAINTAINER_CAN_MODIFY=$(echo "$PR_DATA" | jq -r '.maintainerCanModify')
CURRENT_USER=$(gh api user -q '.login')
```

## Determine Permission

```bash
if [ "$PR_AUTHOR" = "$CURRENT_USER" ]; then
  PERMISSION="owner"
elif [ "$HEAD_REPO_OWNER" = "$PR_REPO_OWNER" ]; then
  PERMISSION="write"
elif [ "$MAINTAINER_CAN_MODIFY" = "true" ]; then
  PERMISSION="maintainer"
else
  echo "Error: No push access to PR #$PR_NUMBER"
  echo "Ask PR author to enable 'Allow edits from maintainers'"
  exit 1
fi
```

## Set Push Target

```bash
case "$PERMISSION" in
  owner|write)
    PUSH_REMOTE="origin"
    WORK_BRANCH="$HEAD_BRANCH"
    ;;
  maintainer)
    if [ "$HEAD_REPO_OWNER" = "$UPSTREAM_OWNER" ]; then
      # PR branch is on canonical repo — use upstream
      FORK_REMOTE="upstream"
    else
      # PR branch is on a fork — use author name as remote
      FORK_REMOTE="$HEAD_REPO_OWNER"
      if ! git remote | grep -q "^${FORK_REMOTE}$"; then
        git remote add "$FORK_REMOTE" \
          "git@github.com:$HEAD_REPO_OWNER/$HEAD_REPO_NAME.git"
      fi
    fi
    git fetch "$FORK_REMOTE" "$HEAD_BRANCH"
    PUSH_REMOTE="$FORK_REMOTE"
    WORK_BRANCH="$HEAD_BRANCH"
    ;;
esac
```

## Cleanup After Push

Do NOT remove the fork remote — it is reused by upstream tracking for auto-detection in `/github-pr` and `/fix-pr`.
