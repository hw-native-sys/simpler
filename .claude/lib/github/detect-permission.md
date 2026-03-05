# Detect Permission

Used by `address-pr-comments` when working on someone else's PR. Determines push access and overrides `PUSH_REMOTE` if needed.

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
    FORK_REMOTE="pr-$PR_NUMBER"
    if ! git remote | grep -q "^${FORK_REMOTE}$"; then
      git remote add "$FORK_REMOTE" \
        "https://github.com/$HEAD_REPO_OWNER/$HEAD_REPO_NAME.git"
    fi
    git fetch "$FORK_REMOTE" "$HEAD_BRANCH"
    PUSH_REMOTE="$FORK_REMOTE"
    WORK_BRANCH="$HEAD_BRANCH"
    ;;
esac
```

## Cleanup After Push

```bash
if [ "$PERMISSION" = "maintainer" ]; then
  git remote remove "$FORK_REMOTE" 2>/dev/null || true
fi
```
