# Detect Permission

Determine whether the current user can push to a PR branch.

```bash
PR_DATA=$(gh pr view "$PR_NUMBER" --repo "$PR_REPO_OWNER/$PR_REPO_NAME" --json \
  number,title,headRefName,headRepository,headRepositoryOwner,baseRefName,state,maintainerCanModify,author)

HEAD_BRANCH=$(echo "$PR_DATA" | jq -r '.headRefName')
HEAD_REPO_OWNER=$(echo "$PR_DATA" | jq -r '.headRepositoryOwner.login')
HEAD_REPO_NAME=$(echo "$PR_DATA" | jq -r '.headRepository.name')
PR_AUTHOR=$(echo "$PR_DATA" | jq -r '.author.login')
MAINTAINER_CAN_MODIFY=$(echo "$PR_DATA" | jq -r '.maintainerCanModify')
CURRENT_USER=$(gh api user -q '.login')
```

```bash
if [ "$PR_AUTHOR" = "$CURRENT_USER" ]; then
  PERMISSION="owner"
  PUSH_REMOTE="origin"
  WORK_BRANCH="$HEAD_BRANCH"
elif [ "$HEAD_REPO_OWNER" = "$PR_REPO_OWNER" ]; then
  PERMISSION="write"
  PUSH_REMOTE="origin"
  WORK_BRANCH="$HEAD_BRANCH"
elif [ "$MAINTAINER_CAN_MODIFY" = "true" ]; then
  PERMISSION="maintainer"
  PUSH_REMOTE="$HEAD_REPO_OWNER"
  WORK_BRANCH="$HEAD_BRANCH"
  if ! git remote | grep -q "^${PUSH_REMOTE}$"; then
    git remote add "$PUSH_REMOTE" "git@github.com:$HEAD_REPO_OWNER/$HEAD_REPO_NAME.git"
  fi
  git fetch "$PUSH_REMOTE" "$HEAD_BRANCH"
else
  echo "No push access to PR #$PR_NUMBER. Ask the author to enable maintainer edits."
  exit 1
fi
```

Do not remove contributor remotes after use; upstream tracking may depend on them.
