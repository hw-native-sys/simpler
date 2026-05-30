# Checkout Fork Branch

Create or update a local work branch for a cross-fork PR.

Requires `PR_NUMBER`, `PUSH_REMOTE`, and `HEAD_BRANCH`.

```bash
LOCAL_BRANCH="pr-$PR_NUMBER-work"

if git show-ref --verify --quiet "refs/heads/$LOCAL_BRANCH"; then
  git checkout "$LOCAL_BRANCH"
  git pull "$PUSH_REMOTE" "$HEAD_BRANCH"
else
  git fetch "$PUSH_REMOTE" "$HEAD_BRANCH:$LOCAL_BRANCH"
  git checkout "$LOCAL_BRANCH"
fi

git branch --set-upstream-to="$PUSH_REMOTE/$HEAD_BRANCH" "$LOCAL_BRANCH"
BRANCH_NAME="$LOCAL_BRANCH:$HEAD_BRANCH"
```

Use `BRANCH_NAME` as the push refspec when updating the PR.
