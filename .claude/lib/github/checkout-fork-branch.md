# Checkout Fork Branch

Create or switch to a local working branch for a cross-fork PR, and set `BRANCH_NAME` to the correct push refspec.

**Requires:** `PR_NUMBER`, `PUSH_REMOTE`, `HEAD_BRANCH` (set by [detect-permission](detect-permission.md)).

```bash
LOCAL_BRANCH="pr-$PR_NUMBER-work"

if git show-ref --verify --quiet "refs/heads/$LOCAL_BRANCH"; then
  git checkout "$LOCAL_BRANCH"
  git pull "$PUSH_REMOTE" "$HEAD_BRANCH"
else
  git fetch "$PUSH_REMOTE" "$HEAD_BRANCH:$LOCAL_BRANCH"
  git checkout "$LOCAL_BRANCH"
fi

# Set refspec so commit-and-push pushes local branch to the fork's remote branch
# e.g. git push --force-with-lease pr-176 pr-176-work:feat/batch-counter
BRANCH_NAME="$LOCAL_BRANCH:$HEAD_BRANCH"
```

**Variables set:** `LOCAL_BRANCH`, `BRANCH_NAME`.
