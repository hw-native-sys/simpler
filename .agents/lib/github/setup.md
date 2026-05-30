# Setup

Initialize GitHub workflow state.

## Authenticate

Run:

```bash
gh auth status
```

If authentication fails, tell the user to run `gh auth login` and stop.

## Detect Canonical Repository

Prefer the current GitHub CLI context:

```bash
PR_REPO_OWNER=$(gh repo view --json owner -q '.owner.login')
PR_REPO_NAME=$(gh repo view --json name -q '.name')
DEFAULT_BRANCH=$(gh repo view --json defaultBranchRef -q '.defaultBranchRef.name')
```

Fallback for this repository is `uv-xiao/pto-cu` and `main`.

## Detect Role And Remotes

```bash
ORIGIN_URL=$(git remote get-url origin 2>/dev/null || echo "")
REPO_OWNER=$(echo "$ORIGIN_URL" | sed -n 's#.*[:/]\([^/]*\)/\([^/]*\)\.git.*#\1#p')
REPO_NAME=$(echo "$ORIGIN_URL" | sed -n 's#.*[:/]\([^/]*\)/\([^/]*\)\.git.*#\2#p')

if [ "$REPO_OWNER" = "$PR_REPO_OWNER" ] && [ "$REPO_NAME" = "$PR_REPO_NAME" ]; then
  ROLE="owner"
  BASE_REMOTE="origin"
  PR_HEAD_PREFIX=""
else
  ROLE="fork"
  BASE_REMOTE="upstream"
  PR_HEAD_PREFIX="$REPO_OWNER:"
  if ! git remote | grep -q '^upstream$'; then
    git remote add upstream "git@github.com:$PR_REPO_OWNER/$PR_REPO_NAME.git"
  fi
fi

git fetch "$BASE_REMOTE" "$DEFAULT_BRANCH"
git fetch origin

BASE_REF="$BASE_REMOTE/$DEFAULT_BRANCH"
PUSH_REMOTE="origin"
BRANCH_NAME=$(git branch --show-current 2>/dev/null || echo "")
UNCOMMITTED=$(git status --porcelain)
COMMITS_AHEAD=$(git rev-list HEAD --not "$BASE_REF" --count 2>/dev/null || echo "0")
```

Never assume local `main` is fresh. Use `BASE_REF`.
