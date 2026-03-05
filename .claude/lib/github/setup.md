# Setup

Initialize GitHub workflow: authenticate and detect repository context.

## 1. Authenticate

```bash
gh auth status
```

If not authenticated, tell user to run `gh auth login` and stop.

## 2. Detect Context

Detects repository role, remotes, and current state. Sets standard variables used by all skills.

### Parse Remotes

```bash
# Get remote URLs
ORIGIN_URL=$(git remote get-url origin 2>/dev/null || echo "")
UPSTREAM_URL=$(git remote get-url upstream 2>/dev/null || echo "")

# Validate origin exists
if [ -z "$ORIGIN_URL" ]; then
  echo "Error: No 'origin' remote found"
  exit 1
fi

# Get default branch
DEFAULT_BRANCH=$(git remote show origin | sed -n 's/.*HEAD branch: \(.*\)/\1/p')

# Parse origin → REPO_OWNER / REPO_NAME
REPO_OWNER=$(echo "$ORIGIN_URL" | sed -n 's#.*[:/]\([^/]*\)/\([^/]*\)\.git.*#\1#p')
REPO_NAME=$(echo "$ORIGIN_URL" | sed -n 's#.*[:/]\([^/]*\)/\([^/]*\)\.git.*#\2#p')

# Parse upstream (if exists)
if [ -n "$UPSTREAM_URL" ]; then
  UPSTREAM_OWNER=$(echo "$UPSTREAM_URL" | sed -n 's#.*[:/]\([^/]*\)/\([^/]*\)\.git.*#\1#p')
  UPSTREAM_NAME=$(echo "$UPSTREAM_URL" | sed -n 's#.*[:/]\([^/]*\)/\([^/]*\)\.git.*#\2#p')
fi
```

### Determine Role

```bash
if [ -n "$UPSTREAM_URL" ]; then
  # Fork contributor
  ROLE="fork"
  BASE_REF="upstream/$DEFAULT_BRANCH"
  PUSH_REMOTE="origin"
  PR_REPO_OWNER="$UPSTREAM_OWNER"
  PR_REPO_NAME="$UPSTREAM_NAME"
  PR_HEAD_PREFIX="$REPO_OWNER:"
else
  # Repo owner
  ROLE="owner"
  BASE_REF="origin/$DEFAULT_BRANCH"
  PUSH_REMOTE="origin"
  PR_REPO_OWNER="$REPO_OWNER"
  PR_REPO_NAME="$REPO_NAME"
  PR_HEAD_PREFIX=""
fi
```

### Fetch Remotes

```bash
git fetch origin
if [ "$ROLE" = "fork" ]; then
  git fetch upstream
fi
```

### Gather State

```bash
BRANCH_NAME=$(git branch --show-current 2>/dev/null || echo "")
UNCOMMITTED=$(git status --porcelain)
if [ -n "$BRANCH_NAME" ]; then
  COMMITS_AHEAD=$(git rev-list HEAD --not "$BASE_REF" --count 2>/dev/null || echo "0")
else
  COMMITS_AHEAD="0"
fi
```

See [README.md](README.md) for the full list of variables set by this procedure.
