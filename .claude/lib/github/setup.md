# Setup

Initialize GitHub workflow: authenticate and detect repository context.

## 1. Authenticate

```bash
gh auth status
```

If not authenticated, tell user to run `gh auth login` and stop.

## 2. Detect Context

Detects repository role, remotes, and current state. Sets standard variables used by all skills.

### Canonical Repo

```bash
UPSTREAM_OWNER="ChaoWao"
UPSTREAM_NAME="simpler"
UPSTREAM_REPO="$UPSTREAM_OWNER/$UPSTREAM_NAME"
DEFAULT_BRANCH="main"
```

### Ensure Upstream Remote

```bash
if ! git remote | grep -q "^upstream$"; then
  git remote add upstream "git@github.com:$UPSTREAM_REPO.git"
fi
git fetch upstream
```

### Parse Origin

```bash
ORIGIN_URL=$(git remote get-url origin 2>/dev/null || echo "")

if [ -z "$ORIGIN_URL" ]; then
  echo "Error: No 'origin' remote found"
  exit 1
fi

REPO_OWNER=$(echo "$ORIGIN_URL" | sed -n 's#.*[:/]\([^/]*\)/\([^/]*\)\.git.*#\1#p')
REPO_NAME=$(echo "$ORIGIN_URL" | sed -n 's#.*[:/]\([^/]*\)/\([^/]*\)\.git.*#\2#p')
```

### Determine Role

`BASE_REF` is always `upstream/main` since upstream always points to the canonical repo.

```bash
BASE_REF="upstream/$DEFAULT_BRANCH"
PUSH_REMOTE="origin"
PR_REPO_OWNER="$UPSTREAM_OWNER"
PR_REPO_NAME="$UPSTREAM_NAME"

if [ "$REPO_OWNER" = "$UPSTREAM_OWNER" ] && [ "$REPO_NAME" = "$UPSTREAM_NAME" ]; then
  ROLE="owner"
  PR_HEAD_PREFIX=""
else
  ROLE="fork"
  PR_HEAD_PREFIX="$REPO_OWNER:"
fi
```

### Fetch Origin

```bash
git fetch origin
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

### Variables Set

| Variable | Owner | Fork |
| -------- | ----- | ---- |
| `ROLE` | `"owner"` | `"fork"` |
| `BASE_REF` | `upstream/main` | `upstream/main` |
| `PUSH_REMOTE` | `origin` | `origin` |
| `PR_REPO_OWNER` | `ChaoWao` | `ChaoWao` |
| `PR_REPO_NAME` | `simpler` | `simpler` |
| `PR_HEAD_PREFIX` | `""` | `"myuser:"` |
| `DEFAULT_BRANCH` | `main` | `main` |
