---
name: github-pr
description: Create a GitHub pull request after committing, rebasing, and pushing changes. Use when the user asks to create a PR, submit changes for review, or open a pull request.
---

# GitHub Pull Request Workflow

Two developer roles use this repo differently:

| Role | Remote setup | Push target | PR target |
| ---- | ------------ | ----------- | --------- |
| **Repo owner** | `origin` = this repo | Feature branch on `origin` | `origin/<default-branch>` |
| **Fork contributor** | `origin` = fork, `upstream` = this repo | Feature branch on `origin` (fork) | `upstream/<default-branch>` |

## Step 1: Detect Role and Default Branch

```bash
ORIGIN_URL=$(git remote get-url origin)
UPSTREAM_URL=$(git remote get-url upstream 2>/dev/null || echo "")
```

Detect the default branch:

```bash
DEFAULT_BRANCH=$(git remote show origin | sed -n 's/.*HEAD branch: \(.*\)/\1/p')
```

**Decision logic**:

| `upstream` exists? | Role | Base ref |
| ------------------ | ---- | -------- |
| No | Repo owner | `origin/$DEFAULT_BRANCH` |
| Yes | Fork contributor | `upstream/$DEFAULT_BRANCH` |

Set `BASE_REF` accordingly for all subsequent steps.

## Step 2: Prepare Branch and Commit

```bash
BRANCH_NAME=$(git branch --show-current)
git status --porcelain
git fetch origin
if [ -n "$UPSTREAM_URL" ]; then git fetch upstream; fi
COMMITS_AHEAD=$(git rev-list HEAD --not "$BASE_REF" --count)
```

A branch "needs a new branch" when the branch name matches `$DEFAULT_BRANCH`.

**Validation**: If `COMMITS_AHEAD == 0` and no uncommitted changes, error — nothing to PR.

**Decision logic**:

| Needs new branch? | Uncommitted changes? | Action |
| ------------------ | -------------------- | ------ |
| Yes | Yes | Create new branch, then commit via `/git-commit` |
| Yes | No (commits ahead) | Create new branch (commits carry over) |
| Yes | No (nothing ahead) | Error — nothing to PR. Tell user to make changes first |
| No | Yes | Commit on current branch via `/git-commit` |
| No | No | Skip — already committed on a feature branch |

**If a new branch is needed:**

1. Auto-generate a branch name with a meaningful prefix based on the changes — do NOT ask the user:

| Change type | Prefix |
| ----------- | ------ |
| New feature | `feat/` |
| Bug fix | `fix/` |
| Refactoring | `refactor/` |
| Documentation | `docs/` |
| Tooling / CI | `support/` |
| Test changes | `test/` |

2. Create and switch:

```bash
git checkout -b <branch-name>
```

3. Commit via `/git-commit` skill (mandatory — runs review and testing).

## Step 3: Check for Existing PR

```bash
BRANCH_NAME=$(git branch --show-current)
gh pr list --head "$BRANCH_NAME" --state open
```

**If PR exists**: Display with `gh pr view`. Check if local branch is ahead of the remote tracking branch:

```bash
git rev-list origin/"$BRANCH_NAME"..HEAD --count
```

- Ahead > 0 → continue to Step 4–5 to push updates, skip Step 6
- Ahead == 0 → already up to date, exit

## Step 4: Rebase onto Base

```bash
git fetch origin
if [ -n "$UPSTREAM_URL" ]; then git fetch upstream; fi
git rebase "$BASE_REF"
```

**On conflicts**:

```bash
git status                       # View conflicts
# Edit files, remove markers
git add path/to/resolved/file
git rebase --continue
# If stuck: git rebase --abort
```

**After rebase**: Verify exactly 1 commit ahead of base (squash if needed):

```bash
git rev-list HEAD --not "$BASE_REF" --count
```

If more than 1 commit ahead, squash into a single commit:

```bash
git reset --soft "$BASE_REF"
git commit  # Re-commit with proper message via /git-commit conventions
```

## Step 5: Push

Use `git` commands (not `gh`) for push.

**Repo owner** — push feature branch to `origin`:

```bash
# First push
git push --set-upstream origin "$BRANCH_NAME"

# After rebase
git push --force-with-lease origin "$BRANCH_NAME"
```

**Fork contributor** — push to `origin` (their fork):

```bash
# First push
git push --set-upstream origin "$BRANCH_NAME"

# After rebase
git push --force-with-lease origin "$BRANCH_NAME"
```

Both roles always push to `origin`. Never push to `upstream` or other remotes.

## Step 6: Create PR

```bash
gh auth status
```

**If gh not authenticated**: Tell user to run `gh auth login` and stop.

**Repo owner** — PR within the same repo:

```bash
gh pr create \
  --base "$DEFAULT_BRANCH" \
  --head "$BRANCH_NAME" \
  --title "Brief description" \
  --body "$(cat <<'EOF'
## Summary
- Key change 1
- Key change 2

## Testing
- [ ] Simulation tests pass
- [ ] Hardware tests pass (if applicable)

Fixes #ISSUE_NUMBER (if applicable)
EOF
)"
```

**Fork contributor** — PR from fork to upstream:

```bash
gh pr create \
  --repo OWNER/REPO \
  --base "$DEFAULT_BRANCH" \
  --head "FORK_OWNER:$BRANCH_NAME" \
  --title "Brief description" \
  --body "$(cat <<'EOF'
## Summary
- Key change 1
- Key change 2

## Testing
- [ ] Simulation tests pass
- [ ] Hardware tests pass (if applicable)

Fixes #ISSUE_NUMBER (if applicable)
EOF
)"
```

Extract `OWNER/REPO` from `upstream` URL, `FORK_OWNER` from `origin` URL.

**PR title/body**: Auto-generate from commit messages. Keep the title under 72 characters.

**Important**: Do NOT add AI co-author footers or branding.

## Step 7: Post-Merge Cleanup (Repo Owner Only)

After the PR is merged, the repo owner should delete the feature branch:

```bash
git checkout "$DEFAULT_BRANCH"
git pull origin "$DEFAULT_BRANCH"
git branch -d "$BRANCH_NAME"                   # Delete local branch
git push origin --delete "$BRANCH_NAME"         # Delete remote branch
```

Fork contributors do not need remote cleanup — their fork branches are independent.

## Common Issues

| Issue | Solution |
| ----- | -------- |
| PR already exists | `gh pr view`, push if ahead, then exit |
| Merge conflicts | Resolve, `git add`, `git rebase --continue` |
| Push rejected | `git push --force-with-lease` |
| gh not authenticated | Tell user to run `gh auth login` |
| More than 1 commit | Squash with `git reset --soft` + re-commit |
| Can't detect role | Check `git remote -v` output |

## Checklist

- [ ] Role detected (repo owner vs fork contributor)
- [ ] Branch prepared (created from main if needed)
- [ ] Changes committed via `/git-commit`
- [ ] No existing PR for this branch
- [ ] Rebased onto latest base ref
- [ ] Exactly 1 commit ahead of base
- [ ] Pushed to `origin` (never other remotes)
- [ ] PR created with clear title and summary
- [ ] (Repo owner) Feature branch deleted after merge
