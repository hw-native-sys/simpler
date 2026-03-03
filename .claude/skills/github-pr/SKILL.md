---
name: github-pr
description: Create or update a GitHub pull request after committing, rebasing, and pushing changes. Use when the user asks to create a PR, submit changes for review, or open a pull request.
---

# GitHub Pull Request Workflow

## Setup: Detect Context

### 1. Authenticate

```bash
gh auth status
```

If not authenticated, tell user to run `gh auth login` and **stop**.

### 2. Detect Role

```bash
ORIGIN_URL=$(git remote get-url origin)
UPSTREAM_URL=$(git remote get-url upstream 2>/dev/null || echo "")
DEFAULT_BRANCH=$(git remote show origin | sed -n 's/.*HEAD branch: \(.*\)/\1/p')
```

Two developer roles use this repo differently:

| Role | `upstream` exists? | Push target | PR target |
| ---- | ------------------ | ----------- | --------- |
| **Repo owner** | No | `origin` | `origin/$DEFAULT_BRANCH` |
| **Fork contributor** | Yes | `origin` (fork) | `upstream/$DEFAULT_BRANCH` |

Compute these variables once — all later steps use them without role conditionals:

| Variable | Repo owner | Fork contributor |
| -------- | ---------- | ---------------- |
| `BASE_REF` | `origin/$DEFAULT_BRANCH` | `upstream/$DEFAULT_BRANCH` |
| `PUSH_REMOTE` | `origin` | `origin` |
| `PR_REPO` | *(omit — same repo)* | `OWNER/REPO` from `upstream` URL |
| `PR_HEAD` | `$BRANCH_NAME` | `FORK_OWNER:$BRANCH_NAME` (`FORK_OWNER` from `origin` URL) |

### 3. Fetch and Gather State

Single fetch for all remotes:

```bash
git fetch origin
if [ -n "$UPSTREAM_URL" ]; then git fetch upstream; fi
```

Gather current state:

```bash
BRANCH_NAME=$(git branch --show-current)
UNCOMMITTED=$(git status --porcelain)
COMMITS_AHEAD=$(git rev-list HEAD --not "$BASE_REF" --count)
```

### 4. Check for Existing PR

```bash
gh pr list --head "$BRANCH_NAME" --state open
```

### 5. Route

| Existing PR? | On default branch? | Commits ahead? | Uncommitted? | Route |
| ------------ | ------------------- | -------------- | ------------ | ----- |
| No | * | * | * | **Path A** (create new PR) |
| Yes | No | Ahead > 0 | * | **Path B** (update existing PR) |
| Yes | No | 0 | Yes | **Path B** (commit + update) |
| Yes | No | 0 | No | Already up to date — exit |

**Validation**: If no existing PR, `COMMITS_AHEAD == 0`, and no uncommitted changes — error. Nothing to PR.

---

## Path A: Create New PR

### A1. Prepare Branch

Check the current branch and its relationship to the changes being PR'd:

| Current branch | Commits ahead of `$BASE_REF`? | Commits related to current changes? | Action |
| -------------- | ----------------------------- | ----------------------------------- | ------ |
| `$DEFAULT_BRANCH` | — | — | Create new branch from HEAD |
| Feature branch | Yes, all related | Yes | Continue on it |
| Feature branch | Yes, unrelated commits exist | No | Stash → new branch from `$BASE_REF` → pop |
| Feature branch | No (0 ahead) | Branch name fits | Continue on it (uncommitted changes only) |
| Feature branch | No (0 ahead) | Branch name unrelated | Stash → new branch from `$BASE_REF` → pop |

**Case 1 — On `$DEFAULT_BRANCH`**: Create and switch to a new branch:

```bash
git checkout -b <branch-name>
```

**Case 2 — Feature branch, all commits related**: Continue on it. No action needed.

**Case 3 — Feature branch, unrelated commits exist**: The branch has commits that are not part of the changes you want to PR. Stash uncommitted work, create a new branch from `$BASE_REF`, and restore:

```bash
git stash
git checkout -b <branch-name> "$BASE_REF"
git stash pop
```

To detect this: inspect `git log $BASE_REF..HEAD --oneline` and compare those commits against the staged/unstaged changes. If any existing commits touch unrelated files or features, treat the branch as unrelated.

**Case 4 — Feature branch, 0 commits ahead, branch name fits**: Continue on it — only uncommitted changes will be committed.

**Case 5 — Feature branch, 0 commits ahead, branch name unrelated**: The branch name doesn't match the intent of the uncommitted changes. Stash, create a new branch, and restore:

```bash
git stash
git checkout -b <branch-name> "$BASE_REF"
git stash pop
```

See **[Branch Naming](#branch-naming)** for how to generate the name. Do NOT ask the user.

### A2. Commit Changes

If there are uncommitted changes, delegate to `/git-commit` (runs review and testing).

If already committed, skip.

### A3. Ensure Single Valid Commit

Follow the **[shared procedure](#shared-ensure-single-valid-commit)** below.

### A4. Rebase

```bash
git rebase "$BASE_REF"
```

On conflicts: resolve files, `git add`, `git rebase --continue`. If stuck: `git rebase --abort`.

After rebase, re-run the **[shared procedure](#shared-ensure-single-valid-commit)** if the rebase introduced changes.

### A5. Push

Always push to `origin`. Never push to `upstream` or other remotes. Use `git` commands (not `gh`).

```bash
git push --set-upstream origin "$BRANCH_NAME"
```

### A6. Create PR

```bash
gh pr create \
  --repo "$PR_REPO" \
  --base "$DEFAULT_BRANCH" \
  --head "$PR_HEAD" \
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

Auto-generate title and body from the commit message. Keep title under 72 characters. Do NOT add AI co-author footers.

### Checklist A

- [ ] `gh auth status` passes
- [ ] Branch created (if on default branch)
- [ ] Changes committed via `/git-commit`
- [ ] Exactly 1 valid commit ahead of base
- [ ] Rebased onto `$BASE_REF`
- [ ] Pushed to `origin`
- [ ] PR created with clear title and summary

---

## Path B: Update Existing PR

Display the existing PR with `gh pr view`.

### B1. Commit Changes

If there are uncommitted changes, delegate to `/git-commit`.

If already committed, skip.

### B2. Ensure Single Valid Commit

Follow the **[shared procedure](#shared-ensure-single-valid-commit)** below.

### B3. Rebase

```bash
git rebase "$BASE_REF"
```

On conflicts: resolve files, `git add`, `git rebase --continue`. If stuck: `git rebase --abort`.

After rebase, re-run the **[shared procedure](#shared-ensure-single-valid-commit)** if the rebase introduced changes.

### B4. Push

```bash
git push --force-with-lease origin "$BRANCH_NAME"
```

Always push to `origin`. Never push to `upstream` or other remotes.

### B5. Update PR Title/Body

If the commit message changed, update the PR to match:

```bash
gh pr edit --title "Updated title" --body "Updated body"
```

### Checklist B

- [ ] `gh auth status` passes
- [ ] Changes committed via `/git-commit`
- [ ] Exactly 1 valid commit ahead of base
- [ ] Rebased onto `$BASE_REF`
- [ ] Force-pushed to `origin`
- [ ] PR title/body updated (if commit changed)

---

## Shared: Ensure Single Valid Commit

### Squash if Needed

```bash
COMMITS_AHEAD=$(git rev-list HEAD --not "$BASE_REF" --count)
```

If more than 1 commit ahead, squash:

```bash
git reset --soft "$BASE_REF"
git commit  # Re-commit with proper message via /git-commit conventions
```

### Validate Commit Message

```bash
git log -1 --format='%s'   # Subject line
git log -1 --format='%b'   # Body
```

| Rule | Check |
| ---- | ----- |
| Subject format | `Type: concise description` |
| Valid types | Add, Fix, Update, Refactor, Support, Sim, CI |
| Length | Under 72 characters |
| Style | Imperative mood, no trailing period |
| Body | Required if commit touches 3+ files |
| Co-author | No AI co-author lines |

If the message does not comply, amend:

```bash
git commit --amend -m "Type: corrected description"
```

For multi-line messages:

```bash
git commit --amend -m "$(cat <<'EOF'
Type: corrected description

- What changed and why
EOF
)"
```

---

## Post-Merge Cleanup (Repo Owner Only)

After the PR is merged:

```bash
git checkout "$DEFAULT_BRANCH"
git pull origin "$DEFAULT_BRANCH"
git branch -d "$BRANCH_NAME"
git push origin --delete "$BRANCH_NAME"
```

Fork contributors do not need remote cleanup.

---

## Reference

### Branch Naming

1. Determine the commit type prefix:

| Commit type | Branch prefix |
| ----------- | ------------- |
| Add | `feat/` |
| Fix | `fix/` |
| Update | `feat/` |
| Refactor | `refactor/` |
| Support | `support/` |
| Sim | `sim/` |
| CI | `support/` |

2. Take the commit subject description (after `Type: `), lowercase it, replace spaces and special characters with hyphens, strip trailing hyphens.

3. Truncate to 50 characters.

Example: `Refactor: inline ring buffer hot paths` → `refactor/inline-ring-buffer-hot-paths`

### Common Issues

| Issue | Solution |
| ----- | -------- |
| `gh auth` fails | Tell user to run `gh auth login` |
| PR already exists | Route to Path B |
| Merge conflicts during rebase | Resolve, `git add`, `git rebase --continue` |
| Push rejected (non-fast-forward) | `git push --force-with-lease` after rebase |
| More than 1 commit ahead | Squash via `git reset --soft` + re-commit |
| Can't detect role | Check `git remote -v` output |
| Nothing to PR | Error — tell user to make changes first |
