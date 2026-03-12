# Commit and Push

Prepares changes for PR: rebases, squashes commits, and pushes.

## 1. Rebase

First, rebase onto the latest base branch to incorporate upstream changes.

```bash
# Save commit SHA before rebase
BEFORE_REBASE=$(git rev-parse HEAD)

# Perform rebase
git rebase "$BASE_REF"

# Check if commit history changed
if [ "$BEFORE_REBASE" != "$(git rev-parse HEAD)" ]; then
  echo "Warning: Rebase changed commit history"
  REBASE_CHANGED=true
else
  REBASE_CHANGED=false
fi
```

On conflicts: resolve files, `git add <file>`, `git rebase --continue`. If stuck: `git rebase --abort`.

If `REBASE_CHANGED == true`, validate the commit message and content.

## 2. Ensure Single Valid Commit

After rebasing, squash multiple commits into one if needed.

```bash
COMMITS_AHEAD=$(git rev-list HEAD --not "$BASE_REF" --count 2>/dev/null || echo "0")
```

| `COMMITS_AHEAD` | Action |
| --------------- | ------ |
| `0` | **Error.** Nothing to push. |
| `1` | **OK.** Proceed to step 3. |
| `> 1` | **Must squash.** See below, then re-verify. |

### Squash procedure (when `COMMITS_AHEAD > 1`)

1. Collect other human authors before squashing (to preserve attribution):
   ```bash
   CURRENT_USER_EMAIL=$(git config user.email)
   OTHER_AUTHORS=$(git log "$BASE_REF"..HEAD --format='%aN <%aE>' \
     | grep -v -F "<$CURRENT_USER_EMAIL>" | sort -u)
   ```
2. Soft-reset to base:
   ```bash
   git reset --soft "$BASE_REF"
   ```
3. All changes are now staged. Delegate to `/git-commit` to create a single commit with a proper message based on the combined diff. If `OTHER_AUTHORS` is non-empty, append `Co-authored-by:` trailers for each human author.
4. **Re-verify** the count:
   ```bash
   COMMITS_AHEAD=$(git rev-list HEAD --not "$BASE_REF" --count)
   # Must be exactly 1. If not, something went wrong — do NOT push.
   ```

**Important:** When squashing, the commit message must be regenerated based on the final combined diff, not copied from any single commit.

## 3. Push

**First push (new branch)**:

```bash
git push --set-upstream "$PUSH_REMOTE" "$BRANCH_NAME"
```

**Update push (existing branch)**:

```bash
git push --force-with-lease "$PUSH_REMOTE" "$BRANCH_NAME"
```

Always use the determined `PUSH_REMOTE`. Never push to `upstream` or other unrelated remotes.
