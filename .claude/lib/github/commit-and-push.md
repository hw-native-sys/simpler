# Commit and Push

Prepares changes for PR: rebases, squashes commits, and pushes.

## 1. Rebase

First, rebase onto the latest base branch to incorporate upstream changes.
**This must happen before the squash in §2, not after** — see the warning there.

```bash
# $BASE_REF is a remote-tracking ref and goes stale during a long session
git fetch upstream

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

On conflicts, follow the
[merge-conflict procedure](../../skills/fix-pr/references/merge-conflicts.md):
resolve files, `git add <file>`, `git rebase --continue`. Abort only when the
rebase target is wrong or the branch is corrupted.

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

1. Capture the original PR commit's message **before** the reset destroys it —
   `/git-commit` regenerates from the diff and has no other way to see it, so
   without this the "evolve the original message" rule below cannot hold:

   ```bash
   # Oldest commit ahead of base = the original PR commit
   ORIG_COMMIT=$(git rev-list --reverse "$BASE_REF"..HEAD | head -1)
   git log -1 --format='%B' "$ORIG_COMMIT" > /tmp/orig_pr_msg.txt
   ```

2. Collect other human authors before squashing (to preserve attribution):

   ```bash
   CURRENT_USER_EMAIL=$(git config user.email)
   OTHER_AUTHORS=$(git log "$BASE_REF"..HEAD --format='%aN <%aE>' \
     | grep -v -F "<$CURRENT_USER_EMAIL>" | sort -u)
   ```

3. Soft-reset to base:

   ```bash
   git reset --soft "$BASE_REF"
   ```

   **Only safe because §1 already rebased onto `$BASE_REF`.** The reset moves
   HEAD to the base but keeps *your* index, so if the branch still sits on an
   older base, every file the base gained meanwhile is recorded as **your
   deletion** — and a later rebase will not undo it, because the revert is by
   then part of your own diff. Step 5 verifies this; do not check it here, where
   `HEAD` is `$BASE_REF` and a `$BASE_REF`-to-`HEAD` diff is empty by
   construction.

4. All changes are now staged. Delegate to `/git-commit`, passing the captured
   `/tmp/orig_pr_msg.txt` as the starting point so the single squashed commit
   **evolves** the original subject/body to cover the combined diff (see the
   message rule below) rather than inventing a generic one. If `OTHER_AUTHORS`
   is non-empty, append `Co-authored-by:` trailers for each human author.
5. **Re-verify** the count, and that the squash reverted nothing:

   ```bash
   COMMITS_AHEAD=$(git rev-list HEAD --not "$BASE_REF" --count)
   # Must be exactly 1. If not, something went wrong — do NOT push.

   # Three dots, and only now that a commit exists — see step 3.
   git diff "$BASE_REF"...HEAD --stat
   ```

   Every file listed must be one you meant to touch. Anything else is the
   stale-base deletion described in step 3: restore it with
   `git checkout "$BASE_REF" -- <path>` and amend.

**Important:** When squashing, the commit message must describe the final combined diff. Start from the original PR commit's message (keep its type/scope/subject and intent) and evolve it to absorb the folded-in fixes — do not replace it with a generic `fix(pr): …` message, and do not leave it frozen when the fix changed the commit's behavior or scope.

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
