# Commit And Push

Prepare a PR branch for update.

## Rebase

```bash
BEFORE_REBASE=$(git rev-parse HEAD)
git rebase "$BASE_REF"
if [ "$BEFORE_REBASE" != "$(git rev-parse HEAD)" ]; then
  REBASE_CHANGED=true
else
  REBASE_CHANGED=false
fi
```

Resolve conflicts manually, then continue with `git rebase --continue`. Use `git rebase --abort`
only when the rebase cannot be completed.

## Ensure One Commit

Before creating or replacing a commit, ensure the checkout-local identity is
intentional:

```bash
git config --local --get user.name
git config --local --get user.email
```

```bash
COMMITS_AHEAD=$(git rev-list HEAD --not "$BASE_REF" --count 2>/dev/null || echo "0")
```

- `0`: stop, nothing to push.
- `1`: proceed.
- `>1`: squash to one commit before pushing.

For squash:

```bash
CURRENT_USER_EMAIL=$(git config user.email)
OTHER_AUTHORS=$(git log "$BASE_REF"..HEAD --format='%aN <%aE>' \
  | grep -v -F "<$CURRENT_USER_EMAIL>" | sort -u)
git reset --soft "$BASE_REF"
```

Then use the `git-commit` skill and preserve human coauthors with `Co-authored-by:` trailers.

## Push

For first push:

```bash
git push --set-upstream "$PUSH_REMOTE" "$BRANCH_NAME"
```

For existing PR branches:

```bash
git push --force-with-lease "$PUSH_REMOTE" "$BRANCH_NAME"
```

Never push feature work to an upstream remote unless it is explicitly the PR branch remote.
