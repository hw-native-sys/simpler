# Common Issues

Troubleshooting reference for GitHub workflows.

| Issue | Solution |
| ----- | -------- |
| `gh auth` fails | Tell user to run `gh auth login` |
| Merge conflicts during rebase | Resolve files, `git add <file>`, `git rebase --continue` |
| Rebase stuck | `git rebase --abort`, investigate manually |
| Push rejected (non-fast-forward) | Use `git push --force-with-lease` after confirming rebase |
| More than 1 commit ahead | Run [commit-and-push](commit-and-push.md) |
| No origin remote | Repository not properly initialized |
| PR not found | Verify PR number; use [lookup-pr](lookup-pr.md) |
| PR is merged | Exit — cannot modify merged PR |
| No unresolved comments | Inform user all comments resolved; exit |
| No push access | Ask PR author to enable "Allow edits from maintainers" |

## Shell Escaping Pitfalls

### `gh api --jq` with `!=`

Bash history expansion treats `!` as special. Use single quotes for jq expressions:

```bash
# BAD — bash escapes != to \!=
gh api ... --jq ".[] | select(.position != null)"

# GOOD — single quotes prevent expansion
gh api ... --jq '.[] | select(.position != null)'
```

### `gh api -f body=` with special characters

Always use single quotes for body text to avoid issues with backticks, `$`, `!`, etc.:

```bash
# BAD — backticks and $ get interpreted
gh api ... -f body="Fixed — changed to \`grep -F \"<$EMAIL>\"\`"

# GOOD — single quotes, no escaping needed
gh api ... -f body='Fixed — changed to grep -F for precise matching.'
```

If the body must contain single quotes, use a heredoc or keep the message simple.

### `gh api graphql` with variables

Do NOT use `-f`/`-F` flags to pass GraphQL `$variables`. Bash mangles `$` signs even inside single-quoted query strings when combined with `-f` flags. Instead, inline values directly:

```bash
# BAD — $owner/$repo/$number clash with bash
gh api graphql -f owner="hw-native-sys" -f repo="simpler" -F number=276 \
  -f query='query($owner: String!, $repo: String!, $number: Int!) { ... }'

# GOOD — inline values, no variables
gh api graphql -f query='query {
  repository(owner: "'"$PR_REPO_OWNER"'", name: "'"$PR_REPO_NAME"'") {
    pullRequest(number: '"$PR_NUMBER"') { ... }
  }
}'
```

### `gh api` output piped to python

`gh api` may output extra lines beyond the JSON payload. Do NOT pipe to `python3 -c "json.load(sys.stdin)"`. Use `--jq` or `jq` instead:

```bash
# BAD — python chokes on extra lines
gh api ... 2>&1 | python3 -c "import json, sys; json.load(sys.stdin)"

# GOOD — use --jq inline or pipe to jq
gh api ... --jq '.data.repository'
gh api ... | jq '.data.repository'
```
