# Common GitHub Workflow Issues

| Issue | Response |
| --- | --- |
| `gh auth` fails | Ask the user to run `gh auth login`. |
| stale local main | Fetch the base remote and use `BASE_REF`; never trust local `main`. |
| rebase conflict | Resolve files, stage them, run `git rebase --continue`. |
| rebase stuck | Use `git rebase --abort` only after explaining the blocker. |
| push rejected | Re-fetch/rebase and push with `--force-with-lease` when updating a PR branch. |
| PR not found | Verify repo, PR number, and head branch lookup. |
| PR is merged | Stop; do not modify merged PRs. |
| no push access | Ask the PR author to enable maintainer edits or submit a follow-up PR. |

## Shell Pitfalls

- Quote `gh api --jq` expressions with single quotes.
- Quote `gh api -f body='...'` with single quotes.
- Avoid piping `gh api` output into `python -c 'json.load(sys.stdin)'`; use `--jq` or `jq`.
- For GraphQL, inline values when shell quoting gets fragile.
