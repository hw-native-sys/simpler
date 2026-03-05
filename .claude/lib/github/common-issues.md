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
