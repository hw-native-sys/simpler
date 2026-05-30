---
name: fix-pr
description: Fix GitHub PR problems by triaging review comments and any automated check failures, applying approved fixes, pushing updates, and rechecking until clean.
---

# Fix PR

Address review comments and any automated check failures for an open PR.

## Inputs

Accept a PR number, branch name, or no input. With no input, auto-detect the PR from the current
branch or upstream tracking.

## Workflow

1. Read [setup](../../lib/github/setup.md).
2. Use [lookup-pr](../../lib/github/lookup-pr.md) to identify the PR.
3. Fetch unresolved review threads with [fetch-comments](../../lib/github/fetch-comments.md).
4. Fetch checks:

```bash
gh pr checks "$PR_NUMBER" --repo "$PR_REPO_OWNER/$PR_REPO_NAME"
```

5. Classify findings:
   - `A`: actionable code/doc/test changes.
   - `B`: discussable suggestions or style preferences.
   - `C`: informational acknowledgments.
   - `CHECK`: failed or pending automated checks requiring investigation.
6. Present all findings and ask what to address. Recommend fixing `A` and `CHECK` items.
7. Use [detect-permission](../../lib/github/detect-permission.md). For cross-fork PRs, use
   [checkout-fork-branch](../../lib/github/checkout-fork-branch.md).
8. For automated check failures, inspect logs before coding:

```bash
gh pr checks "$PR_NUMBER" --json name,state,link
gh run view "$RUN_ID" --log-failed
```

9. Implement approved fixes, run verification, commit, and use
   [commit-and-push](../../lib/github/commit-and-push.md).
10. For addressed review comments, use [reply-and-resolve](../../lib/github/reply-and-resolve.md).
11. Re-check until all selected issues are resolved and checks are green.

## Constraints

- Do not auto-resolve comments without user approval.
- Pending checks are not clean.
- Do not bypass hooks or use destructive git commands.
- For bot comments, classify by content, not author.
