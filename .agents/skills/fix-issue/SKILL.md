---
name: fix-issue
description: Fix a GitHub issue by fetching issue context, creating a branch, implementing the fix, verifying it, committing, and opening a PR.
---

# Fix Issue

Use this when the user asks to fix a specific GitHub issue.

## Workflow

1. Read [setup](../../lib/github/setup.md).
2. Fetch the issue:

```bash
gh issue view "$ISSUE_NUMBER" --repo "$PR_REPO_OWNER/$PR_REPO_NAME"
gh issue view "$ISSUE_NUMBER" --repo "$PR_REPO_OWNER/$PR_REPO_NAME" \
  --json number,title,body,state,labels,assignees
```

3. If closed, ask whether to continue.
4. If assigned to someone else, ask before proceeding.
5. If unassigned, best-effort assign yourself:
   `gh issue edit "$ISSUE_NUMBER" --repo "$PR_REPO_OWNER/$PR_REPO_NAME" --add-assignee @me`.
6. Create a branch from `BASE_REF` with a prefix matching the issue type:
   `fix/`, `feat/`, `refactor/`, `docs/`, or `support/`.
7. Investigate root cause or required implementation.
8. Implement the change following `.agents/` rules.
9. Run the relevant tests and broader verification.
10. Use the `git-commit` skill. Include `Fixes #<issue>` when the PR should close the issue.
11. Use the `github-pr` skill to open the PR.

## Constraints

- Do not work directly on `main`.
- Keep changes focused on the issue.
- Update docs and tests when the public contract changes.
