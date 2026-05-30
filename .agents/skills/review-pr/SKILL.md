---
name: review-pr
description: Review a GitHub PR by computing the correct merge-base diff, reading relevant context, and reporting prioritized findings.
---

# Review PR

Use a code-review mindset: findings first, ordered by severity, with file/line references.

## Workflow

1. Read [setup](../../lib/github/setup.md).
2. If reviewing a PR number, fetch metadata:

```bash
gh pr view "$PR_NUMBER" --repo "$PR_REPO_OWNER/$PR_REPO_NAME" \
  --json number,title,body,headRefName,baseRefName,commits
```

3. Fetch the base branch and compute the merge base. Never diff against stale local `main`.

```bash
git fetch "$BASE_REMOTE" "$DEFAULT_BRANCH" --quiet
MERGE_BASE=$(git merge-base "$BASE_REF" HEAD)
```

For a PR with a different base branch, fetch that base and compute against it.

4. Gather:

```bash
git diff "$MERGE_BASE"...HEAD --stat
git diff "$MERGE_BASE"...HEAD --name-only
git log --oneline "$MERGE_BASE"..HEAD
```

5. Read diffs by logical area. For large PRs, inspect per-directory or per-file chunks.
6. Read surrounding source when the diff is insufficient.
7. Check tests, docs, API boundaries, failure paths, and compatibility.

## Response Format

Start with findings. If no findings, state that explicitly and mention residual risk.

Use:

- `Must fix`: correctness, data loss, security, broken tests, or behavioral regressions.
- `Should fix`: missing validation, weak error handling, missing tests, docs drift.
- `Consider`: optional improvements.

Then include brief open questions and a short summary only if useful.

## Pitfalls

- Do not review against stale local `main`.
- Do not summarize before findings.
- Do not approve based only on test success.
- Do not ignore generated files if they are checked in and consumed.
