---
name: fix-issue
description: Fix a GitHub issue by fetching content, creating a branch, planning the fix, and implementing it. Use when the user asks to fix a specific issue number or work on a GitHub issue.
---

# Fix Issue Workflow

Fetch GitHub issue, create branch, plan, and implement the fix.

## Workflow

1. Check gh CLI authentication
2. Fetch issue content
3. Assign issue to me
4. Create issue branch
5. Enter plan mode to design fix
6. Implement the fix
7. Run tests (use `testing` skill)
8. Commit changes (use `git-commit` skill)

## Step 1: Check gh CLI Authentication

```bash
gh auth status
```

**If not authenticated**, prompt user:

```text
gh CLI is not authenticated. Please run: gh auth login
```

Stop here if not authenticated.

## Step 2: Fetch Issue Content

```bash
gh issue view ISSUE_NUMBER
gh issue view ISSUE_NUMBER --json number,title,body,state,labels
```

**Parse**: Issue number, title, description, state (open/closed), labels

**If issue is closed**: Ask user if they still want to work on it.

## Step 3: Assign Issue to Me

Before assigning, check if someone is already working on the issue:

```bash
gh issue view ISSUE_NUMBER --json assignees --jq '.assignees[].login'
```

**If assigned to current user**: Continue — already claimed.

**If assigned to someone else**: Ask the user whether to proceed or pick a different issue.

**If unassigned**: Assign to yourself (best-effort — skip gracefully if permissions are insufficient):

```bash
gh issue edit ISSUE_NUMBER --add-assignee @me
```

If the assignment fails due to permissions, continue with the workflow — do not block.

## Step 4: Create Issue Branch

**Branch naming**: `fix/issue-{number}-{short-description}`

Use a prefix that matches the issue type:

| Issue Type | Branch Prefix |
| ---------- | ------------- |
| Bug fix | `fix/` |
| New feature | `feat/` |
| Refactoring | `refactor/` |
| Documentation | `docs/` |
| Other | `support/` |

```bash
git checkout main && git pull origin main
BRANCH_NAME="fix/issue-${ISSUE_NUM}-short-description"
git checkout -b "$BRANCH_NAME"
```

**Important**: Always branch from `origin/main`. Never use other remotes.

## Step 5: Enter Plan Mode

Use `EnterPlanMode` to design the fix.

**Plan should cover**:

- Root cause analysis (for bugs)
- Files that need changes
- Implementation strategy
- Testing approach (simulation, hardware, or both)
- Which runtime(s) are affected

## Step 6: Implement the Fix

After plan approval, follow project conventions:

1. Make code changes following plan
2. Follow `.claude/rules/` conventions
3. Stay within your assigned directory (see `CLAUDE.md` Directory Ownership)
4. Add/update tests if applicable

## Step 7: Run Tests

```text
/testing
```

Fix any failures before committing.

## Step 8: Commit Changes

```text
/git-commit
```

**Commit message format**:

```text
Fix: brief description

Fixes #ISSUE_NUMBER

Detailed explanation of the fix.
```

## Common Issue Types

| Type | Approach |
| ---- | -------- |
| Bug fix | Reproduce, root cause, fix, add regression test |
| Feature request | Plan design, implement, add tests |
| Refactoring | Plan changes, ensure tests pass |
| Documentation | Fix/improve docs, verify examples work |

## Checklist

- [ ] gh CLI authenticated
- [ ] Issue content fetched and understood
- [ ] Issue assignment attempted (best-effort)
- [ ] Issue branch created from latest `origin/main`
- [ ] Plan created and approved
- [ ] Fix implemented following `.claude/rules/`
- [ ] Tests passing
- [ ] Changes committed with issue reference (`Fixes #N`)

## Remember

**Reference the issue number** in commit messages using `Fixes #ISSUE_NUMBER` for auto-linking.
