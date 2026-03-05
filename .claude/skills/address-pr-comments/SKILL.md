---
name: address-pr-comments
description: Analyze and address GitHub PR review comments intelligently, distinguishing between actionable feedback and comments that can be resolved without changes. Use when addressing PR feedback or review comments.
---

# Address PR Comments Workflow

Intelligently triage PR review comments, address actionable feedback, and resolve informational comments with full permission awareness and worktree safety.

## Input

Accept multiple input formats:
- PR number: `123`, `#123`
- PR URL: `https://github.com/owner/repo/pull/123`
- Branch name: `feature-branch`
- No input: Interactive selection from open PRs

## Workflow Overview

0. Detect repository and permissions
1. Match input to PR (with worktree safety checks)
2. Fetch unresolved comments
3. Classify comments
4. Get user confirmation on ALL comments
5. Address user-selected comments with code changes
6. Push changes (with permission-aware fallback)
7. Reply and resolve threads
8. Output evidence summary

---

## Step 0: Detect Repository and Permissions

### 0.1 Parse Repository from Git Remote

**Never hardcode repository owner/name.** Always parse dynamically:

```bash
# Try origin first, fallback to upstream
REMOTE_URL=$(git remote get-url origin 2>/dev/null || git remote get-url upstream 2>/dev/null)

# Parse owner/repo from URL (supports HTTPS and SSH)
# HTTPS: https://github.com/owner/repo.git
# SSH: git@github.com:owner/repo.git
REPO_FULL=$(echo "$REMOTE_URL" | sed -E 's#.*github\.com[:/]([^/]+/[^/]+)(\.git)?$#\1#' | sed 's/\.git$//')
OWNER=$(echo "$REPO_FULL" | cut -d'/' -f1)
REPO=$(echo "$REPO_FULL" | cut -d'/' -f2)
```

Store as `$OWNER/$REPO` for all subsequent API calls.

### 0.2 Check User Permissions

```bash
gh api repos/$OWNER/$REPO --jq '.permissions'
```

**Permission levels:**
- `admin: true` or `maintain: true` or `push: true` → **write access** (can push)
- `pull: true` only → **read-only** (cannot push)

**Store permission level** for use in Step 6.

---

## Step 1: Match Input to PR (with Worktree Safety)

### 1.1 Detect Current Git State

```bash
CURRENT_BRANCH=$(git branch --show-current)
HEAD_COMMIT=$(git rev-parse HEAD)
```

**Worktree safety check:**
- If `$CURRENT_BRANCH` is empty (detached HEAD) → **STOP**: Prompt user to select/create worktree
- If `$CURRENT_BRANCH` is `main` or `master` → **STOP**: Prompt user to select/create worktree for PR branch
- Otherwise → Safe to proceed

**Worktree prompt template:**
```
⚠️  You are on <main/detached HEAD>. Working on PR comments requires a feature branch.

Options:
1. Create new worktree for PR branch
2. Switch to existing worktree
3. Cancel

Run: git worktree add <path> <branch>
```

### 1.2 Parse Input to PR Number

**Case 1: No input provided**

List open PRs and let user select:

```bash
gh pr list --repo $OWNER/$REPO --json number,title,headRefName,author --limit 20
```

Present numbered list:
```
Open PRs:
1. #123: Fix memory leak (feature/fix-leak) by @user1
2. #124: Add new API (feature/new-api) by @user2
...

Select PR number (1-20):
```

**Case 2: PR number** (`123` or `#123`)

```bash
PR_NUM=$(echo "$INPUT" | sed 's/^#//')
gh pr view $PR_NUM --repo $OWNER/$REPO --json number,title,headRefName,state,author
```

**Case 3: PR URL**

```bash
# Extract PR number from URL
PR_NUM=$(echo "$INPUT" | grep -oP 'pull/\K\d+')
gh pr view $PR_NUM --repo $OWNER/$REPO --json number,title,headRefName,state,author
```

**Case 4: Branch name**

```bash
gh pr list --repo $OWNER/$REPO --head "$INPUT" --json number,title,headRefName,state
```

If multiple PRs found, prompt user to select. If none found, error: "No PR found for branch `$INPUT`".

### 1.3 Validate PR State

- If `state != "OPEN"` → Warn user: "PR #$PR_NUM is $STATE. Continue? (y/n)"
- Store `headRefName` (target branch for push in Step 6)

---

## Step 2: Fetch Unresolved Comments

```bash
gh api graphql -f query='
query($owner: String!, $repo: String!, $number: Int!) {
  repository(owner: $owner, name: $repo) {
    pullRequest(number: $number) {
      reviewThreads(first: 50) {
        nodes {
          id
          isResolved
          comments(first: 1) {
            nodes {
              id
              databaseId
              body
              path
              line
              author { login }
            }
          }
        }
      }
    }
  }
}' -f owner="$OWNER" -f repo="$REPO" -F number=$PR_NUM
```

Filter to `isResolved: false` only.

**If no unresolved comments exist:** Inform the user that all PR comments have been resolved and exit the workflow. Do not proceed to subsequent steps.

---

## Step 3: Classify Comments

| Category | Description | Examples |
| -------- | ----------- | -------- |
| **A: Actionable** | Code changes required | Bugs, missing validation, race conditions, incorrect logic |
| **B: Discussable** | May skip if follows `.claude/rules/` | Style preferences, premature optimizations |
| **C: Informational** | Resolve without changes | Acknowledgments, "optional" suggestions |

Present summary showing category, file:line, and issue for each comment. For Category B, explain why code may already comply with `.claude/rules/`.

---

## Step 4: Get User Confirmation

**Always let the user decide which comments to address and which to skip.** Present ALL unresolved comments (A, B, and C) in a numbered list with their classification and brief summary.

Ask the user to specify which comments to address, skip, or discuss:

- Recommend addressing Category A items
- Mark Category B with rationale for skipping or addressing
- Mark Category C as skippable by default

**User choices per comment:** Address (make changes) / Skip (resolve as-is) / Discuss (need clarification)

Only proceed with the comments the user explicitly selects. Do NOT auto-resolve any comment without user consent.

---

## Step 5: Address Comments

For user-selected comments only:

1. Read files with Read tool
2. Make changes with Edit tool
3. Commit using `/git-commit` skill

**Track changed files** for evidence output (Step 8).

---

## Step 6: Push Changes (Permission-Aware Strategy)

### 6.1 Default Push Strategy (Write Access)

If user has **write access** (from Step 0.2):

```bash
# Push to original PR branch with force-with-lease
git push origin HEAD:$HEAD_REF_NAME --force-with-lease
```

**On success:** Record push status = "success", commit hash, proceed to Step 7.

**On failure** (network error, protected branch, etc.):
- Record push status = "failed: <error>"
- Fall through to Step 6.2 (Alternative PR mechanism)

### 6.2 Alternative PR Mechanism (Read-Only or Push Failure)

If user has **read-only access** OR push failed:

**Do NOT attempt to push.** Instead:

1. **Create alternative branch:**
   ```bash
   ALT_BRANCH="pr-$PR_NUM-comments-$(date +%s)"
   git checkout -b $ALT_BRANCH
   git push origin $ALT_BRANCH
   ```

2. **Create new PR:**
   ```bash
   gh pr create --repo $OWNER/$REPO \
     --base $HEAD_REF_NAME \
     --head $ALT_BRANCH \
     --title "Address PR #$PR_NUM review comments" \
     --body "Addresses review comments from #$PR_NUM

   Changes:
   - <list changed files>

   @<original-pr-author> Please review and merge into #$PR_NUM if acceptable."
   ```

3. **Require comment on original PR:**
   ```bash
   gh pr comment $PR_NUM --repo $OWNER/$REPO --body \
     "I've addressed the review comments in alternative PR #<new-pr-num> due to <permission/push-failure reason>.

   Please review: <new-pr-url>"
   ```

4. **Record evidence:**
   - Push status = "alternative-pr"
   - Alternative PR URL
   - Comment URL on original PR

**Do NOT resolve threads in Step 7** — let original PR author resolve after merging alternative PR.

---

## Step 7: Reply and Resolve Threads

**Only execute if push succeeded in Step 6.1** (direct push to original PR branch).

For each addressed comment:

```bash
gh api repos/$OWNER/$REPO/pulls/$PR_NUM/comments/<comment_id>/replies \
  -f body="<response>"
```

Then resolve thread:

```bash
gh api graphql -f query='
mutation($threadId: ID!) {
  resolveReviewThread(input: {threadId: $threadId}) {
    thread { id }
  }
}' -f threadId="<thread_id>"
```

**Response templates:**

- Fixed: "Fixed in `<commit>` — <description>"
- Skip: "Current code follows `.claude/rules/<file>` — <rationale>"
- Acknowledged: "Acknowledged, thank you!"

---

## Step 8: Output Evidence Summary

**Always output** at end of workflow:

```
✅ PR Comments Addressed

Repository: <owner>/<repo>
PR: #<number> (<pr-url>)
Branch: <headRefName>

Commits:
- <commit-hash-1>: <message>
- <commit-hash-2>: <message>

Changed Files:
- <file1>
- <file2>

Push Status: <success | failed: <reason> | alternative-pr>
[If alternative-pr:]
  Alternative PR: #<num> (<url>)
  Original PR Comment: <comment-url>

Resolved Threads: <count>
Skipped Threads: <count>

Responses:
- Thread <id>: <reply-text> [<reply-url>]
- Thread <id>: <reply-text> [<reply-url>]
```

---

## Best Practices

| Area | Guidelines |
| ---- | ---------- |
| **Repository** | Always parse from git remote; never hardcode |
| **Permissions** | Check before push; gracefully fallback to alternative PR |
| **Worktree** | Enforce branch safety; never work on main/detached HEAD |
| **Analysis** | Reference `.claude/rules/`; when unsure → Category B |
| **Changes** | Read full context; minimal edits; follow project conventions |
| **Communication** | Be respectful; explain reasoning; reference rules |
| **Evidence** | Always output complete summary with URLs and hashes |

---

## Error Handling

| Error | Action |
| ----- | ------ |
| No git remote | "Run: `git remote add origin <url>`" |
| PR not found | `gh pr list`; ask user to confirm |
| Not authenticated | "Run: `gh auth login`" |
| No unresolved comments | Inform user all comments resolved; exit workflow |
| Unclear comment | Mark Category B for discussion |
| Detached HEAD / main branch | Prompt worktree selection/creation; do not proceed |
| Push failed | Fallback to alternative PR mechanism |
| Read-only access | Use alternative PR mechanism; do not attempt push |

---

## Checklist

- [ ] Repository parsed from git remote (Step 0.1)
- [ ] User permissions checked (Step 0.2)
- [ ] Worktree safety validated (Step 1.1)
- [ ] PR matched and validated (Step 1.2-1.3)
- [ ] Unresolved comments fetched and classified (Step 2-3)
- [ ] ALL comments presented to user for selection (Step 4)
- [ ] Code changes made and committed (Step 5, use `/git-commit`)
- [ ] Changes pushed with appropriate strategy (Step 6)
- [ ] Comments replied to and resolved (Step 7, if applicable)
- [ ] Evidence summary output (Step 8)

---

## Remember

- **This skill is the single source of truth** for PR comment workflows. Do not duplicate logic in work/tasks.
- **Not all comments require code changes.** Evaluate against `.claude/rules/` first.
- **Respect permissions.** Read-only users get alternative PR mechanism, not push failures.
- **Worktree safety first.** Never work on main/detached HEAD.
- **Always output evidence.** Users need URLs, hashes, and status for verification.
