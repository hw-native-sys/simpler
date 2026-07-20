---
name: fix-pr
description: Fix GitHub PR issues — address review comments and resolve CI failures in a loop until the PR is fully clean. Fetches CI errors online and triages review feedback. Use when fixing PR problems, addressing review comments, or resolving CI failures.
---

# Fix PR Workflow

Fix PR issues (review comments, CI failures) in a loop until the PR is fully clean.

## Task Tracking

Create tasks to track progress through this workflow:

1. Match input to PR
2. Detect & classify issues
3. Get user confirmation
4. Fix issues & push (fold into one commit — amend/squash, never append)
5. Resolve comment threads
6. Re-check (loop until clean)

## Input

Accept PR number (`123`, `#123`), branch name, or no argument (uses current branch).

## Setup

1. [Setup](../../lib/github/setup.md) — authenticate and detect context (role, remotes, state)
2. **Auto-detect cross-fork PR context**: If no PR number provided, check upstream tracking to detect cross-fork push target:

   ```bash
   UPSTREAM=$(git rev-parse --abbrev-ref "@{upstream}" 2>/dev/null || echo "")
   if [ -n "$UPSTREAM" ]; then
     UPSTREAM_REMOTE=$(echo "$UPSTREAM" | cut -d'/' -f1)
     if [ "$UPSTREAM_REMOTE" != "origin" ] && [ "$UPSTREAM_REMOTE" != "upstream" ]; then
       PUSH_REMOTE="$UPSTREAM_REMOTE"
       HEAD_BRANCH=$(echo "$UPSTREAM" | cut -d'/' -f2-)
     fi
   fi
   ```

## Loop: Steps 1→8, repeat until clean or max 5 iterations

### Step 1: Match Input to PR

Use [lookup-pr](../../lib/github/lookup-pr.md) to find the PR.

- If PR number or branch name provided: use "By PR number" or "By branch name" lookup
- If no input and cross-fork detected (from Setup): search with `--head "$UPSTREAM_REMOTE:$HEAD_BRANCH"`
- If no input and no cross-fork: auto-detect from current branch, or list open PRs for user selection

Validate PR state: OPEN (continue), CLOSED (warn), MERGED (exit).

### Step 2: Detect Issues (run in parallel)

**A) Review comments:**

Run [fetch-comments](../../lib/github/fetch-comments.md).

```bash
OWNER=$(gh repo view --json owner -q '.owner.login')
NAME=$(gh repo view --json name -q '.name')

# Fetch review threads — save to file, then grep (see pitfalls below)
gh api graphql \
  -F owner="$OWNER" -F name="$NAME" -F number=<NUMBER> \
  -f query='
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      reviewThreads(first: 100) {
        nodes {
          id isResolved
          comments(last: 1) {
            nodes { id databaseId body author { login } path line }
          }
        }
        pageInfo { hasNextPage endCursor }
      }
    }
  }
}' > /tmp/threads.json

# Count unresolved threads (use whitespace-tolerant pattern)
grep -Ec '"isResolved":[[:space:]]*false' /tmp/threads.json

# Paginate: if hasNextPage is true, re-run with -F cursor="<endCursor>" until done
```

**B) CI status:**

```bash
gh pr checks <NUMBER>
```

**Shell pitfalls to avoid:**

- Do NOT pipe `gh api graphql` to `python3 -c` with `json.load(sys.stdin)` — `gh` may emit extra metadata that breaks JSON parsing with `JSONDecodeError: Extra data`
- Do NOT use `gh api graphql --jq` with `$` in filter expressions — `gh`'s jq processor interprets `$` as a jq variable sign, causing `Expected VAR_SIGN` errors even when shell quoting is correct
- Also see [common-issues](../../lib/github/common-issues.md) for `gh api` quoting pitfalls (use single quotes for `--jq` and `-f body=`)
- Use `grep -c` for simple counts; save to a temp file first if complex parsing is needed

Present: "**Iteration N** — Found X unresolved comments and Y failed/pending checks."

**Exit condition:** All checks green AND no unresolved comments → done. Pending checks do NOT count as clean.

### Step 3: Detect Permission

Run [detect-permission](../../lib/github/detect-permission.md) to determine push access.

### Step 4: Classify Issues

**Review comments** — filter `isResolved: false`, classify:

| Category | Description | Examples |
| -------- | ----------- | -------- |
| **A: Actionable** | Code changes required | Bugs, missing validation, race conditions, incorrect logic |
| **B: Discussable** | May skip if follows `.claude/rules/` | Style preferences, premature optimizations |
| **C: Informational** | Resolve without changes | Acknowledgments, "optional" suggestions |

Treat bot reviewers (CodeRabbit, Copilot, Gemini) same as human — classify by content.

For Category B, explain why code may already comply with `.claude/rules/`.

**CI failures:**

```bash
# List failed checks to get the link for each failed job
gh pr checks <NUMBER> --json name,state,link

# Extract IDs from a failed check's link
# Link format: https://github.com/<owner>/<repo>/actions/runs/<RUN_ID>/job/<JOB_ID>
RUN_ID=$(echo "$LINK" | sed -En 's|.*/runs/([0-9]+)/.*|\1|p')
JOB_ID=$(echo "$LINK" | sed -En 's|.*/job/([0-9]+).*|\1|p')

# Whole-run logs (requires run to be complete — see note below)
gh run view "$RUN_ID" --log-failed

# Single-job logs — works even while the run is still in progress
gh run view --job "$JOB_ID" --log-failed
```

**`gh run view <RUN_ID> --log-failed` requires the entire run to be complete** (all jobs, not just the failed one). If any job is still pending, `gh` returns "run is still in progress". Check first: `gh run view <RUN_ID> --json status --jq '.status'` — must return `"completed"`.

**When the run is still partially running but some jobs have already failed**, prefer `gh run view --job <JOB_ID> --log-failed` — it pulls the single job's log without waiting for siblings. List job IDs with `gh run view <RUN_ID> --json jobs --jq '.jobs[] | {name, status, conclusion, databaseId}'`.

For large logs: `gh run view --job <JOB_ID> --log-failed 2>&1 | grep -E "error:|FAILED|fatal" | head -20`

**External checks** (non-GitHub Actions): no run ID exists — open the `link` URL directly to view logs from the external provider.

### Step 5: Get User Confirmation

Present ALL issues in a numbered list:

```text
Review Comments:
  1. [A] src/foo.cpp:42 — Missing null check (reviewer: alice)
  2. [B] src/bar.py:15 — Style suggestion (reviewer: coderabbitai)
CI Failures:
  3. [CI] build — error: 'Foo' is not a member of 'pto2'
```

Ask which to address/skip:

- Recommend addressing Category A + CI items
- Mark Category B with rationale for skipping or addressing
- Mark Category C as skippable by default

**User choices per comment:** Address (make changes) / Skip (resolve as-is) / Discuss (need clarification)

Only proceed with the comments the user explicitly selects. Do NOT auto-resolve any comment without user consent.

On subsequent iterations, reuse prior "address all" policy for same categories. When unsure about a comment's category, default to B.

### Step 6: Work Location Setup & Fix Issues

Work directly on the PR branch. Setup depends on permission level:

**For owner/write permission:**

```bash
git checkout $HEAD_BRANCH
git pull "$PUSH_REMOTE" "$HEAD_BRANCH"
```

**For maintainer permission (cross-fork PR):**

Run [checkout-fork-branch](../../lib/github/checkout-fork-branch.md) to create/switch to the local working branch and set the push refspec.

**Fix:**

1. Read affected files, make changes with Edit tool
2. For CI: analyze logs online first, reproduce locally only as last resort

**Fold the fix into the PR — never append a standalone "fix(pr)" commit.**
The PR must stay as **one commit** (its original commit + your fixes squashed
in), so reviewers see a single clean diff, not a running log of review
churn. This is the default on **every** iteration.

Stage all changes, then squash based on how many commits the PR has ahead of
`$BASE_REF`:

```bash
git add -A
COMMITS_AHEAD=$(git rev-list HEAD --not "$BASE_REF" --count)
```

| `COMMITS_AHEAD` | How to fold the fix in |
| --------------- | ---------------------- |
| `0` | **Error — stop.** Nothing ahead of base to fold into; do not rebase or push. Matches [commit-and-push](../../lib/github/commit-and-push.md) §2. |
| `1` | **Amend with an updated message.** Pass the message non-interactively — `git commit --amend -F <msgfile>` (or `-m`), never bare `--amend` (it opens an editor and hangs in a non-interactive/agent shell) and never `--no-edit` (leaves the message stale). Write the evolved message per the rule below into `<msgfile>` first. |
| `> 1` | **Squash to one** via the [commit-and-push](../../lib/github/commit-and-push.md) squash procedure (capture the original message, soft-reset to `$BASE_REF`, recommit once) |

Do NOT run `/git-commit` to create a *new* commit here — that is what causes
the appended-commit problem. `/git-commit` is only for regenerating the
squashed message when `COMMITS_AHEAD > 1`.

Then push and verify exactly one commit landed:

1. Rebase onto `$BASE_REF` (see [commit-and-push](../../lib/github/commit-and-push.md) §1)
2. **Verify single commit:** `git rev-list HEAD --not "$BASE_REF" --count` must print `1`. If not, squash again before pushing — never push a multi-commit PR.
3. Push (update push with `--force-with-lease` to `$PUSH_REMOTE`)

**Commit message — evolve it, don't replace or freeze it.** The message must
describe the commit's *final combined diff*, so it has to change when the fix
changes the code's behavior or scope. Two failure modes to avoid equally:

- ❌ **Frozen** (`--no-edit`): the message now under-describes what the commit
  does — a reviewer reads it and the diff disagrees.
- ❌ **Replaced** (`fix(pr): resolve review comments`): the original intent and
  its scope/type/subject are lost, and the message becomes a changelog of
  review churn rather than a description of the code.

Instead: **keep the original subject and body, then edit them to absorb the
fix.** Match the type/scope/style of the original message.

- Small fix that doesn't change what the commit does (typo, lint, comment) →
  message may stay as-is; amend without message edits only in this case.
- Fix that adds/changes behavior → work it into the existing body (extend or
  correct the relevant bullet/sentence); adjust the subject only if the
  headline scope actually changed.

Example — original `feat(runtime): add predicated dispatch`; review found a
missing null check on the predicate. Update the body to note the guard, keep
the `feat(runtime): add predicated dispatch` subject — not `fix(pr): address
comments`, not the untouched original.

### Step 7: Reply and Resolve Comment Threads

For each comment, **both steps are mandatory** (see [reply-and-resolve](../../lib/github/reply-and-resolve.md)):

1. **Reply** using the comment's `databaseId`:

   ```bash
   gh api "repos/${PR_REPO_OWNER}/${PR_REPO_NAME}/pulls/${PR_NUMBER}/comments/${COMMENT_DATABASE_ID}/replies" \
     -f body='Fixed — description of change'
   ```

2. **Resolve the thread** using the thread's GraphQL node `id` (from fetch-comments, NOT the databaseId):

   ```bash
   gh api graphql -f query='
   mutation { resolveReviewThread(input: {threadId: "THREAD_NODE_ID"}) {
     thread { isResolved }
   }}'
   ```

**Important:** The thread `id` comes from `reviewThreads.nodes[].id` in the fetch-comments GraphQL response. Each thread contains comments — use the thread's `id` to resolve, and the comment's `databaseId` to reply.

Reply templates:

- **Fixed** → "Fixed — description of change" (do **not** cite a commit SHA: the PR commit is amended/squashed and force-pushed each iteration, so any SHA you quote is immediately stale)
- **Skip** → "Follows `.claude/rules/<file>` — explanation"
- **Ack** → "Acknowledged!"

### Step 8: Wait and Re-check

```bash
# Verify run is complete before fetching whole-run logs
gh run view <RUN_ID> --json status --jq '.status'  # must be "completed"
```

Poll with `gh pr checks <NUMBER>` — proceed early if all checks finish. **For whole-run logs, wait until status is "completed".** If a job has already failed and you only need that job's output, fetch it now via `gh run view --job <JOB_ID> --log-failed` (see Step 4) rather than blocking on the rest of the run.

Then loop back to Step 2.

**Loop safeguards:** Max 5 iterations. Flag stuck issues (same failure reappears) to user instead of retrying.

## Reference Tables

| Area | Guidelines |
| ---- | ---------- |
| CI errors | Fetch logs online first; reproduce locally as last resort |
| Bot reviews | Classify by content, not author |
| Changes | Read full context; minimal edits; follow project conventions |

| Error | Action |
| ----- | ------ |
| PR not found | `gh pr list`; ask user |
| CI logs unavailable / run in progress | Wait for run completion; if still unavailable, fall back to local reproduction |
| CI logs too large | `grep -E "error:\|FAILED\|fatal"` |
| Max iterations reached | Stop, report remaining issues |
| Same failure persists | Flag to user, do not retry |

## Checklist

- [ ] PR matched and validated
- [ ] Review comments and CI status fetched
- [ ] ALL issues presented to user for selection
- [ ] Fixes folded into the PR (amended/squashed — **no appended `fix(pr)` commit**)
- [ ] Verified `git rev-list HEAD --not "$BASE_REF" --count` == 1 before pushing
- [ ] Changes force-pushed with `--force-with-lease` (single commit)
- [ ] Review comment threads replied to and resolved
- [ ] Waited for CI/reviews and re-checked
- [ ] Loop exited: all clean OR max iterations reached

## Remember

**Not all comments require code changes.** Evaluate against `.claude/rules/` first. When in doubt, consult user.
