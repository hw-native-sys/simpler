---
name: fix-pr
description: Fix GitHub PR issues — address all PR feedback (inline review threads, review summary bodies, and conversation comments) and resolve CI failures in a loop until the PR is fully clean. Fetches CI errors online and triages review feedback. Use when fixing PR problems, addressing review comments, or resolving CI failures.
---

# Fix PR Workflow

Fix PR issues (review comments, CI failures) in a loop until the PR is fully clean.

## Task Tracking

Create tasks to track progress through this workflow:

1. Match input to PR
2. Detect & classify issues
3. Reconcile scope and existing authorization
4. Fix issues & push (fold into one commit — amend/squash, never append)
5. Reply to actionable feedback; resolve only settled threads
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

**A) All PR feedback — three surfaces, not just inline threads:**

Run [fetch-comments](../../lib/github/fetch-comments.md) and fetch **all three**
surfaces in one query:

| Surface | What it is | Why it is easy to miss |
| ------- | ---------- | ---------------------- |
| **A** inline review threads | `reviewThreads.nodes` — anchored to file/line | — (the only one with `isResolved`) |
| **B** review summary bodies | `reviews.nodes[].body` — text submitted with CHANGES_REQUESTED / COMMENTED / APPROVED | Not a thread; never appears in a `reviewThreads` query |
| **C** PR conversation comments | `comments.nodes` — issue comments, `@`-mentions, most bot posts | Not a thread; holds "please rebase", scope objections, maintainer instructions |

**Fetching only surface A is the classic failure of this workflow** — a
maintainer's "this approach is wrong, see my comment above" lives on B or C and
gets silently skipped while the PR is declared clean.

```bash
OWNER=$(gh repo view --json owner -q '.owner.login')
NAME=$(gh repo view --json name -q '.name')
P='.data.repository.pullRequest'

# Save the combined query from fetch-comments.md to a file, then jq the file
# (see pitfalls below)
gh api graphql -f query='...' > /tmp/pr-feedback.json

# Count each surface
jq "[${P}.reviewThreads.nodes[] | select(.isResolved == false)] | length" /tmp/pr-feedback.json
jq "[${P}.reviews.nodes[]      | select(.body != \"\")]          | length" /tmp/pr-feedback.json
jq "${P}.comments.nodes | length" /tmp/pr-feedback.json

# Paginate: if any connection's hasNextPage is true, re-query with after: "<endCursor>"
```

Surfaces B and C carry no `isResolved` bit — filter them against the
handled-ID ledger (`/tmp/pr-<NUMBER>-handled.txt`, see fetch-comments) so
already-answered feedback is not re-presented every iteration.

**B) CI status:**

```bash
gh pr checks <NUMBER>
```

**Shell pitfalls to avoid:**

- Do NOT pipe `gh api graphql` to `python3 -c` with `json.load(sys.stdin)` — `gh` may emit extra metadata that breaks JSON parsing with `JSONDecodeError: Extra data`
- Do NOT use `gh api graphql --jq` with `$` in filter expressions — `gh`'s jq processor interprets `$` as a jq variable sign, causing `Expected VAR_SIGN` errors even when shell quoting is correct
- Also see [common-issues](../../lib/github/common-issues.md) for `gh api` quoting pitfalls (use single quotes for `--jq` and `-f body=`)
- Use `grep -c` for simple counts; save to a temp file first if complex parsing is needed

Present: "**Iteration N** — Found X unresolved review threads, Y unhandled review bodies / conversation comments, and Z failed/pending checks."

**Exit condition:** Required checks pass under repository policy at the current
head, non-required failures are triaged, and no blocking feedback remains across
all three surfaces. Pending or unexplained skipped required checks are not success.
Informational comments need no reply. A handled-ID entry only records processing;
it does not prove a disputed finding accepted or a changes-requested review cleared.
Re-read edited comments and new replies even if their earlier IDs were handled.
A new head requires reconciling review and CI again. Author-side repairs complete
does not mean maintainer approval or merge; hand that state to
[process-pr](../process-pr/SKILL.md).

### Step 3: Detect Permission

Run [detect-permission](../../lib/github/detect-permission.md) to determine push access.

### Step 4: Classify Issues

**Feedback** — take unresolved surface-A threads plus unhandled surface-B/C
items, and classify **every** item by content, regardless of which surface it
arrived on:

| Category | Description | Examples |
| -------- | ----------- | -------- |
| **A: Actionable** | Code changes required | Bugs, missing validation, race conditions, incorrect logic, "rebase onto main", "split this PR" |
| **B: Discussable** | May skip if follows `.claude/rules/` | Style preferences, premature optimizations |
| **C: Informational** | Answer without changes | Acknowledgments, "optional" suggestions, bot walkthroughs |

Treat bot reviewers (CodeRabbit, Copilot, Gemini) same as human — classify by content.

For Category B, explain why code may already comply with `.claude/rules/`.

Surface-specific handling before classifying:

- **B (review bodies):** a `CHANGES_REQUESTED` body is actionable by default — it blocks merge even when every inline thread is resolved. A body that only enumerates its own inline threads is informational; address the threads instead.
- **C (conversation comments):** drop bot status/walkthrough noise per the filter list in [fetch-comments](../../lib/github/fetch-comments.md); classify everything else. Short human one-liners here are frequently the most consequential feedback on the PR.
- **Dedup across surfaces** — the same point raised inline and repeated in a summary is one item, addressed once.

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

### Step 5: Reconcile Scope and Authorization

Summarize findings by feedback surface and CI failure, with the planned disposition:
fix, explain, discuss or informational. Reuse the user's existing authorization.
An instruction to fix the PR covers routine repairs within its agreed scope; do not
require the user to select every null check, build repair or already-approved rebase.

Ask only about unresolved material design/scope trade-offs, disputed requirements
or actions outside the authorized scope. Continue independent authorized fixes.
Permission to write the fork is not permission to take over another author's work:
confirm the writing scope and inspect active work before an authorized maintainer
repair. Review-only requests do not enter this mutation workflow.

Keep permission to publish replies and resolve threads explicit in the task scope;
reuse it when already granted. If missing, prepare the fix and proposed replies
before asking about publication. Do not equate disagreement or a decision to skip
an item with reviewer acceptance. Informational feedback needs no repetitive reply.

### Step 6: Work Location Setup & Fix Issues

Work directly on the PR branch. Setup depends on permission level:

**For owner/write permission:**

```bash
git checkout $HEAD_BRANCH
git pull "$PUSH_REMOTE" "$HEAD_BRANCH"
```

**For maintainer permission (cross-fork PR):**

Run [checkout-fork-branch](../../lib/github/checkout-fork-branch.md) to create/switch to the local working branch and set the push refspec.

**Land on the current base before editing anything.** `$BASE_REF` moves while a
PR is open and everything below measures against it. Doing it here also keeps the
rebase off a dirty worktree, which git refuses to rebase:

```bash
git fetch upstream                        # $BASE_REF is a remote-tracking ref — refresh it
git rebase "$BASE_REF"                    # see [commit-and-push](../../lib/github/commit-and-push.md) §1
```

**Fix:**

1. Read affected files, make changes with Edit tool
2. For CI: analyze logs online first, reproduce locally only as last resort

**Fold the fix into the PR — never append a standalone "fix(pr)" commit.**
The PR must stay as **one commit** (its original commit + your fixes squashed
in), so reviewers see a single clean diff, not a running log of review
churn. This is the default on **every** iteration.

Stage the fix and count what is ahead of the base you rebased onto above:

```bash
git add <intended-files>
COMMITS_AHEAD=$(git rev-list HEAD --not "$BASE_REF" --count)
```

That rebase is why this is safe. `git reset --soft "$BASE_REF"` moves HEAD to the
base but keeps *your* index, so on a stale base every file the base gained since
you branched is recorded as **your deletion**. Rebasing afterwards does not undo
it — the revert is already part of your diff and replays cleanly. It surfaces as
unrelated files being reverted in the PR, easy to miss in a large diff.

| `COMMITS_AHEAD` | How to fold the fix in |
| --------------- | ---------------------- |
| `0` | **Error — stop.** Nothing ahead of base to fold into; do not rebase or push. Matches [commit-and-push](../../lib/github/commit-and-push.md) §2. |
| `1` | **Amend with an updated message.** Pass the message non-interactively — `git commit --amend -F <msgfile>` (or `-m`), never bare `--amend` (it opens an editor and hangs in a non-interactive/agent shell) and never `--no-edit` (leaves the message stale). Write the evolved message per the rule below into `<msgfile>` first. |
| `> 1` | **Squash to one** via the [commit-and-push](../../lib/github/commit-and-push.md) squash procedure (capture the original message, soft-reset to `$BASE_REF`, recommit once) |

Do NOT run `/git-commit` to create a *new* commit here — that is what causes
the appended-commit problem. `/git-commit` is only for regenerating the
squashed message when `COMMITS_AHEAD > 1`.

Then verify and push:

1. **Verify single commit:** `git rev-list HEAD --not "$BASE_REF" --count` must print `1` — never push a multi-commit PR. `> 1` means squash again; `0` means the fold produced nothing and is the same stop-and-investigate as in the table above, not something to re-squash.
2. **Verify you reverted nothing.** Review the file list with a **three-dot** diff:

   ```bash
   git diff "$BASE_REF"...HEAD --stat     # three dots: your changes only
   ```

   Two dots (`$BASE_REF..HEAD`) also reports files the base has and you do not,
   rendering them as deletions you made — so it hides a real revert among noise
   and invents fake ones. Any file here you did not intend to touch is a bug:
   restore it with `git checkout "$BASE_REF" -- <path>` and amend.
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

### Step 7: Reply and Resolve

Every actionable item in Step 5 gets a disposition; informational noise needs no reply. How depends on its surface (see
[reply-and-resolve](../../lib/github/reply-and-resolve.md)).

**Surface A — inline review threads.** For an accepted fix, reply with the
change and validation, then resolve within the authorized scope. Leave disputed
or unverified findings open until the reviewer or authorized maintainer accepts
the disposition. A reply alone does not settle a disagreement. For settled items:

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

**Surfaces B and C — review bodies and conversation comments.** There is no
resolve mutation for these (`resolveReviewThread` takes only a review-thread
node ID). Reply with a single batched PR conversation comment, then record each
handled ID in the ledger:

```bash
gh pr comment "$PR_NUMBER" --body "@alice Addressed:
- Rebased onto main.
- Null check added on the predicate path."

echo "$NODE_ID" >> "/tmp/pr-${PR_NUMBER}-handled.txt"   # one line per item, incl. skipped/informational
```

Recording the ID is what makes the answer stick — skip it and Step 2 re-presents
the same feedback next iteration and the loop never converges.

Reply templates:

- **Fixed** → "Fixed — description of change" (do **not** cite a commit SHA: the PR commit is amended/squashed and force-pushed each iteration, so any SHA you quote is immediately stale)
- **Discuss** → Explain the evidence and proposed disposition; leave the finding pending until accepted
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
- [ ] **All three feedback surfaces fetched** — inline threads, review bodies, conversation comments — plus CI status
- [ ] Findings classified by surface; routine fixes use existing authority, unresolved material decisions are surfaced
- [ ] Fixes folded into the PR (amended/squashed — **no appended `fix(pr)` commit**)
- [ ] Verified `git rev-list HEAD --not "$BASE_REF" --count` == 1 before pushing
- [ ] Changes force-pushed with `--force-with-lease` (single commit)
- [ ] Settled inline threads replied to and resolved; disputed findings remain explicitly pending
- [ ] Review bodies / conversation comments replied to and recorded in the handled-ID ledger
- [ ] Waited for CI/reviews and re-checked
- [ ] Loop exited: all clean OR max iterations reached

## Remember

**Not all comments require code changes.** Evaluate against `.claude/rules/` first. Ask about material unresolved decisions; handle routine authorized repairs directly.

**Feedback is not only inline review threads.** A review summary body or a plain
conversation comment carries no `isResolved` flag and no file anchor, but it can
be the most consequential thing on the PR. Fetch all three surfaces every
iteration, or the loop will report "clean" over unread feedback.
