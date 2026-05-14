---
name: review-pr
description: Review a GitHub PR by analyzing the correct diff (merge-base to HEAD), reconciling stated vs. real goal, and applying type-specific scrutiny. Optionally folds in independent reviews from local `codex` / `gemini` CLIs when the invocation explicitly opts in (`codex`, `gemini`, or `all` in the arguments). Use when the user asks to review a PR, analyze PR changes, or give feedback on a pull request.
---

# Review PR Workflow

Analyze and review PR changes using the correct merge-base diff.
Optionally cross-check against independent local CLI reviewers when
the user explicitly opts in — see [Input](#input).

## Input

Accept PR number (`123`, `#123`), or no input (review current branch
against its base).

**Optional external reviewer opt-in.** The arguments may also contain
the literal tokens `codex`, `gemini`, or `all` — these turn on the
Step 7 cross-check with the named CLIs. Without an explicit opt-in,
**Step 7 is skipped** even if the binaries are installed; cross-check
is slow (3–5 minutes per CLI) and noisy, so don't run it by default.

Examples:

- `/review-pr 773` — your own review only.
- `/review-pr 773 codex` — also run `codex review`.
- `/review-pr 773 codex gemini` or `/review-pr 773 all` — both.
- `/review-pr with codex` — review current branch, also run codex.

## Step 1: Setup

1. [Setup](../../lib/github/setup.md) — authenticate and detect context
   (role, remotes, state)

## Step 2: Determine Diff Base

**This is the critical step.** Never diff against a stale local branch.

```bash
# Default base when reviewing the current branch (no PR number).
# This must be set unconditionally so later steps (Step 7's codex/gemini
# snippets) can reference $BASE_BRANCH safely.
BASE_BRANCH="$DEFAULT_BRANCH"

# Ensure upstream is fresh
git fetch upstream "$BASE_BRANCH" --quiet

# Find the true fork point (merge-base)
MERGE_BASE=$(git merge-base "upstream/$BASE_BRANCH" HEAD)
echo "Merge base: $MERGE_BASE"
```

If reviewing a PR by number (not the current branch), override
`$BASE_BRANCH` from PR metadata:

```bash
# Fetch PR metadata
PR_DATA=$(gh pr view $PR_NUMBER --repo "$PR_REPO_OWNER/$PR_REPO_NAME" \
  --json headRefName,baseRefName,title,body)
BASE_BRANCH=$(echo "$PR_DATA" | jq -r '.baseRefName')

# Fetch and compute merge-base against the PR's actual base branch
git fetch upstream "$BASE_BRANCH" --quiet
MERGE_BASE=$(git merge-base "upstream/$BASE_BRANCH" HEAD)
```

**Validation:** Verify the merge-base is NOT a commit on the PR branch
itself:

```bash
PR_COMMITS=$(git log --oneline "$MERGE_BASE"..HEAD)
echo "$PR_COMMITS"

if git log --oneline "$MERGE_BASE"..HEAD | grep -q "$(git rev-parse --short $MERGE_BASE)"; then
  echo "ERROR: merge-base is on the PR branch — check upstream fetch"
fi
```

## Step 3: Gather Diff

All diffs use the three-dot syntax against the merge-base:

```bash
git diff "$MERGE_BASE"...HEAD --stat
git diff "$MERGE_BASE"...HEAD --name-only
git diff "$MERGE_BASE"...HEAD
```

For large PRs, read diffs per-category (by directory or file type) to
avoid overwhelming context.

## Step 4: Extract the Stated Goal

Read the PR title, body, and commit messages. Write **one paragraph** in
your own words restating what the author says the PR does and why.

```bash
if [ -n "$PR_NUMBER" ]; then
  gh pr view "$PR_NUMBER" --repo "$PR_REPO_OWNER/$PR_REPO_NAME"
fi
git log --format='%s%n%n%b%n---' "$MERGE_BASE"..HEAD
```

If reviewing the current branch with no open PR, the commit messages
are the only stated goal you have — read them as the PR body.

If the PR body is empty or just a template, say so — that itself is
worth flagging.

## Step 5: Derive the Real Goal from the Code

Independently of Step 4, read the diff and write **one paragraph**
describing what the code actually does. Do not look back at the stated
goal while writing this — the point is an independent read.

Then **compare the two paragraphs**:

- **Match** → proceed to Step 6 with the agreed goal.
- **Mismatch** → record it as a Must-fix-or-explain issue. Common
  mismatches:
  - Stated as bugfix, but adds new functionality
  - Stated as refactor, but changes behavior
  - Stated scope is narrower than the diff (scope creep)
  - Stated scope is wider than the diff (incomplete work)

The mismatch is the finding — do **not** silently rewrite the stated
goal to match the code.

## Step 6: Type-Specific Analysis

Classify the PR as one of the following and apply the matching checklist.
A PR with multiple types gets multiple checklists.

### Bugfix

CI was green before this PR, which means the existing test suite does
**not** trigger the bug. Your review must answer:

1. **What is the bug?** State the wrong behavior in one sentence.
2. **How is it triggered?** Identify the exact input/state/race/timing
   that exposes it. Explain why existing tests miss this trigger
   (untested code path, unobserved interleaving, platform-specific gate,
   etc.).
3. **How does the fix work?** Walk through the changed lines and explain
   why they restore correct behavior under the trigger.
4. **Is there a regression test?** If not, flag it. A bugfix without a
   test that fails pre-fix and passes post-fix is incomplete (see
   `.claude/rules/discipline.md` §3).

### Feature

1. **Is it needed?** Cross-reference the stated motivation against
   existing code — does an equivalent already exist? Is the use case
   real (linked issue / user request) or speculative?
2. **How does it fit existing design?** Identify the extension points it
   touches. Does it follow the patterns in `.claude/rules/architecture.md`
   and the surrounding files, or invent a new pattern?
3. **What is the blast radius?** Which files/runtimes/platforms does it
   touch? Are tests added for each platform it claims to support?
4. **Is it complete?** Half-finished features (TODOs, stubs, disabled
   code paths) should be flagged.

### Other (refactor / docs / config / chore / test-only)

1. **Behavior preservation** (refactor): does any line change semantics?
   If yes, it is not a pure refactor — re-classify.
2. **Doc/comment consistency** (any type, but especially docs): apply
   `.claude/rules/doc-consistency.md` — are referenced identifiers,
   flags, and paths still valid?
3. **Codestyle** (any type): apply `.claude/rules/codestyle.md` and any
   arch-specific rules (`.claude/rules/ascend-device.md`, etc.).

## Step 7: Optional Cross-Check with External CLI Reviewers

**Opt-in only.** Skip this entire step unless the user's `/review-pr`
arguments explicitly include `codex`, `gemini`, or `all`. The
cross-check is slow (3–5 minutes per CLI), noisy, and unnecessary for
small or routine PRs — your own analysis from Steps 5–6 is the review
by default.

Parse the request into flags. The slash-command harness exposes the
caller's argument string as `$ARGUMENTS`; the line below copies it
into `$REVIEW_ARGS` so the rest of the snippet is independent of the
harness's variable name:

```bash
REVIEW_ARGS="${ARGUMENTS:-}"

WANT_CODEX=0
WANT_GEMINI=0
case " $REVIEW_ARGS " in
  *" all "*)              WANT_CODEX=1; WANT_GEMINI=1 ;;
esac
case " $REVIEW_ARGS " in *" codex "*)  WANT_CODEX=1  ;; esac
case " $REVIEW_ARGS " in *" gemini "*) WANT_GEMINI=1 ;; esac

# Only check availability for the ones the user asked for.
[ "$WANT_CODEX"  = 1 ] && command -v codex  >/dev/null && HAS_CODEX=1
[ "$WANT_GEMINI" = 1 ] && command -v gemini >/dev/null && HAS_GEMINI=1
```

If the user opted in but the requested binary isn't on PATH, note it
in "Independent Reviewer Notes" and proceed without it. If neither
flag is set, skip the rest of Step 7 entirely.

The CLIs run as **independent** second opinions — each runs in its
own process and does not see your analysis. Run them in parallel
(single message, multiple Bash calls). A missing binary, non-zero
exit, or empty capture file is treated as "skip with note" — same
as if the user hadn't asked for it.

### Common setup: output capture

Both CLIs take minutes on real PRs (codex 3–5 min, gemini comparable).
Write each one's output to a fixed scratch path under `.docs/review/`
— that directory is already in the repo's `.gitignore` (`# Mid-work
documentation`), so the files persist across the review session for
inspection but never reach a commit. Do not pipe through `tail`:
line buffering hides progress until the process exits.

```bash
mkdir -p .docs/review
CODEX_OUT=.docs/review/codex.out
GEMINI_OUT=.docs/review/gemini.out
```

These paths overwrite on each run. If you want to keep prior output
for comparison, rename before re-running.

No wall-clock timeout is applied. The caller (you, while running this
skill) is expected to monitor progress with `wc -c "$CODEX_OUT"` /
`tail "$CODEX_OUT"` and `kill` the process if it hangs (e.g. gemini's
backoff loop on a 503 from the model proxy). Hardcoded timeouts in
the past killed legitimate runs that simply needed more time.

### Codex

`codex review` takes either `--base <branch>` (auto-diff mode) **or** a
custom `[PROMPT]` — never both; the CLI rejects the combination at
parse time. Use auto-diff mode for PR review:

```bash
if [ -n "$HAS_CODEX" ]; then
  codex review \
    --base "upstream/$BASE_BRANCH" \
    --title "PR #${PR_NUMBER:-current}: $(printf '%s' "${PR_DATA:-}" | jq -e -r .title 2>/dev/null || git log -1 --format=%s "$MERGE_BASE"..HEAD)" \
    >"$CODEX_OUT" 2>&1 \
    || echo "[codex skipped: exit $?]" >>"$CODEX_OUT"
fi
```

If you need to pass a custom prompt instead of `--base` auto-mode, use
`codex exec "<prompt>"` — `codex exec` accepts a free-form prompt and
runs non-interactively.

### Gemini

`gemini` is an agent CLI with shell and file access — same model as
`codex review --base`. Do **not** inline `$(git diff …)` into the
prompt; that hits `ARG_MAX` on large PRs and is the wrong primitive
anyway. Pass the merge-base SHA and let gemini run `git diff` itself
in the current worktree:

```bash
if [ -n "$HAS_GEMINI" ]; then
  gemini --yolo -p "$(cat <<EOF
You are an independent PR reviewer. The working directory is a git
worktree of the PR branch. Run \`git diff $MERGE_BASE...HEAD\` (and
inspect surrounding files as needed) to read the changes.

Stated goal of the PR (from its body, may be empty for current-branch
review):
$(printf '%s' "${PR_DATA:-}" | jq -e -r .body 2>/dev/null || git log --format='%s%n%n%b%n---' "$MERGE_BASE"..HEAD)

Stated title: $(printf '%s' "${PR_DATA:-}" | jq -e -r .title 2>/dev/null || git log -1 --format=%s "$MERGE_BASE"..HEAD)

Report — terse bullets only:
- correctness bugs you can verify by reading the diff and the files
- scope mismatches between the stated goal and the actual diff
- missing tests for any bugfix in the diff
Skip style nits and skip anything you cannot verify against the code.
EOF
)" >"$GEMINI_OUT" 2>&1 \
    || echo "[gemini skipped: exit $?]" >>"$GEMINI_OUT"
fi
```

`--yolo` auto-approves gemini's shell/file reads so it can run `git
diff` without prompting; the worktree is read-only from gemini's
perspective for this use case (it has no instruction to modify
anything). Drop `--yolo` if you want to inspect each tool call.

Read the captured output with `cat "$CODEX_OUT"` and `cat "$GEMINI_OUT"`
— do **not** pipe these through `tail` while the process is still
running; you will block on buffer flush and see no progress.

### Synthesis rules

- **Treat each CLI's output as one independent reviewer's notes**, not
  ground truth. Verify each claim against the actual code before
  including it.
- **Agreement across reviewers** (your read + codex + gemini) raises
  confidence — surface those findings prominently.
- **Disagreement** is worth investigating: if codex says a line is
  buggy and you don't see it, re-read that line before dismissing.
- **Reviewer hallucinations** (claims about code that isn't in the
  diff) get dropped silently — do not pass them to the user.
- If a CLI is unavailable or errors out, note it once in the final
  review and continue.

## Step 8: Write the Review

Structure:

### Stated Goal

One paragraph from Step 4.

### Real Goal (as read from the code)

One paragraph from Step 5. If it matches the stated goal, say so in one
line and move on. If it mismatches, this section is the headline finding.

### Type-specific Analysis

The checklist output from Step 6, organized by type if the PR mixes
several.

### Issues Found

Categorize by severity:

- **Must fix**: bugs, correctness issues, security problems,
  unresolved goal-vs-code mismatches
- **Should fix**: style violations per `.claude/rules/`, missing
  tests for bugfixes, doc/comment drift
- **Consider**: suggestions, optional improvements

### Independent Reviewer Notes

A short subsection per external reviewer (codex, gemini) summarizing
which of their findings you verified and included, and which you
dropped (and why, in one line).

### Verdict

approve / request changes / needs discussion.

## Common Pitfalls

1. **Stale local main** — Always `git fetch upstream` before computing
   merge-base. Never use local `main` directly.
2. **Squash-merged upstream commits** — If upstream squash-merges, the
   merge-base may include commits that look like they are on the PR
   branch. Verify with `git log --oneline MERGE_BASE..HEAD`.
3. **Cross-fork PRs** — The PR head may be on a different remote. Use
   `/checkout-pr` first if not already on the PR branch.
4. **Large PRs** — Read diffs in chunks by directory. Do not try to
   read the entire diff in one tool call if it exceeds ~2000 lines.
   For `codex` and `gemini` in Step 7, pass only the merge-base SHA
   and let each CLI run `git diff` itself in the worktree — never
   inline `$(git diff …)` into the prompt argv (hits `ARG_MAX`, ~128
   KiB on Linux).
5. **Silent goal rewriting** — Never paper over a stated-vs-real goal
   mismatch by rephrasing the stated goal. The mismatch is the finding.
6. **Trusting external reviewers** — `codex` and `gemini` hallucinate.
   Verify every claim against the diff before surfacing it.
7. **Misconfigured CLI ≠ unavailable** — a CLI binary can be on PATH
   but fail every call (no model route, expired token, upstream 5xx
   retry loop). The `command -v` gate alone won't catch this. Monitor
   `wc -c "$CODEX_OUT"` / `wc -c "$GEMINI_OUT"` while the call runs;
   if a process produces nothing for several minutes, `kill` it and
   treat the empty file as "skip with note" — same as a missing
   binary. Avoid hardcoded timeouts: codex review legitimately takes
   3–5 minutes on real PRs and a tight timeout kills good runs.
8. **Piping CLI output through `tail`** — line buffering means `tail`
   only flushes at EOF. A stuck process with active stderr looks like
   silence and breaks the empty-output heuristic. Always capture to
   a file with `>"$CAPTURE" 2>&1` and read post-hoc.
