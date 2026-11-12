---
name: review-pr
description: Review a GitHub PR or a fixed commit range from its background, existing pain points, and expected benefits through correctness, architecture, decoupling, performance, code quality, and repository standards. Uses the correct diff (fixed range or merge-base) and checks claims against implementation and evidence. Optionally folds in independent reviews from local `codex` / `claude` CLIs or read-only sub-agents when the invocation explicitly opts in. Use when the user asks to review a PR, a fixed range, or changes.
---

# Review PR Workflow

Analyze and review PR changes using the correct diff: a user-supplied
fixed range, a PR's merge-base, or the current branch against its base.
Optionally cross-check against independent local CLI reviewers or
read-only sub-agents when the user explicitly opts in — see [Input](#input).

Understand the current system, its concrete pain points, and the intended
benefit before judging the solution. Review both functional behavior and the
quality of the resulting design. A passing test suite alone does not establish
that the architecture, performance, or maintenance cost is acceptable.

## Role and review freshness

This skill performs technical review. Maintainer routing, readiness and merge
follow-through belong to [process-pr](../process-pr/SKILL.md); author repairs belong
to [fix-pr](../fix-pr/SKILL.md). Review-only work does not authorize changing the
author's branch, posting a GitHub review or merging. Reuse explicit publication
authority if supplied; otherwise return the review to the user.

Record the actual reviewed base and head SHA (`$BASE_REF` / `$HEAD_REF` from
Step 2) and read code at that head. Read existing discussion and findings
before repeating them.
If the head changes during review, reconcile the new diff or label the result as
covering the old head. CI green is not a substitute for review; an agent's favorable
verdict is not approval by a required GitHub reviewer. Preserve unresolved disputes
and distinguish verified defects from hypotheses or optional improvements.

State the evidence level for each finding: code inspection, diff-only, or not
run. Never present an untested behavior as verified; a missing test, build, or
hardware run is a limit, not a pass.

The main agent always performs the review in Steps 1–7. Step 8 only adds
independent second opinions. The `claude` token launches a separate
`claude -p` process; it never makes the main agent's own review the Claude
external reviewer. If that CLI is unavailable, record the skip in Appendix C;
do not substitute self-review.

## Input

### Review target

Accept one of:

- **PR number** (`123`, `#123`): review that PR against its base branch.
- **Fixed range**: `<base-ref>...<head-ref>` (three-dot) or
  `<base-ref>..<head-ref>` (two-dot). Use the range exactly as supplied;
  never substitute a merge-base or a remote PR diff for it. This is the right
  mode for a local, unpublished correction.
- **No target**: review the current branch against its base.

**Argument grammar.** The target and option tokens come first, in any order.
The first word that is not one of them starts the user-stated goal, and
everything from there on is goal text: a `#N`, a range, or an option word
inside the goal is never parsed as a target or option. For example,
`/review-pr 123 codex check all callers` reviews PR 123 with codex, and its
goal is "check all callers". Option words are deliberately not common English
words, so a goal that starts right after the target is not mistaken for one.

A PR number and a fixed range may both be given: the range decides the diff,
and the PR supplies the stated goal and metadata.

**Review-only is the default.** The skill reads the diff and repository files,
performs `gh` lookups when a PR target requires them, and never checks out,
edits, builds, tests, runs hardware, posts to GitHub, or invokes external CLIs
or sub-agents. Optional cross-checks are opt-in tokens. Fetching refs is the
only local Git change (see Step 1).

The invocation arguments do not reach the shell as a variable. Paste them
verbatim into the single-quoted `REVIEW_ARGS` assignment below, writing each
embedded `'` as `'\''`.

Shell state does not persist between tool calls, so the snippet saves the
parsed state to `.docs/review/state.env`. Start every later snippet in this
skill with `. .docs/review/state.env`, and append each variable a later step
sets (`DIFF_SPEC`, `LOG_RANGE`, `BASE_REF`, `HEAD_REF`, `MERGE_BASE`,
`BASE_BRANCH`) with `declare -p`.

```bash
REVIEW_ARGS='<invocation arguments, verbatim>'

PR_NUMBER=""; FIXED_RANGE=""; SCOPE=""; GOAL_TEXT=""
WANT_CODEX=0; WANT_CLAUDE=0; WANT_SUBAGENTS=0; IGNORED_TOKENS=""
IN_GOAL=0
set -f                                  # no pathname expansion while splitting
for tok in $REVIEW_ARGS; do
  if [ "$IN_GOAL" = 1 ]; then GOAL_TEXT="$GOAL_TEXT $tok"; continue; fi
  case "$tok" in
    codex)     WANT_CODEX=1 ;;
    claude)    WANT_CLAUDE=1 ;;
    subagents) WANT_SUBAGENTS=1 ;;
    gemini)    IGNORED_TOKENS="$IGNORED_TOKENS gemini(unsupported)" ;;
    scope:?*)  SCOPE=${tok#scope:} ;;
    ?*..?*)    FIXED_RANGE=$tok ;;      # <base-ref>..<head-ref> or <base-ref>...<head-ref>
    *)
      num=${tok#\#}
      case "$num" in
        ''|*[!0-9]*) IN_GOAL=1; GOAL_TEXT=$tok ;;
        *)           PR_NUMBER=$num ;;
      esac ;;
  esac
done
set +f

mkdir -p .docs/review
declare -p PR_NUMBER FIXED_RANGE SCOPE GOAL_TEXT WANT_CODEX WANT_CLAUDE \
  WANT_SUBAGENTS IGNORED_TOKENS > .docs/review/state.env
```

Optional tokens:

- `codex` — opt in to the Step 8 codex cross-check when this session is not
  Codex. If this session is Codex, ignore the token: the main review already
  is Codex.
- `claude` — opt in to the Step 8 Claude CLI cross-check when this session is
  not Claude Code. If this session is Claude Code, ignore the token: the main
  review already is Claude.
- `subagents` — opt in to the Step 8.5 read-only sub-agent cross-check.
  Without the token, stay single-agent.
- `scope:<path>` — narrow the review to that path, its direct callers, and the
  definitions it depends on.
- `gemini` — no longer supported; recorded as ignored, not run.

You know your harness from the system context; no shell detection is needed.
If this session is Claude Code, treat `claude` as absent; if this session is
Codex, treat `codex` as absent. Record ignored tokens, including
`$IGNORED_TOKENS`, in Appendix C.

Do not require `no-*` tokens. Review-only is the default: no checkout, edits,
builds, tests, hardware, GitHub writes, external CLIs, or sub-agents. If the
user asks for builds, tests, or hardware as part of the review, route to
`testing`, `fix-pr`, or `develop-task`; do not add an opt-in token here.

Echo the interpreted target and opt-in extras in the final Review limits.

**Validate before analysis.** For a fixed range, verify both endpoints and the
non-empty diff before reading any code:

```bash
. .docs/review/state.env
case "$FIXED_RANGE" in
  *"..."*) SEP="..." ;;
  *".."*)  SEP=".." ;;
  *)       SEP="" ;;
esac
if [ -n "$FIXED_RANGE" ]; then
  BASE_REF=${FIXED_RANGE%%"$SEP"*}
  HEAD_REF=${FIXED_RANGE#*"$SEP"}
  git rev-parse --verify "$BASE_REF^{commit}" >/dev/null || { echo "bad base: $BASE_REF"; exit 1; }
  git rev-parse --verify "$HEAD_REF^{commit}" >/dev/null || { echo "bad head: $HEAD_REF"; exit 1; }
  [ -n "$(git diff "$FIXED_RANGE" --name-only)" ] || { echo "empty diff: $FIXED_RANGE"; exit 1; }
  if [ "$SEP" = ".." ] && ! git merge-base --is-ancestor "$BASE_REF" "$HEAD_REF"; then
    echo "WARN: two-dot range with diverged endpoints: base-side commits appear reversed"
    git log --oneline "$HEAD_REF..$BASE_REF"
  fi
  declare -p SEP BASE_REF HEAD_REF >> .docs/review/state.env
fi
```

If validation fails, stop and report the bad ref or empty range. Do not
silently fall back to merge-base.

A two-dot range compares the two endpoint trees. When the endpoints have
diverged, the commits printed by the warning are base-side history: their
changes appear reversed in the diff, and `$LOG_RANGE` does not include them.
Read them as context, do not report their reversal as the reviewed change, and
record the divergence in Review limits.

**Read context at the review head.** For a fixed range or a PR reviewed
without checkout, the working tree is not the reviewed code. Read surrounding
files with `git show "$HEAD_REF:<path>"` and search with
`git grep <pattern> "$HEAD_REF"`. Use working-tree files only when
`git rev-parse HEAD` equals `$HEAD_REF` and `git status --porcelain` is empty.

### User-stated expected goal (optional but high-priority)

When the user invokes `/review-pr` they may describe in natural language what
they think this PR is *supposed* to achieve — possibly more ambitious or
differently-framed than the PR body itself claims. Treat that description as a
first-class source of the stated goal (see Step 4), ranked **above** the PR
body. The most common reason this matters: the PR author downgraded the
original goal in their own description, and the user — who is reviewing —
knows the original target.

Examples of user-stated goals to watch for:

- "I thought this PR was supposed to do X" — X is now part of stated goal.
- "The original design was Y" — Y is now part of stated goal.
- "The team agreed to Z" — Z is now part of stated goal.
- "Per issue #N this should do W" — go read issue #N before Step 4.

### Optional cross-checks

The arguments may also contain:

- `codex` or `claude` — Step 8 external CLI cross-check. Without an
  explicit opt-in, **Step 8 is skipped** even if the binaries are installed;
  the cross-check is slow (3–5 minutes per CLI) and noisy. The token matching
  the current harness is ignored: `claude` under Claude Code, `codex` under
  Codex.
- `subagents` — optional Step 8.5 read-only sub-agent cross-check. Without
  the token, stay single-agent.

Examples:

- `/review-pr 773` — your own review only.
- `/review-pr 773 codex` — also run the codex cross-check.
- `/review-pr 3f4b5541af...bea407de05` — fixed local range, no PR.
- `/review-pr upstream/main...my-fix scope:src/a5 subagents`
- `/review-pr codex` — review current branch, also run codex.
- `/review-pr 773 — this PR is supposed to add symmetric IPC between
  X and Y` — user-stated goal wins over PR body for Step 4.

## Step 1: Setup

If the target is a fixed range and no PR number was supplied, skip Setup: the
review is local and needs no GitHub context. If a PR number is present, run
Setup to resolve its base and metadata.

1. [Setup](../../lib/github/setup.md) — authenticate and detect context
   (role, remotes, state)

Save the Setup variables later steps use:

```bash
declare -p DEFAULT_BRANCH PR_REPO_OWNER PR_REPO_NAME >> .docs/review/state.env
```

Setup may add the `upstream` remote and fetch remote-tracking refs; it moves no
branch, edits no working-tree file, and writes nothing to GitHub. This skill
never posts, edits, labels, or merges; publishing belongs to `process-pr` or an
explicit user instruction.

## Step 2: Determine Diff Base or Fixed Range

**This is the critical step.** Never diff against a stale local branch, and
never substitute a different diff for the user's fixed range. A failed fetch
stops the review: a stale `upstream/<base>` silently yields the wrong diff.

```bash
. .docs/review/state.env
if [ -n "$FIXED_RANGE" ]; then
  # Input already validated both endpoints and the non-empty diff.
  # A PR number given alongside supplies only goal and metadata (Step 4).
  DIFF_SPEC="$FIXED_RANGE"
  LOG_RANGE="$BASE_REF..$HEAD_REF"
  echo "Using fixed range: $DIFF_SPEC"
elif [ -n "$PR_NUMBER" ]; then
  # Review the PR's own head without checking it out.
  PR_DATA=$(gh pr view "$PR_NUMBER" --repo "$PR_REPO_OWNER/$PR_REPO_NAME" \
    --json headRefOid,baseRefName,title,body) || { echo "PR lookup failed"; exit 1; }
  BASE_BRANCH=$(printf '%s' "$PR_DATA" | jq -r '.baseRefName')
  HEAD_REF=$(printf '%s' "$PR_DATA" | jq -r '.headRefOid')
  git fetch upstream "$BASE_BRANCH" --quiet || { echo "fetch failed: $BASE_BRANCH"; exit 1; }
  git fetch upstream "pull/$PR_NUMBER/head" --quiet || { echo "fetch failed: PR head"; exit 1; }
  [ "$(git rev-parse FETCH_HEAD)" = "$HEAD_REF" ] || { echo "PR head moved; re-run"; exit 1; }
  MERGE_BASE=$(git merge-base "upstream/$BASE_BRANCH" "$HEAD_REF")
  DIFF_SPEC="$MERGE_BASE...$HEAD_REF"
  LOG_RANGE="$MERGE_BASE..$HEAD_REF"
  BASE_REF="$MERGE_BASE"
else
  # Current branch against the default base.
  BASE_BRANCH="$DEFAULT_BRANCH"
  git fetch upstream "$BASE_BRANCH" --quiet || { echo "fetch failed: $BASE_BRANCH"; exit 1; }
  MERGE_BASE=$(git merge-base "upstream/$BASE_BRANCH" HEAD)
  HEAD_REF=$(git rev-parse HEAD)
  DIFF_SPEC="$MERGE_BASE...$HEAD_REF"
  LOG_RANGE="$MERGE_BASE..$HEAD_REF"
  BASE_REF="$MERGE_BASE"
fi
echo "Diff: $DIFF_SPEC"
declare -p DIFF_SPEC LOG_RANGE BASE_REF HEAD_REF >> .docs/review/state.env
[ -n "${MERGE_BASE:-}" ] && declare -p MERGE_BASE BASE_BRANCH >> .docs/review/state.env
```

`$HEAD_REF` is a commit SHA in both merge-base modes, so a later checkout or
fetch cannot change what is being reviewed. Record it as the reviewed head.

**Validation (merge-base mode only).** Verify the merge-base is NOT a commit on
the PR branch itself:

```bash
. .docs/review/state.env
if [ -z "$FIXED_RANGE" ]; then
  git log --oneline "$LOG_RANGE"

  if git log --oneline "$LOG_RANGE" | grep -q "$(git rev-parse --short "$MERGE_BASE")"; then
    echo "ERROR: merge-base is on the PR branch — check upstream fetch"
  fi
fi
```

A fixed range is the user's contract. If it is invalid or empty, stop; do not
recompute a merge-base and continue.

## Step 3: Gather Diff

Use `$DIFF_SPEC` from Step 2 for every diff:

```bash
. .docs/review/state.env
git diff "$DIFF_SPEC" --stat
git diff "$DIFF_SPEC" --name-only
git diff "$DIFF_SPEC"
```

If `$SCOPE` is non-empty, start with `git diff "$DIFF_SPEC" -- "$SCOPE"`. Read
outside the scope only what is needed to judge it: direct callers, and the
definitions and contracts the scoped code depends on. Findings must concern
the scoped change.

For large PRs, read diffs per-category (by directory or file type) to
avoid overwhelming context.

## Step 3.5: Categorize and Size the Changes

Break the diff into buckets and count changed lines (added + deleted,
i.e. churn) per bucket. This feeds Step 9's Appendix A (Change
Breakdown) and drives the oversized-PR warnings.

**Precedence:** classify by file type / purpose first (Test/Examples,
Build, Docs), then by source-tree location (Core), then everything
left over is Uncategorized. First match wins — so a `CMakeLists.txt`
under `src/` is still Build, a `README.md` under `python/` is still
Docs, and a `.py` outside the Core dirs is Uncategorized, not Core.

**Buckets:**

- **Test/Examples** — anything under `test/`, `tests/`, `__tests__/`,
  `fixtures/`, `testdata/`, `snapshots/`, **`examples/`**; files named
  `test_*.*`, `*_test.*`, `*Test.*`, `*_spec.*`, `conftest.py`.
- **Build** — `CMakeLists.txt`, `*.cmake`, `Makefile`/`*.mk`,
  `*.am`/`*.ac`/`*.in`, `configure`, `config.sub`/`config.guess`,
  `aclocal.m4`, `meson.build`, `setup.py`, `setup.cfg`,
  `pyproject.toml`, `requirements*.txt`, `Pipfile*`, `MANIFEST.in`,
  `package.json`, `*-lock.json`, `yarn.lock`, `pnpm-lock.yaml`,
  `tsconfig.json`, `BUILD`, `BUILD.bazel`, `*.bzl`, `WORKSPACE`.
- **Docs** — `*.md`, `*.rst`, `*.txt`, `*.adoc`, `*.tex`, images
  (`png`/`jpg`/`jpeg`/`svg`/`gif`/`webp`); files under `docs/`/`doc/`;
  `CHANGELOG`, `README`, `CONTRIBUTING`, `LICENSE`, `NOTICE`.
- **Core** — real source / logic, **only if** the path is under `src/`,
  `python/`, or `simpler_setup/` (root-anchored). Anything outside
  those three dirs is not Core even if it looks like source.
- **Uncategorized** — everything else not matched above (e.g. agent
  config under `.claude/`, scripts, CI workflows, repo-root config).
  This bucket exists so odd / out-of-scope files stay visible instead
  of being silently absorbed into Core or Docs.

Awk field references in this skill are written `$(N)`, never as a bare dollar
followed by a digit: the skill loader replaces that bare form with an
invocation argument before the snippet runs.

```bash
. .docs/review/state.env
git diff "$DIFF_SPEC" --numstat | awk -F'\t' '
  $(1)=="-" { next }                             # skip binary
  {
    a=$(1); d=$(2); path=$(3); tot=a+d;
    if (path ~ /(^|\/)(test|tests|__tests__|fixtures|testdata|snapshots|examples)\// ||
        path ~ /(^|[_\/])(test_|conftest\.py)/ ||
        path ~ /(_test|Test|_spec)\.[A-Za-z0-9]+$/)
      c="test";
    else if (path ~ /(^|\/)CMakeLists\.txt$/ || path ~ /\.cmake$/ ||
             path ~ /(^|\/)Makefile/ || path ~ /\.(mk|am|ac|in)$/ ||
             path ~ /(^|\/)(aclocal\.m4|configure|config\.(sub|guess))$/ ||
             path ~ /(^|\/)meson\.build$/ ||
             path ~ /(^|\/)(setup\.py|setup\.cfg|pyproject\.toml|requirements.*\.txt|Pipfile.*|MANIFEST\.in)$/ ||
             path ~ /(^|\/)(package\.json|package-lock\.json|yarn\.lock|pnpm-lock\.yaml|tsconfig\.json)$/ ||
             path ~ /(^|\/)(BUILD|BUILD\.bazel|WORKSPACE)$/ || path ~ /\.bzl$/)
      c="build";
    else if (path ~ /\.(md|rst|txt|adoc|tex)$/ ||
             path ~ /(^|\/)(CHANGELOG|README|CONTRIBUTING|LICENSE|NOTICE)/ ||
             path ~ /(^|\/)docs?\// ||
             path ~ /\.(png|jpg|jpeg|svg|gif|webp)$/)
      c="docs";
    else if (path ~ /^(src|python|simpler_setup)\//)
      c="core";
    else
      c="uncat";
    F[c]++; A[c]+=a; D[c]+=d; T[c]+=tot;
  }
  END {
    split("core build test docs uncat", o, " ");
    label["core"]="Core"; label["build"]="Build"; label["test"]="Test/Ex";
    label["docs"]="Docs"; label["uncat"]="Uncategorized";
    tf=ta=td=tt=0;
    for (i=1;i<=5;i++){k=o[i];
      printf "  %-13s %3d files  +%5d  -%5d  =%d\n", label[k], F[k]+0, A[k]+0, D[k]+0, T[k]+0;
      tf+=F[k]; ta+=A[k]; td+=D[k]; tt+=T[k];}
    printf "  %-13s %3d files  +%5d  -%5d  =%d\n", "TOTAL", tf, ta, td, tt;
  }
'
```

**Thresholds (carry into Step 9):**

- **Total churn > 1000 lines** → ⚠️ oversized-PR warning (review burden + merge-conflict risk; suggest splitting by concern).
- **Core churn > 1000 lines** → ❌ stronger oversized-PR warning. Core
  is the real logic under `src/` / `python/` / `simpler_setup/`; when
  it alone exceeds 1000 lines the PR cannot be held in head in one
  pass. Recommend split and require the Mechanism Brief (Step 5.5,
  rendered in §2) plus a reviewer-friendly commit review order.

### Review depth

Set `REVIEW_DEPTH` from the churn and risk:

- `quick` — single file, under ~50 changed lines, and no concurrency, ABI,
  ownership, or security surface. The trivial-PR exception in Step 9 may apply.
- `standard` — the default.
- `deep` — over ~500 changed lines or 3+ files, or any concurrency, ABI,
  ownership, or security surface. Require the Mechanism Brief (Step 5.5), the
  traceability table (Step 5.7), and the Standards check (Step 6.5).

Depth decides which later steps are mandatory; it does not decide the verdict.

## Step 4: Extract the Stated Goal(s)

The stated goal can come from **multiple sources**. Gather all of them
before judging. They are not equal in weight — when they disagree, the
disagreement itself is a finding.

### Sources, in priority order

1. **User's natural-language description in the `/review-pr` invocation.**
   See Input. If the user said "this PR is supposed to do X", X is the
   highest-priority stated goal — they're the reviewer asking for the
   review, and they may know context (linked design doc, team agreement,
   parent issue) the PR body omits.

2. **Linked GitHub issue or design doc.** If PR body, commit messages,
   or the user references an issue number / design doc, **read it
   before judging**. Issues often carry the original goal the PR body
   later compressed or downgraded.

   ```bash
   # If PR body / commits reference an issue:
   gh issue view "$ISSUE_NUMBER" --repo "$PR_REPO_OWNER/$PR_REPO_NAME"
   ```

3. **PR body and title.** What the author wrote when opening the PR.

4. **Commit messages.** Often the most precise per-commit intent;
   useful when PR body is empty or a template.

```bash
if [ -n "$PR_NUMBER" ]; then
  gh pr view "$PR_NUMBER" --repo "$PR_REPO_OWNER/$PR_REPO_NAME"
fi
git log --format='%s%n%n%b%n---' "$LOG_RANGE"
```

### Producing the Stated Goal

Record the stated goal (rendered in §1 of the review) using **the most
ambitious / most authoritative** source that's coherent with the
others. Then explicitly note any *narrower* sources:

- If the user said "should do X" and the PR body says "adds primitive
  for X-helper", **state X as the goal** and record "PR body downgrades
  to X-helper" as a discrepancy to track through Steps 5.7 and 9.
- If a linked issue says "implement Y" and PR body says "first step
  toward Y", **state Y as the goal** and record "PR body scopes to
  first step" as a discrepancy.
- If only the PR body exists and it's a coherent goal statement, use it.
- If only commit messages exist (current-branch review), synthesize
  from them.
- If the PR body is empty or just a template **and** no user
  description and no linked issue, say so — that itself is a finding.

**Goal downgrade is a high-value finding.** The single most common
cause of "approved PR turns out to not have done what we wanted" is
the author silently narrowing the goal in their own PR body. The
extra Step 4 work catches it before approval.

If reviewing the current branch with no open PR, commit messages and
any user-stated description are your only sources.

If no user description, linked issue/design doc, PR body, or commit message
yields a coherent goal, state "no spec available" in Review coverage and
review against the user's explicit contract. Do not invent a spec or silently
promote an implementation detail into a goal.

## Step 4.5: Establish the Background, Pain, and Expected Benefit

Before judging the changed implementation, inspect the base version
(`$BASE_REF`) and the relevant callers, consumers, contracts, and existing
alternatives. Establish:

- **Current behavior and constraints:** how the affected mechanism works today
  and why it has this shape. Read relevant design docs and investigations.
- **Concrete pain:** an input, workflow, dependency, or maintenance operation
  that exposes the limitation; identify who pays the cost and where.
- **Expected benefit:** what becomes possible, faster, simpler, or easier to
  maintain, including better architectural boundaries, less coupling, and
  clearer code. These quality improvements are benefits in their own right.
  State the success criterion when known; do not invent a numerical target
  or turn the author's claim into an observed result.
- **Evidence and uncertainty:** cite base-code locations, issue/design context,
  reproductions, or measurements. Keep verified facts, author claims, and your
  hypotheses distinct; say what is missing when the motivation is unclear.

Carry the background and pain into report §1, then evaluate the benefit against
the actual mechanism in §2. Keep this proportional to the change, but do not
skip it because the PR is small or labeled refactor.

## Step 5: Derive the Real Goal from the Code

Independently of Step 4, read the diff and describe what the code
actually does. Do not look back at the stated goal while writing this —
the point is an independent read.

**Length scales with PR size.** One paragraph for a small PR; several
paragraphs for a medium PR; a full Mechanism Brief (Step 5.5) for any
PR over ~500 changed lines or 3+ files. A "one paragraph" summary of a
5000-line PR is not a summary, it's an abdication.

Then **compare your description against the stated goal**:

- **Match** → proceed to Step 5.5 / 5.7 with the agreed goal.
- **Mismatch** → record it as a Must-fix-or-explain issue. Common
  mismatches:
  - Stated as bugfix, but adds new functionality
  - Stated as refactor, but changes behavior
  - Stated scope is narrower than the diff (scope creep)
  - Stated scope is wider than the diff (incomplete work)

The mismatch is the finding — do **not** silently rewrite the stated
goal to match the code.

## Step 5.5: Mechanism Brief

**Required for PRs over ~500 lines or 3+ files; recommended
otherwise; a trivial PR (single-file, < ~50 lines) may omit it.**

Write a Mechanism Brief that walks a reader through the PR's design from
scratch. The audience is anyone who will later need to reason about this
code — including the PR author re-reading their own work in six months.

This is not the place for findings. It is the place where you prove you
understood the PR before judging it. Skipping it on a large PR makes the
rest of the review unreliable: bug-hunting without a mental model
produces low-signal nitpicks and misses architectural issues.

Cover, in order:

1. **What problem the PR solves** — in your own words, not the PR
   body's. If you can't restate it, you don't understand it yet.
2. **Central abstractions introduced or modified** — data structures,
   state machines, ABI surfaces, on-disk / on-wire formats.
3. **Lifetime and ownership rules** — who allocates, who frees, what
   prevents use-after-free / leaks.
4. **Concurrency / threading model** — invariants, critical sections,
   what's serialized vs. parallel.
5. **Cross-boundary contracts** — process / language / ABI / network /
   IPC. What's the wire format? What's the version story?
6. **How the existing system absorbed the change** — which extension
   points were used, what was generalized, what was duplicated.
7. **Benefit and cost** — how the mechanism removes the identified pain,
   which costs it adds or moves elsewhere, and what evidence supports the
   expected improvement. Explain relevant alternatives and trade-offs.

## Step 5.7: Goal-Method Traceability

Start a table mapping every stated goal to its design choice and code location.
In Step 6, extend the same assessment with quality dimensions and their benefits
and costs relative to the baseline:

| Goal / dimension | Design / code location | Before → after | Benefit / cost | Assessment / evidence |
| :--------------- | :--------------------- | :------------- | :------------- | :-------------------- |

For goal coverage, use the following assessments:

- ✅ **Solid** — choice clearly supports the goal
- ⚠️ **Underspecified** — choice exists but rationale missing from PR
  body, comments, or docs
- ⚠️ **Weak** — choice exists but doesn't fully support the goal
  (explain how it falls short)
- ❌ **Missing** — goal stated but no corresponding choice in the diff
- ➕ **Implicit** — code introduces behavior or obligations beyond the stated
  scope. Discuss whether to include that scope or remove the extra behavior.

Distinguish author-stated goals from reviewer-observed quality effects. A quality
improvement does not become scope creep merely because the author did not
advertise it; explain its benefit and any associated cost.

❌ and ➕ rows are must-discuss before approval. ⚠️ rows feed Step 9's
§4 issue list (typically as Should-fix or Consider). ✅ rows need no
further action but document that the goal landed.

This table also makes asymmetries obvious: if a PR has 8 stated goals
and only 3 traceable design choices, the other 5 goals are either
hand-waving or done implicitly via mechanisms the reader has to guess.

## Step 6: Comprehensive Quality Review

Read and apply [the quality review guide](references/quality-review.md).
For every code-changing PR, assess architecture, coupling, performance,
simplicity, reuse, naming, data structures, concurrency/ownership, failure
handling, compatibility, test quality, and documentation/operability. These
dimensions also apply to bugfixes and refactors, not only features.

Evaluate each dimension as part of the change's benefit/cost analysis. Better
architecture, weaker coupling, clearer code, and stronger performance are
benefits; architectural degradation, tighter coupling, complexity, and resource
regressions are costs. Trace **baseline pain → design change → quality effect →
benefit/cost → evidence**, including effects beyond the author's stated claims.
A dimension can improve in one respect and regress in another; record both.

Inspect relevant surrounding and analogous code with `rg` (at `$HEAD_REF` when
the working tree is not the reviewed head; see Input); a diff alone cannot
establish that a new abstraction fits or that no equivalent already exists.
Scale the depth to the changed paths and risks. For docs/config-only work,
explain which dimensions are inapplicable; do not manufacture code findings.

Use these results to substantiate the benefits and costs explained in report §2
and the combined assessment in §3. Route actionable issues to §4 and explain
which trade-offs support the recommendation in §5. Never replace the assessment
with a blanket "looks good" or present an unmeasured performance expectation as
a verified improvement.

## Step 6.5: Standards Check (cross-cutting)

The Standards axis asks whether the diff follows the repository's documented
rules, independently of whether it implements the stated goal. Run it at every
depth; at `quick` depth, limit it to the changed files. Step 6 weighs those
rules as design benefits and costs; this step records compliance — each rule is
either met or violated at a `file:line`.

**Sources:** `.claude/rules/codestyle.md`, `comments.md`, `doc-consistency.md`,
`project-layout.md`, `ascend.md`, `env-macro-gating.md`, `discipline.md`,
`running-onboard.md`, `ci-change-detection.md`, plus any other rule relevant to
the changed paths; `CLAUDE.md`, `docs/developer-guide.md`, and
`docs/testing.md`.

Rules:

- Repository rules override generic smell heuristics (duplicated code, feature
  envy, data clumps, primitive obsession, speculative generality, and similar).
  Smells are judgement calls, not violations.
- Skip anything tooling already enforces (clang-format, markdownlint, lint
  hooks); a clean hook is not a review finding.
- Hard-constraint violations (for example `ascend.md` MMIO/AICore limits, or an
  unauthorized environment/macro gate) are **Must-fix**. Style, documentation,
  and missing-test gaps are Should/Consider.
- Report each violation with `file:line` and the rule it violates. If the diff
  is clean, say so in Review coverage.

The codestyle and doc-consistency checks live here; Step 7 "Other" refers to
this step rather than repeating them.

## Step 7: Type-Specific Analysis

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
   touches. Does it follow the patterns in `.claude/rules/ascend.md` and
   `.claude/rules/project-layout.md` and the surrounding files, or invent
   a new pattern?
3. **What is the blast radius?** Which files/runtimes/platforms does it
   touch? Are tests added for each platform it claims to support?
4. **Is it complete?** Half-finished features (TODOs, stubs, disabled
   code paths) should be flagged.
5. **Stable-boundary discipline.** Identify every boundary the PR
   touches (C ABI, IPC protocol, on-disk format, public Python API,
   wire format). For each:
   - Is there a version field? Is its evolution rule documented?
   - Are reserved fields constrained at the validator (rejected if
     non-zero)?
   - What's the deprecation path if v2 needs to differ?
   Reserved-but-unvalidated fields and absent version stories are
   findings.
6. **Error propagation paths.** Trace each error condition from where
   it is raised to where the end user sees it. For cross-boundary
   paths (process / language / ABI):
   - Are integer codes preserved, or translated to strings and back?
   - Is the translation rule documented or hardcoded?
   - Are there any string-matching error translations? Those are bug
     magnets — flag them.
7. **Concurrency model.** Enumerate every critical section and every
   happens-before relation the code relies on. For each:
   - Where is the invariant stated (comment, doc)?
   - Which test exercises the path under contention?
   - What's the failure mode if the invariant breaks?
   A concurrency-relevant mechanism with no contention test is
   incomplete, even if the linear-case tests pass.
8. **Alternatives considered.** What other shape could this feature
   have taken? For each plausible alternative (extending an existing
   primitive, reusing an adjacent mechanism, picking a different
   ownership model), why was this shape chosen? If the PR body
   doesn't answer this, flag it — the next contributor will have to
   re-derive the trade-off, and missing rationale is the single most
   common cause of architectural regressions.

### Other (refactor / docs / config / chore / test-only)

1. **Behavior preservation** (refactor): does any line change semantics?
   If yes, it is not a pure refactor — re-classify.
2. **Standards**: apply Step 6.5 (codestyle, comments, doc consistency,
   layout, and architecture rules). Do not repeat the checks here.

## Step 7.5: pto-isa Pin Check (repo-specific)

This repo vendors the `pto-isa` headers from a **pinned** git commit. When the pin is active (a concrete SHA, not the "latest main/master" default), the review always carries a pto-isa pin note: a baseline **advisory** that the pinned commit exists and may need bumping, escalating to a **recommendation** when the PR adds or relocates a pto-isa header reference — the case where the pinned revision likely no longer matches what the PR references and a bump is probably required. The risk being mitigated: a PR can silently rely on pto-isa surface the pinned revision does not provide, whose only signal is a broken build or runtime on device.

**Pin source of truth:** repo-root `pto_isa.pin` (a 40-hex SHA). CI
reads it via `.github/actions/read-pto-isa` and feeds it both to
test-time (`--pto-isa-commit`) and to the onboard `host_runtime.so`
build (`SIMPLER_PTO_ISA_BUILD_COMMIT` cmake define; path is
`-DPTO_ISA_ROOT=` from the managed pin checkout, not ambient env). Unpinned sentinel
values (use latest HEAD): `""`, `head`, `latest`, `none` — see
`simpler_setup/pto_isa.py` `_UNPINNED_COMMIT_VALUES`.

```bash
# 1. Is the pin active? (read the review head's pto_isa.pin)
. .docs/review/state.env
PIN_VALUE=$(git show "$HEAD_REF:pto_isa.pin" 2>/dev/null | tr -d '[:space:]')
PIN_ACTIVE=0
case "$(printf '%s' "$PIN_VALUE" | tr 'A-F' 'a-f')" in
  ""|head|latest|none) PIN_ACTIVE=0 ;;
  *) printf '%s' "$PIN_VALUE" | grep -qE '^[0-9a-f]{40}$' && PIN_ACTIVE=1 ;;
esac
echo "pto_isa.pin=$PIN_VALUE  active=$PIN_ACTIVE"

# 2. Detect pto-isa include-path changes in the diff.
#    Classify every +/- pto-isa #include by header basename:
#      CHANGED  = same basename, different path   (pto-isa header reorg)
#      ADDED    = a pto-isa header newly referenced
#      REMOVED  = a pto-isa header no longer referenced
INC_REPORT=$(git diff "$DIFF_SPEC" -- '*.h' '*.hpp' '*.c' '*.cpp' '*.cc' '*.cxx' '*.cu' '*.cuh' | awk '
  /^---/ { next }  /^\+\+\+/ { next }
  /^[+-][[:space:]]*#[[:space:]]*include[[:space:]]*[<"][^>"]*pto/ {
    sign = substr($(0),1,1); line = substr($(0),2)
    if (!match(line, /[<"][^>"]*pto[^>"]*[>"]/)) next
    hdr = substr(line, RSTART+1, RLENGTH-2)   # strip < > or " "
    bn = hdr; sub(/^.*\//, "", bn)
    if (sign == "-") { rem[bn] = hdr } else { add[bn] = hdr }
    seen[bn] = 1
  }
  END {
    for (b in seen) {
      if (b in rem && b in add) { if (rem[b] != add[b]) printf "  CHANGED  %s -> %s\n", rem[b], add[b] }
      else if (b in add)        printf "  ADDED    %s\n", add[b]
      else                      printf "  REMOVED  %s\n", rem[b]
    }
  }')

# (a) the pin file itself touched?
git diff "$DIFF_SPEC" --name-only | grep -qx 'pto_isa.pin' \
  && PIN_TOUCHED=1 || PIN_TOUCHED=0

# (b) referenced pto-isa header PATH changed (the rename/reorg case)
PTO_REFS=$(printf '%s\n' "$INC_REPORT" | grep '^  CHANGED' || true)
# (c) a pto-isa header newly referenced (no matching removal)
NEW_PTO_INC=$(printf '%s\n' "$INC_REPORT" | grep '^  ADDED' || true)

echo "pin_touched=$PIN_TOUCHED"
echo "pto_refs (path changes):"; printf '%s\n' "$PTO_REFS"
echo "new pto includes:";        printf '%s\n' "$NEW_PTO_INC"
```

**Decision — two severity levels:**

- `PIN_ACTIVE=0` → pin check is **skipped** (repo runs on latest main/master). Record one line ("pto-isa unpinned — using latest; pin check skipped") and stop this step.
- `PIN_ACTIVE=1` **and no** pto-isa header-reference change (`PTO_REFS` and `NEW_PTO_INC` both empty) → render the Step 9 Appendix B (pto-isa Pin Check) paragraph at the default level, an **advisory**: a light, always-on reminder that the pin exists and may need bumping. This level surfaces whenever the pin is active — even when the diff does not touch the pto-isa cone — so the human is nudged to confirm the pinned commit is still adequate. It does not by itself block the merge.
- `PIN_ACTIVE=1` **and** a pto-isa header reference changed (`PTO_REFS` or `NEW_PTO_INC` non-empty) → escalate one level to a **recommendation**: the pinned commit likely no longer matches the headers the PR references, so a bump is probably required, not merely worth checking. Escalating signals:
  1. `PTO_REFS` — a referenced pto-isa header **path** changed (e.g. `pto/npu/comm/async/sdma/sdma_workspace_manager.hpp` → `pto/comm/async/sdma/sdma_workspace_manager.hpp`). The PR is adapting to a pto-isa header-tree reorg → the pinned commit almost certainly must bump to the revision with the new layout. This is the strongest pin-bump signal.
  2. `NEW_PTO_INC` — the PR newly references a pto-isa header not used before → may need a newer pin that provides it.

`PIN_TOUCHED` (the PR itself edited `pto_isa.pin`) is independent of the two levels: the author already bumped explicitly, so the paragraph is rendered at whichever level the header-reference signals dictate (default advisory if no header-ref change), and you additionally verify the new SHA is intentional and that onboard `host_runtime.so` gets rebuilt against it.

Note: this detects *include-path* changes, **not** edits to files that merely happen to `#include` pto-isa. A logic-only change inside an existing pto-isa consumer keeps the paragraph at the default advisory level — it does not by itself escalate.

**What to convey (both levels):** the current pinned SHA from `pto_isa.pin`; and that a pin bump requires rebuilding onboard `a2a3` `host_runtime.so` against the new commit via `--config-settings=cmake.define.SIMPLER_PTO_ISA_BUILD_COMMIT=<sha>` (the SDMA headers are compiled into `host_runtime.so`; see `docs/developer-guide.md`, issue #1067). The exact wording per level is given by the Appendix B blockquotes in Step 9.

## Step 7.6: Feature Coverage Declarations (repo-specific)

The onboard ST lanes (`_st-npu-a2a3.yml`, `_st-npu-a5.yml`) and the framework
lanes (`_st-framework*.yml`) run pytest with `--environment-receipts
--verify-feature-coverage --coverage-partition <partition>`. A selected case
with a missing or conflicting declaration fails **collection**, turning the
whole lane red before any test runs. Trigger this step when the diff adds or
edits a case under `examples/` or `tests/st/`, or touches
`simpler_setup/scene_test.py`, `simpler_setup/test_coverage.py`, or those
workflows.

For every added or modified case, check:

1. **Declarations present.** Program is the default: a case declares
   `runtime_product("kernel"|"auxiliary")` only when it is not Program. Kernel
   cases also declare `execution_mode("eager"|"capture_replay")`; Program and
   auxiliary cases are mode-free and an explicit mode on them is an error.
   `SceneTestCase` defaults to Program; its `_st_execution_mode` default applies
   only to Kernel. Plain Program tests need no product or mode marker.
2. **Exactly one runtime identity.** From the `runtime` marker, the scene class
   `_st_runtime`, or the parametrized `runtime` parameter. `coverage_runtime`
   is for runtime-independent probes / legacy tests and must not carry a
   feature claim.
3. **Honest product.** Program = borrowed-context / limited-resource; Kernel =
   worker-owned; Auxiliary probes must not claim delivered Kernel features or
   satisfy mandatory groups. Framework (`torch_npu`) consumers must be Kernel.
4. **Skip exceptions.** Only `coverage_skip(platforms=[...], reason="<exact>")`
   on the default environment; the JUnit skip reason must match exactly; no
   xfail; no framework skip exceptions; an exemption cannot replace a required
   group member. New exceptions must be justified against the scoped-skip table
   in `docs/ci.md`.
5. **Obligations stay satisfiable.** Coverage groups are `program/<runtime>`,
   `auxiliary/<runtime>` and `kernel/<runtime>/<mode>`. Default onboard sweeps
   require Program and Kernel eager for both runtimes (plus native HBG capture
   on A2/A3); framework sweeps require Kernel eager and their declared capture
   groups. Flag removal or rename of the only case satisfying a mandatory
   group, or a partition narrowed without CI-visible justification.
6. **Receipt invariants** (only when the PR edits the coverage machinery): an
   unexecuted case is never a pass; failure/incomplete summaries preserve
   source-checked partial counts; a retry counts a node ID once and a failed
   attempt cannot become a recovered pass.

Authoritative detail: `docs/ci.md` → "Feature coverage verification",
`simpler_setup/test_coverage.py` and `simpler_setup/test_environments.py`. To
verify the actual selection before flagging a collection gap, use the
`collection_inventory` collect-only command with the lane's explicit
`--coverage-partition`.

A missing declaration is a **blocking finding**, not an optional improvement:
the lane cannot collect until it is fixed.

## Step 7.7: A5 Device Stack-Realignment Check (repo-specific)

Trigger this step when the diff adds or edits anything under `src/a5/**` that
runs on the device. The device compiler has a known stack-realignment defect
([simpler#2973](https://github.com/hw-native-sys/simpler/issues/2973)): a stack
object with a large alignment can get inconsistent address fixups once its
address escapes into an outlined call, so a predicate may read a shifted object
and report an error that never happened. The source looks wrong to nobody, the
build is clean and unit tests pass; only an A5 run shows it.

Point the scanner at the build directory of the configuration under review:

```bash
./tools/a5-ir-check/a5_ir_check.py <build-dir>
```

Run it only where this session already authorizes building or testing (a
development task, or an explicit ask): the scanner compiles each device TU, so a
read-only review must not launch it. In that case read existing evidence whose
configuration matches the reviewed head, or record the scan as not run.

Record the exit status as evidence (0 clean, 1 findings, 2 tooling problem or an
incomplete scan — a `SKIP`ped TU must not read as clean).

- **A hit is a precondition, not a proven defect.** The object may never cross an
  outlined boundary at that build's level. Removal or an explicit justification
  is acceptable; silence is not.
- **A clean run covers only the scanned configuration.** With upstream `inline`
  nothing escapes and nothing is reported, so run it against the build the
  configuration under review actually produces. Escapes routed through
  `phi`/`select` are not traced at all, so a clean result is not a proof.

Do not treat the scanner as evidence about hardware: it cannot see the
miscompile, only the shape that exposes it. When the PR claims a fix for a
device-functional failure, the A5 run remains the only signal.

## Step 8: Optional Cross-Check with External CLI Reviewers

**Opt-in only.** Skip this entire step unless the user's `/review-pr`
arguments explicitly include `codex` or `claude`. The
cross-check is slow (3–5 minutes per CLI), noisy, and unnecessary for
small or routine PRs — your own analysis from Steps 5–7 is the review
by default.

The main agent's Steps 1–7 review still runs; Step 8 adds independent second
opinions. Ignore the external token that names the current harness: `claude`
under Claude Code, `codex` under Codex. The model knows its harness from the
system context; no shell detection is needed. Record ignored tokens in
Appendix C. Do not feed the main agent's analysis into an external reviewer's
prompt.

The target and flags were parsed once in Input. Use the resulting
`$WANT_CODEX` and `$WANT_CLAUDE` from `.docs/review/state.env`; do not
re-parse the invocation arguments here.

Each section checks its own binary. If the user opted in but the requested
binary isn't on PATH, note it in Appendix C (Independent Reviewer Notes) and
proceed without it. If no external reviewer remains after ignoring the current
harness, skip the rest of Step 8 entirely.

The CLIs run as **independent** second opinions — each runs in its
own process and does not see your analysis. Run them in parallel
(single message, multiple Bash calls). A missing binary, non-zero
exit, or empty capture file is treated as "skip with note" — same
as if the user hadn't asked for it.

### Common setup: output capture

Both CLIs take minutes on real PRs (codex 3–5 min, claude comparable).
Write each one's output to a fixed scratch path under `.docs/review/`
— that directory is already in the repo's `.gitignore` (`# Mid-work
documentation`), so the files persist across the review session for
inspection but never reach a commit. Do not pipe through `tail`:
line buffering hides progress until the process exits.

```bash
. .docs/review/state.env
CODEX_OUT=.docs/review/codex.out
CLAUDE_OUT=.docs/review/claude.out
PROMPT_FILE=.docs/review/xcheck-prompt.txt
if [ -n "$PR_NUMBER" ]; then
  GOAL_SRC=$(gh pr view "$PR_NUMBER" --repo "$PR_REPO_OWNER/$PR_REPO_NAME" \
    --json title,body -q '.title + "\n\n" + .body')
else
  GOAL_SRC=$(git log --format='%s%n%n%b%n---' "$LOG_RANGE")
fi
cat >"$PROMPT_FILE" <<EOF
You are an independent PR reviewer working read-only in a git worktree.
Read the change with \`git diff $DIFF_SPEC${SCOPE:+ -- $SCOPE}\`. The working
tree may not match the reviewed head: read other files with
\`git show $HEAD_REF:<path>\` and search with \`git grep <pattern> $HEAD_REF\`.
${SCOPE:+Report only findings about changes under $SCOPE; read outside it only to verify them.}

User-stated goal (may be empty):${GOAL_TEXT}

PR title/body or commit messages:
$GOAL_SRC

Report — terse bullets only, each with file:line:
- correctness bugs you can verify by reading the diff and the files
- scope mismatches between the stated goal and the actual diff
- missing tests for any bugfix in the diff
- design and code-quality concerns supported by the changed code and the
  relevant existing contracts or analogous implementations; consult
  .claude/skills/review-pr/references/quality-review.md
- performance risks, clearly distinguishing measurements from hypotheses
Skip style nits and skip anything you cannot verify against the code.
EOF
declare -p CODEX_OUT CLAUDE_OUT PROMPT_FILE >> .docs/review/state.env
```

The prompt carries only the diff spec, scope and stated goal — never the main
agent's findings. Do **not** inline `$(git diff …)` into it; that hits
`ARG_MAX` on large PRs. Each CLI runs `git diff` itself.

These paths overwrite on each run. If you want to keep prior output
for comparison, rename before re-running.

No wall-clock timeout is applied. The caller (you, while running this
skill) is expected to monitor progress with `wc -c "$CODEX_OUT"` /
`tail "$CODEX_OUT"` and `kill` the process if it hangs (e.g. a CLI's
backoff loop on a 503 from the model proxy). Hardcoded timeouts in
the past killed legitimate runs that simply needed more time.

Both reviewers run under an enforced read-only boundary, not a prompt
request: repository content they read can contain instructions, so the
boundary must hold even if the reviewer follows them.

### Codex

Run this section only when this session is not Codex and `$WANT_CODEX=1`.
If this session is Codex, skip it and record "codex ignored: current session
is Codex" in Appendix C.

`codex exec -s read-only` runs the shared prompt in a read-only sandbox, so
one invocation covers every target mode. (`codex review --base` diffs the
current checkout, which is not the reviewed head for a fixed range or a PR
reviewed without checkout.)

```bash
. .docs/review/state.env
if [ "$WANT_CODEX" = 1 ]; then
  if command -v codex >/dev/null 2>&1; then
    codex exec -s read-only "$(cat "$PROMPT_FILE")" >"$CODEX_OUT" 2>&1 \
      || echo "[codex skipped: exit $?]" >>"$CODEX_OUT"
  else
    echo "[codex skipped: not on PATH]" >"$CODEX_OUT"
  fi
fi
```

### Claude

Run this section only when this session is not Claude Code and `$WANT_CLAUDE=1`.
If this session is Claude Code, skip it and record "claude ignored: current
session is Claude Code" in Appendix C.

`--setting-sources ""` loads no user, project or local settings, so no allow
rule from them can widen the session; `--tools` limits it to read/search tools
plus Bash, `--allowedTools` pre-approves only read-only git subcommands, and
`--permission-mode dontAsk` denies every other call instead of prompting:

```bash
. .docs/review/state.env
if [ "$WANT_CLAUDE" = 1 ]; then
  if command -v claude >/dev/null 2>&1; then
    claude -p \
      --setting-sources "" \
      --tools "Read,Grep,Glob,Bash" \
      --allowedTools "Read,Grep,Glob,Bash(git diff:*),Bash(git show:*),Bash(git log:*),Bash(git grep:*)" \
      --permission-mode dontAsk \
      "$(cat "$PROMPT_FILE")" >"$CLAUDE_OUT" 2>&1 \
      || echo "[claude skipped: exit $?]" >>"$CLAUDE_OUT"
  else
    echo "[claude skipped: not on PATH]" >"$CLAUDE_OUT"
  fi
fi
```

Never add a permission-bypass flag to either command. This is a separate
process with its own context, not the current session.

Read the captured output with `cat "$CODEX_OUT"` and `cat "$CLAUDE_OUT"`
— do **not** pipe these through `tail` while the process is still
running; you will block on buffer flush and see no progress.

### Synthesis rules

- **Treat each CLI's output as one independent reviewer's notes**, not
  ground truth. Verify each claim against the actual code before
  including it.
- **Agreement across reviewers** (your read plus any external reviewer that
  actually ran) raises confidence — surface those findings prominently.
- **Disagreement** is worth investigating: if codex says a line is
  buggy and you don't see it, re-read that line before dismissing.
- **Reviewer hallucinations** (claims about code that isn't in the
  diff) get dropped silently — do not pass them to the user.
- **Scope:** when `$SCOPE` is set, drop findings about changes outside it
  and list them as dropped in Appendix C.
- **Same-model caveat:** the harness-matching token is ignored, but a
  different harness can still use the same model family (for example, an
  opencode session backed by Claude with the `claude` token). Treat
  same-family agreement as weaker evidence than a genuinely different model;
  verify every claim against the diff and state the limitation in Appendix C.
- If a CLI is unavailable or errors out, note it once in the final
  review and continue.

### Verification before surfacing

External reviewers routinely confuse "code in the file I'm reading" with
"code this PR added" — the most common hallucination class. Before
including any external finding in your final review, run the matching
check:

1. **Tie the finding to the PR diff.** Identify the changed lines or behavior
   in `git diff "$DIFF_SPEC" -- <path>` that introduce or worsen the issue.
   Existing callers, contracts, or duplicate implementations outside the diff
   can supply evidence, but an unchanged pre-existing defect is not a finding
   against this PR. When unchanged code no longer conforms to an interface or
   contract the diff changed, anchor the finding to that changed line and keep
   it.

2. **For "missing test for bugfix" claims, verify the alleged bugfix is
   in your diff.** Take a distinctive substring from the change the
   reviewer cites and search the reviewed head's history:

   ```bash
   git log -S "distinctive substring" --oneline "$HEAD_REF" -- "path/to/file"
   ```

   If the introducing commit is outside `"$LOG_RANGE"`, the
   "bugfix" was merged earlier — the reviewer is hallucinating PR
   authorship. Drop it. For a diverged two-dot range, a hit on the
   base-side commits listed by the Input warning is a reversed base
   change, not reviewed work.

3. **For "race condition" / "thread safety" claims, identify the
   specific interleaving.** If the reviewer can't name two concurrent
   call paths and the shared state between them, the claim is
   speculative. Drop it.

4. **For "memory leak" claims, identify the missing free.** A claim
   without a concrete "allocated here, never freed on path X" trace is
   speculative. Drop it.

5. **For design or performance claims, verify the criterion and impact.**
   Name the violated contract, dependency, duplicated implementation, or
   measured workload. A personal preference is optional; an unmeasured speed
   claim is a hypothesis, not a confirmed regression.

Only findings that survive these checks may be surfaced in the review
itself — they enter the §4 issue list like any of your own findings.
Step 9's Appendix C (Independent Reviewer Notes) then records, per
reviewer, both the findings you surfaced and the ones you dropped (one
line each) so the human can audit your filtering.

## Step 8.5: Optional Internal Sub-agent Cross-check

**Opt-in only.** Run this step only when the arguments include `subagents`
(Input parsed `$WANT_SUBAGENTS`). Without it, stay single-agent and
record "subagents not requested" in Review limits.

This is the host's native subagent path, separate from the external `claude`
CLI process in Step 8. A `claude` token does not enable this step, and a
`subagents` token does not launch the external CLI.

If the host cannot spawn read-only sub-agents, note "subagents unavailable" in
Appendix C and continue with your own review.

When enabled, spawn two independent, read-only reviewers in parallel:

- **Standards reviewer** — read `.claude/rules/*.md`, `CLAUDE.md`, and the
  developer/testing guides; inspect the changed files and report rule
  violations with `file:line`. Treat tool-enforced checks as already handled.
- **Spec reviewer** — read the stated goal, linked issue/design doc, and PR
  body; inspect the diff and report missing, partial, scope-creep, or
  incorrectly implemented goal items with `file:line`.

Bounded brief for each:

- Read-only: do not edit, checkout, build, test, run hardware, post to GitHub,
  or invoke external CLIs.
- Inspect `$DIFF_SPEC` (limited to `$SCOPE` when set) and read code at
  `$HEAD_REF`; read callers and the definitions the changed code depends on
  when needed to verify a claim.
- Return findings with severity (Must/Should/Consider), `file:line`, a concrete
  failure example for each Must, and the evidence state (code inspection,
  diff-only, or not run). State "no findings" explicitly.
- Cap the report at 10 findings; rank correctness and contract issues first.

Merge with the Step 8 "Synthesis rules" and "Verification before surfacing":
verify every claim against the diff, drop hallucinations, and record surfaced
and dropped findings in Appendix C. Count findings per axis in Review coverage;
merge them into §4 by severity without letting one axis mask the other.

## Step 9: Write the Review

Order the review by **reader logic** — background/problem → solution →
benefit/cost evidence → residual → verdict — not by the order the analysis
steps ran. Sections 0–5 are a fixed frame: fill each one, in order, and put
each piece of analysis output in the section named below — the frame
exists precisely so the review cannot drift into free-form commentary.
The mechanical outputs (change breakdown, pin check, independent-reviewer
notes) go in the appendices so they support the narrative without
interrupting it. Everything Steps 3.5–8.5 produced still lands somewhere
below — nothing is dropped, only reordered.

**Trivial-PR exception**: a trivial PR (single-file, < ~50 changed
lines) may collapse the frame to §0, §4, and §5 — §0 then carries the
problem, solution, and expected benefit concisely; §4 records any material
quality concerns or missing evidence. Anything larger fills
every section, in order. Review coverage (in §0) and Review limits are
kept at every size. The appendices follow their own rules
regardless of size.

### 0. Summary

One or two lines: *this PR addresses pain P by method M, with benefit B and cost C;
remaining R; verdict: approve / request changes / needs discussion.* After
this line the reader knows whether they need to read further.

**Review coverage** — three short lines, not a second report:

- **Spec:** which sources were checked (user contract, issue/design doc, PR
  body, commits) and which goal items remain uncovered. If no spec exists, say
  "no spec available".
- **Standards:** which `.claude/rules/` areas were checked and whether any
  violation was found; say "clean" when none.
- **Limits:** the reviewed base/head SHAs and a pointer to Review limits,
  which holds the full list.

### 1. Background and Problem Statement

Written for a reviewer who does **not** know this corner of the
codebase. Two layers, in this order:

- **Background first.** What the relevant mechanism does today and why
  it has its current shape — the minimum context (from your Step 5.5
  reading of the surrounding code) without which the problem statement
  reads as jargon. The test: after the background paragraph, the
  reader must be able to see *why this problem can arise at all*.
- **Then the problem.** What is wrong or missing in the current
  behavior. Merge Step 4 (stated goal) here: state the goal in your
  own words, and if the goal sources disagree — user description vs.
  linked issue vs. PR body, especially a goal downgrade — that
  discrepancy is the **headline of this section**.

**A concrete example is mandatory:**

- Bugfix → the triggering input/state and the wrong behavior ("with
  input X in state Y, output is Z instead of W").
- Feature → one concrete use case impossible before and possible
  after (ideally a short code/command snippet).
- Refactor/chore → a snippet of current code that shows the pain.

### 2. Solution and Rationale

Three questions, all answered here:

- **What the method is** — the Mechanism Brief (Step 5.5): mechanism,
  central abstractions, key design choices. Depth scales with PR size
  exactly as Step 5.5 prescribes — required over ~500 lines or 3+
  files; recommended otherwise; omittable only for trivial PRs
  (single-file, < ~50 lines).
- **Why this method solves the §1 problem** — the causal argument
  from mechanism to problem. This absorbs the type-specific
  correctness reasoning from Step 7 (bugfix: how the changed lines
  restore correct behavior under the trigger; feature: how it fits
  the existing design and extension points). Also: why this shape
  rather than the plausible alternatives (Feature checklist #8).
  The routing rule for all of Step 7's output: checklist items that
  yield design reasoning (needed-ness, design fit, blast radius,
  boundary and error-path walkthroughs, the concurrency model) are
  woven into this section; items that yield findings go to the §4
  issue list.

- **What improves and what it costs** — evaluate the expected benefit from
  Step 4.5 using the quality analysis from Step 6. Improvements in architecture,
  coupling, performance, and maintainability contribute directly to benefits;
  degradation in those dimensions contributes directly to costs. Explain who
  gains and who bears each cost, including work shifted to other phases or
  maintainers. Separate observed results from expectations, cite measurements
  for speed or resource claims, and state unverified gaps.

**Reuse the §1 example as a before/after**: show the behavior or code
path with and without the PR. One example threaded through §1 and §2
costs the reader one mental context, not two. This is the section
that proves you understood the PR; omitting it on a large PR makes
the rest of the review look like nitpicking.

### 3. Implementation–Claim Consistency and Benefit/Cost Evidence

A solution narrative can read as valid while the diff does not deliver
it. This section checks the implementation against the PR body and
commit messages, in **both** directions:

- The combined goal, quality, and benefit/cost table (Steps 5.7 and 6), rendered as a
  **markdown table plus footnotes — never reformatted into a
  vertical list or card layout**, no matter how long the content
  is. Cells hold short phrases only; the detailed explanation of any
  ⚠️ / ❌ / ➕ row goes into a numbered footnote (`[1]`, `[2]`, ...)
  directly below the table, referenced from the Assessment / evidence cell.
  ❌ rows = claimed but not implemented (incomplete work); ➕ rows =
  extra behavior or obligations outside the stated scope. Both are must-discuss.
  Reviewer-observed quality benefits and costs are labeled as observations,
  not attributed to the author as claims they never made.
- The stated-vs-real mismatch classes from Step 5 (stated as bugfix
  but adds functionality, stated as refactor but changes behavior)
  are reported here — never silently papered over. When the stated
  and real goals match, say so explicitly in one line: the reader
  must be able to tell "checked and consistent" from "not checked".
- Test coverage as claimed: does the PR body's Testing section match
  what the diff and CI actually ran? For a bugfix, does a regression
  test exist that fails pre-fix and passes post-fix (Bugfix checklist
  #4)? A gap found here is recorded and carried into §4 as an issue.

In that same table, cover every dimension from Step 6 relative to the baseline.
State the benefit, cost, or lack of material change, including both gains and
losses within a dimension when applicable. Mark missing evidence as **Not
verified** and irrelevant dimensions as **Not applicable**, with the reason.
Support conclusions with concrete code/contracts/tests/measurements. Related
goals and dimensions may share a row when their relationship remains clear;
reference shared evidence rather than counting one improvement twice. Keep
cells short, put details below, and link actionable costs to §4. "Not verified"
does not establish either a benefit or the absence of a cost.

### 4. Residual Issues and Risks

Three sub-lists; give each entry a concrete example where possible:

- **Uncovered cases** — ❌ traceability rows and stated-goal parts not
  done; name an input/scenario that still fails after merge.
- **New risks/costs introduced** — ➕ rows, architecture or coupling
  regressions, concurrency or ABI hazards, performance/resource regressions,
  maintenance weight from churn. Include deliberate trade-offs from §3.
- **Follow-up work** — TODOs, deferred items, docs to sync.

Then the severity-ranked issue list:

- **Must fix**: bugs, correctness issues, security problems,
  unresolved goal-vs-code mismatches, ❌ / ➕ traceability rows, or
  demonstrated violations of required architecture/data contracts or agreed
  performance budgets. **Every Must-fix carries concrete evidence:** a failing
  input/interleaving, a reproducible budget violation, or the required design
  rule plus the changed dependency/layout and its impact. "There is a race"
  without an interleaving or "this is inelegant" without an objective criterion
  is not a Must-fix. An unresolved design preference belongs in discussion.
  Every Must/Should finding also carries `file:line` and an evidence state:
  code inspection, diff-only, or not run. A hypothesis without evidence is a
  Consider item or explicitly "needs verification", never a confirmed defect.
- **Should fix**: style violations per `.claude/rules/`, missing
  tests for bugfixes, doc/comment drift, ⚠️ traceability rows with
  user-visible impact, string-matching error translations across
  boundaries, missing version stories on new ABIs, avoidable coupling or
  redundancy with demonstrated maintenance cost, and missing evidence for
  material performance claims.
- **Consider**: suggestions, optional improvements, ⚠️ rows that
  only affect future maintainability.

### 5. Merge Recommendation

Not a bare word — an explicit chain over the sections above:

> problem is real (§1) → method solves it (§2) → implementation
> and benefit/cost evidence hold (§3) → benefits justify the costs and
> residual risks (§4), with required contracts intact →
> **approve** (list follow-ups)

Decision rule:

- Real problem + working method + claims-matching implementation + benefits
  justify the costs + residual is only acceptable follow-up → **approve**.
- Any Must-fix, an unexplained goal-vs-code mismatch, or an
  unresolved ❌ coverage row (§3) → **request changes**.
- Method works but a design choice is contested, a goal downgrade
  has not been confirmed with the author, or an unresolved ➕
  coverage row still awaits the author's rationale →
  **needs discussion**.

Explain why each material cost is justified, must be reduced, or requires
agreement. Functional gains do not automatically compensate for architectural
degradation or tighter coupling, and no benefit cancels a required contract
violation. Do not invent a numerical score that hides these trade-offs.

If the chain breaks, say at which link and why — that link *is* the
verdict's justification. State whether each material unverified benefit or
quality concern blocks the recommendation or can be an explicit follow-up.
Do not treat functional tests as evidence for unmeasured performance or
unexamined architecture; do not block solely on personal taste.
The oversized-PR warnings from Appendix A
feed in here: Core churn > 1000 lines biases the verdict toward
"request changes / needs discussion" unless the author justifies the
size.

### Review limits

State what this review did not cover: files or directories outside the
requested scope, tests/builds/hardware not run, external CLIs or sub-agents
skipped, a missing or partial spec, and whether the reviewed head changed
during the review. The default is review-only; name which opt-in extras were
requested and which were not. Do not present a limit as a pass.

### Appendix A: Change Breakdown

From Step 3.5. Render the bucket table (Core / Build / Test-Ex / Docs /
Uncategorized, each with file count and +/−/= churn, plus TOTAL) so the
reviewer can see at a glance how big the PR is and how much churn is
mechanical vs. real logic.

Apply the oversized-PR warnings from Step 3.5 at the top of this
section:

- **TOTAL > 1000 lines** → `⚠️ Oversized PR`: review burden and
  merge-conflict risk are high; recommend splitting by concern. May
  still be approvable.
- **Core > 1000 lines** → `❌ Oversized PR (core logic)`: stronger
  signal — the PR cannot be reviewed in one pass. Recommend split
  *and* require the Mechanism Brief (§2) plus a commit review order;
  the §5 decision rule picks this warning up.

Both warnings quote the actual numbers (e.g. "Core 1180 lines").

If **Uncategorized** is non-trivial, list what landed there and
sanity-check it — it may be mis-bucketed real logic (living outside
`src/`/`python/`/`simpler_setup/`) or out-of-scope files the reviewer
should question before approval.

### Appendix B: pto-isa Pin Check

From Step 7.5. **Always rendered when the pin is active** (`PIN_ACTIVE=1`), at one of two severity levels chosen by whether a pto-isa header reference changed — the level is the whole point of this section, so state it explicitly in the heading:

- **No header-reference change** (`PTO_REFS` and `NEW_PTO_INC` empty) → **advisory**. A light, always-on nudge that the pin exists; does not by itself block the merge. Render as a single info line, e.g.:

  > ℹ️ **pto-isa pin:** `pto_isa.pin` is pinned to `<sha>`. No pto-isa header references changed in this PR — confirm the pinned commit is still adequate; bump + rebuild onboard `a2a3` `host_runtime.so` (`SIMPLER_PTO_ISA_BUILD_COMMIT`) only if needed.

- **Header-reference changed** (`PTO_REFS` or `NEW_PTO_INC` non-empty) → escalate to **recommendation**. Surface a visible reminder and mirror it as a **Should-fix / check** row in the §4 issue list so it cannot be missed when writing the §5 Merge Recommendation:

  > ⚠️ **pto-isa pin check:** this PR changes how it references pto-isa while `pto_isa.pin` is pinned to `<sha>`. Verify the pinned commit still provides every pto-isa header the PR references — a **changed pto-isa include path** (e.g. `pto/npu/...` → `pto/...`) is the strongest hint a bump is needed; a newly added pto-isa include is a weaker hint. If the pin is bumped, rebuild onboard `a2a3` `host_runtime.so` against the new commit (`SIMPLER_PTO_ISA_BUILD_COMMIT`).

  List the triggering signals verbatim (changed pto-isa include paths / newly added pto-isa includes). `PIN_TOUCHED` (the PR edited `pto_isa.pin`) is reported alongside, at whichever level applies, with a note to verify the new SHA and the `host_runtime.so` rebuild.

When the pin is not active, emit the one-line skip note and move on.

### Appendix C: Independent Reviewer Notes

A short subsection per independent reviewer (codex, claude, or an internal
Standards/Spec sub-agent) listing:

- which of their findings you verified and surfaced above
- which you dropped, with a one-line reason each (using the
  "Verification before surfacing" checks from Step 8)

The dropped-list is part of the review's audit trail — do not omit it
when there were dropped findings. A reader needs to see what filtering
you did.

Record any external token ignored because it names the current harness:
`claude` under Claude Code, `codex` under Codex. When an external reviewer
does run, note whether it shares the main reviewer's model family; if so,
agreement is weaker independent evidence.

## Common Pitfalls

1. **Stale local main** — Always `git fetch upstream` before computing
   merge-base. Never use local `main` directly.
2. **Squash-merged upstream commits** — If upstream squash-merges, the
   merge-base may include commits that look like they are on the PR
   branch. Verify with `git log --oneline "$LOG_RANGE"`.
3. **Cross-fork PRs** — No checkout is needed: Step 2 fetches
   `pull/<N>/head` from `upstream` and reviews that SHA. Use
   `/checkout-pr` only when you will go on to change the branch.
4. **Large PRs** — Read diffs in chunks by directory. Do not try to
   read the entire diff in one tool call if it exceeds ~2000 lines.
   For `codex` and `claude` in Step 8, pass only `$DIFF_SPEC` and let
   each CLI run `git diff` itself in the worktree — never inline
   `$(git diff …)` into the prompt argv (hits `ARG_MAX`, ~128 KiB on
   Linux).
5. **Silent goal rewriting** — Never paper over a stated-vs-real goal
   mismatch by rephrasing the stated goal. The mismatch is the finding.
   Equally: never silently *accept* the PR body's framing of the goal
   when other sources (user description, linked issue, design doc) give
   a broader or different target. Step 4 ranks user description above
   PR body for exactly this reason — authors routinely downgrade their
   own goal description, and a review that adopts the downgraded version
   approves a PR that doesn't do what was intended. When in doubt, ask
   the user "is X the goal, or is Y?" before judging.
6. **Trusting external reviewers** — `codex` and `claude` hallucinate.
   Verify every claim against the diff before surfacing it.
7. **Misconfigured CLI ≠ unavailable** — a CLI binary can be on PATH
   but fail every call (no model route, expired token, upstream 5xx
   retry loop). The `command -v` gate alone won't catch this. Monitor
   `wc -c "$CODEX_OUT"` / `wc -c "$CLAUDE_OUT"` while the call runs;
   if a process produces nothing for several minutes, `kill` it and
   treat the empty file as "skip with note" — same as a missing
   binary. Avoid hardcoded timeouts: codex review legitimately takes
   3–5 minutes on real PRs and a tight timeout kills good runs.
8. **Piping CLI output through `tail`** — line buffering means `tail`
   only flushes at EOF. A stuck process with active stderr looks like
   silence and breaks the empty-output heuristic. Always capture to
   a file with `>"$CAPTURE" 2>&1` and read post-hoc.
9. **Substituting a different diff for a fixed range** — When the user
   supplies `A...B`, use it verbatim. Do not recompute a merge-base, and do
   not use a remote PR diff in its place. If the range is invalid or empty,
   stop and report; do not silently fall back.
10. **Treating a sub-agent's report as verified** — Internal reviewers
    hallucinate like external CLIs. Run every finding through the Step 8
    verification checks before surfacing it; record dropped findings in
    Appendix C.
11. **Letting one axis mask the other** — Report Standards and Spec coverage
    separately in Review coverage. A clean Standards pass does not excuse a
    missing spec item, and a correct spec implementation does not excuse a
    hard-rule violation.
