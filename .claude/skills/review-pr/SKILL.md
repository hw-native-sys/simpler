---
name: review-pr
description: Review a GitHub PR, a fixed commit range, or commits on the current branch. Prioritize code quality, compare the code with its baseline, and self-audit each review pass against the review goal and all applicable requirements. Read-only by default.
---

# Review PR

## Mandatory rules for the reviewing agent

1. **Preserve the requester's review intent.** Use their instructions and prior
   context to establish the scope and stopping conditions. A short invocation
   such as `review 3305` sets the target and invokes the full review workflow.
   Base any decision to stop after one pass on the agreed completion conditions.
   Ask the requester to resolve any material ambiguity about when the review
   ends, including whether the Goal requires follow-through on later revisions.
2. **Start every review with a persistent Goal.** Create it for the agreed
   objective before inspecting the diff. Use authorization already granted.
   When explicit authorization is required and still outstanding, ask the
   requester before starting the review. Report unavailable Goal tooling as
   a blocker.
3. **Complete the agreed review scope.** Treat a first pass and any early
   finding as interim results while required work remains. Continue through
   the required checks, including follow-ups covered by the Goal, and run
   the review self-audit before delivery.
4. **Report every review item to the requester.** Give each item a separate
   checklist entry with its result and evidence or verification limit. State
   whether the review Goal was achieved and why. If work remains, name the
   open items and the blocker.

**Code quality takes priority over completing the feature as proposed.**
Verify functional correctness and judge how the design affects the next
supported change. When a routine feature needs extensive coordination,
trace the complexity to the requirements or the existing structure. Evaluate
a bounded refactor when the change compounds avoidable complexity. Use the
[quality guide](references/quality-review.md), baseline, repository rules,
and CI evidence.

## Start the review

### Scope and permissions

- Accept a PR number (with or without `#`), an explicit `A..B` or `A...B`,
  or the current branch when the target is omitted. With both a PR number and
  an explicit range, the range defines the diff and the PR supplies context.
- Parse target and option tokens before the user's goal text. The first other
  token starts that text, including any target-like words that follow it.
  `scope:<path>` limits findings to the change under that path. Read direct
  callers and dependencies as needed to verify them.
- `codex` and `claude` opt in to the named external CLI. `subagents` opts in to
  native read-only subagents. Load [independent review](references/independent-review.md)
  only when one of those options is requested. Record `gemini` as unsupported
  if supplied.
- By default, read code and GitHub evidence and fetch refs as needed. Checkout,
  edits, builds, tests, hardware runs, publication, and merge require separate
  authorization. For requested verification use [testing](../testing/SKILL.md).
  Route maintainer follow-through to [process-pr](../process-pr/SKILL.md) and
  author-side repairs to [fix-pr](../fix-pr/SKILL.md).

### Create the review Goal

Once the target and scope are known, create a persistent Goal to deliver an
evidence-based verdict after assessing the agreed scope and passing the
[review self-audit](#review-self-audit). Continue the existing Goal for
follow-ups within the same review scope while it is active.

Maintain a working checkpoint with the Goal, pinned base/head SHAs, changed call
paths, stated requirements, quality dimensions, and repository checks. Add
checks and evidence as you trace the change. Keep the checkpoint current through
follow-ups and context changes. Use a task-specific directory under `~/tmp/` for
scratch files as required by the workspace instructions.

### Pin the diff

| Target | Action |
| ------ | ------ |
| Explicit range | Resolve both endpoints to commits and use the supplied range semantics. Require a nonempty diff. For a diverged two-dot range, record the divergence and identify base-side reversals before assigning authorship. |
| GitHub PR | Read the base branch, head SHA, body, linked records, discussion, and checks. Fetch the current base and `pull/<number>/head` from upstream, verify the fetched head against GitHub, and diff from their merge-base. This also works for fork PRs. |
| Current branch | Fetch its intended upstream base and diff committed changes from the merge-base to the pinned `HEAD`. Record any excluded uncommitted changes in Review limits. |

Use the repository's [GitHub setup](../../lib/github/setup.md) when remote
resolution needs it. A failed fetch, invalid range, empty diff, or inconsistent
head must be resolved before analysis proceeds. Use the commit log for the
pinned range to check the actual commits and exclude unrelated base-side history.
Inspect existing context at the pinned revisions via `git show <sha>:<path>`
and `git grep <pattern> <sha>`. A clean checkout at the exact SHA can supply
that revision's files. Read large diffs by subsystem.

Before delivery recheck a PR's head. If it moved, reconcile the new diff and
rerun affected checks, or label the result with the reviewed head and its limits.

Use [change breakdown](references/repository-checks.md#change-breakdown) to
classify churn and flag oversized PRs. Use a brief review for a single file with
under about 50 changed lines and no concurrency, ABI, ownership, or security
impact. Review deeply for over about 500 changed lines, three or more files,
or any of those risk areas. Review depth scales the analysis and report.

## Inspect the change

### Read the baseline and requirements

Establish how the baseline works, the concrete pain it causes, and what success
would change for the caller or maintainer. Use an example such as a failing
input or interleaving, a newly supported use case, or a maintenance operation.
Check relevant design docs and investigations.

Read stated requirements in this order:

1. The user's goal and constraints.
2. Linked Issues and design contracts.
3. The PR title and body.
4. The commits in the pinned range.

Record material disagreements between sources. For example, if the user expects
end-to-end retry and the PR only supplies a helper, preserve both scopes and
explain the unmet behavior. If no coherent spec is available, record that limit.

Describe the actual behavior from the diff and its callers, then compare it
with the stated goal in both directions. Check for unmet requirements and
extra behavior or obligations. Capture each in the [review checklist](#3-implementation-and-evidence).
An independently observed quality improvement belongs in the quality
assessment when it fits the agreed scope.

### Inspect the code and callers

Start with the highest-risk changed paths. Apply **every applicable dimension**
in the [quality guide](references/quality-review.md), covering architecture,
coupling, reuse, performance and resources, data and ABI, concurrency and
lifetime, failure paths, tests, and documentation. Also assess simplicity,
naming, compatibility and change scope, and operability.
For each, compare the before and after, cite the evidence, and state the
benefit, cost, or lack of material change. Assess the actual effects of docs
and config changes too. Give a reason for each dimension that is inapplicable.

Apply the guide's checks for **security and privacy, build and dependency
integrity, rollout and data migration, numerical correctness, and
accessibility and internationalization** when they affect the change. Derive
further relevant criteria from the diff, callers, supported configurations,
and deployment boundaries even when the request omits them. Verify unfamiliar
or version-specific external contracts against primary sources and link the
evidence. Apply the repository's own constraints to the result.

For incremental changes, check [reuse and consolidation](references/quality-review.md#review-dimensions)
even when the new logic is correct. Find existing implementations, compare
their contracts, and evaluate where common behavior belongs. Check whether
[a bounded refactor is necessary](references/quality-review.md#when-an-extension-spreads-a-rule)
when an extension spreads an invariant across call sites or exposes a weak
type or interface. Independently check whether newly introduced types enforce
their intended constraints, including when their design was previously agreed.

Follow normal entry points and failure, cancellation, retry, and cleanup paths.
For shared code and protocols, check affected producers and callers, including
existing variants. Verify that new work fits their contracts under locks, on
hot or nonblocking paths, and at handoffs between layers. For concurrency and
ownership, identify who publishes and observes each state, the happens-before
relations, and when reuse and release are allowed. Tie a race or resource finding
to a specific interleaving or allocation and cleanup path.

For a deep review, describe the mechanism in the report. Cover the central
types and state transitions, concurrency and ownership, cross-boundary
contracts, extension points, and realistic alternatives. Explain the costs
and benefits with the same example used for the baseline. For a large diff,
include a practical commit or subsystem review order.

Apply the checks that match the change:

- **Bugfix:** Verify the trigger, why existing coverage missed it, and how the
  fix restores correctness. Check for a regression test that fails before the
  fix and passes after it. Name the evidence source.
- **Feature:** Check the real use case, reuse opportunities, affected runtimes
  and platforms, and unfinished paths. For stable external contracts, review
  layout, validation of reserved fields, and evolution rules. Apply the
  repository's current-only contract for internal formats. Trace error codes
  to callers and check coverage of concurrency and contention.
- **Refactor:** Verify caller-visible behavior preservation. Treat any
  behavior change as part of the review's actual scope.

Distinguish CI status from tests that actually ran. Verify the PR's test claims
against selected cases, platform, revision, and skips. Qualify performance
claims using the baseline and candidate, workload, metric, timing boundary,
and data. Record the missing evidence and its effect on the verdict when an
important claim remains open.

### Check repository rules

Record **Standards** separately from **Spec**. Check the relevant
[repository rules](../../rules/), [CLAUDE.md](../../../CLAUDE.md),
[developer guide](../../../docs/developer-guide.md), and
[testing guide](../../../docs/testing.md). Keep quick checks scoped to the
changed files. Cite the rule, changed line, and impact for each violation.
Hard contract violations are Must-fix. Tool-enforced formatting can rely on
the relevant check. Record when the standards inspection is clean.

Apply these repository checks alongside the source review:

| Check | Trigger and evidence |
| ----- | -------------------- |
| [PTO-ISA pin](references/repository-checks.md#pto-isa-pin) | Read the pin at the reviewed head. Record the advisory or recommendation, the added or relocated includes, and any pin change. |
| [Feature coverage](references/repository-checks.md#feature-coverage) | A scene case, coverage implementation, or onboard/framework workflow changes. Verify declarations, actual selection, and applicable execution evidence. |
| [A5 stack realignment](references/repository-checks.md#a5-device-stack-realignment) | A device source under `src/a5/` changes. Use matching scanner evidence or record the limit. The scanner compiles device translation units and needs build authorization. |

## Review self-audit

After each pass, compare the findings and checkpoint against the review goal,
the full quality guide, and the applicable requirements here. In particular:

- Reconcile all changed call paths, goals, quality dimensions, Standards,
  and triggered checks. Record the evidence and its revision. Check that the
  baseline comparison and each material conclusion use that evidence.
- Verify the changed lines behind each finding, its concrete effect, and its
  severity. Apply the same check to findings from any independent reviewer.
- Use **Not verified** where inspected code needs missing test, build, or
  measurement evidence. Use **Not reviewed** for source or behavior that could
  not be inspected. Use **Not applicable** with the reason. State the open
  judgment for each evidence or inspection limit.

Investigate every gap that available evidence can close, update the conclusion,
and repeat the self-audit before delivery. Finding a blocker settles the
recommendation under the [merge rules](#5-merge-recommendation). Finish the
remaining in-scope review and report all substantiated findings together.

Mark the Goal complete only when the report accounts for every review item,
the agreed scope is covered, and the verdict follows from the evidence.
If a blocker leaves any required work unresolved, report the confirmed findings,
the review items still open, and that the Goal was not achieved.

## Deliver the review

For a full review, use §§0–5 below. A brief review may condense the background
and solution. Include the review checklist, Goal outcome, review limits, and
applicable appendices at either depth.

### 0. Summary

Lead with the verdict and the quality findings or trade-offs driving it.
Name the reviewed base/head and summarize coverage of **Quality**, **Spec**,
and **Standards**, with the material evidence limits.
State **Review Goal achieved** or **Review Goal not achieved** and give the
reason. A fully substantiated **Request changes** verdict can satisfy the
review Goal. Report the change's acceptance criteria in the checklist.

### 1. Background and problem

Explain the baseline, concrete pain, intended outcome, and disagreements
between requirement sources. Include the triggering input, use case, or
maintenance example.

### 2. Solution and rationale

Explain the mechanism and why it addresses the example. Include relevant
alternatives, responsibilities, invariants, and the observed or expected
benefits and costs. Keep the design brief proportional to the review depth.

### 3. Implementation and evidence

Give the requester a Markdown checklist grouped under **Quality**, **Spec**,
and **Standards**. List each quality dimension, acceptance criterion, and
repository check individually. For each item, give its result, cite supporting
evidence or state what is missing, and explain the material impact. For code
changes, summarize the relevant difference from the baseline. Include items
with no findings and use the self-audit statuses for unverified, unreviewed, or
inapplicable items. Keep the author's claims and reviewer observations clearly
attributed.

### 4. Findings and risks

Include uncovered cases, introduced risks or costs, and worthwhile follow-ups.
Order findings by severity. For Must-fix and Should-fix give a `file:line`,
the evidence level, the triggering case or affected contract, and the impact.
Tie each finding to behavior introduced or worsened by the reviewed diff.

- **Must-fix** includes confirmed defects, security or hard-contract
  violations, unresolved requirement gaps, and demonstrated budget breaches.
  Show the failing condition, interleaving, violated rule, or measured breach.
- **Should-fix** includes material test and doc gaps, avoidable coupling or
  duplication with a concrete cost, and missing evidence for important claims.
- **Consider** includes optional improvements and hypotheses with a named
  verification gap.

### 5. Merge recommendation

Explain how the evidence and trade-offs justify the verdict.

- **Approve** when the change meets the goals and contracts and the residual
  costs and risks are acceptable. List applicable follow-ups.
- **Request changes** for any confirmed Must-fix, unexplained scope mismatch,
  or unresolved missing requirement. The first confirmed Must-fix settles the
  recommendation. Continue through the full review scope, close gaps from the
  [self-audit](#review-self-audit), and report all substantiated findings in one review.
- **Needs discussion** when the method can work and a contested design choice,
  deliberate scope reduction, or extra obligation needs agreement.

### Review limits and appendices

- **Review limits:** State the scope and pinned revisions, material coverage
  or validation limits, and any opted-in reviewer or verification that could
  not run. Distinguish a partial review from a completed review with known
  evidence limits.
- **A. Change breakdown:** Show the categories, totals, and applicable
  size warnings from the [change breakdown instructions](references/repository-checks.md#change-breakdown).
- **B. PTO-ISA pin:** Record the result specified by the
  [pin check](references/repository-checks.md#pto-isa-pin).
- **C. Independent reviewers:** When requested, record what ran or was skipped
  and which findings were confirmed or dropped. Follow the
  [synthesis instructions](references/independent-review.md#synthesize-the-results).
