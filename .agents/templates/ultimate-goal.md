# Ultimate Goal Template

Use this template when a goal is too large for one PR, one Codex session, or one stable design
slice. The purpose is to make future dispatcher `/goal` sessions reproducible: the goal file defines
the destination and operating contract, while the dispatcher discovers and logs the concrete child
PR route during execution.

This template is generic. Copy the relevant sections into:

- `docs/in_progress/NN-<goal-slug>.md` for the umbrella goal file;
- `docs/in_progress/<goal_slug>/dispatch_log.md` for the dispatcher log;
- `docs/in_progress/<goal_slug>/work_preparation.md` for goal-specific read order and operating
  rules, when the goal needs more than the umbrella file;
- `docs/in_progress/<goal_slug>/shared_contracts.md` when workers need common schemas,
  vocabulary, directory layout, or artifact conventions.

Do not copy domain-specific examples from another goal unless they are actually part of the new
goal. The template should preserve the pattern, not the previous problem.

## When To Use Ultimate-Goal Mode

Use ultimate-goal mode when at least one of these is true:

- the desired outcome naturally spans multiple reviewable PRs;
- several Codex sessions should work on independent slices;
- a dispatcher must audit current state before it can name the exact child PR backlog;
- the work needs reusable contracts, worker prompts, progress reports, or dependency PRs;
- the final acceptance criteria cannot be proven by one narrow branch.

Do not use ultimate-goal mode for a small feature that can be implemented, verified, and reviewed in
one PR. Use normal PR-sized mode for that.

## How To Write The Umbrella Goal File

The umbrella file is a development contract, not a full implementation plan. It should make the
destination clear enough that a dispatcher can start `/goal`, audit the current repository state,
propose child PRs, launch workers, and keep human reviewers aligned.

Write it with these properties:

- **Outcome first:** describe the final capability, behavior, or artifact state.
- **Evidence discipline:** state what must be proven, cited, tested, traced, or reviewed.
- **Ownership boundaries:** explain which artifacts belong to the dispatcher, shared contracts,
  dependency PRs, and workers.
- **Dispatcher freedom:** allow the dispatcher to discover missing preparation as first-class goal
  progress.
- **Worker constraints:** state that workers may own child PRs but must not dispatch nested workers.
- **Durable logs:** require dispatch commands, child PRs, verification, merges, handoffs, and scope
  cuts to be recorded.
- **Acceptance over backlog:** final acceptance criteria should be stable; child PR sequencing can
  change as the dispatcher learns.

Avoid writing a giant fixed checklist that pretends the dispatcher already knows every child PR.
Instead, define the loop and the review evidence expected at each step.

## Umbrella Goal File Skeleton

```markdown
# <Goal Name>

## Purpose

This ultimate goal <creates/enables/removes/migrates> <final outcome>. The goal is larger than one
PR and should be delivered through a dispatcher-managed sequence of child PRs.

The final result should <describe what the repository can do or prove when the goal is complete>.

## Problem

Describe the current limitation, why a single PR is the wrong shape, and what pressure the goal
should apply to the repository architecture, examples, tests, or docs.

## Non-Goals

- <Behavior or scope that should not be solved in this ultimate goal.>
- <A tempting adjacent refactor or feature that should become a follow-up.>

## Source, Requirement, Or Evidence Discipline

State the evidence rules for this goal. Examples:

- source claims must cite primary docs;
- public contracts must have tests and docs in the same slice;
- generated artifacts must record their inputs and verification commands;
- mismatches must be recorded with failing command, fix, and passing command.

## Scope And Artifact Map

List the artifact families that the final goal may touch. Keep this as ownership guidance rather
than a frozen path list.

- `<artifact-family>`: <responsibility>
- `<artifact-family>`: <responsibility>

The concrete paths can change during implementation, but the ownership boundaries should remain
visible.

## Goal Mode And PR Slicing

This file is an ultimate-goal umbrella note, not a single-PR contract. Work proceeds through
dispatcher-managed child PRs:

- The dispatcher owns this umbrella note, child-PR sequencing, dispatch log, progress reports,
  shared-contract coordination, and final promotion into stable docs.
- Each child PR owns one coherent slice with its own branch, PR description, local verification, and
  merge decision.
- Workers may open and merge their own child PRs when their scope is satisfied and verification is
  recorded.
- Workers must not dispatch nested workers; they propose new child slices back to the dispatcher.
- Dependency PRs are required before adding reusable framework behavior or shared machinery that is
  not local to one child slice.
- Every dispatch, branch, PR, merge decision, scope change, and handoff is recorded in
  `docs/in_progress/<goal_slug>/dispatch_log.md`.

The dispatcher is expected to discover missing preparation while running the goal. Missing child
backlog entries, worker prompt templates, verification matrices, stale references, or current-state
audits are not blockers to starting `/goal`; they are valid first child PRs when they reduce risk
for the parent goal.

## Work Preparation And Operating Rules

Before launching workers, the dispatcher should ensure the goal has enough preparation for a new
Codex session to continue without private context:

- read order;
- branch and PR policy;
- identity policy;
- dispatcher-worker policy;
- logging policy;
- dependency PR policy;
- documentation policy;
- evidence policy;
- verification policy;
- debug or mismatch policy, when relevant;
- skill or reusable-pattern policy, when relevant.

## Dispatcher First-Run Contract

When a dispatcher starts `/goal`, its first milestone is to make the next child PR sequence
reviewable enough to run:

1. Read the goal file, dispatch rule, and goal-specific preparation files.
2. Audit current repository state against the acceptance criteria.
3. Update the dispatch log with the dispatcher session, audit summary, and proposed next child PRs.
4. Decide whether the next child PR is preparation/shared-contract work, a dependency PR, or a
   worker slice.
5. If launching a worker, record the exact objective, branch, allowed files, expected PR, and
   verification commands before running the Codex command.

This audit is real goal progress. Commit it through a child PR when it changes reviewable docs or
contracts.

## Worker Prompt Minimum Contract

Each worker prompt must include:

- parent goal file;
- worker role, explicitly not dispatcher;
- no nested dispatch;
- child slice objective and target maturity;
- branch name and expected child PR relationship to the parent goal;
- files or directories the worker owns;
- source, evidence, benchmark, trace, debug, progress-report, or other domain-specific
  requirements;
- exact local verification commands expected before handoff;
- handoff location, usually the dispatch log or a worker handoff file named in the log.

## Progress Reports

State how human-review progress should be reported. Include deck or report location, update
cadence, and which decisions the reports should make reviewable.

## Dispatcher Development Loop

The dispatcher repeats this loop until acceptance is met:

1. Audit current state.
2. Choose the next smallest child PR, worker, dependency PR, progress report, or scope cut.
3. Record the plan in the dispatch log.
4. Run local work or launch a worker.
5. Review the result, run verification, and merge when appropriate.
6. Update docs, dispatch log, and progress reports.
7. Re-audit before choosing the next step.

Each loop iteration should leave one durable outcome:

- a merged child PR;
- a dispatch log entry explaining why the next worker was launched or deferred;
- a dependency PR for newly discovered shared machinery;
- a progress report for human review;
- an explicit scope cut with follow-up criteria.

## Acceptance Criteria

- <Final acceptance criterion that proves the goal outcome.>
- <Final acceptance criterion for evidence and verification.>
- <Final acceptance criterion for docs/design or todo promotion.>
- <Final acceptance criterion for dispatch log completeness.>

## Review Questions

- <Decision that humans should make before or during the dispatcher run.>
- <Tradeoff that may affect child PR sequencing.>
```

## Dispatch Log Skeleton

```markdown
# <Goal Name> Dispatch Log

This log records dispatcher-worker activity for `<goal name>`. It is required review evidence; do
not rely on private terminal scrollback or unstated session memory.

## Logging Schema

Each dispatch entry must include:

- timestamp;
- dispatcher session or PR;
- worker id and objective;
- exact Codex command or script invocation;
- parent goal and child slice;
- branch name and PR URL or planned PR slot;
- allowed scope and files;
- dependencies and blocked assumptions;
- verification commands and results;
- merge decision and merge commit, when applicable;
- handoff summary and remaining gaps.

## Entries

### <timestamp> - <entry title>

- Dispatcher session or PR: <branch, PR, or session id>.
- Worker id and objective: <worker id, or "no worker dispatched" for dispatcher-owned work>.
- Exact Codex command or script invocation: <command, script path, or "not applicable" with reason>.
- Parent goal and child slice: <goal file and child slice name>.
- Branch name and PR URL: <branch and URL, or planned PR slot before creation>.
- Allowed scope and files: <paths or artifact families>.
- Dependencies and blocked assumptions: <dependencies or "none known">.
- Verification commands and results: <commands and observed results>.
- Merge decision and merge commit: <pending, merged with commit, abandoned with reason>.
- Handoff summary and remaining gaps: <what the next dispatcher or worker should know>.
```

## Worker Prompt Skeleton

Use this structure when the dispatcher launches a Codex worker:

```text
/goal
Parent ultimate goal: docs/in_progress/<goal-file>.md
Role: worker, not dispatcher. Do not dispatch nested workers.

Objective:
<One child-slice objective.>

Branch and PR:
- Start from updated main.
- Use branch: feat-<child-slice>.
- Open a child PR into main when the slice is verified.
- Keep the PR description linked to the parent goal and dispatch log.

Owned scope:
- <paths or artifact families this worker may edit>

Out of scope:
- <paths or behaviors this worker must not edit>
- Do not launch other Codex sessions.
- If new parallel work is discovered, record a proposed child slice for the dispatcher.

Required read order:
1. AGENTS.md
2. .agents/AGENT.md
3. .agents/coding-guidance.md
4. .agents/rules/ultimate-goal-dispatch.md
5. docs/in_progress/<goal-file>.md
6. docs/in_progress/<goal_slug>/dispatch_log.md
7. <slice-specific files>

Implementation requirements:
- <domain-specific requirements>

Evidence and logging:
- Update <dispatch log or handoff file> with scope changes, verification, PR URL, and remaining
  gaps.

Verification:
- <exact command>
- <exact command>

Completion:
- Commit with repository identity.
- Push branch and open/update PR.
- Do not describe the PR as ready or mergeable until local verification is recorded.
```

## Child PR Lifecycle

Each child PR should follow this lifecycle:

1. Dispatcher records the intended child slice in the dispatch log.
2. Dispatcher or worker creates a `feat-xxx` branch from updated `main`.
3. Implementation stays inside the allowed scope.
4. Verification is run and recorded in the PR description and dispatch log.
5. PR is reviewed and merged when its slice is complete.
6. Dispatcher records merge decision and remaining gaps.
7. Dispatcher re-audits before launching or opening the next child slice.

Do not batch unrelated child slices only to reduce PR count. The goal is forward progress with
reviewable evidence, not a single large branch.

## Completion Pattern

An ultimate goal is complete only when:

- acceptance criteria in the umbrella file are met or intentionally revised with human-visible
  rationale;
- all child PRs are merged, closed with reason, or converted into explicit follow-up goals;
- stable results are promoted into `docs/design/`;
- remaining gaps are represented in `docs/todo/`;
- `docs/in_progress/` no longer presents completed work as active;
- the dispatch log records final verification and completion status.
