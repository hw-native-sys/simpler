---
name: fix-issue
description: Implement a GitHub Issue as its assignee or delegated contributor, selecting direct repair or develop-task planning and following through to a reviewed PR. Use when asked to fix or work on an Issue; use process-issue for maintainer triage.
---

# Fix an Issue

## Establish the task and owner

Use [GitHub setup](../../lib/github/setup.md) for repository, account and remotes.
Read the latest Issue body, comments, assignee, state, related design, children,
dependencies and linked PRs. Check current main and overlapping open/draft PRs
before starting another implementation. Resume existing authorized work where
appropriate. For a closed Issue, reconcile its resolution with the request: an
explicit follow-up may proceed, but do not silently reopen or redo completed work.

- If the user is the assignee, continue within the agreed scope.
- If unassigned, follow maintainer/module-owner routing from
  [process-issue](../process-issue/SKILL.md). Explicit delegation or permission to
  claim may authorize self-assignment; do not auto-assign every unowned Issue.
- If another person owns it, preserve that assignment. Explicit delegation or
  independent investigation may proceed with clear scope and integration ownership;
  otherwise clarify before competing implementation. Read-only investigation can
  still establish what is needed.
- Assignment failure is not a successful claim. Report it; separately authorized
  implementation may continue without pretending the business owner changed.

If personal WudiTask coordination is enabled, use its installed skill to acquire
or verify this execution before writing, and recheck before publication. Preserve
the task/run identity through downstream skills. Do not require team-wide locks,
a second business registration or separate claims for child agents.

## Decide whether to use develop-task

Choose based on the actual work, not merely that it came from an Issue:

| Situation | Execution path |
| --------- | -------------- |
| Bounded repair, known approach, focused validation | Implement directly with the focused skills below; no mandatory DAG or subagents |
| Several dependent deliverables, material design choices, useful authorized parallel work, or substantial continuation across sessions | Use [develop-task](../develop-task/SKILL.md) to align design, plan dependencies and integrate results |
| Existing accepted plan or active develop-task execution | Resume it and preserve approvals; do not start another manager or replan from scratch |
| Initially small work reveals wider dependencies or unresolved architecture | Escalate to develop-task with evidence already collected |

A missing include with a clear build failure normally goes direct; a resource-lifetime
change across backends normally needs `develop-task`. Neither path mandates a
particular Plan Mode API. Ask about unresolved material scope/design decisions,
not routine implementation already authorized. State the chosen path and why briefly.

On handoff, pass Issue identity, ownership, acceptance, approvals, overlap findings
and any personal execution identity. `develop-task` then owns execution; do not
have it invoke this entry again for the same established context.

## Implement and validate

For direct work, state intended behavior, likely cause and focused validation.
Follow `CLAUDE.md` and applicable rules. Use an isolated branch/worktree based on
current `$BASE_REF`, or resume the established task branch. Inspect worktrees before
creating or switching branches; never reset a branch another writer holds.

Reproduce defects where practical, make the bounded change and use
[testing](../testing/SKILL.md) for appropriate checks. Keep one meaningful delivery
per PR; do not split one owner's work into trivial child Issues or PRs to populate
a DAG. If scope changes, reconcile acceptance and design before claiming completion.
Documentation-only work needs document checks, not runtime tests.

## Deliver and follow through

Use [git-commit](../git-commit/SKILL.md) and [github-pr](../github-pr/SKILL.md) within
the user's publication authorization. Describe final behavior, evidence and limits;
preserve the repository's single final commit convention.

Link the Issue in the PR. Use `Fixes #N` only when the PR fully satisfies that Issue
and closure on merge is appropriate. For partial delivery, parent tracks or pending
post-merge acceptance, use `Related to #N` and state what remains. Do not automatically
add closing keywords to every Issue reference.

Use [fix-pr](../fix-pr/SKILL.md) for authorized feedback and CI follow-through.
Report readiness for the exact current head. The maintainer's
[process-pr](../process-pr/SKILL.md) owns review routing and authorized merge;
creating a PR does not authorize merging it or taking over that role.

Update confirmed progress in the Issue body and design discussion in comments when
authorized. Reread before editing, preserve concurrent changes and verify writes.
Report separately what is implemented, published, reviewed, merged and accepted;
do not close a parent Issue because one delivery PR landed.
