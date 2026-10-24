---
name: develop-task
description: Plan and deliver a complete simpler task with design alignment, an internal dependency graph, parallel agents, reviewed results and PR follow-through. Use for substantial implementation or continuation; use the existing focused skills directly for small edits or review-only requests.
---

# Develop a complete task

The current session is the main agent. Keep GitHub Issues and PRs as the business
record; the local plan describes how one owner executes that work. Do not start
another autonomous manager or require a second task registration.

## Establish the agreement

Read `CLAUDE.md`, applicable rules, the current Issue/PR and the latest user
instructions. Check current main and open **including draft** PRs before planning
new implementation. Preserve existing approvals and limits; do not reopen an
approved decision merely because the session resumed.

Define the useful delivery, supported scope, normal/error/close behavior,
resource costs and observable acceptance. Internally review the concrete design
before asking the user to decide an unresolved material trade-off. Continue
unaffected work while that decision is pending. A short repair can go directly
to the existing implementation and test workflow without a DAG or subagents.

## Plan, delegate and integrate

For multi-step work read [planning](references/planning.md). Divide by testable
artifacts and dependencies, with one meaningful feature per PR. Different people
owning independently deliverable parts may need sub-Issues; one owner's internal
subtasks normally stay in the same Issue.

When delegation is useful and authorized, read [agents](references/agents.md).
Use native subagents or a verified local Claude/Codex CLI. Record assignments
before launch; isolate writers and shared hardware. A child's completion is a
candidate result. Review its actual diff, provenance and tests in the same working
turn before accepting it or releasing dependent work. Reuse its author for rework
when safe. Do not simply forward a Completed notification and stop.

Choose the plan format, board presentation and agent invocation from the current
task, user preferences and available host capabilities. The references describe
what information and behavior to preserve; they do not prescribe a storage schema,
renderer, launcher or fixed CLI command. Reuse suitable existing tools.

## Deliver and resume

Read [delivery](references/delivery.md) for review, CI, Ready, merge and recovery.
Use the repository's existing skills for each operation:

| Work | Existing workflow |
| ---- | ----------------- |
| Establish Issue context, only if not already supplied | [fix-issue](../fix-issue/SKILL.md) |
| Choose and run checks | [testing](../testing/SKILL.md) |
| Commit and publish a PR | [git-commit](../git-commit/SKILL.md), [github-pr](../github-pr/SKILL.md) |
| Review the implementation | [review-pr](../review-pr/SKILL.md) |
| Address feedback and failed CI | [fix-pr](../fix-pr/SKILL.md) |

When entered from `fix-issue`, reuse its established context and chosen plan. Do
not call back into it for the same task or create another main agent. Maintainer
triage and lifecycle management belong to [process-issue](../process-issue/SKILL.md)
and [process-pr](../process-pr/SKILL.md), not to child implementation agents.

These workflows retain their own permissions and project requirements. Design
approval alone does not authorize public messages, merging or unrelated changes.
Use authorization already present in the conversation rather than asking again.

## Optional personal coordination

If the user has enabled WudiTask coordination, use its installed `wuditask` skill
to check the current personal execution before dispatch and publication; preserve
its task/run identity and checkpoint link locally. Children do not claim separate
business tasks. If execution rights changed, reconcile before further writes.
This repository skill also works without WudiTask, but provides **no cross-machine
exclusion**. Never silently enable coordination or write WudiTask's storage format.

On takeover, preserve accessible evidence and isolate the new workspace if an old
writer may still run. Takeover does not require proof that the old process stopped,
and does not revoke writes already sent to GitHub. Recheck remote state before
publishing. This skill does not migrate or remove existing WudiTask installations.
