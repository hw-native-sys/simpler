# Delegate and consume results

## Reserve an assignment

Use the user's available/approved backend. Before any spawn, save an assignment
with task/item/attempt identity, exact cwd and base, final prompt, permitted writes,
inputs, expected report, validation limits and approval references. Check the latest
base/PR overlap against that final prompt. A saved assignment is intent, not proof
of a started worker. If launch acknowledgement is lost, inspect the original
attempt before retrying; never create a second writer merely because a call timed out.

One writer owns each overlapping file scope. Native agents may share the filesystem,
Git refs and device pools; separate directories alone do not reserve hardware or
prevent logical conflicts. Use isolated worktrees for overlapping code changes,
review immutable commits/artifacts, and integrate deliberately. Never reset a branch
another worktree owns. Independent reviewers may repeat analysis when useful or
explicitly requested, but must not silently compete as authors.

## Choose execution from the current environment

Inspect the available host tools, installed agent versions and the user's backend
preference at the time of execution. Choose native subagents, a local Claude/Codex
CLI or another explicitly authorized backend accordingly. Reuse existing host
facilities; this skill does not require a launcher, wrapper or fixed command line.

For native subagents, use the host's actual spawn, message, wait and notification
APIs. Save the returned handle with the assignment and candidate report so the
main agent can follow up. Keep pending, delivered and acknowledged feedback
distinct. A successful send does not show the fix was applied. If the child did
not save its report, the main agent can persist the received response, identified
as a manager-saved copy.

For external agents, inspect the installed CLI's current help, authentication and
local provider configuration without exposing secrets. Derive the invocation from
that evidence instead of copying a historical command. Use safe argument passing
and a supported prompt-input mechanism. Do not guess model flags, credentials or
session IDs, or add permission-bypass flags. Reuse an existing configured entry
when appropriate; keep tokens and shell configuration out of the repository and
prompts.

Use the host's process/session management and log facilities to retain enough
information to distinguish launch intent, a confirmed start, terminal outcome and
the actual result. Choose the record format and monitoring method for that host.
Keep separate attempts distinguishable and avoid overwriting earlier evidence.
A zero exit is only candidate success, not proof every descendant stopped writing.
Missing terminal evidence means unknown; reconcile the original host session and
artifacts before allowing another writer. A PID alone is insufficient identity,
particularly after restart or a move to another machine.

## Main-agent completion loop

While active, prefer completion notifications when supported; otherwise use the
host's wait or bounded polling facilities at an appropriate cadence. On each meaningful result:

1. Recheck user steering, authority and attempt identity. Persist the candidate.
2. Confirm the writer stopped, or keep its scope reserved. Verify the actual diff,
   allowed scope, base, tests, evidence and limitations; read errors even on exit 0.
3. Accept, request rework, or record the concrete blocker. Rework goes to the original
   author/session if safe, in a new attempt for external commands. If the original
   writer is active, send feedback instead of launching a competing author.
4. Integrate accepted artifacts, run relevant integration checks, update the plan
   and launch authorized independent successors. Do not wait for unrelated siblings.
5. Recheck pending results before yielding. No runnable item is not proof the task
   is complete: distinguish running work, unresolved decisions, merge waits and blockers.

Before compression, switching backend or ending a session, checkpoint current
approval, assignments, author handles, artifacts, pending feedback, commands/tests,
unknown operations and next actions. Do not reuse an unrelated idle session.

## Wake and recovery limits

Completion notifications are useful while the main session is active or waiting.
Do not assume a notification can start a new turn after that session ends.
Choose host-supported, authorized wake mechanisms when available and verify
the intended session; queue acceptance is not proof the main agent resumed. Without
verified wake support, explicitly report **manual resume required** and the record
location. On resume, consume saved results before redispatching. Routine unchanged
polls need no user messages or GitHub comments.

For a different machine, a local path is insufficient: include accessible artifact
locations and fingerprints in the private handoff. Late results are candidates to
review, never permission to overwrite a newer accepted result. Preserve uncertain
old work and use an isolated checkout for takeover.
