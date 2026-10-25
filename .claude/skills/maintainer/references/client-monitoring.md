# Run maintainer in different clients

Read this when starting, testing, stopping or recovering continuous maintenance.
The client supplies wakeups; [maintainer](../SKILL.md) supplies the decisions.
Use the capabilities actually exposed by the installed client, not its model name.
Check current client help and the official references below before relying on
version-sensitive scheduling behavior.

## Shared operating agreement

Before scheduling, resolve and persist:

- Repository/track/personal scope, PR target-branch filter and allowed actions.
- A stable local directory for the action table, evidence and pending decisions,
  readable by every invocation. Keep it outside disposable worktrees. Record its
  resolved path in the scheduled prompt; a relative path can point elsewhere on
  the next run. A cloud runner needs an explicitly accessible equivalent, not an
  assumption that it can read the workstation's files.
- The scheduler/session identity, cadence, next due time when available, last
  successful iteration and stop conditions. Reuse an existing monitor for the scope.
- The user-facing destination for decisions and the route for replies. Prefer the
  same chat when supported. For independent runs, record the user's actual reply
  against the proposed action before the next invocation may act. GitHub comments
  from other actors and scheduler permission settings are not user approval.

Use one writer for this table and write scope, including foreground handling of
user replies. Reuse personal execution coordination when available; otherwise
serialize invocations through the chosen scheduler/runner. A tick arriving during
an active run should be coalesced or skipped for a later reconciliation, not start
another writer. Record a changed decision before running a dependent action.

A suitable scheduled prompt, adapted with the resolved paths and scope, is:

> Read the maintenance agreement and persistent action table at the agreed
> location. Use the maintainer skill to perform exactly one iteration for the
> recorded scope and authority. Reconcile current evidence and pending actions,
> update the same table, and raise only new actionable findings or decisions.
> Keep unchanged/no-action items quiet. Preserve unassessed work and failures.
> Do not create another schedule or broaden authorization. Return after this
> iteration; the existing scheduler owns the next wakeup.

Do not embed an approval only in a transient run's conversation when the next run
starts fresh. Store proposal, user decision, scope and outcome together. If a
head/scope change invalidates the decision's assumptions, reassess it before acting.
A missing reply leaves that action pending, while independent work can proceed.

## Choose the client mechanism

### Claude Code

For session polling, supply both interval and explicit single-iteration instructions:

```text
/loop 10m Read the saved maintenance agreement and table; use maintainer for one iteration only. Do not create schedules. Notify only on new actionable findings or required decisions.
```

Resolve the agreement/table paths before using this example. Avoid bare `/loop`:
its built-in maintenance prompt may continue branch work or perform cleanup beyond
this scope. Fixed-interval loops use `CronCreate`; inspect `CronList` to verify the
job and cancel that specific ID with `CronDelete`. Natural-language list/cancel
requests are also supported. Do not assume interrupting one turn cancels a schedule.

The session process must remain running for fires. Current documentation describes
expiry and conditional restoration with `--resume` / `--continue`; self-paced loops
have different recovery behavior. Inspect restored jobs before recreating any.
Do not treat a backgrounded session as a durable system service. If work must
survive session exit, select an available Desktop/cloud scheduler or external runner,
then recheck file access, permissions and decision delivery for that environment.

Source: [Claude Code scheduled tasks](https://code.claude.com/docs/en/scheduled-tasks).

### Codex in the desktop app

Use the exposed scheduled-task/automation controls. Prefer a task returning to the
same chat for interactive maintenance; standalone tasks can start a new chat each
run. Explicitly select the project and single-iteration prompt, including the stable
state location. Local project tasks require the machine and app to remain running.

Discover available automation tools or UI before creating anything; do not invent
a tool name or write undocumented automation configuration files. Read back the
saved identity, destination and schedule, and inspect actual run history. If this
session exposes no management tool, explain the UI setup needed and mark automatic
startup unverified rather than substituting an unrelated product's scheduler.

Source: [Scheduled tasks](https://learn.chatgpt.com/docs/automations?surface=app).

### Codex CLI

`codex exec` runs one non-interactive task; it is not a recurring scheduler. Use an
existing external scheduler, such as a user timer or cron, to invoke the agreed
single-iteration prompt. Check `codex exec --help` on the installed version.

```sh
codex exec -C /path/to/maintenance-workspace \
  --output-last-message /path/to/maintenance-state/latest-result.md \
  'Read /path/to/maintenance-state/agreement.md and perform one maintainer iteration.' </dev/null
```

These are placeholders, not installation commands. Choose the required sandbox and
preauthorized access without broadly disabling approvals. A terminal session's PATH,
authentication and environment may differ from a timer's; verify them without
printing credentials. Close unused stdin so a scheduled call cannot wait for input
from its launcher. If resuming a conversation, use its explicit session ID rather
than `--last`, which might select unrelated work.

Source: [Codex non-interactive mode](https://learn.chatgpt.com/docs/non-interactive-mode).

Configure the external runner to prevent overlap, bound execution time, retain exit
status and deliver actionable results to the agreed user destination. A process
exit or output file does not by itself mean the user received a decision. If an
unattended run cannot obtain required input/access, persist the pending action and
return; never treat inability to prompt as permission. Do not install a new runner
when an existing scheduler already covers this scope.

## Verify startup, recovery and shutdown

A created schedule is only **configured**. Report monitoring as **verified running**
after at least two automatic invocations successfully read and update the same table,
with observable timestamps and outcomes. A manual first pass plus a saved job is not
two automatic invocations. Record which client, mechanism and scope were tested.

For a bounded smoke test, use a local fixture and isolated state with no GitHub
mutations. A temporary external timer may validate the CLI integration; it does not
prove a production cron service or desktop automation works. Keep fixture tests
separate from a real deployment's status. No permanent test harness is required.

Check the behavior, not just process launches:

1. On the first automatic iteration, inspect the table and a concrete pending action.
2. Let another automatic iteration read that table. Unchanged evidence must not
   re-propose the same decision. Add a fixture change and verify it is reconciled.
3. Record a user deferral through the agreed reply route. Verify the next iteration
   preserves it while processing an independent change.
4. Stop the scheduler and inspect its state. Check the table remains unchanged after
   the next expected fire, accounting for documented jitter. Cancelling future ticks
   does not cancel a running action: let it finish or explicitly stop it and mark
   uncertain outcomes for reconciliation.
5. Resume with the same state and inspect the existing schedule before replacing it.
   Reconcile the missed observation window once, without duplicate actions or writers.

Track these separately: configured, verified running, paused/stopped, and failed or
unverified. On failure preserve the last successful observation and unprocessed work;
a failed heartbeat must not advance the discovery checkpoint. State inability to
observe scheduler health rather than claiming the monitor will detect its own death.

Do not leave test loops running. Cancel only the test-owned job/process, retain the
small evidence report, and report remaining limitations. A client blocked by auth,
missing tools or unavailable scheduling has not passed; success in another client
cannot substitute for that missing evidence.
