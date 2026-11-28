# PR hardware test commands

On an open PR targeting this repository's `main`, post one command:

| Command | Hardware tests | Aggregate check |
| ------- | -------------- | --------------- |
| `/run a5` or `/run-a5` | A5 UT and onboard scene tests | `ci/run/a5` |
| `/run a3-network1` or `/run-a3-network1` | Daily's A3 two-machine Network1 suite | `ci/run/a3-network1` |

Surrounding whitespace is accepted. Only maintainers and admins can
admit hardware work; edited comments, extra arguments and other commands do
not admit tests. A substring prefilter may start the small hosted router when
someone quotes an alias; the Python controller checks the complete command
before any dispatch or status write.

## Execution and feedback

The controller follows PyPTO's `/ci full` protocol:

1. `ci-command.yml` validates permission, the PR head and GitHub's test merge.
   It pins the head, base and test merge before queueing. If any of them changes
   before dispatch, post a new command for the new snapshot.
2. It writes a pending request marker on the admitted head before dispatching
   the selected workflow from trusted `main`, then binds the returned run ID before
   releasing the queue. A failed pending write prevents dispatch.
   If GitHub rejects the dispatch, the request's check reports failure and links
   to the command logs. If the response is lost, the controller looks up that
   request and leaves its marker available for reconciliation.
3. The A5 workflow verifies the dispatch and merge, then calls the existing
   `_ut-npu-a5.yml` with CANN examples and `_st-npu-a5.yml` with DFX smokes.
   Its `a5-result` finalizer requires preparation and both suites to succeed;
   a skipped or cancelled required job never passes the aggregate result.
   The Network1 workflow uses the same admission and calls `_st-network1.yml`
   with platform `a2a3` on the Daily runner pool. Its result requires the two-machine suite to pass.
4. `ci-command-status.yml` publishes the selected command’s aggregate check.
   `ci-command-progress.yml` directly mirrors suite progress under
   `ci/run/<command> / <suite>` with job-log links. Both statuses are informational;
   repository merge rules are unchanged.

Progress appears in PR Checks. Command validation/dispatch failures receive a
comment with the reason and command-log link. There is no separate running
comment to maintain. Head, base and tested merge are recorded in the source
run's summary; its title identifies the PR, head, base and request attempt.

Repeated commands reuse an already running or passing run for the same PR
head and current base. If main advances, the next command tests the new merge
even while an earlier run is in progress. A failed run can be requested again. Native failed-job
reruns preserve earlier successful dependencies. Re-running comment admission
is rejected; post a new comment to admit another revision. A full native test
rerun rechecks the original pinned inputs; it cannot silently move to another
head. Hardware jobs for the same PR and suite queue across revisions and reruns.
Retrying an older run leaves newer tests running.
Pushes do not automatically request the A5 sweep. Automatic PR and Daily
coverage, including framework validation, are unchanged.

## Ownership and recovery

Command admission and aggregate publication use the same short writer queue.
Progress has a separate queue, so a long monitor cannot block admission or
final aggregate publication. The queue keys use command plus head SHA because
commit statuses are shared by every PR pointing to that commit. The latest
admitted request owns the shared statuses; each run retains its own tested
merge and logs.

A pending marker and its bound run/attempt protect against an older publisher,
including when run searches lag or the final aggregate write fails. Publishers
verify source workflow, repository, default branch and run identity. They read
Actions job metadata rather than PR files, comments or artifacts. Status URLs
and request markers locate the owner; verified workflow metadata establishes
its identity. A manual run must pass the authorization step before taking over
the aggregate or suite statuses. If metadata lookup fails, only a run with
a matching status binding can report a failure. Otherwise the current status
is preserved for reconciliation.

Progress rereads persisted statuses after API failures, including a write whose
response was lost. It retries five times with exponential backoff, shares a
500-request/hour polling budget, reserves 200 API requests for aggregate writers,
and waits for quota reset when capacity is insufficient. The watch window is
five hours; an `always()` notification requests final aggregate reconciliation
on exit. Metadata pagination fails closed at its documented API bounds.

To recover without running hardware tests again, manually dispatch
**CI Command Status** on `main` with the command and its existing `run_id`. Dispatch
**CI Command Progress** with the same inputs to resume suite visibility for the current attempt. The
finalizer explicitly notifies both workflows because `GITHUB_TOKEN` dispatches
may not emit downstream `workflow_run` events reliably.

## Source and adaptation

The Python helpers and command/status/progress workflow structure are adapted
from [PyPTO commit c84b5866](https://github.com/hw-native-sys/pypto/tree/c84b5866b7cbc11b8bdfd35a15f7368a8a3f1f3d/.github).
The shared API, ownership, rerun reconstruction, routing, publication and polling
contracts retain that implementation and its regression tests. A5 adaptations
are its maintain/admin permission policy, workflow/job names, pinned test merge,
status labels and the commit-wide ownership key. PyPTO's lint prerequisite and
branch-distance gate are not A5 policy.

`.github/ci-commands.json` supplies aliases, workflow names, required suites and
status labels. A new command adds its profile and workflow calls while sharing
`ci_control.py` and `ci_progress.py`; its admission policy must be explicit.
Control jobs check out helpers from `github.workflow_sha`, and hardware jobs
receive read-only credentials and only the validated merge checkout.

The workflows and their helpers must exist on `main` before comment
commands can use this protocol. Local mocked-API tests validate control logic;
end-to-end delivery and hosted token permissions still need validation after
deployment.
