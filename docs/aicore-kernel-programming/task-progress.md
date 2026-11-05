# TMR task dependencies and progress

This page applies to the A2/A3 and A5 `tensormap_and_ringbuffer` runtimes.
Their core tasks execute non-preemptively: once a kernel occupies an AICore,
the scheduler cannot suspend it to run another task. An ordinary task must run
to completion without depending on work whose execution it prevents.

The communication library or compiler must express notification dependencies
in the orchestration. Task arguments describe declared dependencies and launch
attributes; they do not identify a kernel's hidden wait event, notifier, signal
generation or cross-device visibility. Submission order does not supply those
semantics. A blocking `TWAIT` therefore needs a protocol whose release work can
execute under the submitted dependencies and available resources.

## WAIT dependencies and early resolve

A `DEP_WAIT` edge gates execution on the producer's logical completion. When
the producer allows early resolve, the runtime can stage its consumer on gated
core slots before the producer logically completes. A kernel that has returned
physically can still be logically pending on a deferred notification condition.
If that task's notifier needs the occupied slots, early staging can prevent it
from running even though the staged consumer's kernel has not executed.

For a one-way protocol that permits producer completion before the consumer
starts, a dependency-only dummy provides a local completion barrier:

```cpp
// producer_args and waiter_args contain the kernel's arguments.
const auto producer = rt_submit_aiv_task(producer_kernel_id, producer_args);

CoreTaskArgs guard_args;
guard_args.set_allow_early_resolve(false);
const TaskId producer_ids[] = {producer.task_id()};
const DepFlags ordering[] = {DEP_WAIT};
guard_args.set_dependencies_with_kinds(producer_ids, ordering, 1);
const auto guard = rt_submit_dummy_task(guard_args);

const TaskId guard_ids[] = {guard.task_id()};
waiter_args.set_dependencies_with_kinds(guard_ids, ordering, 1);
rt_submit_aiv_task(waiter_kernel_id, waiter_args);
```

The dummy has no AICore work. It becomes ready after its WAIT predecessors
logically complete and completes through the dependency-only queue. Its
disabled early-resolve hint prevents the consumer from entering early core
staging. The producer's hint for its other consumers and the consumer's own
hint for its successors remain available. The dummy must depend on the actual
completion-producing work; an unrelated or invalid task id is not a
notification dependency.

The example has one producer and no other explicit dependencies on the consumer.
Preserve existing dependencies when constructing the lists. Give the dummy all
required WAIT edges when several tasks produce the release. WAIT-edge reduction
preserves the early-dispatch restriction of an unflagged predecessor even when
a direct edge becomes redundant. The dummy consumes one task-window entry and
its dependency capacity, but no execution core.

An alternative is to disable the relevant producers' early-resolve hints and
attach ordinary WAIT edges. That affects those producers' other consumers too.
Changing only the consumer's own hint controls its successors; it does not
prevent that consumer from being staged early.

## Dependency and storage lifetime

Use valid task ids from the current orchestration run and the appropriate scope
and handle lifetime. Dependency arrays must outlive the submit call. `DEP_WAIT`
orders completion but does not retain producer storage: tensor access tags or
retaining dependencies (`DEP_WAIT | DEP_RETAIN`) must keep data and notification
storage alive. A scalar containing an address creates no tensor dependency.

The graph must identify the actual release work. These dependencies do not
validate a notification predicate, remote progress, memory visibility or signal
generation. A predicated producer can complete without sending a notification;
the communication protocol must account for that behavior. Depending on work
that itself needs the consumer to execute can create a logical cycle.

## Completion and coordination at other levels

### Deferred notification completion

A kernel that can return before a notification arrives can register a counter
condition with `get_async_ctx(args)` and
`save_expected_notification_counter(ctx, addr, expected)`. Computation that
consumes the notified data belongs in a separate completion-dependent task; see
the [notification-counter scene](../../tests/st/worker/comm_domain/async_notify/kernels/orchestration/async_notify_orchestration.cpp).

Registration does not suspend and resume the kernel. A subsequent blocking
`TWAIT` still occupies its core. The counter storage and the protocol's data
visibility and generation rules must remain valid until completion. Releasing
an AICore does not imply that the enclosing Chip task or its parent Worker can
accept another submission.

### Worker groups and resident participants

The [group-reservation scene](../../tests/st/worker/collectives/group_reservation/README.md)
coordinates independent workers. Worker-level reservation and a core task's
`require_sync_start` operate at different levels: the latter admits one fitting
task's blocks, not another task's or a remote producer's release work. Neither
is a general cross-device wait-cycle policy. A protocol with intentionally
resident participants needs an explicit resource guarantee for the complete
set of release work; ordinary task submission supplies no such general policy.

## Diagnostic boundary

Undeclared kernel waits have no automatic pre-admission identification or
complete event/notifier/resource-wait graph. Registered deferred conditions
and owner-local dispatch placement provide observations for correlation with
the submitted graph and kernel signal accesses. Their fields and limitations
are described in [stall diagnostics](../troubleshooting/device-error-codes/stall.md).
A timeout or a placement record does not prove a notification relationship,
safe cancellation, or that resources can be freed and a Worker reused.
