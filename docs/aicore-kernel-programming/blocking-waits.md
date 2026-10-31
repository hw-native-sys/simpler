# Blocking waits and task progress

This contract applies to core tasks submitted through the A2/A3 and A5
`tensormap_and_ringbuffer` orchestration API. A core executes a kernel call
without preemption. Its pending dispatch bank stores the next task; it is
not a second execution resource. A kernel blocked in `pto::comm::TWAIT` does
not return to the executor to pick up that next task or handle its EXIT signal.

## Supported waiting patterns

| Task pattern | Caller obligation and runtime guarantee |
| ------------ | --------------------------------------- |
| Ordinary task | The task must be able to finish without work whose progress it prevents. The runtime does not infer blocking operations from its kernel binary or scalar arguments. |
| Waiter with a producer that can finish first | Declare producer-to-waiter ordering in the task graph. A `DEP_WAIT` edge keeps the waiter from executing before the declared producer reaches runtime completion. This is local task ordering, not proof that a remote signal arrived or that its value is visible. |
| Declared completion-first waiter | Pass `BlockingWait::after_producers(...)` to a TMR submit overload. The runtime adds `DEP_WAIT` for each named local predecessor and prevents this waiter from occupying a running, pending or gated core slot before its logical dependencies are ready. The caller must establish that completion of those predecessors suffices to release the kernel's wait. |
| Explicit deferred completion | Register a supported completion condition and return from the kernel. The core becomes available after FIN; the task's consumers remain gated until all its kernels have returned and all registered conditions are satisfied. The notifier must be able to run independently of that deferred task's completion. |
| Intentionally resident waiter | The caller must establish that all work needed to release the wait can progress using resources the waiter cannot prevent from running. There is no generic reservation, co-residency, or cross-device admission guarantee for an arbitrary set of tasks. |

The unsafe placement is possible even when the submitted graph is acyclic:

```text
W occupies core C and waits for P's notification
P is pending on C and cannot execute until W returns
```

The missing semantic W-to-P wait is not represented in that graph. Changing
task order, leaving one arbitrary core free, or serializing all waits does
not prove that every producer and its own dependencies can run. A producer
can need a different core type, a whole cohort, or another device.

For a one-way protocol whose producer can finish before the waiter starts,
the existing explicit-dependency API provides a conservative path. Given
populated `producer_args` and `waiter_args`, with no other unfinished WAIT
predecessors:

```cpp
producer_args.set_allow_early_resolve(false);
const auto producer = rt_submit_aiv_task(producer_kernel_id, producer_args);
const TaskId dependencies[] = {producer.task_id()};
const DepFlags kinds[] = {DEP_WAIT};
waiter_args.set_dependencies_with_kinds(dependencies, kinds, 1);
rt_submit_aiv_task(waiter_kernel_id, waiter_args);
```

The arrays must outlive the submit call. `DEP_WAIT` provides ordering only;
it does not retain storage allocated by the producer. Use the tensor access
tags or the retaining `set_dependencies` form (`DEP_WAIT | DEP_RETAIN`) when
the waiter also needs that storage to remain alive. A scalar containing a
signal address does not automatically create a tensor dependency.

Disabling the producer's hint prevents its early fanin propagation to consumers.
Set it before submitting that producer. For a waiter with multiple predecessors,
apply this restriction to every relevant unfinished WAIT producer; disabling only
an already completed or unrelated producer does not constrain the remaining
flagged predecessors. Dependency reduction preserves the restriction on an
unflagged edge. This approach also changes early eligibility for the producer's
other consumers.

### Completion ordering and early core occupancy

A normal `DEP_WAIT` edge gates execution, but a flagged producer can permit
its consumers to occupy gated core slots before it logically completes. This
matters when the producer's kernels have returned but deferred completion still
needs another task: a gated consumer can hold the cores that task needs.
Turning off the waiter's own `allow_early_resolve` hint does not prevent that
placement; the hint describes this task as a producer of later tasks.

Use the completion-first submit overload when the waiter must not reserve a
core before the named predecessors complete:

```cpp
producer_args.set_allow_early_resolve(true);
const auto producer = rt_submit_aiv_task(producer_kernel_id, producer_args);
const TaskId completion_predecessors[] = {producer.task_id()};
rt_submit_aiv_task(
    waiter_kernel_id, waiter_args,
    BlockingWait::after_producers(completion_predecessors, 1)
);
```

The same descriptor works with `rt_submit_task`, `rt_submit_aic_task` and
`CoreTaskArgsWithDeps`. Its producer array is borrowed only until submit returns.
The named edges are ordering-only; tensor tags or explicit retaining dependencies
must separately keep any required storage alive. Existing dependencies are
merged with the named WAIT edges, and transitive reduction may remove a direct
edge while preserving its completion order. The waiter's own early-resolve hint
still applies to its successors after the waiter is admitted.

This guarantee concerns core occupancy before logical readiness. Submission
still consumes task-window and output-storage resources. It does not validate
notifications, predicates, remote progress, visibility, signal generations or
the completeness of the caller's release-predecessor list. A skipped producer
can complete without notifying; the declaration must account for that protocol.
Every named id must come from the current local orchestration run. The runtime
rejects invalid, out-of-range or not-yet-submitted ids before allocating the
task; equal-valued ids from different runs cannot be distinguished by `TaskId`.

An empty/null declaration, an unknown policy, or `BlockingWait::resident()` is
rejected with `SIMPLER_ERROR_INVALID_ARGS` before task allocation or publication.
This submit entry provides no new resident resource policy. Existing group and
cohort interfaces retain their own contracts. Old unannotated submissions retain
their run-to-completion obligations and existing early-dispatch behavior.

The overload uses an independently bound, versioned extension. Old task
arguments and the legacy runtime ops layout are unchanged. A new orchestration
SO on an old runtime can use the old submit overloads; the new overload reports
unsupported admission instead of silently using the legacy submit operation.
An old orchestration SO continues to use its legacy binder on a new runtime.

### Deferred completion

When a kernel can return before a notification arrives, the existing
`get_async_ctx(args)` and `save_expected_notification_counter(ctx, addr, expected)`
API registers an unsigned 32-bit counter condition (`value >= expected`).
Split any computation that reads the notified data into a separate consumer
with a completion dependency on the registering task. The
[notification-counter scene](../../tests/st/worker/comm_domain/async_notify/kernels/orchestration/async_notify_orchestration.cpp)
shows this separation. Registration does not suspend and resume the kernel:
calling a blocking `TWAIT` after registering still occupies the core.

Keep the counter storage valid until the deferred task completes, and ensure
that its generation and notification/data visibility satisfy the protocol.
The runtime accepts at most 64 registered conditions per task and 64 deferred
tasks per runtime wait list. Exceeding these bounds is an error, not a
reservation for additional waiters. A producer that depends on the deferred
task's completion still forms a logical cycle. Releasing an AICore also does
not imply that the enclosing Chip task or its parent Worker is available for
another submission.

An early-resolve hint permits runtime-controlled gated staging; it is not a
declaration that a kernel cannot block, and it does not make an undeclared
communication dependency safe. `require_sync_start` describes the known
blocks of one task and rejects a cohort larger than the available resource
count. It does not identify or reserve an arbitrary producer closure.

Some cooperative algorithms require concurrent resident participants, so
forcing producer completion before every wait is not a general replacement.
For example, the [group-reservation scene](../../tests/st/worker/collectives/group_reservation/README.md)
deliberately lets one worker wait while an independent worker releases it.
Worker-level group reservation and a core task's `sync_start` are different
mechanisms; neither establishes a general cross-device wait-cycle policy.

## What can be rejected or diagnosed

The core task submission API carries task dependencies, launch shape,
predicates and timing attributes. It does not declare the event, comparison
or notifier used by a hidden kernel `TWAIT`. Explicit deferred completion
does carry a condition address, expected value and completion type; it does
not identify the notifying task or its producer dependency/resource closure.
The runtime does not automatically convert a blocking wait into deferred
completion. An undeclared blocking wait cannot be reliably identified or
rejected before admission. The completion-first declaration validates its
explicit local predecessor range and admission policy; it does not identify a
hidden event or provide a general hidden-wait-cycle detector.

For explicitly registered deferred conditions, the existing `ASYNC_WAIT`
warnings can report the condition address, expected value and observed
counter value. They do not identify the notifying task and cannot describe
an unregistered wait inside a running kernel.

An incorrectly admitted waiter can still stall. The scheduler timeout reports
the lack of progress; it is not guaranteed cancellation of the blocked kernel,
nor proof that its resources can be freed or its Worker reused safely. Keep
ordinary failure and teardown error results distinct.

For TMR, [stall diagnostics](../troubleshooting/device-error-codes/stall.md)
include owner-local publication records for running and pending banks. They
identify task ids, core ids, dispatch tokens, banks and logical block indices
without following task pointers. An owner that reaches the tensor-data timeout
fatal branch also reports its own records at WARN before calling emergency
retirement. An owner that exits on the completion flag may leave no such record.
Use the available records to identify a pending candidate behind a blocked
running task, then map the kernels' actual signal accesses.
The report explicitly marks implicit wait metadata as unavailable: it cannot
name the awaited event or its producer, or prove that two recorded tasks form
a cycle. A task id is scoped to its run; dispatch tokens can wrap, and neither
is a global cross-device identity.

## Relation to the blocking-wait progress contract

The supported patterns implement the task-progress contract requested by
[#1798](https://github.com/hw-native-sys/simpler/issues/1798):

| Requirement | Supported behavior |
| ----------- | ------------------ |
| Ordinary tasks run to completion | The caller must not depend on work whose execution the task prevents. |
| Blocking tasks declare dependencies or a supported progress policy | Use completion-producing WAIT dependencies with the conservative producer hints, the completion-first declaration, or explicit deferred completion. |
| An unmet wait does not prevent its release work from obtaining a core | Completion-first admission leaves running, pending and gated core slots unoccupied until logical readiness; deferred kernels return to free their cores while their logical completion waits. |
| Fail early or provide actionable diagnostics | Known unsupported or malformed declarations fail before allocation/publication. Registered conditions and owner-local placement records report the metadata the runtime actually has. |

The resident policy at this submit entry is unsupported and rejected; specific
group and cohort interfaces keep their own contracts. The runtime does not infer
MoE-specific event semantics from a kernel binary. A caller's undeclared or
incorrect notification relationship can still violate the contract, and the
diagnostics do not claim a complete hidden-event/notifier graph. Reproducing and
attributing a particular historical multi-device failure is separate from these
supported submission guarantees.

Dependency capture includes the declaration's added WAIT edges in its existing
explicit-edge records. It does not record the completion-first target policy,
so replaying that graph alone does not reproduce speculative admission choices.
