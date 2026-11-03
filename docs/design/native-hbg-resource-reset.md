# HBG per-execution state restoration

[Execution and API](native-capture-common-contract.md) ·
[Resource lifetime](native-hbg-prepared-lifetime.md)

## Prepared information and mutable state

HBG preparation materializes a fixed graph, task arguments and publication
sources. RTS replay executes the recorded submissions without rerunning Host
orchestration. Prepared information survives each execution; mutable scheduling
state is initialized at the device boundary that excludes its previous writers.
There is no separate Host-submitted reset task or new AICore launch ABI.

| Region | Preserved information | Per-execution state |
| ------ | --------------------- | ------------------- |
| Graph/task descriptions | Task topology, arguments, tensor/scalar bindings, shape, attributes and initial task kind | Scheduling progress and temporary execution links |
| Shared task table | Used task count and prepared initial state for each task | Completion bytes, block claims, wake/early-dispatch lists and payload progress |
| Shared header | Layout and prepared bindings | Scheduler error bitmap, code and reporting thread |
| Runtime entry | Callable/code owners and prepared resources | Completion gate, initialization/classification counters and readiness flags |
| Handshake | Core identity/report payload for observation | Report-valid and READY markers after acknowledged retirement |
| Native result storage | Allocation and retained fault evidence | Current result publication and callback aggregation |

Only used task entries are restored. Inline-completed tasks retain their
prepared completed initial state instead of being treated as pending device
work. Weights and caller-owned persistent tensor contents are not workspace to
clear. Address/scalar changes and Host-control changes are preparation changes,
not per-execution reset operations.

## Entry and exit ordering

Every invocation requires prior users of shared execution state to have ended.
A callback count, report epoch or successful enqueue is not that physical proof.
Caller/graph dependencies must prevent simultaneous execution against the same
mutable state.

The affinity entry gate rendezvous identifies native callbacks and invokes one
initializer before releasing runtime participants. It resets invocation-private
entry coordination and begins native result aggregation. Runtime cleanup hands
its terminal selection to the native completion owner; the last callback
publishes after its own runtime work and local cleanup.

For A3 and A5 legacy HBG, device runtime attachment restores shared-header and
task progress before classification and dispatch. Queue slots and the AICore
mailbox use their own initializers; zeroing raw memory is not a substitute.
`SharedMemoryHeader::reset_execution_state` and the task/payload reset methods
preserve prepared arguments while clearing mutable progress and scheduler errors.

A5 resident HBG has a different startup path. Before READY, Core can read stable
configuration but must not consume a previous invocation's mutable error as its
own. A recorded context-wait timeout is reported only against the context made
available for this invocation. Initialization failure reports through its own
owner and retirement path. A bounded watchdog is not permission to overwrite a
context still used by a Core.

AICPU invalidates a Core's report and READY marker only after acknowledged
retirement establishes that the Core has finished the relevant report writes.
It publishes that invalidation with cache maintenance. Unacknowledged workers
retain their unresolved state. This protects serial reuse even when a recorded
binding reuses its report epoch; it does not turn an ACK into proof of complete
kernel exit or replace the inter-execution completion dependency.

## Functional results and fault evidence

`NativeInvocationCompletion` collects the actual callback contributions and the
runtime terminal selection. It does not authorize reuse of execution storage.
`NativeInvocationResultStorage` keeps the existing result prefix and a separate
`NativeInvocationEvidence` tail.

Cold preparation initializes that storage outside capture. Device entry
invalidates the current result publication before new production. Recorded
initialization must not erase the retained evidence tail on each replay.
Missing required terminal contributions, invalid handoff state and execution
errors are distinct from missing optional diagnostic detail.

Retained fault evidence survives healthy replays and per-invocation state reset.
Reading or retaining a copy is not acknowledgement. Only explicit owner
acknowledgement or final destruction of its backing ends retention, after
physical exit; this introduces no public acknowledgement API. It is not an
Execute/Skip latch: a later independent invocation does not consult it to decide
whether to run. Real execution dependencies, device health and unproved prior users
still constrain subsequent use.

Ordinary eager progress and successful result observation require no private
result D2H. A failed run may request optional structured detail after the required
observation. Retained recording backing reads replay evidence during external
resource cleanup; a failed read keeps the owner for retry. None of these reads
establishes physical completion or replaces the RTS execution error.

## Supported boundary

The [execution contract](native-capture-common-contract.md#support-and-qualification-limits)
states the platform and validation limits. A3 bounded replay and timeout cases
do not prove recovery from permanent hangs, missing handshake reports or absent
native callbacks. Acceptance of a retirement request is not completed retirement.
Unknown physical use continues to protect its backing.

The A3/legacy and A5 resident startup mechanisms are distinct; source changes or
CPU controls in both do not qualify A5 replay hardware. Host construction
statistics are not per-replay measurements. DFX recording, concurrent-model
execution and in-place parameter updates are not provided by this restoration
mechanism.
