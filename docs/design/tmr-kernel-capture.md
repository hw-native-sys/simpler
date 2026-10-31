# Native TMR execution and graph resource references

## One execution mechanism

Native TMR eager enqueue and RTS capture recording use the ordinary
`simpler_prepare_run` / `simpler_launch_run_on_stream` / `simpler_wait_run` /
`simpler_finalize_run` lifecycle. Both submit the same registered invocation,
device admission, Core/CPU launches and whole-operator join. RTS decides whether
the submitted commands execute or are recorded. Replay invokes those recorded
device commands without returning to Host preparation.

The feature builds on the unified registration and native eager contracts.
`simpler_kernel_capture_supported` reports support for TMR onboard builds;
HBG and simulation refuse graph resource retention. The legacy
`simpler_kernel_mode_launch` remains a refusing stub.

## Cold preparation and graph lifetime

Ordinary eager execution needs no graph reference. A graph caller acquires an
additional reference before capture so allocations, code publication,
registration and event arming finish outside the recording interval:

```text
register callable
prepare_run(context, run, callable, fixed arguments, configuration, descriptor)
retain_run_resources(context, run, caller_stream, &graph_reference)
RTS CaptureBegin(caller_stream)
launch_run_on_stream(context, run, caller_stream, no_predecessor)
RTS CaptureEnd(caller_stream, &model)
finalize_run(context, run)

RTS ExecuteAsync(model, caller_stream)
wait_run(context, graph_reference)
... repeat ExecuteAsync and ordinary wait ...

RTS Destroy(model)
release_run_resources(context, graph_reference)
```

The caller owns construction and destruction of the native model. Graph
retention supports one fixed TMR invocation/model on a borrowed kernel context,
pipeline slot zero, diagnostics off and the same serialized caller stream.
Another native execution cannot reuse that context while the graph reference
holds its fixed resources. Multiple models, cross-stream replay, graph updates,
HBG capture and public Worker adapters are outside this interface.

The original opaque Runtime storage can be reused after ordinary finalize.
An independent resource-reference handle keeps the address-stable prepared
Runtime, HostApi, workspace reservation, fixed argument image and code alive.
Both handles resolve to the same execution state; the reference is not a second
executor. Its handle can be used with ordinary poll/wait. A successful resource
release invalidates the reference. Lifecycle and registry calls on a context
are serialized by its caller.

After successful release, the caller discards the resource-reference handle;
it cannot be polled, waited on or released again. Failed release leaves the
reference valid for cleanup retry.

`launch_run_on_stream` rejects a null caller stream with `INVALID_STATE`
before querying the SDK or changing the prepared run. A recording stream
without cold-retained resources also returns `INVALID_STATE`; this is a
submission-state refusal rather than the previous `UNSUPPORTED` classification.
Stream-query or capture-state rejection preserves the previous caller and
recording flag, so an owned PROGRAM run can still launch without a caller.
Host allocation failure before retention accepts a claim also preserves those
fields. Once cold work may be queued, its ownership remains until quiescent
cleanup succeeds.

| Resource | Lifetime owner |
| -------- | -------------- |
| Original Runtime handle storage | Native caller, through ordinary finalize |
| Prepared Runtime, arenas, parameters, control, events and code pin | Native execution state, extended by the graph reference |
| Context, caller stream, addressed Tensor allocations and runtime module | Caller, through successful graph resource release |
| Native CANN model | Caller; destroyed and prevented from replaying before resource release |

A public graph adapter must also retain its existing RuntimeModule and external
Tensor owners. A device code pin does not retain the Host shared library.

## Common submission and completion

The architecture launch transactions call the same production
`DeviceRunnerBase::enqueue_execution_core` and `enqueue_execution_cpu` methods.
`RunCallerOrder`, the native completion fence and queued-wait owners maintain
the actual dependencies:

```text
caller: Start ------------------------------------ wait CpuDone -> Tail
CPU:    wait Start -> admission -> Fork -> scheduler -> wait CoreDone -> CpuDone
Core:                             wait Fork -> kernel -> CoreDone
```

Cold image publication and registration precede this sequence. Eager cold
registration is queued without synchronously draining an older native run.
Single-use eager registrations retire in the last CPU callback. Host frees their
control only after the ordinary run drain proves the whole operator complete;
retirement adds no stream-wide wait behind queued successors. Graph registrations
remain until explicit graph resource release.

Graph resource acquisition additionally proves cold work quiescent before
CaptureBegin. Admission validates the retained platform image and decides user
dispatch before the fork. It clears the previous Core report markers only after
the preceding whole-operator join. Rejected invocations still complete the
ordinary platform handshake and retirement without creating a user runtime or
dispatching user tasks.

The same transaction distinguishes a complete command submission from Partial
or NotStarted progress. Recorded commands are not evidence of execution.
Original-handle retirement after recording can finish while the graph reference
continues to hold future-use obligations. Passive Host timestamp attribution is
not recorded for graph-retained invocations; it cannot count raw replay events.
Ordinary eager diagnostics keep their existing ownership and timing contracts.

Common wait obtains whole-operator completion and the current device terminal
result. A graph replay does not update Host phases, so graph-held completion
never reuses a cached successful Host wait to decide a later replay. Waiting
before any replay is not successful execution. The signed runtime status is the
ordinary integer wait result; raw epochs and admission decisions stay internal.
Native kernel payloads publish their runtime fault in that control and return
normally through RTS. Both ordinary eager wait and graph-held wait read that
status after completion proof. Eager cleanup retains an earlier submission
error and reuses the completed single-execution observation; it does not read
the control twice or replace an error with cleanup success.

## Registered invocation and device protocol

The shared launch envelope remains 192 bytes and the shared invocation header
stays at that offset. TMR's binding carries the immutable Runtime address,
binding epoch and independently registered execution reference. Eager and
recorded invocations use only `KernelInvocation`; a packet cannot register its
own pointers during execution. An empty reference is allowed only while encoding
a cold staging image; device consumption requires its valid registered identity.
The AICore entry, launch ABI and handshake layout are the mainline ones.
Host and AICPU must come from compatible builds.

Cold registration validates callable occupancy generation, context/execution
identity, argument geometry and armed diagnostic metadata. Every request and
control reserved field must be zero, including revocation requests. Protocol
extensions require explicit revision and compatibility handling. A3 invalidates
the freshly uploaded control before its first cold read, after identity checks;
A5 uses the coherent host-DMA contract. Replay does not repeat cold invalidation.

The CPU-only registration protocol has an explicit version and header extent.
Version2 retains a complete trusted KernelArgs snapshot, with unused envelope
bytes zero. A recognized rejected frame restores its CPU handshake/diagnostic
addresses from that snapshot before entry; rejected input cannot supply those
addresses. The CPU completion sequence advances separately on each replay. Unknown
references and protocol versions cannot create a registration.

The initial execution/completion sequence is binding_epoch minus one. Device
admission advances that sequence on the AICPU; it never replaces the immutable
native result identity or diagnostic stamp. The result stays bound to the Host
run epoch. After the preceding whole-operator join, admission clears its
publication marker before the next fork. A terminal is therefore valid only
after this execution has published it again under the fixed binding.
TMR kernel launches use the existing unstamped Core handshake: admission clears
aicore_done before the fork, and the scheduler accepts the fresh nonzero report.
PROGRAM and HBG keep their existing report-epoch rules. No capture control or
generation is passed to Core. Every original CPU participant contributes completion,
including filtered or rejected participants. Final completion consumes the
mainline signed terminal for that execution; a missing terminal is an error.

At final quiescent release, a graph's native carrier takes its own result
snapshot. The observation and its first transfer status survive registration
release and the original public handle. Common outcome checks and runtime error
payload inspection consume this same snapshot. Graph replay sequences never
enter the ordinary per-slot result-read cache, and cleanup retries do not
replace a failed first read with a later successful copy.

The first failed execution closes admission persistently. Already queued later
replays perform only the ordinary platform handshake and retirement. Unregister first quiesces graph use and revokes the
fixed registration. Numeric callable-ID reuse cannot retarget an existing model;
its code remains pinned until resource release. Ordinary unretired native runs
continue to refuse registry mutation.

## Failure and release evidence

Partial or unknown submission retains every possibly referenced owner. Failed
cold publication, registration, synchronization, revocation or free cannot
silently drop the last owner. Release is explicit and retryable, outside capture,
after the caller attests that the model is destroyed and cannot enqueue future
uses. A failed completion proof does not authorize borrowed-device reset.

After all participating streams are quiescent and future graph use has ended,
common wait-reference owners can discharge recorded edges without waiting for
a Tail event in a model that was never executed. Only then may physical native
retirement revoke registration, release code/control and reuse workspace/events.
Graph-held resources survive ordinary logical retirement and caller-storage
reuse until that proof succeeds. Resource release reports cleanup status;
execution errors are obtained through ordinary wait/status operations.

The existing device fault channel stops later admission without rewriting an
already observed execution result. Historical CPU-watchdog investigation #2669
and shared native-drain isolation #2690/#2695 remain separate; this execution
refactor does not claim to diagnose or fix the historical watchdog.

## Validation boundaries

CPU controls exercise both architecture registries, the common packet and Core
predicate, cold reserved fields, stale/reused identities, epochs and signed
terminal results. Common Host tests trace the actual production command methods
for eager enqueue and active RTS recording, including exact event dependencies
and Partial ownership. Lifetime controls retain a graph across original-handle
retirement/storage reuse, reject cached success after a later replay fault and
retain ownership across failed quiescence or cleanup.

Native hardware tests use real CANN capture/replay, numeric and parameter
snapshots, empty/active reset, queued first-fault and stale-callable refusal,
release retry, control-generation reuse and subsequent ordinary eager execution.
A3 and A5 results are separate. Simulation validates unsupported resource
retention. Tests do not establish physical stop after a device timeout; failed
proofs retain resources. Build/test results are bound to their actual source and
compatible artifacts rather than to an older PR's green checks.
