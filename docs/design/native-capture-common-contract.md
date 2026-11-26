# Native capture execution and API contract

[Resource lifetime](native-hbg-prepared-lifetime.md) ·
[HBG state restoration](native-hbg-resource-reset.md)

## Execution scope

Kernel has no eager/capture execution-mode selection. Ordinary submission and
framework graph recording use the same native preparation and launch path.
Capture records the publication, stream dependencies
and actual device launches; RTS model execution performs the recorded work.
Replay does not call Host orchestration again. Preparing and recording a model
require no preliminary eager execution of the user's computation.

Program has no eager/capture mode selection. Simpler's Program layer manages its device environment
and necessary input/output allocation and transfers. Kernel borrows the caller's device context, stream and
tensor storage; retaining an address does not transfer ownership of those objects.
There is no separate capture executor or prepared-call cache.

The framework adapter attaches the common submission token to its graph's
lifetime. Simpler's native model-notice helper validates that attachment and
reports model destruction; it does not select another execution engine.

The native interfaces are declared in
[`runtime_c_api.h`](../../src/common/worker/runtime_c_api.h).
The Python framework adapter also uses these interfaces. Its model ticket
holds the original run and the caller's tensor owners until model destruction.
The common API shape does not establish capture support for every backend.

## Preparation and eligibility

Onboard Kernel callers use `simpler_prepare_caller_run_v2` when they own the
original run through its last stream or model use. The same entry accepts
eager streams and recording streams. Ordinary lane-managed runs continue to
use `simpler_prepare_run_v2`. Both entries take the current `NativeRunDescriptor`.
Rebuild native callers against that layout. Simulation returns `UNSUPPORTED`
for caller-owned stream submissions.

Caller-owned preparation rejects writable HOST/NONE tensor arguments, including OUT
and INOUT, before user Host orchestration. Read-only IN controls may determine
fixed graph structure and task arguments. Their source values are consumed at
prepare time; changing those values afterwards does not change an existing
model. Device arguments retain the existing Kernel address/transfer contract.
No implicit Kernel input/output copies or private Host-write staging are added.

Preparation constructs the runtime image and required backing and provisions
Simpler-owned binary registration and completion/caller-order events. The caller
must already supply a valid borrowed execution environment. Resource allocation
is separate from starting user tasks.

## Recording and replay

The framework queue registers a model-destroy callback before enqueueing its
first native node. Further nodes in the same model can use that registration.
The registration identifies the model and owns no native execution slot.
When the queue invokes the node, the framework calls `simpler_prepare_caller_run_v2`
and calls `simpler_submit_run_on_stream` on the caller stream. A successful submission returns the slot
for another node and leaves the original run with the ticket. The recorded
tasks can execute zero or more times. After the caller ends all model uses,
checked cleanup finalizes the original run and releases its tensor owners.
The destroy callback itself only records the notification. When a live run still owns the previous
slot backing, prepare initializes the replacement worker reports once on a
private stream. Replay preserves the worker reports written by the device.

Native callers can prepare before entering capture. Each node has its own
original run handle and storage address through finalization.

```text
simpler_prepare_caller_run_v2 -> RTS CaptureBegin -> simpler_submit_run_on_stream -> RTS CaptureEnd
  -> RTS model execute / synchronize (zero or more times)
  -> destroy the model -> simpler_finalize_run
```

Supported native diagnostic families use this same original run. Prepare
reserves and arms its collection window before recording. Finalization settles
the window after the caller has ended all uses. The existing diagnostic flush
reports when artifact writing completes.

Publication precedes the entry fork. The AICore stream waits for that fork;
completion events follow the corresponding device kernels, and the caller's
exit dependency follows both sides. Recorded waits must have valid records in
the captured dependency graph. Caller-stream input copies, model execution and
output copies must preserve their data dependencies.

HBG Kernel eager calls and captured models in one context share GM heap scratch.
The caller must order complete invocations, including publication copies and
both device roles, in a single execution domain. Use one ordered caller stream
or dependencies that establish the equivalent order across streams. Retaining
several models does not permit concurrent execution against that heap. Separate
contexts have separate pools. Preparation itself does not overwrite shared heap;
A5 Graph argument publication into the heap is recorded and replayed before its
consumers. Heap contents are invocation-local; persistent outputs remain in
caller-owned tensors. No new model owner or implicit cross-stream ordering is added.

The caller establishes the last-use boundary, including model destruction or
cancellation and all outstanding executions on other streams. Poll and wait
cannot establish this boundary. Finalization drains the caller stream and
keeps resources on a failed drain for retry. A partial submission also needs
checked cleanup on its actual execution streams.

Diagnostic-enabled models and eager runs sharing a collector pool execute in
one caller-ordered domain. Stop all producers using that pool before finalizing
any one of its runs. Each run closes its own window. Aggregate sample and
publication semantics are specified in the
[ScopeStats capture contract](../dfx/scope-stats.md#native-tmr-capturereplay-collection),
the [PMU capture contract](../dfx/pmu-profiling.md#native-a3-capturereplay),
the [ArgsDump capture contract](../dfx/args-dump.md#35-native-a3-capturereplay-collection)
and the [Swimlane capture contract](../dfx/chip-swimlane-profiling.md#recorded-kernel-model-windows).

## Completion and errors

Physical device completion, functional success and output delivery are distinct.
An event boundary or callback count cannot establish all three. Unknown device
use keeps its backing and queued-wait obligations alive.

Each identified native AICPU callback contributes its status once after its work
and local cleanup. This includes filtered and controlled early-error callbacks.
The last contributor aggregates the runtime terminal selection and callback
errors and returns the final status to RTS. Missing required contributions or
an invalid required handoff cannot establish success. The count itself is not
proof that every device kernel has physically exited.

Private terminal publication distinguishes `Published`, `NotRequested` and
`InvalidState`. A consistent unrequested binding does not change a functional
result. An invalid requested binding is an internal producer-contract error;
a known functional error keeps precedence. Optional diagnostic payload may be
absent without replacing the original signed code and provenance.

Program execution preserves its structured error record. For Kernel eager and
capture alike, enabling any diagnostic in the call's `CallConfig` requests the
private native result and retained replay history; with all diagnostics off,
execution has no private record. The existing run epoch still identifies the
packet and Core reports, independently of the optional result address. Required
terminal handoff and actual signed RTS errors remain mandatory. No missing
private record establishes functional success or physical completion.

Requested retained fault evidence does not decide whether a later independent invocation
executes or skips. Device health, real dependencies and unresolved ownership
still constrain execution; independent replay is not permission to reuse unsafe
storage or operate on a quarantined device. RTS errors remain observable through
the actual execution/caller stream contract. Kernel does not reset the caller's
device.

## Support and qualification limits

A3 HBG coverage exercises bounded dependency, early-dispatch and nested-Graph
computations, changing device input contents, original-run finalization after model destruction,
multiple retained models with persistent INOUT tensors, and model/resource
cleanup. Separate native TMR scenarios exercise cold replay, serial retained
models, multiple prepared nodes, unexecuted-model destruction and a controlled
RTS execution error. Neither runtime's coverage substitutes for the other's
reconstruction and lifetime checks. Bounded capture-refusal, partial-submission
and timeout cases do not establish arbitrary device-fault recovery or kernel-stop
guarantees.

ScopeStats alone is supported for native A3 TMR capture, under the serialized
producer contract above. Its separate consumer checks actual replay-produced
samples and a subsequent ordinary eager run; DFX-off capture tests and ordinary
ScopeStats tests do not establish that combination. Recorded windows export
aggregate raw boundary samples, not exact per-replay totals or paired scopes.
HBG has no ScopeStats device producer. PMU alone uses the real A3 AICPU
task-completion producer on TMR and HBG; its aggregate raw rows carry frozen
event columns and explicit unknown cumulative device accounting. Neither PMU
rows nor fixed task IDs identify individual replays. ArgsDump alone samples
actual task input/output boundaries on A3 TMR/HBG, retaining each occurrence
and its payload independently. Its recorded manifest explicitly leaves
aggregate device totals unknown; task IDs do not define replay boundaries.
Swimlane alone preserves repeated raw device intervals on A3 TMR/HBG.
Its Host preparation context uses a separate clock and does not describe replay
Host activity. Fixed IDs do not define replay boundaries or dependency joins.
Other capture diagnostic families and combinations remain unsupported.

A3 qualification does not qualify A5 or framework graph integration. Native CPU
tests and SIM behavior do not establish RTS model support. Permanent missing
handshake/callback, invalid register bindings and unknown physical exit remain
outside the qualified recovery scope.

There is no in-place model address/scalar-update API here. A prepared model fixes
those bindings and Host-derived controls; new bindings need separate preparation
and model construction. Updating contents at an unchanged device address remains
a caller-ordering responsibility. This contract makes no framework-integration,
performance, concurrent-model or complete platform-matrix guarantee.
