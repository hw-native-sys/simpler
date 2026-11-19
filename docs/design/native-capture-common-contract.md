# Native capture execution and API contract

[Resource lifetime](native-hbg-prepared-lifetime.md) ·
[HBG state restoration](native-hbg-resource-reset.md)

## Execution scope

Kernel eager submission and capture recording use the same native preparation,
launch and ownership path. Capture records the publication, stream dependencies
and actual device launches; RTS model execution performs the recorded work.
Replay does not call Host orchestration again. Preparing and recording a model
require no preliminary eager execution of the user's computation.

Program is eager-only. Simpler's Program layer manages its device environment
and necessary input/output allocation and transfers. Kernel borrows the caller's device context, stream and
tensor storage; retaining an address does not transfer ownership of those objects.
There is no separate capture executor or prepared-call cache.

The native interfaces are declared in
[`runtime_c_api.h`](../../src/common/worker/runtime_c_api.h).
The Python framework adapter also uses these interfaces. Its model ticket
holds the original run and the caller's tensor owners until model destruction.
The common API shape does not establish capture support for every backend.

## Preparation and eligibility

The native caller supplies a `NativeRunDescriptor` to `simpler_prepare_run`.

| `prepare_purpose` | Contract |
| ----------------- | -------- |
| `SIMPLER_PREPARE_EAGER` (zero) | The original run owns the submission. Kernel callers can record a diagnostic-free run on their stream and keep the run alive through model destruction. |
| `SIMPLER_PREPARE_CAPTURE` | Checks Host effects before Kernel orchestration. The caller can retain a separate owner to finalize the original run before replay. |

All native callers must rebuild against the current `NativeRunDescriptor`
layout. `simpler_prepare_run` and `simpler_run` use that same descriptor.
A zero purpose preserves eager behavior; it does not make an older descriptor
layout compatible.
Simulation rejects capture-purpose preparation with `UNSUPPORTED`.

Capture preparation rejects writable HOST/NONE tensor arguments, including OUT
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
When the queue invokes the node, the framework uses ordinary Kernel prepare
and calls `simpler_submit_run_on_stream` on the caller stream. A successful submission returns the slot
for another node and leaves the original run with the ticket. The recorded
tasks can execute zero or more times. After the caller ends all model uses,
checked cleanup finalizes the original run and releases its tensor owners.
The destroy callback itself only records the notification. When a live run still owns the previous
slot backing, prepare initializes the replacement worker reports once on a
private stream. Replay preserves the worker reports written by the device.

A diagnostic-free Kernel caller can keep its ordinary run alive for the model's lifetime. The same launch entry submits to an eager stream or a recording stream.

```text
prepare_run -> RTS CaptureBegin -> launch_run_on_stream -> RTS CaptureEnd
  -> RTS model execute / synchronize (zero or more times)
  -> destroy the model -> finalize_run
```

When the caller needs to finalize the original run before replay or collect a
recorded diagnostic window, it acquires a separate reference before `CaptureBegin`.

```text
prepare_run(CAPTURE) -> retain_run_resources
  -> RTS CaptureBegin -> launch_run_on_stream -> RTS CaptureEnd
  -> finalize the original Host run
  -> RTS model execute / ordered copies / synchronization
  -> destroy or cancel the model and end every external use
  -> release_run_resources
```

The exact native entry names are `simpler_prepare_run`,
`simpler_retain_run_resources`, `simpler_launch_run_on_stream`,
`simpler_finalize_run` (or `simpler_finalize_run_with_delivery`) and
`simpler_release_run_resources`. Model operations belong to the caller's RTS API.

The detached submission uses Kernel mode, capture-purpose preparation, a matching
retained reference and caller stream, and no predecessor argument. Diagnostics
allow ScopeStats alone on native A3 TMR, or one of PMU, ArgsDump and Swimlane
on native A3 TMR/HBG.
Each supported family reserves and arms its collection window during retention,
before submission. Simpler selects the ownership path from the caller's
preparation and retained reference. RTS decides whether to enqueue or record
the tasks and reports stream errors. The same externally owned submission can
be used on an eager stream. To observe an eager run with Simpler after retaining
a preparation, call `simpler_launch_run` with the bound stream.

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

For an externally owned submission, `simpler_poll_run` reports a poll error and both native wait
entries return `UNSUPPORTED`; none reports execution completion. Finalization
ends the original Host run without synchronizing recorded device execution or
copying Program outputs. The retained reference remains responsible for future
model uses. Resource and failed-release rules are specified in the
[resource lifetime contract](native-hbg-prepared-lifetime.md).

Diagnostic-enabled models and eager runs sharing one context's collector pool
must execute in one caller-ordered domain, with no concurrent producers. Multiple
resource references and collection windows may coexist. Before releasing any
such reference, the caller must finish **all** producers sharing that
pool and start none until release returns. Closing one window does not close
another live window. Synchronizing the retained caller stream or counting
references cannot prove that a raw model replay on another stream has stopped;
the caller establishes that ordering. This contract adds neither a second owner
nor a whole-context stream synchronization operation. Aggregate sample and
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
computations, changing device input contents, original Host-run finalization,
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
