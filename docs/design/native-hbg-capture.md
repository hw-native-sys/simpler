# Borrowed HBG native capture and replay

## Scope and execution ownership

A2/A3 onboard HBG kernel mode borrows the caller's device, RTS context, stream
and DEVICE tensor storage. Program and kernel initialization retain different
ownership contracts, while callable registration, native preparation, submission,
completion and cleanup share the existing runtime implementation. Program graph-task
capture adaptation is outside this change. TMR, A5 and simulation capture remain
unsupported. All onboard native eager launches share the device preparation and
callback aggregation contract described below; simulation remains unchanged.

The dedicated HBG packet executor, device slots, graph-token leases and submission
binder have been removed together with their C ABI and Worker/Python entry points.
Callers register through the common callable registry and execute through native
prepare/launch/completion. No rejecting compatibility stubs remain.

## Host/device sequence

```mermaid
sequenceDiagram
    participant U as Caller / framework
    participant H as Common native Host runtime
    participant R as RTS streams / graph
    participant C as AICPU native executor
    participant D as AICore
    U->>H: Borrowed init + common callable registration
    U->>H: Native eager prepare / launch / wait / finalize
    H->>R: Existing eager publication and submission
    R->>C: Native Runtime + graph definition
    C->>D: Existing mailbox scheduling
    D-->>C: Task completion
    U->>H: reserve_capture_resources (outside capture)
    H->>H: Reserve descriptor, pinned image, workspace, streams/events
    H->>H: Reclaim obsolete workspace outside capture
    U->>R: CaptureBegin(caller stream)
    U->>H: Common prepare_run + launch_run_on_stream
    H->>H: Validate current bindings and reserved capacity
    H->>R: Record Runtime/image restoration + common preparation + native launches
    Note over H,R: Caller/AICPU/AICore event ordering uses existing execution streams
    H-->>U: Recorded receipt (not completion)
    U->>H: finalize_run
    H->>H: Attach capture resource owners; no device wait/free/unmap
    U->>R: CaptureEnd
    loop Serial replay on the capture stream
        U->>R: Graph replay
        R->>C: Preparation: inspect prior result, freeze Execute/Skip, reset N callbacks
        R->>C: Native executor entries (all N RTS callbacks)
        C->>D: Existing mailbox scheduling
        D-->>C: Task completion
        C-->>R: Terminal status and participant retirement
    end
    U->>R: Destroy graph after replay completion
    R-->>H: Model destruction notification
    U->>H: retire_capture_resources
    H->>R: Drain retained streams
    H->>H: Release graph-held resources after drain proof
```

RTS records stream operations. The actual task scheduler remains Simpler's native
AICPU executor and mailbox communication with AICore. Capture does not introduce
a second task executor. Ordinary eager phase order is preserved.

## Cold preparation and capture restrictions

A completed and finalized eager warmup establishes the native resource demand.
`reserve_capture_resources` reserves remaining descriptor/pinned publication
storage, execution streams and events outside capture, and reclaims obsolete
workspace before marking the reservation ready. Subsequent eager preparation
invalidates readiness; callers reserve again before the next capture.

Capture refuses missing resources or insufficient capacity rather than allocating,
replacing backing, or reclaiming obsolete storage. Both recorded-run finalize and
prepared-but-unlaunched capture finalize defer obsolete workspace reclamation.
The retained full Runtime/image is restored on every replay. A separate aligned
`NativeExecutionControl` tail is never restored from the image. Its sequence,
frozen decision, callback count and first error survive publication. Context-wide
first-error storage is independently retained and is not reset between runs.
Each captured graph has an independent aligned `DeviceRunResultRegion`, followed
by the common execution control. Simpler retains descriptors,
publication storage, workspace, code and events; callers retain borrowed contexts,
streams and DEVICE tensor storage.

After the previous whole-operator join, common device preparation checks its
terminal before clearing publication, resets worker handshake state and initializes
exactly the number of native callbacks RTS will dispatch. It freezes Execute or
Skip using the context first error, flushes shared state, and only then releases
the AICPU/AICore fork. Both ordinary onboard execution and capture use the existing
unstamped Core handshake. The AICore launch ABI is unchanged and reads no new
control block. Native result identities remain for structured result validation;
recorded identities are fixed, with no dynamic replay-epoch handshake.

Every native callback contributes after its local work and destructors finish,
including filtered callbacks and controllable entry failures. Non-last callbacks
return zero; the last callback validates the runtime result, preserves the first
signed error and returns the aggregate status. Missing terminal evidence remains
Unknown rather than being reported as a known execution failure. A subsequent
preparation freezes Skip: runtimes perform their existing empty handshake and
shutdown without dispatching user tasks. A5 resident scheduling uses its existing
scheduler stop state before the fork. No AICore interface is added.

The callback counter is semantic aggregation, not physical completion proof.
Host stream joins still cover both AICPU and AICore exits. If an unhandled failure
or missing participant prevents that proof, resources remain retained and a
successor cannot safely reset the shared state. RTS may stop the queue after an
error; Skip is the safe path if it continues dispatching.

Same-stream serial replay is supported. Cross-stream replay, concurrent replay,
TMR capture, diagnostics during capture and program graph-task adaptation are not
claimed. A Recorded receipt (-1008) must not be interpreted as completed output.

The reservation names one pipeline slot and arena bank, and a recorded run must
take that same slot: a direct run that lands on another slot has no reserved
descriptor, events or publication capacity and is refused during preparation.
The public `Worker` reserves slot 0 / bank 0 for **one recorded invocation per
reserved resource generation**. Finalizing that recorded run does not make its
generation reusable: the graph still holds the descriptor and publication source.
A second captured submit on that generation is refused, including within the
same capture. Another generation must be established by successful eager warmup
and reservation outside capture. The C ABI accepts an explicit slot/bank for
callers that warm and select their own; this does not establish arbitrary
multi-slot or multi-node capture qualification.

Parameter changes use a new immutable graph generation: submit an eager run
with the new DEVICE tensor addresses and scalar values on the same caller stream,
wait for it, reserve again, and record into a new caller-owned RTS model. The
successful CaptureEnd publishes the new model handle. Select that handle for
subsequent serial replay; an older handle continues to use its original values.
This is **not an in-place update of an existing RTS graph**. Keep the topology,
tensor geometry, callable and launch configuration fixed for this parameter-only
contract. Each version owns its descriptor, publication source and workspace
backing until model destruction plus a successful drain/retirement; external
DEVICE buffers remain caller-owned and must survive their last use. A failed
new-version preparation must not release the old model or its backing. The
host-only tests cover ownership; the native capture chain tests exercise changed
addresses and scalars through the C ABI, ChipWorker and Worker entry points.

Graph-held callable protection is object-specific: the active callable image
(including its child code and function tables) cannot be unregistered/replaced.
Unrelated registration and unregister retain ordinary live-run guards. A new
code upload may rotate the AICore stream, but graph-held stream generations are
pinned through graph retirement. Old unpinned stream handles are reclaimed on
the next stream preparation or context teardown, retaining failed destroys for
retry. Context teardown still refuses while graph resources are retained.

The RTS event work mode is process-wide and has one value. Kernel-context
initialization raises it from software events (0) to hardware events (1) when
the runtime supports capture, and it is not restored at close. Program-mode
contexts in the same process therefore share mode 1, and a context that already
observes mode 1 changes nothing.

A recording that fails after its first queued operation leaves an incomplete
model. It is not retried into that model: the caller abandons capture, destroys
the model, retires the retained resources and reserves again before recording a
new one.

Replay status is not host-visible through poll or wait: recorded runs are already
host-finalized, and the caller's stream supplies per-replay completion. Common
preparation freezes Execute or Skip from context-wide first-error storage. A Skip
still completes the existing Core handshake and accounts for every RTS callback;
the last callback publishes the selected terminal result. An unfinished prior
attempt refuses preparation without clearing its evidence. Callback accounting
alone is not proof of physical device exit; resource retirement requires the
retained stream fences. Device fault notices matched to the context are consumed
by a later host-supervised finalize or close. A public per-replay Host result API
is still a C4 item.

## Common native capture resource attachment

Finalizing the run transfers resource ownership to the existing captured-graph
attachment. It neither waits for an unplayed graph nor releases the resources
referenced by recorded commands. Destruction notification alone is not a completion
proof: explicit retirement also drains retained streams. Retirement is
context-wide: `retire_capture_resources` requires every retained model destroyed
before it runs, so a dead graph cannot be released while another captured model
is still live. Failed capture end, never-replayed graphs and failed drain follow
the same ownership rules. Failed free keeps the allocation in the existing ledger
and fails that reservation. A later reservation may proceed with retained
accounting; it does not retry an unconfirmed free or pretend the allocation was
released.

## Validation responsibilities

| Boundary / owner | Validated fact and timing | Regression coverage |
| ---------------- | ------------------------- | ------------------- |
| Common callable registry | Canonical image bounds before upload; ID occupancy and code ownership | `test_kernel_mode_c_api.py`, native admission C++ tests |
| Native prepare / invocation validation | Current borrowed context, DEVICE tensors, geometry and config before binding; generation belongs to the actual registry/context | Native admission, kernel caller-stream tests |
| Native HBG builder and bind ledger | Graph definition sizing, overflow, offsets and native resource requirements | HBG bind-ledger and definition-packing C++ tests |
| Cold reservation / capture prepare | Existing backing capacity and ready streams/events; capture may not grow resources | Native admission reservation/growth/refusal tests; cold probe |
| Common runtime publication | Stable full Runtime/image backing through asynchronous consumption; graph attachment pins it beyond finalize | Runtime launch-metadata and captured-resource tests; real replay |
| Common device preparation / terminal executor | Frozen Execute/Skip, all dispatched callbacks and context-wide first error; terminal error survives publication | Terminal-retirement C++ tests; real repeated replay (not exhaustive device fault injection) |
| Common finalize / capture retirement | Recorded is not completion; model destruction plus drain before release; capture finalize defers reclamation | Captured-resource and native admission tests; never-replayed/failed-end board cases |

The obsolete framing facade had no production callers and
was removed with its private test suite. TMR retains its actual
`consume_kernel_invocation` framing, registered generation, runtime-address/epoch
and tensor checks; HBG uses native launch metadata and GraphDefinition validation.
The old whole-image `validate_images` replay scan is gone with the second packet
format. Native preparation validates the privately constructed image before
publication; graph retention pins its immutable replay source. Device launch
metadata and runtime access checks remain on the actual consumption path.

These checks establish different facts at ownership boundaries; there is no second
whole-invocation packet parser or fixed-slot validation pipeline. Gate arguments
are privately constructed by the Host runtime, not a new public packet ABI.

## Test migration and evidence

Old packet parser, slot-control, dedicated launch binder and fixed-slot close tests
were removed with their production implementation. They are not counted as passing
native tests. Shared native admission, publication, graph attachment and completion
tests cover the current owners. `test_native_capture_chain.py` exercises the real
C ABI, ChipWorker and Worker capture paths.

The cold probe refuses device/pinned allocations and frees, stream/event creation,
and HAL unmap while recording. A CPU self-test proves each interception, including
`halHostUnregister` obtained through an explicit local library handle. Board checks
then require zero observed cold operations across real capture and finalize.

Batch evidence and exact validation results are stored under
`build/review-validation/native-capture-convergence/`; final API-removal evidence
is under `build/review-validation/native-capture-api-cleanup/`. CPU tests do not qualify
hardware behavior; A2/A3 board results do not qualify A5 capture or arbitrary device
fault recovery.
