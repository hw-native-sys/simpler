# Enabling capture by relaxing the native launch guard

**Date**: 2026-10-26

## Proposal and disposition

Do not enable native capture merely by removing the caller-stream capture refusal.
At PR #2647 head `65f88d75d`, `simpler_prepare_run` has no caller stream,
but HBG `publish_run_image_impl` already performs a synchronous device copy.
In that revision the stream first reaches the runtime C ABI at
`simpler_launch_run_on_stream` after publication. The public run lane already
receives and checks it before prepare; the gap is its handoff to the runtime.
The local common descriptor adaptation supplies that handoff and preserves early
refusal while full capture publication and resource retention remain unimplemented.
Correct capture needs preparation intent, retained publication sources and
recorded restoration; a successful CaptureEnd alone cannot establish these facts.
Ordinary eager's phase order need not change to add a capture-specific contract.

## Observations

Raw CANN 9.0.0 probes on A2/A3, using locked devices, found:

| Case | Observation | Limit |
| ---- | ----------- | ----- |
| Synchronous H2D during capture | Returns 107030; capture remains ACTIVE; subsequent async copy and CaptureEnd succeed | A graph can be created after an earlier operation was refused |
| Pinned source changed after capture | Source changes from 23 to 47; replay produces 47 | The graph does not provide an immutable snapshot of source contents |
| Destroy with a queued replay behind a bounded host gate | Destroy waits about 1001 ms; callback observes the graph's D2H result 47 | One tested stream/model, not a universal SDK lifetime guarantee |
| Failed CaptureEnd with an unjoined fork | Returns 107025 and null output model; destroy callback has already fired once; GetInfo reports NONE | Covers this failed-end path only |

The fork probe uses `aclrtCreateEventExWithFlag(ACL_EVENT_SYNC)`, matching the
runtime. Two earlier setup attempts with ordinary and progress events failed
at RecordEvent with 207000 and did not reach CaptureEnd; they are not runtime
regression results. The probes retain all storage through explicit drain and
do not free storage from the callback.

Local evidence is under `build/review-validation/unified-api-step4/`: probe
sources, logs, `REPORT.md` and `manifest.json`. The probes do not load Simpler's
executor, and establish neither HBG restoration nor A5/framework support.

## Implementation consequences

A graph owner must exist before recording can establish a resource reference.
CaptureEnd failure cannot be handled by assuming a missing output model means
no callback or no retained reference. Destruction notification should only mark
an independently retained owner for retirement; taking registry locks, draining
streams or unloading modules inside it needs a separate reentry/lifetime proof.
Output and diagnostic consumers can outlive device completion.

Revisit enabling native capture after the common prepare contract distinguishes
capture before device-visible effects, the graph retains immutable publication
sources and actual resource references, and replay restores HBG state. Keep
ordinary eager's phase order and existing capture compatibility until the new
path is qualified. No capture production path was enabled by this investigation.

## Common metadata publication boundary

`RuntimeLaunchMetadata::materialize` consumes a preparation snapshot into pinned
storage and a separate `Publication` record. `Publication::enqueue` only submits
copies from those addresses; it does not rewrite the sources, consume the record,
or infer completion from an enqueue result. The existing eager `publish` operation
uses these same two operations, at its existing call site and in its existing order.
No additional pinned allocation or metadata copy is introduced for eager.

The record owns its region descriptions, **not** the pinned allocation or device
targets. Keeping a record alone is insufficient for capture: the graph owner must
also retain those allocations and prevent source content changes until graph
retirement. Partial enqueue failure likewise leaves the submission owner responsible
for any accepted readers; keeping the record does not authorize an automatic retry.
The common native capture refusal remains in place. HBG image restoration, replay
identity, graph ownership transfer and recording-only finalization are not supplied
by this metadata refactor.
