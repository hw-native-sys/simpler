# Prepared capture resources and lifetime

For A5 Scalar scheduling, apply the [shared hardware and scheduling premises](../zh-cn/design/a5-hbg-graph-task/build-execute-separation.md#a5-hardware-principles). A future STARS event must retain its actual resource through all notification and observation uses; it cannot be reset or released merely because a task returned. Mapped SPRs are not a substitute without an allocation and release protocol.

[Execution and API](native-capture-common-contract.md) ·
[HBG state restoration](native-hbg-resource-reset.md)

## Owners and backing

A prepared execution refers to immutable `PreparedRunResources`; the runner
separately tracks `PreparedRunUse` obligations. Destroying a Host run does not
remove an unresolved device, queued-wait, binding or external-use obligation.
A completed execution is not authorization to destroy a model's backing.

| Resource | Ownership and last use |
| -------- | ---------------------- |
| Caller context, stream and external tensors | Caller-owned; valid through all recorded and in-flight uses. |
| Callable code and registration | Simpler retains the actual callable lease; referenced code cannot be unregistered. Unrelated callables remain independent. |
| Runtime descriptor and initial HBG execution image | Onboard Program and Kernel pass these as RTS Host inputs. RTS owns the snapshot for execution and retains it with a captured model for replay. |
| HBG writable execution arena | The run owns the writable backing, or a retained resource reference can extend its lifetime. The ordinary A5 scheduler restores a 64-byte status prefix. A2/A3 restore the runtime and shared headers and initialize writable task slots from RTS records. |
| Pinned sources for explicit copies | Simpler submission storage retains pinned sources for the lifetime of recorded copies and any in-flight DMA. |
| Shared state and scheduler resources | Simpler retains the shared AICPU/AICore backing, each workspace lease, and scheduler notification and counter owners until the last use. Both device roles receive the same shared addresses. |
| HBG GM heap | Preparation retains a fixed heap lease through its last use, including external model uses. |
| Internal streams, fences and caller-order events | Actual stream generation and event storage survive their recording/execution/wait references. |
| Program input staging | Retained as real backing for ordinary Program resource retention; this does not enable Program capture. |

Host orchestration scratch and draft mirrors can be reused after extraction has
ended their readers. RTS launch snapshots and the Simpler-owned sources of
recorded DMA carry their respective replay lifetimes. Borrowed device addresses
in either carrier refer to separately retained backing.

Slot and bank caches do not own permission to overwrite a retained version.
Equal-capacity writes can require new storage. Image, scheduler, arena, temporary
and submission storage detach from retained predecessors before reuse. Retired
owners keep their raw addresses and failed-release records reachable; workspace
accounting cannot recycle an address still held by a prepared owner.

HBG heap leases protect allocation lifetime, not historical scratch contents.
Same-context ordered preparations share one fixed-capacity allocation even while
models remain retained. Capacity and base remain fixed from the first successful
prepare allocation until context teardown. Oversized graphs fail before device
submission. Neither normal retirement nor workspace pressure trims the heap.
Failed external release preserves the preparation and its heap lease for retry.
Heap sharing does not apply to
caller-owned persistent tensors, immutable images or retained diagnostic evidence.

## Retaining a preparation

Call the existing C interface before capture starts:

```cpp
int simpler_retain_run_resources(
    DeviceContextHandle ctx, RuntimeHandle runtime,
    void *caller_stream, RuntimeHandle *out_reference);
```

The run must be Prepared. `out_reference` receives an independent resource token,
not an executable run handle. Exactly one retention is permitted per original
run, including if that reference is released before the original run ends.
The original run still requires its own finalization.

Onboard retention covers actual non-DFX backing for Kernel and ordinary Program
preparations. Native A3 TMR capture additionally retains ScopeStats-only backing;
native A3 TMR/HBG capture retains PMU-only backing through the same prepared
resource owner. See the
[ScopeStats capture contract](../dfx/scope-stats.md#native-tmr-capturereplay-collection)
and [PMU capture contract](../dfx/pmu-profiling.md#native-a3-capturereplay)
for shared-pool ordering, driver protection and release obligations. Other
diagnostic combinations remain refused. Program remains eager-only.
SIM returns `UNSUPPORTED` and clears a supplied output reference. Retention
success alone does not advertise a backend's device replay support; use the
[execution support limits](native-capture-common-contract.md#support-and-qualification-limits).

Retaining code, internal streams and backing does not retain an external device
context, caller stream or tensor owner on the caller's behalf. Those objects must
remain valid. A context cannot be destroyed or have its grant changed while
prepared resources remain in use.

## Ending external use and releasing

```cpp
int simpler_release_run_resources(
    DeviceContextHandle ctx, RuntimeHandle reference);
```

Before calling release, the caller must finish model destruction or capture
cancellation, stop future uses, and complete all in-flight uses on every execution
stream. Ending one recording or destroying the original Host run is insufficient.
The API cannot infer these external facts from a reference count. Calling
release promises that the reference will not be used for future execution,
even if synchronization fails and ordinary release remains retryable.

Release validates the context and caller stream and synchronizes that retained
caller stream with the configured timeout. That synchronization supplements the
caller's promise; it does not prove completion on other replay streams. A failed
synchronization returns its error without consuming the reference. Do not invoke
release from an RTS callback that forbids synchronization/device operations.

Once external use ends, internal obligations remain separate: actual device exit
or a proven no-submission outcome, consumed waits, returned copyback, released
bindings and ended Host ownership. Only their combined completion permits
reclamation. Returning storage to an idle cache is a valid release; immediate
freeing of every allocation is not required.

## Failures and cleanup

The caller completes the RTS-required capture/model teardown after recording
failure. Simpler reports the submission error and retains its own potentially
referenced resources. Partial submission is not rollback or no-submission proof.
Recording finalization does not synchronize, reset the caller's device or invent
execution completion.

An ordinary resource-release failure keeps an unconsumed token and failed
addresses for retry. Once release has been requested, the token cannot be used
for further execution. Successful frees are not repeated. A binding-cleanup
failure keeps the original carrier alive; finalization retries do not repeat
copyback that already returned.

External release and retry reclaim only that reference's obsolete managed
backing, preserve reusable cached storage, and do not retry another reference's
failed frees. Successful recovery leaves unrelated quarantined backing and
unconfirmed use obligations intact.

Ending external use cannot discharge unproved internal use. Unrequested external
references still block an owned Program fatal reset. When that reset is confirmed
and the existing confirmed-reset resource retirement consumes release-requested
references, their tokens become invalid. They must not be released or retried
after that retirement, even if close ultimately returns a nonzero code. The final
close return code alone proves neither reset confirmation nor continued token
validity. A borrowed Kernel context stays isolated and is never reset by Simpler.
Missing or dropped diagnostic records do not revoke a runtime's independent
physical proof, and log contents do not grant reclamation.
