# Kernel launch submission module

This module implements the three-stream submission and failure-compensation part
of the [v9 design](https://icc.gt.tc/vllm-pto?i=1#v9-design). It compiles on main
without K1 or HBG headers. All interfaces are internal C++ types in
`simpler::kernel_launch`; no public C ABI or runtime packet layout is introduced.

## Owner contract

`launch_bound_kernel(binding, caller, gate, ops)` consumes a readable Host packet,
trusted placeholder descriptors and prepared stream/event handles. The native
variant also consumes registered function handles, prepared AICore/HostArgs
buffers and clear/cancel regions. Nothing is allocated or registered in launch.

`KernelLaunchGateOps::acquire` obtains an exclusive submission lease and validates
context phase, current device, callable registration/generation, frozen capacity,
packet ABI and runtime-specific bindings. It returns a `KernelLaunchAdmission`
snapshot with the two internal streams, five events, previous caller identity
and whether preparation is pending. Acquisition failure retains no lease and
must leave persistent context state unchanged.

Every successful acquisition is paired with exactly one `finish(result)`:

| Result | Owner action before releasing the lease |
| ------ | --------------------------------------- |
| Success | Store caller identity and consume the pending preparation dependency |
| Rejection before enqueue | Preserve phase, previous caller and preparation state |
| Enqueue failure | Poison context; retain submission and compensation errors separately |

The owner serializes preparation, close and submission through this lease. It
must not call binder recursively. Neither owner callback may allocate, enqueue,
synchronize or query capture. The native adapter performs its argument checks
inside the lease and releases it even if native preflight rejects the packet.
Native tail queries use `aclrtQueryEventStatus`; the injected query callback is
used only by the generic entry.

Preparation records `PrepareTail` after initialization and slot registration;
ready publication means submitted, not completed. Events must be created outside
launch with `ACL_EVENT_SYNC`; AICPU is a dedicated **non-hidden** stream and AICore
is **hidden**. Caller is borrowed, and all three handles must differ. The binder
validates distinct/non-null handles but cannot establish their creation flags.

The owner retains device resources and function/stream/event handles until all
executions and captured graphs end and external quiescence is established.
HostArgs is writable exclusive staging. CANN copies it into task-owned storage
before returning; the adapter restores zero placeholders for reuse. Runtime
validation must reject placeholders that overlap its metadata, because this
module only checks descriptor agreement, zero addresses and buffer bounds.

## Submission order

A caller change queries the previous SerialTail before any enqueue. Not-ready or
query failure rejects without poisoning. Same-caller submission neither queries
nor waits on old tail; caller FIFO and the branch joins provide serialization.
Not-ready/distinct-handle rejection uses the existing
`PTO_RUNTIME_ERR_PREPARED_INCOMPATIBLE`; a future public owner maps its API status.

| Stream | Operations in Host submission order |
| ------ | ----------------------------------- |
| caller | Optionally wait PrepareTail, asynchronously clear prepared launch/handshake/report regions, record Start |
| hidden AICore | Wait Start, launch AICore, record AicoreDone |
| non-hidden AICPU | Wait Start, launch with HostArgs, record AicpuDone |
| caller | Wait AicpuDone, wait AicoreDone, record SerialTail |

AICore-first avoids a waiting AICPU scheduler obstructing its AICore SQE and
permits pre-AICPU cancellation after checking the core branch submissions.
Both branches remain gated by Start. Host success means submission only.

## Failure behavior

Every failure after entry into the enqueue sequence tells the owner to poison.
Before successful AICore launch, stop. Failure recording AicoreDone cancels the
waiting core, retries the record once, then joins and records SerialTail.
AICPU Start-wait failure cancels and joins core. AICPU launch failure additionally
records/joins the already-forked AICPU wait branch. Cancellation is one all-ones
async fill of the prepared 32-bit cancel words, publishing `UINT32_MAX`.
After successful AICPU launch, stop on errors without Host cancel.

Compensation stops on its first error. `cleanup_status` never replaces the
original `status`, and `tail_recorded` identifies whether a tail was established.
Failed compensation can leave no provable join and an uncancelled waiter; this
is a terminal execution failure requiring external quiescence/reset ownership,
not permission for binder to synchronize, reset or reuse the context.

## Integration and tests

The module does not provide K1 context/resource ownership, callable residency,
HBG packet/slot validation, or a public launch entry. Those owner adapters remain
with the separate integration work and must be connected after prerequisites
land. The context adapter must apply the finish table above, reject prepare and
launch after poison, and retain resources through graph destruction. There is
no duplicate context phase machine in this module.

Fake-owner tests verify acquire/finish pairing, same/cross-caller behavior,
concurrent submission rejection and all enqueue/compensation failure positions.
The SDK-enabled native unit target uses real CANN declarations and fake symbols.
Source guards reject forbidden ACL/RTS operations and dependency guards prevent
importing the unmerged K1/HBG interfaces. These tests establish Host protocol
behavior, not real capture/replay, event flags, precision or performance. The
native owner, event Probe B and mixed-operator device tests remain prerequisites
for that claim. Program-mode execution is unchanged.
