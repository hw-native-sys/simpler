# Kernel launch submission module

This module implements the three-stream submission and fail-closed ownership part
of the [v9 design](https://icc.gt.tc/vllm-pto?i=1#v9-design). It compiles on main
without K1 or HBG headers. All interfaces are internal C++ types in
`simpler::kernel_launch`; no public C ABI or runtime packet layout is introduced.

## Owner contract

`launch_bound_kernel(binding, caller, gate, ops)` consumes a readable Host packet,
trusted placeholder descriptors and prepared stream/event handles. The native
variant also consumes registered function handles, prepared AICore/HostArgs
buffers and clear regions. Nothing is allocated or registered in launch.

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
| Enqueue failure | Poison context; retain all device-visible resources until external quiescence |

The owner serializes preparation, close and submission through this lease. It
must not call binder recursively. Neither owner callback may allocate, enqueue,
synchronize or query capture. The native adapter performs its argument checks
inside the lease and releases it even if native preflight rejects the packet.
Native tail queries use `aclrtQueryEventStatus`; the injected query callback is
used only by the generic entry.

Preparation enqueues on the context's own AICPU stream and publishes no event
for a launch to consume: every launch enqueues on that same stream, so stream
FIFO already orders registration ahead of it. Ready publication means submitted,
not completed. Events must be created outside launch with `ACL_EVENT_SYNC`;
AICPU is a dedicated **non-hidden** stream and AICore is **hidden**. Caller is
borrowed, and all three handles must differ. The binder validates
distinct/non-null handles but cannot establish their creation flags.

Kernel-mode initialization selects CANN's process-wide **hardware capture event**
mode after validating the borrowed device and before creating streams or events.
An existing hardware setting is reused. If the application explicitly fixed
software event mode, initialization logs a warning at ERROR severity and
continues using software events. Other query/set errors still fail initialization.
A failed setter is rechecked in case another
initializer selected hardware mode concurrently. Platforms where CANN fixes
hardware mode and reports the mode API unsupported retain that native behavior.
The setting affects other framework operators in the same process, survives
context teardown, and is not changed by program-mode initialization. Configure
any application-level event policy before initializing a kernel-mode Worker;
kernel initialization must run outside graph capture.

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
| caller | Record Start |
| non-hidden AICPU | Wait Start, asynchronously clear prepared launch/handshake/report regions, record AicoreStart |
| hidden AICore | Wait AicoreStart, launch AICore, record AicoreDone |
| non-hidden AICPU | Launch with HostArgs, wait AicoreDone, record AicpuDone |
| caller | Wait AicpuDone, record SerialTail |

The chain is caller ⇄ AICPU ⇄ AICore: caller and AICore are never adjacent, so
ACLGraph capture propagates in two hops rather than forking from the caller.
`AicoreStart` must be recorded before the AICPU launch — the AICPU orchestrator
spins on AICore's handshake report, so recording it after would close a cycle.
AICore-first in Host enqueue order keeps the AICore binary-load path ahead of
the resident AICPU work; device cooperation still runs through the handshake
protocol, and the event chain fixes only entry and exit order. Host success
means submission only.

## Failure behavior

Every failure after entry into the enqueue sequence tells the owner to poison
and stops further Host submission. The binder does not publish a Host-side
cancel, retry an enqueue, or fabricate a completion tail. In particular, a
failure after AICore launch may leave device work waiting for a peer task that
was never submitted. `tail_recorded` remains false because no trustworthy join
exists.

This is a terminal partial-submission failure. All device-visible arguments,
streams, events and execution storage remain retained until the caller has
established external quiescence/reset ownership. Poison is an admission guard,
not proof that stop-on-failure has stopped already-running cores, and it never
permits the binder to synchronize, reset or reuse the context.

## Integration and tests

The module does not provide K1 context/resource ownership, callable residency,
HBG packet/slot validation, or a public launch entry. Those owner adapters remain
with the separate integration work and must be connected after prerequisites
land. The context adapter must apply the finish table above, reject prepare and
launch after poison, and retain resources through graph destruction. There is
no duplicate context phase machine in this module.

Fake-owner tests verify acquire/finish pairing, same/cross-caller behavior,
concurrent submission rejection and every enqueue failure position.
The SDK-enabled native unit target uses real CANN declarations and fake symbols.
Source guards reject forbidden ACL/RTS operations and dependency guards prevent
importing the unmerged K1/HBG interfaces. These tests establish Host protocol
behavior, not real capture/replay, event flags, precision or performance. The
native owner, event Probe B and mixed-operator device tests remain prerequisites
for that claim. Program-mode execution is unchanged.
