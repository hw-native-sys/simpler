# TMR kernel capture and replay

## Dependency and supported contract

The native C API builds on unified callable registration (#2619) and TMR eager
invocations (#2624). `simpler_kernel_capture_supported` reports one for TMR
onboard builds and zero for HBG and simulation. The legacy
`simpler_kernel_mode_launch` remains a refusing stub.

A lease owns one fixed invocation, recorded once into one native CANN model.
It requires a borrowed kernel context, pipeline slot zero and diagnostics off.
No other native run or capture lease can use that context until release.
The caller serializes lifecycle and registry operations, owns native model
construction and destruction, and executes every replay on the recorded caller
stream. Concurrent models or replay on another stream violate this contract.

This is a native fixed-binding interface. It does not expose capture through
the public Worker or create a second Worker/loader. A consumer using this C API
keeps the host runtime module loaded until the lease and any cleanup debt have
ended; pinning the device callable buffer alone does not retain that module.
Public graph integration must hold the existing `RuntimeModule` owner together
with graph resources and external Tensor references.

Registration, unregistration and context finalization share the registry lock.
Ordinary native runs continue to refuse registry mutation until finalized.
For a capture lease, each mutation validates the lease and proves completion
of the recorded caller stream before changing the registry. Explicit IDs and
the unified `simpler_register_callable_v2` automatic entry follow this same
rule. An invalid lease or failed completion proof refuses the mutation.
Explicit unregister revokes the captured callable identity; a later replay of
that model is rejected even if the same numeric ID has been registered again.
It does not silently retarget the model to the new callable. The retained code
pin survives revocation until release. No automatic eviction is introduced.

## Native lifecycle

1. Register the callable outside capture. Allocate aligned Runtime storage
   using `get_runtime_size` and `get_runtime_alignment`.
2. Call `simpler_kernel_capture_prepare` outside capture with DEVICE Tensor
   descriptors and scalars. It reuses ordinary prepared-run arenas, retains
   code and argument storage, activates the launch shape and registers an
   independent device capture identity. All capacities and addresses stay fixed.
   Descriptors and scalars are copied at preparation; later source-object
   changes do not update the invocation. Tensor contents can change between
   replays through operations ordered on the caller stream.
3. Begin native CANN capture on the caller stream and call
   `simpler_kernel_capture_record` exactly once. It queries capture status and
   records events and launches without allocating device/runtime resources,
   uploading execution images, synchronizing or attaching a model. The launcher
   can allocate host storage. Callback exceptions become integer errors with
   partial ownership retained and admission poisoned. The caller ends capture
   and instantiates the model.
4. Execute the model on that same caller stream. `simpler_kernel_capture_wait`,
   outside capture, synchronizes the native whole-operator join and returns the
   retained signed runtime status together with replay and completion epochs.
5. Destroy every captured model and ensure it can never replay again, then
   call `simpler_kernel_capture_release`. This is an explicit caller lifetime
   obligation: the C API cannot inspect external model ownership. Keep Runtime,
   context, stream and addressed Tensor storage alive through successful release.

Preparation failure can retain a lease after ownership was established. Release
is retryable; ordinary native finalize and context close refuse a live lease.
A failed wait or free retains the resources it cannot safely reclaim. Partial
recording requires quiescence of the caller and both private streams. A wait
failure does not prove hardware stopped and does not authorize borrowed-device
reset or freeing memory that accepted work may still reference.

## Retirement evidence and health

The prepared native carrier remains reserved for the graph's entire lifetime.
Its retained-device-use state distinguishes an open lease from a quiescent,
released lease and binds the final proof to the native run identity. Recording
a graph retains possible future device references; it is not evidence that a
replay executed. A validated nonzero device replay epoch records observed
execution in the common workspace ledger. Waiting on a live graph does not
retire its workspace, because another raw replay can still use it.

Only release, after the caller has destroyed the model and all participating
streams have quiesced, supplies the common retirement path with the device
completion proof. That path never labels a retained graph as
`NoDeviceSubmission`. Failed synchronization or control readback retains the
lease and its resource references for retry. Release reports cleanup status;
the retained signed runtime result is obtained through capture wait.

Caller and private stream identities use the ordinary runner's fault-channel
history. Capture wait, registry quiescence and release consume that same health
channel. A matching late backend notice refuses later native admission but
does not rewrite the observed replay's runtime status or authorize a device
reset. A raw CANN model execution bypasses host admission; its caller must keep
the same ordering and monitoring obligations, and host checks do not cancel
already queued replay work.

| Resource | Owner and last-use boundary |
| -------- | --------------------------- |
| Runtime storage, host runtime module, caller stream and external Tensor allocations | The native caller keeps them valid through successful release, including cleanup retries |
| Prepared arenas, invocation snapshot, control, private streams/events and code pin | The lease/context retains them while the model can replay or completion is unknown |
| Native CANN model | The caller destroys it and prevents every future replay before release |
| Public graph/module/callback references | A public adapter must connect these to the existing Worker/module owner; this native API does not provide that adapter |

## Three-stream ordering

The binder records this topology using context-owned streams and five events:

```text
caller: Start ------------------------------------ wait CpuDone -> Tail
CPU:    wait Start -> admit -> Fork -> scheduler -> wait CoreDone -> CpuDone
Core:                         wait Fork -> kernel -> CoreDone
```

Admission precedes both processors. The caller's whole-operator join serializes
each replay before the next admission. Every attempted submission is retained
even when a later record or launch fails. There is no unconditional per-replay
clear ahead of admission. The existing scheduler, READY/REQUESTED handshake,
retirement nonce and native terminal publication remain responsible for execution.

## Device identity and failure propagation

Cold registration independently retains context generation, capture generation,
callable occupancy generation, Runtime and terminal-region addresses, argument
counts and participant counts. Replay packets cannot self-register from their
own pointers. Admission validates the complete invocation before either processor
can modify shared Runtime or arena state. The existing invocation wire header
keeps its generation field, and the launch-envelope header remains 192 bytes.
Capture packets place the full 24-byte control/context/generation reference at
the start of their payload, followed by the unchanged invocation. They do not
store the reference in architecture-dependent header padding. Ordinary eager
and PROGRAM packet layouts keep their existing offsets.

Before cold registration first reads the H2D-initialized control, a2a3 performs
`cache_invalidate_range` on exactly that aligned 128-byte allocation, including
its completion barrier. An allocation can reuse an old lease's address; it does
not prove cold cache. The previous lease must already be quiescent and its
control writes cleaned by replay completion before its allocation is released.
A5 uses the project's coherent host-DMA contract and adds no cold invalidation.
This operation is absent from replay and revocation; it is not an assumed RTS
launch-boundary guarantee. A3 and A5 hardware results must be reported separately.

Each admitted replay receives a nonzero device epoch. The fixed argument binding
epoch remains separate. Every raw AICPU participant reports completion, including
affinity-filtered and rejected participants. Before publishing group completion,
the last participant consumes the mainline terminal record for that replay.
A missing terminal is an error; the validated terminal preserves the original
signed failure code. Native stream synchronization also covers AICore return.

The first failed replay permanently closes that lease's admission. Already
queued later replays complete passively without entering the shared Runtime.
An incomplete prior CPU group also prevents new admission. Callable unregister
first quiesces the recorded caller and revokes the device registration; same-ID
reuse cannot revive an old graph. Retained code memory is pinned until release.

## Validation and scope boundaries

CPU tests cover packet validation, independent registration, stale context,
callable reuse, replay epochs, missing and failed terminal records, participant
completion, return-code and exception exits at every binder operation, and
the exported record entry's host-allocation/loader exception paths. Hardware tests use native
CANN capture and replay for repeated execution, empty/active transitions,
descriptor/scalar snapshots, revoked graphs, prequeued work after orchestration
failure, release retry and subsequent eager execution. The recapture case
retains only the released control backing in a test allocator and reuses its
real device address for four sequential leases. It reports each address and
generation, verifies fresh replay epochs and checks outputs before release.
This controlled reuse case does not measure natural allocator reuse frequency
or reproduce a stale-cache failure on the old implementation. Report A2/A3 and A5
results separately; simulation only validates capability refusal.

Registration interaction tests cover all three registration entries with a
live lease, duplicate IDs, wrong borrowed context without allocation, mutation
behind queued replay, and automatic same-ID reuse followed by three prequeued
old-graph refusals. CPU tests retain ordinary-run refusal and reject registry
mutation when a non-capture owner is presented as a lease.

These tests do not establish physical stop after a native kernel skip or timeout.
Such failed completion proofs retain the lease. HBG graph templates and restore,
framework adapters, multiple graph leases, cross-stream concurrent replay and
capture DFX are separate changes.

These boundaries are explicit: ordered changes to the contents of an already
bound DEVICE Tensor are supported. Changing a scalar, Tensor address or shape
in the original host argument object does not patch the recorded invocation.
A second prepare on the leased context, nonzero pipeline slot or requested
diagnostics is refused. Metadata publication occurs on the cold path before
recording; this fixed-image contract is not graph-owned per-node publication.
Multi-node publication banks, graph-local Root/Ready dependencies, explicit
graph updates and replay diagnostics require their own complete implementation
and acceptance. The native fixed-binding result does not establish that those
features or the public graph adapter are available.
