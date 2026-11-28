# AICPU Kernel Launch Mechanisms

How a host process makes a custom AICPU SO runnable on the device. There
are **three known methods** in CANN; this repo's runtime uses one of
them, the tool at
[`tools/cann-examples/aicpu-kernel-launch/`](../tools/cann-examples/aicpu-kernel-launch/)
implements the same one as a standalone reference, and a third was
attempted in PR #537 but is unusable due to a CANN-side cache coherency
bug ([issue #822](https://github.com/hw-native-sys/simpler/issues/822)).

This doc records all three so the failure lore from #822 doesn't have
to be re-derived if anyone reaches for Path B again.

## Comparison

| Method | Where the SO lands | Sudo? | Multi-runtime per process? | Iterative dev? | Status |
| ------ | ------------------ | ----- | -------------------------- | -------------- | ------ |
| **1. tar.gz pre-deployment** | `/usr/lib64/aicpu_kernels/.../` at CANN install time | Yes (root-owned) | Yes | No — redeploy per change | Works, but obsolete for this repo |
| **2. Path A — dispatcher Mode A bootstrap** | `/usr/lib64/aicpu_kernels/<vf>/aicpu_kernels_device/<hostPid>_<localDeviceId>/simpler_inner_<fp>_<dev>.so` written at runtime by the AICPU OS process | No | **No** (one SO per host process — see "Path A latch") | Yes | **Production runtime + reference tool** |
| **3. Path B — `KERNEL_TYPE_AICPU_CUSTOM`** | `/home/CustAiCpuUser/cust_aicpu_<dev>_<vf>_<pid>/` written by the `aicpu_custom_scheduler` subprocess | No | Yes (latch lifted) | Yes | **Broken** — cust subprocess L1 stale on AICore HBM writes (#822) |

The columns above frame the trade-offs. The rest of this doc explains
each row, then the failure forensics for Path B.

## Method 1: tar.gz pre-deployment (classical)

Ship the inner SO inside a CANN custom-AICPU tarball. The tarball is
extracted into `/usr/lib64/aicpu_kernels/` at CANN install time (or by
the operator running an unpack script). At runtime CANN loads it as if
it were a built-in kernel.

- **Mechanism**: out-of-band file deployment, no special API
- **Deploy time**: install-time, requires root (destination is
  root-owned); for dev iteration the operator has to redeploy the tar
  by hand
- **Multi-runtime**: works fine. The CANN-side `firstCreatSo_` one-shot
  latch (see Path A) is bypassed because the SO was already loaded
  during CANN's own init — there's no second `SaveSoFile` call to
  silently no-op
- **Cache coherency**: not affected. The kernel runs inside the main
  `aicpu_scheduler` cluster, which shares an L1 snoop domain with
  AICore HBM writes
- **Used by**: pre-2024 CANN AICPU custom kernels and any environment
  that has a fleet provisioning pipeline. Not used by this repo —
  iterative development on shared dev boxes makes sudo + per-change
  redeploy a non-starter

## Method 2: Path A — Mode A dispatcher bootstrap (this repo)

The host wraps an inner SO's bytes inside a `DeviceArgs` payload and
invokes
`rtAicpuKernelLaunchExWithArgs(KERNEL_TYPE_AICPU_KFC, "AST_DYN_AICPU", ..., "libaicpu_extend_kernels.so")`.
CANN's preinstalled `libaicpu_extend_kernels.so` dlopens the dispatcher
SO inside the AICPU OS process; the dispatcher writes the inner SO
bytes to
`/usr/lib64/aicpu_kernels/<vf>/aicpu_kernels_device/<hostPid>_<localDeviceId>/simpler_inner_<fp>_<dev>.so`,
then returns. The host then issues a normal
`rtsBinaryLoadFromFile(json, ...)` + `rtsFuncGetByName` +
`rtsLaunchCpuKernel` sequence pointing at the preinstall basename.

Each runner uploads again, including after a device reset. CANN's TSD removes
the process directory at session teardown. Historical files in the shared root
remain outside this cleanup; see [the loader contract](../src/common/aicpu_loader/README.md).

- **Mechanism**: AICPU OS process has write access to the preinstall
  subtree even though the directory is root-owned (the AICPU OS runs
  with a CANN-managed uid that holds the write capability). The host
  process itself never touches the filesystem destination
- **Deploy time**: at runtime, no operator action
- **Sudo**: not needed — the file write happens device-side
- **Multi-runtime — Path A latch**: this is the core limitation.
  CANN's preinstalled `libaicpu_processer.so` (the AICPU OS scheduler
  driving `KERNEL_TYPE_AICPU_KFC`) holds a process-wide
  `BackendServerHandleManager::firstCreatSo_` one-shot latch inside
  `SaveSoFile`. The first successful `SaveSoFile` flips the latch;
  subsequent calls return success **without writing the SO**, so a
  second runtime's inner SO load silently no-ops, and the second
  runtime's kernel either fails to load or runs the first runtime's
  code. **Within one host process you can ship exactly one inner SO
  via Path A.** This forces this repo's
  [`ChipWorker`](../src/common/worker/) model: one host process binds
  one (arch, runtime) pair; multi-runtime work fans out across
  processes
- **Cache coherency**: not affected. Same cluster as AICore snoop
  domain, same as Method 1
- **Used by**: this repo's production runtime
  (`src/{a2a3,a5}/runtime/.../host/runtime_maker.cpp`); the
  [`aicpu-kernel-launch`](../tools/cann-examples/aicpu-kernel-launch/)
  reference tool; and the
  [`aicpu-device-query`](../tools/cann-examples/aicpu-device-query/)
  probe tool

The dispatcher exports three symbols
(`StaticTileFwkBackendKernelServer` +
`DynTileFwkBackendKernelServerInit` +
`DynTileFwkBackendKernelServer`) — see
[`src/common/aicpu_loader/device/`](../src/common/aicpu_loader/device/)
for the symbol-level contract.

### Bootstrap ABI vs per-task launch ABI

There are two similarly shaped but separate argument channels in the current
onboard Path A implementation:

1. **Dispatcher bootstrap private ABI.**
   `LoadAicpuOp::BootstrapDispatcher()` builds a private argument blob for
   `rtAicpuKernelLaunchExWithArgs(KERNEL_TYPE_AICPU_KFC, ...)` and writes a
   `device_args_ptr` at offset 40. The dispatcher reads that pointer as an
   extended `DeviceArgs` object carrying dispatcher SO bytes, inner runtime SO
   bytes, and `device_id`. This ABI belongs only to bootstrap and is kept in
   `src/common/aicpu_loader/{host,device}`.
2. **AICPU per-task launch ABI.**
   The per-task launch hands the front-less `KernelArgs` payload directly to
   `rtsLaunchCpuKernel()`. `argsSize` includes the header and any trailing
   invocation, graph-definition and runtime-descriptor sections. The AICPU entry
   receives the packet base directly. Onboard HBG and TMR Kernel locate the
   delivered descriptor by its packet-relative offset and byte length.
   Two sibling entries reuse the same launch mechanism with their own, smaller
   payloads instead of `KernelArgs`:
   - `simpler_aicpu_init` takes an `InitArgs` (device id + log config), launched
     once per device at `ensure_device_initialized` time. It latches the
     per-device invariants into the resident AICPU SO globals, so `exec` and
     `register_callable` no longer re-push them.
   - `simpler_aicpu_register_callable` takes a `RegisterCallableArgs` (orch-SO
     descriptor extracted from `Runtime`), launched per callable on the
     device-orchestration prepare path so the AICPU (re)dlopens its orch SO. hbg
     is a no-op (host-side orchestration).
3. **Shared per-task `KernelArgs` payload.**
   `src/{a2a3,a5}/platform/include/common/kernel_args.h` is front-less.
   The header carries the Runtime location, profiling buffer bases and register tables. **AICPU** receives this payload by value, as the
   launch argument blob. **AICore** receives `AicoreLaunchArgs`, a projection of
   the same struct holding the subset its entry reads, also by value in its own
   launch argument block. Its runtime fields are a scalar projection: shared
   handshake address, worker count and architecture-specific bootstrap values. Neither side reads a device-resident copy. Per-device
   invariants (device id, log config) are NOT on `KernelArgs` — they travel once
   via `InitArgs`.

The per-task launch envelope reserves 256 bytes for its header. A gated A2/A3
launch carries an execution-gate address, native invocation identity and lease
generation in both `KernelArgs` and the positional AICore argument block.
The gate has four separate 64-byte lines: Host decision, AICPU outcome, AIC
outcome and AIV outcome. The Host publishes the immutable GO/CANCEL decision
before releasing the stream notifications; device entries invalidate and check
it before changing runtime, profiling or handshake state. CANCEL returns from
both entries without invoking the runtime. PENDING, an unknown decision or an
identity mismatch does not execute user work. Host completion checks the entry
outcomes after the submitted streams drain; an entry outcome alone is not a
completion fence. An all-zero address/identity/generation tuple selects ordinary
execution. A5 entries reject nonzero gate tuples.

The executable entry ABI is version 2 (`simpler_aicpu_exec_abi2` and the
`aicore_kernel_abi2` mixed AIC/AIV entries). The Host checks defined executable
symbols in the actual ELF symbol tables before accepting the pair, so an old
or mixed device image cannot silently ignore the gate fields.

Retained chip-swimlane may reserve an identity-bound bucket while a gate is
held. Cancellation does not run ordinary terminal reconciliation: after both
private streams and publication complete, the native owner checks the CANCEL
entry outcomes and issues a private producer-never-started receipt. Withdrawal
checks that receipt against the still-admitting bucket, then waits for reader
reference acknowledgements before returning its bank and bucket. A failed
proof or acknowledgement retains the unresolved ownership. Other device
profiling channels do not use this gate path.

Keep those channels distinct. The bootstrap ABI still uses the `DeviceArgs`
name because the dispatcher really reads that structure. The platform per-task
ABI passes only the front-less `KernelArgs` and carries no CANN launch front.

The per-task launch buffer begins with these `KernelArgs` fields:

```text
KernelArgs + 0:  borrowed runtime_args or inline image section (offset, bytes)
KernelArgs + 8:  regs
```

An earlier revision wrapped this payload behind a 48-byte CANN launch front
(`AicpuLaunchArgs`, an opaque pointer at offset 40 plus the payload at offset
48) on the belief that `rtsLaunchCpuKernel` required it: removing the front had
produced CANN `507018` on a2a3 onboard. That failure was traced to build
skew — the AICPU inner SO and AICore `.o` had not been rebuilt against the
moved offsets — not a CANN requirement. With a clean rebuild, passing the bare
`KernelArgs` (`runtime_args` @ 0, `argsSize = sizeof(KernelArgs)`) is verified
working on a2a3 onboard across HBG and TRB, so the front was removed.

### Ownership boundaries

- `LoadAicpuOp` still owns both dispatcher bootstrap registration and cached
  per-task AICPU entry launches. Splitting those into separate loader objects
  would be a larger lifecycle refactor.

### Runtime descriptors and shared state

`DeviceRuntimeLaunchDesc` contains the invocation's delivered control values and
borrowed device addresses. `RuntimeSharedState` contains the AICPU/AICore
handshake reports and teardown gates. Both processors use the same shared-state
address; the platform initializes its reports once before use and retains the
allocation until all native or captured consumers have ended.

Onboard HBG Program and Kernel execution append the compact Runtime descriptor
to the AICPU launch package after any Graph Definition. The header gives the
descriptor offset and length and, when present, the execution image offset and
length. The image length covers its header, snapshot and reference table.
RTS copies the complete packet by `argsSize` with zero host-input relocations.
TMR Program publishes a separately allocated descriptor on the AICPU stream.

In TMR Kernel mode the host appends the KernelInvocation at byte 256 and the
Runtime control descriptor after the invocation. The packet identifies the
descriptor by its byte offset from the packet base and its length. RTS copies
the complete packet according to `argsSize`, with zero host-input relocations.
The AICPU validates the layout before reading the descriptor. Host preparation
snapshots the control values and invocation arguments for this launch.

AICore receives the shared-state address and required bootstrap values directly
as scalar launch arguments. It never borrows a descriptor from an AICPU argument
package. AICPU keeps a thread-local-lifetime aligned copy of its delivered
control prefix: RTS argument pools can be only eight-byte aligned, while Runtime
control access requires cache-line alignment. Copying the control view preserves
its borrowed device addresses. No per-thread view is published into
shared runtime state or retained after its device entry returns.

TMR KernelInvocation carries tensor descriptors and scalar values, not tensor
storage. HBG Graph Definitions keep their per-reader launch-envelope view.
In onboard HBG Program and Kernel modes, the packed initial execution image is
a separate section of the same packet. The native AICPU entry checks the
section bounds, restores the mutable prefix to the working arena and binds
the device fields to their offsets in the RTS snapshot before releasing the
scheduling threads on each execution. When these references are present, the
snapshot data starts at an actual 64-byte-aligned device address. Simpler
retains this arena separately so it stays valid through device retirement and
any retained model use. Simulation delivers the image through a synchronous copy.
Caller input/output buffers remain caller-owned.

Kernel eager and capture use the same argument layout. Capture retains the RTS
task arguments and simpler's referenced shared allocations for future replays.
The caller holds the original run until the model and its in-flight replays have finished. Replay does not reread a modified host argument packet. Each graph keeps
its own delivered values while its external references obey the existing caller
ordering and lifetime contract.

RTS argument storage carries the descriptor and HBG image without a Simpler-owned
pinned upload source for those bytes. HBG uses the same RTS route for Program,
Kernel eager and capture. AICPU and AICore receive the address of the
same Simpler-owned handshake allocation. The mutable image prefix is copied on device
into the writable execution arena. Shared scheduler storage and notification or
counter handles stay with Simpler's prepared resource owner. Staged scheduler
metadata still uses retained pinned sources and ordered publication. A submitted
copy or kernel can survive a failing API call. Partial-launch cleanup retains
those resources until physical completion or confirmed device reset. Normal
shared-state release preserves failed-free ownership for retry.

## Method 3: Path B — `KERNEL_TYPE_AICPU_CUSTOM` (broken — #822)

The path that PR #537 attempted in order to lift Path A's latch and
allow one host process to bind both `host_build_graph` and
`tensormap_and_ringbuffer` runtimes simultaneously. Used in production
by other CANN customers; never made it to green in this repo due to a
CANN-side cache coherency bug.

### How it was supposed to work

JSON descriptor declares each inner-SO function with
`opKernelLib=AICPUKernel + userDefined=True`. CANN
(`cann/runtime/src/runtime/core/src/kernel/program_common.cc`) routes
`userDefined=True` to `KERNEL_TYPE_AICPU_CUSTOM (4)`. The
`aicpu_custom_scheduler` subprocess (separate from the main
`aicpu_scheduler` used by Path A) handles `KERNEL_TYPE_AICPU_CUSTOM`
calls.

`cann/runtime/src/aicpu_sched/aicpu_processer/ae_so_manager.cc::GetSoPath`
makes `KERNEL_TYPE_AICPU_CUSTOM` the **only** kernel type that looks
under `/home/CustAiCpuUser/cust_aicpu_<dev>_<vf>_<pid>/` — every other
type only searches `/usr/lib64/aicpu_kernels/...`, which is unwritable
without root. A gate at `ae_so_manager.cc:514` (`IsCustAicpuSd()`)
enforces that `KERNEL_TYPE_AICPU_CUSTOM` must execute inside the cust
subprocess; a violation aborts the load.

The latch problem is genuinely solved: `firstCreatSo_` lives in
`libaicpu_processer.so`, which the cust subprocess does not link.
Multiple inner SOs can coexist.

### How it actually fails

PR #537 reached the point where all the routing was correct: CANN
dispatched the `Dyn*` exports to the cust subprocess, the inner
`libsimpler_aicpu_<runtime>.so` was dlopen'd, all three phases
(Null / Init / Run) entered our code, and
`SchedulerContext::handshake_partition` step 1 successfully wrote
`complete=1` to all 9 cores' `Handshake` slots in HBM. **The host's D2H
readback confirmed the AICPU's writes landed in HBM**, AICore picked
them up, ran past its phase 1, and wrote its report (`aicore_done` +
`physical_core_id` + `core_type`) back into the same `Handshake` slots.

Step 2 of `handshake_partition` is then a spin-wait:

```cpp
// src/{arch}/runtime/tensormap_and_ringbuffer/runtime/scheduler/scheduler_cold_path.cpp
while (hank->aicore_done == 0) {}   // ← cust AICPU stuck here forever
```

The cust AICPU's L1 cache holds a stale `0` for `aicore_done`.
HBM has `1`, the host's D2H readback sees `1`, but the cust AICPU never
observes the change. After 2 s,
`aclrtSynchronizeStreamWithTimeout(stream_aicpu_)` reports
**`ACL_ERROR_RT_AICPU_EXCEPTION (507018)`**.

The mechanism: `cann/runtime/src/aicpu_sched/aicpu_schedule/core/aicpusd_worker.cpp::SetAffinity`
binds the cust subprocess's worker threads to `cpuId=0` (the AICPU
cluster reserved for OS-side work) rather than to the same AICPU
cluster that drives AICore. **The cust cluster's L1 is not in
AICore's HBM-write snoop domain**, so the cust AICPU never sees
AICore's writes until something explicitly invalidates the line.

Method 1 and Method 2 dodge this because they run on the main
`aicpu_scheduler` cluster, which **is** in AICore's snoop domain.

### User-space workarounds that DO NOT work

For anyone tempted to re-attempt: all four standard ARM64 cache-bypass
primitives have been tried and fail. Documenting why each one fails
saves a future session from running the same experiment.

| Attempt | Result | Why it doesn't help |
| ------- | ------ | ------------------- |
| `volatile uint32_t` field qualifier | No effect | Prevents the compiler from caching the value in a register / reusing it across statements. The CPU still reads from L1 (or whatever level holds the stale line). Cache coherency is an architectural issue below the C abstract machine. |
| `__atomic_load_n(..., __ATOMIC_ACQUIRE)` (compiles to `ldar`) | No effect | `ldar` is an ordering instruction — it orders this load with respect to later loads/stores. It does **not** force a coherent reload from memory or invalidate the L1 line. |
| `dc civac` (clean + invalidate by VA to Point of Coherency) in spin loop | Worse — corrupts AICore data | The same cache line holds AICPU-written fields (`aicpu_ready`, `task`) and the AICore-written field (`aicore_done`). `civac` writes back the AICPU's dirty stale view of the line, **clobbering AICore's HBM writes** for the AICPU-owned fields. |
| `dc ivac` (invalidate-only by VA to Point of Coherency) in spin loop | Silently NOP'd | EL0 access to `dc ivac` is gated by `SCTLR_EL1.UCI`. The Linux kernel inside the cust subprocess has `UCI=0`, so the instruction traps and the kernel handler silently turns it into a NOP rather than emulating it. From userspace it looks like the instruction ran with no effect. |

The fix has to live somewhere with the privilege to either change the
memory attribute, change the affinity, or enable the EL0 cache op. All
four candidates are CANN-side or driver-side:

| # | Where | Change |
| - | ----- | ------ |
| A | CANN device kernel / driver | Set `SCTLR_EL1.UCI=1` for the cust subprocess so EL0 `dc ivac` works; user-space spin loops can then explicitly invalidate |
| B | CANN runtime / driver | Allocate handshake HBM with non-cacheable / write-through attribute when called from cust subprocess context. Small per-access HBM latency cost |
| C | CANN cust scheduler | Bind cust worker threads (`aicpusd_worker.cpp::SetAffinity`) to the same AICPU cluster as AICore's snoop domain instead of `cpuId=0` |
| D | this repo's runtime | Split `Handshake` so AICPU-written and AICore-written fields live on disjoint cache lines, then `dc civac` only the AICore-written line. Insufficient on its own because EL0 invalidate is still NOP'd — only works combined with A. Alternatively: replace the spin-wait protocol with a device event/notify primitive that bypasses shared-memory polling (substantial refactor) |

Issue #822 was closed (2026-05-20) as COMPLETED after CANN-side
mitigation landed. **The failure modes documented above remain the
durable knowledge.** If a CANN upgrade ever silently regresses one of
the A/B/C fixes, the same symptom will return; the diagnosis recipe
above stays valid.

## How to choose

- **Default to Method 2 (Path A)** for any new AICPU work in this
  repo. The reference tool at
  [`tools/cann-examples/aicpu-kernel-launch/`](../tools/cann-examples/aicpu-kernel-launch/)
  is the template
- **Use Method 1 (tar.gz)** only if you cannot avoid sudo at runtime
  and you have an existing fleet provisioning pipeline to redeploy
  through. No active use case in this repo
- **Do not reach for Method 3 (Path B)** unless you have an
  independent reason to believe at least one of CANN-side fixes A/B/C
  has shipped on your stack, AND you have a fresh end-to-end repro to
  confirm. The 507018 deadlock is silent — your test will look like a
  normal stream timeout from any other cause

## References

- Issue [#822](https://github.com/hw-native-sys/simpler/issues/822) —
  the bug report with the diagnostic D2H recipe and CANN source
  pointers
- PR #537 — the migration that shipped Path A (dispatcher bootstrap plus
  per-task `rtsLaunchCpuKernel`)
- [`src/common/aicpu_loader/`](../src/common/aicpu_loader/) —
  the dispatcher bootstrap, three-symbol device contract, and per-task
  launch loader
- [`tools/cann-examples/aicpu-kernel-launch/`](../tools/cann-examples/aicpu-kernel-launch/) —
  the standalone reference tool implementing Path A
- [`tools/cann-examples/aicpu-device-query/`](../tools/cann-examples/aicpu-device-query/) —
  another Path A consumer, but device-side HAL probe rather than
  generic launch
- CANN sources that defined the failure surface (paths inside the
  CANN open-source release):
  - `cann/runtime/src/runtime/core/src/kernel/program_common.cc` —
    `opKernelLib` → `kernelType` routing table
  - `cann/runtime/src/aicpu_sched/aicpu_processer/ae_so_manager.cc` —
    `GetSoPath` cust-vs-inner routing, `IsCustAicpuSd` gate, the
    `SaveSoFile` latch
  - `cann/runtime/src/aicpu_sched/aicpu_schedule/core/aicpusd_worker.cpp` —
    `SetAffinity` thread binding
  - `cann/runtime/src/aicpu_sched/aicpu_schedule/core/aicpusd_cust_so_manager.cpp` —
    cust SO upload destination
