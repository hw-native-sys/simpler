# Runtime Capability Survey — Topology, Launch, Communication

**Snapshot: verified against `main` @ `e6a61f15`, 2026-07-27.**

This is a **status** document: for each mechanism it answers *is this shipped,
exercised, gated, or only designed?* It deliberately does not re-explain
architecture — the reference docs below own that, and this page links to them
rather than restating them:

- Hardware substrate → [hardware/chip-architecture.md](hardware/chip-architecture.md),
  [hardware/mmio-performance.md](hardware/mmio-performance.md),
  [hardware/cache-coherency.md](hardware/cache-coherency.md)
- Level model → [hierarchical-level-runtime.md](hierarchical-level-runtime.md)
- Three-program model → [chip-level-arch.md](chip-level-arch.md)
- AICPU launch mechanics → [aicpu-kernel-launch-mechanisms.md](aicpu-kernel-launch-mechanisms.md)
- Comm domains / overlays → [comm-domain.md](comm-domain.md)
- Remote L3 / L4 → [remote-l3-worker-design.md](remote-l3-worker-design.md)

Status claims rot faster than architecture. Re-derive rather than trust this
page when the snapshot SHA is old; every claim carries a `file:line` so it can
be re-checked directly.

## Status vocabulary

| Status | Meaning |
| ------ | ------- |
| **Shipped** | Implemented, reachable in a default build, covered by CI |
| **Shipped, not CI-run** | Implemented and reachable, but no CI job exercises it |
| **Gated** | Implemented but unreachable in any configuration available here |
| **Design only** | Documented contract or plan; no working data path |
| **Name only** | An identifier exists (enum, macro, string) with no implementation behind it |

## Topology: L2 (host / AICPU / AICore) → L4 (host → remote host)

The 7-level model (L6 Cluster … L0 Core) is declared in
[hierarchical-level-runtime.md](hierarchical-level-runtime.md). Its own status
table is accurate: L3 implemented; L4 local implemented, remote host_tcp shipped;
L5/L6 untested.

**The level is a label in C++ and a real branch in Python.** `Worker` stores
`level_` (`src/common/hierarchical/worker.cpp:56`) and nothing else reads it, so
L3–L6 execute byte-identical Orchestrator / Scheduler / WorkerManager code. The
Python facade *does* branch: `level == 2` → single `ChipWorker`, `level >= 3` →
hierarchical, anything else `ValueError`
(`python/simpler/worker.py:4117-4123`); `add_remote_worker` and the
remote-memory APIs require `level >= 4` (`:2367`, `:2603`). A worker level is a
position in the orchestration tree and carries no physical meaning by itself.

### L2 — one chip (shipped)

The host cannot dispatch to AICore at all; all on-chip work is submitted to
AICPU, which runs the graph (`hardware/chip-architecture.md:132-136`). GM is
shared across a device's AICPU/AICore tiers but exclusive per `device_id`.

| Aspect | a2a3 | a5 |
| ------ | ---- | -- |
| Die ↔ `device_id` | a2: 1 die → 1 id. a3: 2 dies → **2 ids**, and both dies **share one AICPU OS** (`src/a2a3/docs/hardware.md:20-24`) | 2 dies → **1 id**, one orchestrator per id (`src/a5/docs/hardware.md:12-14`) |
| AICore per die | 24 clusters → 24 AIC / 48 AIV | 18 clusters → 18 AIC / 36 AIV |
| `PLATFORM_MAX_BLOCKDIM` | 24 → 72 cores | 36 → 108 cores |
| AICPU cores per die | 2 clusters × 4 = 8 (`src/a2a3/docs/hardware.md:30-32`) | 2 clusters × 2 = 4 (`src/a5/docs/hardware.md:17-19`) |
| AICPU threads | max 4, launch bound 6 | max 7, launch bound 14 (7 physical × SMT) |
| Host-map (SVM) | `halHostRegister(DEV_SVM_MAP_HOST)`, dlsym'd (`src/a2a3/platform/onboard/host/device_runner.cpp:78-102`) | not overridden |
| Comm window | Fabric V2 handles, VMM-IPC fallback | VMM shareable handles only |

Two numbers surprise readers who skip the arch docs: the default
`aicpu_thread_num` is **0 (auto)** — each architecture resolves it against its
usable AICPU topology — and AICore geometry is not selectable at all, because
`resolve_block_dim()` unconditionally takes the device maximum
(`src/common/platform/onboard/host/device_runner_base.cpp:1201-1212`).
Each arch doc's "three views of how many cores" section explains why the spec
count, the silicon count, and the runtime-visible count differ
(`src/a2a3/docs/hardware.md:42-51`).

The AICPU↔AICore control path is the AIC_CTRL MMIO window: one LDR costs
95–105 ns and serializes within a thread, so polling N cores from one thread
costs ~95 ns × N and only more AICPU threads shrink the round
([hardware/mmio-performance.md](hardware/mmio-performance.md)).

The **AICPU role split is per-runtime, not per-arch**: `tensormap_and_ringbuffer`
decouples an orchestrator thread at `tidx == nthreads-1` when `nthreads > 1`
and `SIMPLER_TMR_SERIAL_ORCH_SCHED_ENABLE` is off
(`src/a2a3/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp:260-273`);
`host_build_graph` runs no device-side orchestrator.

### L3 — one host, many device ids (shipped)

One forked process per `device_ids` entry, each binding its own `ChipWorker` in
the child (`python/simpler/worker.py:4366-4409`). Every parent↔child edge is a
POSIX-shm mailbox, and the HeapRing is mmap'd **before** any fork so children
inherit it (`src/common/hierarchical/worker.cpp:55-66`) — that shared region,
not just the mailbox, is what makes parent-allocated tensors visible to chip
children. Collectives are explicitly single-host: both backends state "Scope:
L3 single-host multi-card only" (`src/a2a3/platform/onboard/host/comm_hccl.cpp:19`).

Two additional shipped L3↔L2 channels bypass the level abstraction and talk to
AICPU directly: [l3-l2-orch-comm.md](l3-l2-orch-comm.md) and
[l3-l2-message-queue.md](l3-l2-message-queue.md), with examples under
`examples/workers/l3/`. Parent-directed targeting of a specific child is
[directed-next-level-scheduling.md](directed-next-level-scheduling.md).

### L4 — host → remote host (host_tcp data plane shipped)

Shipped: `add_remote_worker(RemoteWorkerSpec)` → JSON manifest over TCP →
`simpler-remote-worker` daemon → session runner → C++ `RemoteL3Endpoint`
registered as a NEXT_LEVEL endpoint and dispatched by the same Scheduler
(`python/simpler/worker.py:2362-2375`; `src/common/hierarchical/worker.cpp:82-93`).
The remote session builds a real `Worker(level=3, device_ids=…)` and initialises
its chip subtree before answering READY.

`transport` defaults to `"host_tcp"`, and the daemon rejects other
profiles. Remote buffers use session-managed host storage, so `REMOTE_WINDOW` /
`UB_LDST` remain protocol placeholders for future peer-memory transports. The
A2 RoCE / A3 HCCS / A5 UB HCOMM profiles are documented contracts marked
hardware-gated
([remote-l3-worker-design.md](remote-l3-worker-design.md):71-77).
`examples/workers/l4/vector_add_mixed_l3/` is the L4 example — one parent
driving a local L3 subtree and a remote one on a second machine — and the
`st-pod-onboard-a2a3` job starts the daemon on that peer.

## Launching AICore and AICPU work (the CANN surface)

The runtime binds the low-level **`rt*` / `rts*`** layer, not the public
`aclrt*` launch API. Seven entry points, one of which is a2a3-only:

| Purpose | CANN call | Site | Status |
| ------- | --------- | ---- | ------ |
| Register AICore ELF | `rtRegisterAllKernel` | `src/common/platform/onboard/host/device_runner_base.cpp:1118` | Shipped |
| Launch AICore | `rtKernelLaunchWithHandleV2` | `device_runner_base.cpp:1139` | Shipped |
| Bootstrap AICPU SO | `rtAicpuKernelLaunchExWithArgs` (`KERNEL_TYPE_AICPU_KFC`) | `src/common/aicpu_loader/host/load_aicpu_op.cpp:201` | Shipped |
| Register AICPU op | `rtsBinaryLoadFromFile` | `load_aicpu_op.cpp:341` | Shipped |
| Resolve AICPU entries | `rtsFuncGetByName` (loops all declared symbols) | `load_aicpu_op.cpp:362` | Shipped |
| Launch AICPU | `rtsLaunchCpuKernel` | `load_aicpu_op.cpp:401` | Shipped |
| Unload AICPU binary | `rtsBinaryUnload` | `load_aicpu_op.cpp:226`, `:334` | Shipped |
| FFTS base address | `rtGetC2cCtrlAddr` → `KernelArgs::ffts_base_addr` | `src/a2a3/platform/onboard/host/device_runner.cpp:121` | Shipped, **a2a3 only** |

The unload asymmetry explains a design that otherwise looks arbitrary: the
AICPU binary *can* be unloaded, the AICore ELF *cannot* (CANN exposes no public
`rtUnregisterAllKernel`), so the ELF handle is registered lazily and cached.
Per-run registration pinned an extra device copy and surfaced as `207001` at
launch with a `507899` cascade at `rtStreamCreate`
(`device_runner_base.cpp:1101-1124`).

Per-*task* AICore kernels are not CANN-registered at all: they are `rtMemcpy`'d
into device GM as a `ChipCallable` image and entered through a raw device
function pointer
(`src/a2a3/runtime/tensormap_and_ringbuffer/aicore/aicore_executor.cpp:37-41`).

### AICPU bootstrap

Three stages, all shipped. A KFC launch targets CANN's preinstalled
`libaicpu_extend_kernels.so` (`DynTileFwkKernelServerInit`) carrying a private
byte-offset ABI; the device-side dispatcher writes the inner SO to
`/usr/lib64/aicpu_kernels/...` keyed by a truncated ELF Build-ID; then JSON
registration with `opKernelLib=AICPUKernel`. Two symbols are launched, not one:
`simpler_aicpu_init` (per-device one-shot `InitArgs`) and `simpler_aicpu_exec`
(`src/common/aicpu_loader/host/load_aicpu_op.h:155-158`).

`numBlocks` is the AICPU thread count, but placement belongs to CANN, so the
host over-launches and a device-side gate reads `sched_getcpu()` and drops every
thread outside a host-chosen allowed-CPU table
(`src/common/platform/onboard/aicpu/platform_aicpu_affinity.cpp:31-40`).

### Launch order and timeouts

AICore is launched **before** the AICPU Run task, because the first
`rtKernelLaunchWithHandleV2` lazily loads the binary on-device: ~1.4 s against a
busy device versus ~0.4 ms idle (`src/a5/platform/onboard/host/device_runner.cpp:360-410`).
The a2a3 mirror of this ordering is self-documented as unverified on a2a3
silicon.

Three timeouts exist, and the constants are **defaults, not the operative
values**: op-execute 45 s, stream-sync 50 s, scheduler 10 s
(`src/a2a3/platform/include/common/platform_config.h:69,85,75`). All three are
env-overridable with ordering validation
(`resolve_onboard_timeout_config`, `device_runner_base.cpp:66-110`;
[troubleshooting/local-timeout-defaults.md](troubleshooting/local-timeout-defaults.md)),
and **CI runs at 2 s / 3 s / 4 s** (the `SIMPLER_*_TIMEOUT_*` env block on each self-hosted job in `.github/workflows/ci.yml`). Triage
a CI timeout against the CI values, not the header constants.

Every failed launch runs a recovery path that is easy to miss:
`recover_device_or_mark_unusable` does a bounded
`aclrtSynchronizeDeviceWithTimeout`, and `force_reset_device()` calls
`aclrtResetDeviceForce` followed by a post-reset stream poison probe
(`src/a5/platform/onboard/host/device_runner.cpp:465-497`, `:567-620`).

### Not used here

- **`KERNEL_TYPE_AICPU_CUSTOM` ("Path B")** — absent from `src/`. Tried during
  PR #537 and abandoned per issue #822: the cust subprocess is bound to
  `cpuId=0`, whose L1 sits outside AICore's HBM snoop domain. Note that
  [aicpu-kernel-launch-mechanisms.md](aicpu-kernel-launch-mechanisms.md)
  **inverts** the Path A/B labels relative to issue #822's own usage. Treat
  Path B as unvalidated on this stack rather than as a live option.
- **ACL launch family** (`aclrtLaunchKernel*`, `aclrtBinaryLoad*`) and
  **`rtStarsTaskLaunch`** — no call sites. STARS appears only as the subsystem
  that reaps the op-execute timeout.
- **a2a3 pipeline slot 1** — plumbed end to end but dead; `run()` hard-codes
  slot 0 (`src/a2a3/platform/onboard/host/device_runner.cpp:218`).
- **Sim platform** uses no CANN at all: `dlopen` + `std::thread`, with
  `SIM_AUTO_BLOCKDIM = 8` and one OS thread per AICore
  (`src/common/platform/sim/host/device_runner_base.h:56-65`).

## Communication engines: SDMA, URMA, CCU, HCCL, Fabric

Four async engines are declared —
`ASYNC_ENGINE_{SDMA,ROCE,URMA,CCU}`
(`src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_types.h:53-59`) — and
**two are name only**. ROCE and CCU appear as enum constants,
`COMPLETION_ENGINE_*` macros, and `return "CCU"` strings in
`async_engine_name()`; there is no backend directory, submit path, poll op, or
test for either, and no doc states what they are intended to be. CANN itself
does treat CCU as a first-class engine (`COMM_ENGINE_CCU = 5` in
`hccl/hcomm_res_defs.h`), but nothing in `src/` references `COMM_ENGINE_*`.

The single wiring point is the completion-ops table: a2a3 registers COUNTER and
SDMA_EVENT_RECORD (`.../runtime/pto_async_wait.h:85-89`); a5 registers those
plus URMA_EVENT_HANDLE (`src/a5/.../pto_async_wait.h:90-98`). Out-of-range
values yield `PTO2_ERROR_ASYNC_COMPLETION_INVALID`.

| Engine | a2a3 | a5 | Status |
| ------ | ---- | -- | ------ |
| COUNTER (default) | registered | registered | **Shipped** — `async_notify_demo` runs onboard on both arches and `deferred_notify_demo` runs in sim on both, through the `st-onboard-*` / `st-sim-*` jobs in `ci.yml`. Routed by `CASES[*]["platforms"]`, no `skipif` |
| SDMA | build macro forced ON; runtime opt-in | `option(... OFF)` | a2a3 **Shipped** (the "SDMA pytest (a2a3)" step in `ci.yml`); a5 not built |
| URMA | absent | full implementation | **Gated** — see below |
| ROCE, CCU | enum only | enum only | **Name only** |

**a2a3 SDMA is opt-in at runtime**, not "always on": the provider is always
compiled, but provisioning the 48 STARS streams requires
`Worker(..., enable_sdma=True)`, default `False`
(`python/simpler/worker.py:4178`, `:4396`;
`python/bindings/task_interface.cpp:1367`). It is quarantined from the general
CI sweep via `@pytest.mark.sdma` for a measured hazard: with 48
device-only SDMA streams an AICore fault takes ~306 s to tear down versus ~0.3 s
without, traced to a single 300,000 ms remote TRS event timeout
([investigations/2026-07-a2a3-sdma-fault-teardown.md](investigations/2026-07-a2a3-sdma-fault-teardown.md),
issue #1425).

**a5 URMA is real code that cannot execute.** The scheduler walks `UrmaCqCtx`
CQEs checking the owner bit, advances the tail and rings the doorbell
(`src/a5/.../backend/urma/urma_completion_scheduler.h:133-215`); the kernel
submits `TGET_ASYNC`/`TPUT_ASYNC<DmaEngine::URMA>` with 256 MB chunking. Both
sit behind `PTO_URMA_SUPPORTED`, which is **defined nowhere in this repo and
nowhere in the installed CANN pto headers**, so the `#else` branch returns
`PTO2_ERROR_ASYNC_COMPLETION_INVALID` immediately. The host overlay macro is
fully wired, so turning it on does not help — the device path stays unreachable.
a5's SDMA and URMA overlays are mutually exclusive by CMake `FATAL_ERROR`
because `CommContext` exposes a single `workSpace` pair
(`src/a5/platform/onboard/host/CMakeLists.txt:49-53`).

**HCCL is bootstrap, not data movement.** The complete set of functions called
is `HcclGetRootInfo`, `HcclCommInitRootInfo`, `HcclBarrier`, `HcclCommDestroy`.
There is no `HcclAllReduce` / `AllGather` / `Send` / `Recv` anywhere; every
shipped collective is a hand-written AIV kernel that computes a peer pointer
from the symmetric window
(`tests/st/worker/collectives/allreduce/kernels/aiv/allreduce_ring_kernel.cpp:59-61`).

**Fabric is undocumented.** a2a3 prefers `alloc_windows_via_fabric()` with
`ACL_MEM_SHARE_HANDLE_TYPE_FABRIC` and falls back to `alloc_windows_via_ipc()`
only when Fabric is unsupported
(`src/a2a3/platform/onboard/host/comm_hccl.cpp:676`, `:762`). a5 has zero
occurrences of "fabric". No `.md` in the repo describes Fabric as a
memory-sharing mechanism.

One ABI fact constrains portability: the AICore→AICPU completion struct
diverged — `DeferredCompletionEntry` is 24 bytes on a2a3 and 32 on a5, the extra
8 being the `backend_cookie` URMA's poll needs. A URMA-on-a2a3 port must widen a
struct that is currently frozen by `static_assert`.

## Open questions

Unresolved after this survey, in rough order of how much they block:

1. **What are `ASYNC_ENGINE_ROCE` and `ASYNC_ENGINE_CCU` for?** No design doc,
   investigation entry, or code comment says whether they are reserved slots or
   leftovers from a dropped design.
2. **Has a5 URMA ever run on silicon?** No CI run, test artifact, or
   investigation attests to it.
3. **Which CANN mitigation closed issue #822, and is Path B usable on CANN
   9.0.0?** The doc says "CANN-side mitigation landed" without naming it, and
   nobody re-ran the repro.
4. **Does CANN's one-inner-SO-per-process latch still exist?** The whole
   one-process-per-`(arch, runtime)` ChipWorker model rests on it, but it is
   asserted only from CANN source paths that are not vendored here and no
   in-repo probe detects it.
5. **Where would `PTO_URMA_SUPPORTED` ever be defined?** Not in this repo, not
   in the installed CANN pto headers.
6. **Which platforms lack Fabric support?** Stated only as a code comment, never
   enumerated, and a5's divergence to the V1 handle route is undocumented.

## Documentation drift found while compiling this survey

Each entry below was re-verified directly at the cited line. Line numbers drift;
re-check before editing.

| Location | Claims | Reality |
| -------- | ------ | ------- |
| `hardware/chip-architecture.md:232` | AIC_CTRL MMIO is `Device-nGnRnE` | `Device-nGnRE`, proven from driver source (`hardware/mmio-performance.md:12`) |
| `worker-manager.md:174`, `:190` | dispatch poll uses `sleep_for(50us)`, "not busy-wait" | every mailbox wait spins; liveness is a wall-clock sample (`src/common/hierarchical/worker_manager.cpp:58`). A dispatch-path sleep is now banned by [codestyle](../.claude/rules/codestyle.md) rule 5 |
| `dynamic-linking.md:259` | AICore ELF is "registered each `launch_aicore_kernel()` call" | registration is lazy and cached (`device_runner_base.cpp:1101-1124`) |
| `dynamic-linking.md:355-361` | AICPU launches before AICore; call is `rtKernelLaunch` | AICore launches first; the call is `rtKernelLaunchWithHandleV2` |
| `comm-domain.md:111` | window is VMM + shareable-handle import with `aclrtDeviceEnablePeerAccess` | a2a3 prefers Fabric V2 (`comm_hccl.cpp:762`); the doc never mentions Fabric |
| `comm-domain.md:262-264` | a producer `CoreCallable` declares the SDMA workspace requirement | that API was removed by PR #1406 |
| `investigations/2026-07-a2a3-sdma-fault-teardown.md:153` | `sdma_async_completion_demo` is "unaffected by this change" | that demo sets `enable_sdma=True` (test:134), as CI's own comment states |
| `src/common/worker/pto_runtime_c_api.h:259` | "`config` carries block_dim (0 = auto)" | `CallConfig` has no such field — "There is no block_dim knob" (`src/common/task_interface/call_config.h:22`) |
| `src/common/platform/sim/host/device_runner_base.h:62-64` | "an explicit block_dim is still honoured" | same as above |
| `src/common/aicpu_loader/README.md:18-28`, `:47-51` | one `rtsFuncGetByName`; `device_id` in per-task `KernelArgs`; dispatcher under `build/lib/<arch>/onboard/<runtime>/` | loops all symbols; `device_id` lives on `InitArgs`; dispatcher is at `build/lib/<arch>/dispatcher/` |
| `tests/ut/cpp/a5/test_aicore_completion_mailbox.cpp:139` | "a5 is counter-only (no SDMA event-record backend)" | a5 registers SDMA and URMA ops (`src/a5/.../pto_async_wait.h:90-98`) |
