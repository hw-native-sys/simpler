# `tensormap_and_ringbuffer`: A2/A3 vs. A5

This document describes the substantive differences in the current code under
`src/{a2a3,a5}/runtime/tensormap_and_ringbuffer/`.

> **Maintenance baseline:** The source layout and classifications were verified
> on 2026-09-13. Recompute the counts and update the affected sections whenever
> the files or constants described here change.

## Comparison Boundary and Classification

The direct comparison covers tracked files under
`src/{a2a3,a5}/runtime/tensormap_and_ringbuffer/`, matched by relative path.
There are 51 paths present on both platforms and two additional paths present
only on A5. Every file in that boundary belongs to exactly one of these three
categories:

| Category | Count | Definition |
| -------- | ----: | ---------- |
| Byte-identical | 24 | The files at the same relative path have identical bytes |
| Compile-time or non-functional differences | 9 | Text differs, but the generated runtime behavior and data semantics are equivalent |
| Functional differences | 20 | Eighteen matching paths and two A5-only paths encode or document differences in runtime behavior, capacity, diagnostics, or supported backends |

A file is classified as functional when any part of its diff changes behavior,
even if the same diff also contains include-order, comment, or formatting
changes. Files outside the direct boundary, such as platform configuration and
PMU collector implementations, are cited only as supporting evidence.

The boundary shrinks when a pair is deduplicated rather than reconciled: three
`host/` paths left it for `src/common/tensormap_and_ringbuffer/host/`, where
there is one file and so nothing to compare.

## Byte-Identical Files

The following 24 files are byte-identical:

```text
build_config.py
common/runtime_status.h
docs/{SCALAR_DATA_ACCESS.md,SUBMIT_BY_CLUSTER.md,device_log_profiling.md,profiling_levels.md}
orchestration/{common.cpp,arg_with_deps.h,orchestration_api.h}
runtime/{common.h,async_kernel_api.h,dep_compute.h,orchestrator.h,
         runtime_core.cpp,runtime_core.h,shared_memory.h,tensor.h,
         tensormap.h,tensor_create_info.h}
runtime/scheduler/{scheduler.cpp,scheduler_types.h}
runtime/shared/{shared_memory.cpp,tensormap.cpp,runtime.cpp}
```

Byte identity is a textual result only. A shared file may still consume
platform-specific constants or APIs supplied by files outside this comparison
boundary.

The `host/` directory holds only `runtime_maker.cpp` and appears nowhere above:
`dep_gen_replay.{cpp,h}` and `runtime_compile_info.cpp` now exist once, in
`src/common/tensormap_and_ringbuffer/host/`, so they have no pair to compare.

## Compile-Time or Non-Functional Differences

The following 9 matching paths differ textually without changing runtime
behavior:

| Files | Difference |
| ----- | ---------- |
| `runtime/constants.h` | Path-derived include-guard macro names only |
| `runtime/backend/sdma/sdma_completion_kernel.h`, `runtime/types.h` | `#pragma once` on A2/A3 versus a path-derived include guard on A5 |
| `common/intrinsic.h` | A5 uses `s_block_idx` and `s_block_num` because the unprefixed names are compiler-reserved; getter semantics and layout are unchanged |
| `runtime/dispatch_payload.h` | Comments follow the platform-specific `LocalContext` field names; payload semantics are unchanged |
| `runtime/aicore_completion_mailbox.h` | Path-derived include guards and comment wording only |
| `runtime/completion_token.h` | Path-derived include guards and an A2/A3-only explanatory comment |
| `runtime/runtime_types.h` | Comments state the corresponding 72- or 108-worker capacity; the mask remains two 64-bit words on both platforms |
| `runtime/submit_types.h` | The launch accessor and backing field are named `block_num` on A2/A3 and `core_num` on A5; both represent the logical SPMD block count |

The first two rows, covering three files, are the strict "compile macro only"
subset. The `s_block_*` names are also a compile-time constraint rather than a
different runtime data model. No standalone cleanup is planned; mechanical
include guards can converge to `#pragma once` when those files are next
modified.

## Files with Functional Differences

The following 18 matching paths have at least one functional difference:

```text
aicore/aicore_executor.cpp
aicpu/aicpu_executor.cpp
docs/MULTI_RING.md
docs/RUNTIME_LOGIC.md
host/runtime_maker.cpp
runtime/aicore_completion_mailbox_types.h
runtime/backend/sdma/sdma_completion_scheduler.h
runtime/async_wait.h
runtime/orchestrator.cpp
runtime/ring_buffer.cpp
runtime/ring_buffer.h
runtime/runtime.h
runtime/scheduler/scheduler.h
runtime/scheduler/scheduler_cold_path.cpp
runtime/scheduler/scheduler_completion.cpp
runtime/scheduler/scheduler_context.h
runtime/scheduler/scheduler_dispatch.cpp
runtime/shared/runtime_init.cpp
```

A5 also has two backend files with no A2/A3 counterpart:

```text
runtime/backend/urma/urma_completion_kernel.h
runtime/backend/urma/urma_completion_scheduler.h
```

The functional differences group into the following themes:

| Difference | Root cause | Must remain platform-specific? | Current decision |
| ---------- | ---------- | ------------------------------ | ---------------- |
| Compute topology | Physical hardware | Yes | Retain each platform's capacity constants and derived layouts |
| AICPU launch plan | Product thread limits, CANN launch ABI, and firmware topology | Yes | Retain the A5 dynamic topology query; do not equate thread limits with physical topology |
| Cache coherence | Hardware coherence model | Yes | Retain the required invalidate/flush operations on A2/A3; do not copy unnecessary maintenance operations to A5 |
| PMU collection | Hardware PMU and platform collection protocol | Yes | Retain the different counter counts, readers, and FIN submission paths |
| System counter and DMB | Hardware timing and register layout | Yes | Use the constants for each platform |
| URMA completion | A5-specific implementation and product capability gate | Yes, for now | Retain the A5 path; do not claim that URMA is available in the default build |
| Next-block prefetch | A2/A3-only performance optimization | No | Retain on A2/A3; validate on A5 before considering a port |
| Scheduler progress publication | AICPU topology and measured publication cost | No | Retain A5's 16-task batching; keep per-advance publication on A2/A3, where the portable implementation showed no significant benefit |
| Terminal task release | Measured end-of-run scheduler cost | No | A5 traces show per-task release blocking the tail after task submission has ended, so successful A5 runs elide deferred release after the graph seal; retain incremental release on A2/A3 because no tail release blocking was found there |
| Fatal teardown | Software reliability strategy | No | Retain the current implementations; decide whether to converge after measuring the worst-case A5 teardown time |
| Scheduler trace attribution | Software diagnostic strategy | No | Preserve the current traces; converge only after comparing generated timelines |

### Compute Topology and AICPU Launch Plan

One A2/A3 runtime device corresponds to one die with 24 clusters, comprising
24 AICs and 48 AIVs. An A3 chip exposes its two dies as two device IDs. One A5
runtime device instead spans both dies of the chip and has 36 clusters,
comprising 36 AICs and 72 AIVs. Because the physical core and cluster counts
visible to one runtime device differ, `PLATFORM_MAX_CORES`,
`RUNTIME_MAX_WORKER`, and the capacities of per-core diagnostic buffers must
also differ.

In the current product configuration, A2/A3 uses at most four active AICPU
threads, while A5 uses at most five. `DeviceRuntimeLaunchDesc` retains
`aicpu_launch_count` on both platforms. A5 uses
`simpler_aicpu_query_topology` to determine the launch plan dynamically from
OCCUPY/FG/PG/SMT, whereas A2/A3 does not currently use the same dynamic query
path.

Physical topology, product thread limits, and firmware launch topology are
three related but distinct constraints. The physical core count determines the
runtime capacity ceiling, while the AICPU thread limit and topology query
determine how the current product launches and shards the scheduler. They are
not interchangeable.

| File | Role |
| ---- | ---- |
| `platform/include/common/platform_config.h` | Defines cluster/core counts, the product AICPU thread limit, and platform capacity constants |
| `runtime/runtime.h` | Maps `RUNTIME_MAX_WORKER` to 72/108 platform cores and defines the shared launch descriptor |
| `host/runtime_maker.cpp` | A5 registers and uses the dynamic topology query; A2/A3 does not use this path |

### Cache Coherence

On A2/A3, after Host DMA or SDMA writes to GM, the AICPU must invalidate its
cache before reading the data. On A5, DMA/HBM is coherent with the AICPU, so
these invalidations are unnecessary at the same points. The current SDMA
completion protocol publishes a monotonic completed post ID: both platforms
read it with acquire semantics and neither clears or retires the shared record.
AICore results are still published by the AICore with `dcci`.

This cache maintenance directly guarantees visibility. It cannot be
mechanically removed from A2/A3, nor should it be copied to A5 as unnecessary
overhead.

| File | Current difference |
| ---- | ------------------ |
| `aicpu/aicpu_executor.cpp` | A2/A3 invalidates `runtime->dev`, which Host DMA writes, before teardown; A5 does not require the corresponding operation |
| `runtime/backend/sdma/sdma_completion_scheduler.h` | A2/A3 invalidates the cache line before the acquire load of the completed post ID; A5 performs the acquire load directly; retirement is a no-op on both platforms |
| `runtime/async_wait.h` | A2/A3 provides a cache-line invalidation helper for async COUNTER polling; A5 does not require it |
| `runtime/scheduler/scheduler.h` | A2/A3 invalidates the cache line before async COUNTER polling; A5 polls directly |

### PMU, System Counter, and DMB

A2/A3 exposes eight PMU counters, which the AICPU reads from MMIO after FIN.
A5 exposes ten counters, which the AICore reads with the PTO-ISA `ld_dev`
operation and writes to a per-core staging slot before the AICPU submits the
record after FIN. The counter counts, MMIO readers, and available instructions
differ, so the record layout, FIN sequencing, and per-core ring must diverge.

The A2/A3 system counter runs at 50 MHz and uses DMB MMIO offset `0xA0`; the A5
values are 1 GHz and `0xD0`, respectively. These facts come from each
platform's `platform/include/common/platform_config.h`.
`docs/RUNTIME_LOGIC.md` only documents the corresponding mapping.

| File | Current difference |
| ---- | ------------------ |
| `platform/include/common/platform_config.h` | Defines the system counter frequency and DMB offset for each platform |
| `aicore/aicore_executor.cpp` | Both platforms bracket kernel execution with the PMU gate; A5 additionally writes the ten-counter staging record before FIN |
| `runtime/scheduler/scheduler_completion.cpp` | After FIN, A2/A3 invokes the AICPU MMIO reader for eight counters; A5 commits the ten-counter slot written by the AICore |
| `platform/shared/aicpu/pmu_collector_aicpu.cpp` | Implements the A2/A3 direct MMIO read and the A5 staging-slot consumption paths |

### Optional A5-Specific URMA Backend

A5 contains the source path for issuing URMA completion requests, creating
deferred entries, forwarding FIN, and polling/retiring CQ entries. A2/A3
currently registers only the COUNTER and SDMA completion backends.

The repository does not currently define `PTO_URMA_SUPPORTED`. A5 therefore
compiles the shared ABI, mailbox, CQ polling/retirement, and related paths, but
the kernel path that successfully issues URMA PTO instructions is unreachable.
The current state is "implemented but disabled by default." It neither means
that the default A5 build supports URMA nor proves that the A2/A3 hardware does
not support URMA.

Both platforms already share `CompletionToken::backend_cookie`,
`ASYNC_ENGINE_URMA`, the 32-byte `DeferredCompletionEntry`, end-to-end cookie
propagation from the AICore slab into the 64-byte mailbox message, and the
generic completion-backend dispatch. The actual platform divergence is that
A2/A3 lacks the URMA completion type, request-issue implementation, registered
backend operations, and CQ polling/retirement path.

| Path stage | File | Current difference |
| ---------- | ---- | ------------------ |
| Request issue | `runtime/backend/urma/urma_completion_kernel.h` | Present only on A5; it invokes `TGET_ASYNC`/`TPUT_ASYNC` only when `PTO_URMA_SUPPORTED` is defined |
| Deferred entry | `runtime/aicore_completion_mailbox_types.h`, `runtime/async_kernel_api.h` | Both platforms use the same 32-byte entry and propagate `backend_cookie`; A5 additionally defines the URMA completion type |
| FIN forwarding | `runtime/scheduler/scheduler_completion.cpp`, `runtime/aicore_completion_mailbox.h` | Both platforms carry `backend_cookie` into the same 64-byte mailbox message; A5 can populate it with URMA workspace metadata |
| CQ polling/retirement | `runtime/backend/urma/urma_completion_scheduler.h`, `runtime/async_wait.h` | A5 registers URMA operations, polls CQE owner/status, advances the CQ/WQ tail, and updates the doorbell; the scheduler header itself is not guarded by the capability macro |

### A2/A3 Next-Block Prefetch

The A2/A3 completion path calls `prefetch_block_dst()` in
`runtime/scheduler/scheduler_completion.cpp`; the corresponding helper is
declared in `runtime/scheduler/scheduler_context.h`. This prefetch reduces the
A2/A3 sync-start drain burst without changing the scheduling protocol.

Prefetch effectiveness depends strongly on platform characteristics such as
cache capacity. The repository does not contain an A5 measurement that
establishes a benefit, so the optimization should not be ported mechanically.
This is a platform-sensitive performance strategy, not a different scheduler
architecture.

### Scheduler Progress Publication: Why A5 Only

A2/A3 publishes `ring->fc.last_task_alive` after every local ring-pointer
advance. A5 tracks `last_published_to_sm` and publishes the shared watermark
every 16 local advances while no reclaim consumer is blocked. This reduces
Scheduler-to-Orchestrator cache-line transfers while allowing the shared
watermark to trail local reclamation by at most 15 tasks on the non-blocking
path.

The optimization is portable, but its benefit is topology-specific. A2/A3 has
four physical AICPU cores per cluster and runs four active roles by default
(`1 Orchestrator + 3 Schedulers`). Its affinity policy prefers one cluster that
can hold all four roles, so progress publication is normally cluster-local. A5
has two physical AICPU cores per cluster and runs five active roles by default
(`1 Orchestrator + 4 Schedulers`). Even with SMT, those five roles cannot all
fit in one cluster. They must span clusters and, depending on SMT and the
available CPU pool, may span dies.

Any A5 Scheduler may win `advance_lock`, advance the authoritative
Scheduler-side `last_task_alive`, and publish `ring->fc.last_task_alive` for the
Orchestrator. A remote publisher transfers or invalidates a cache line that the
Orchestrator repeatedly reads for slot, heap, and TensorMap reclamation. K=16
does not remove Scheduler-to-Scheduler contention on `advance_lock`; it reduces
the frequency of Scheduler-to-Orchestrator publication by up to 16x.

| Property | A2/A3 | A5 |
| -------- | ----- | -- |
| Physical AICPU cores per cluster | 4 | 2 |
| Default active roles | 1 Orchestrator + 3 Schedulers = 4 | 1 Orchestrator + 4 Schedulers = 5 |
| Normal placement | All active roles fit in one cluster | Active roles must span clusters and may span dies |
| Shared-watermark publication | Normally cluster-local | Can require cross-cluster or cross-die cache-line transfer |
| Port benchmark | Mean Effective change `-0.24%`; all workloads within approximately +/-2% | Mean Effective change `-2.81%`; all eight workloads improved |
| Decision | Keep per-advance publication | Enable K=16 batching |

A5 force-publishes when the local watermark reaches `current_task_index` or an
orchestrator reclaim consumer has observed no reclaim progress for 10 ms and
requests an exact watermark. Task-slot, heap, dependency-list, fanin-spill, and
TensorMap pressure then set per-ring request bits; scheduler thread 0 services
them under `advance_lock` from both productive and idle iterations, publishes
the local watermark, and acknowledges completion. Structural deadlock checks
run only after this acknowledgment, so their reclaim head is not a batched
lower bound. These request bits are cache-line-isolated from scheduler
lock-contention retries, which remain deferred to idle loops. K=16 batching is
enabled only after all reclaim request/ack pointers are wired to the current
scheduler; incomplete wiring keeps per-advance publication. Initialization,
arena relocation, and ring reuse reset local progress and disable batching
until that validation succeeds again.

This pressure handshake follows the same liveness rule recorded in
[`2026-06-cross-task-batched-publish.md`](investigations/2026-06-cross-task-batched-publish.md):
delaying a publication is safe only while no peer is waiting on it for forward
progress.

| File | A5-only difference introduced by PR #1575 |
| ---- | ----------------------------------------- |
| `runtime/pto_ring_buffer.{h,cpp}` | A5 reclaim consumers request and await exact watermark publication after 10 ms without progress and before structural classification |
| `runtime/orchestrator.cpp` | A5 TensorMap pressure requests publication from every ring after the same 10 ms no-progress interval |
| `runtime/scheduler/{scheduler.h,scheduler_dispatch.cpp}` | A5 batches non-blocking publication at K=16 and services pressure requests in productive and idle loops; A2/A3 continues to publish every local advance |
| `runtime/shared/runtime_init.cpp` | A5 initializes, resets, and wires the publication shadow and pressure handshake |

The A2/A3 result is not a correctness limitation. A local port passed targeted
and complete non-hardware tests, but its eight workload deltas were all within
the approximately +/-2% noise band, with an unweighted mean of `-0.24%`.
Batching would therefore add up to 15 tasks of non-blocking reclamation lag
without a demonstrated payoff. The same-device A5
measurements instead showed lower Effective time in all eight workloads, with
an unweighted mean reduction of `2.81%`. Full A2/A3 measurements are recorded
in the [PR benchmark follow-up](https://github.com/hw-native-sys/simpler/pull/1575#issuecomment-5310909143).

### Terminal Task Release: A5 Seal and Elision

Task completion and task release are separate scheduler operations. Completion
records the finished AICore work and unlocks dependent tasks. Release later
drops the completed task's retained references, advances the ring's reclaim
head across consumed slots, resets reusable slot state, and publishes reclaim
progress. Both platforms defer this release work in a per-scheduler array with
a capacity of 256 entries.

A2/A3 preserves the incremental protocol for the whole run. It drains the
array when it becomes full, during idle cleanup, and when dispatch exits. Every
completed task therefore reaches `on_task_release()` before the scheduler
returns.

A5 follows the same protocol while the Orchestrator can still submit tasks.
After `orchestrator_done_` marks the orchestration terminal, however, no new
task can require a reclaimed ring slot. At the next existing full-array,
idle-drain, or exit-drain boundary, A5 discards the deferred-release backlog
instead of calling `on_task_release()` once per entry. Shared helper
`drain_or_elide_deferred_releases` owns that decision. The seal is deliberately
not loaded on every scheduler-loop iteration or every completion: completion
and dependency unlocking remain unchanged, and the added acquire loads stay on
boundaries that already perform release bookkeeping. At a sealed capacity
boundary, the overflowing completed slot is also not deferred.

The terminal flag is stored whenever orchestration exits, including after an
orchestration error. At that point no later submission can consume reclaimed
capacity, while `orch_error_code` independently carries the failure through
emergency shutdown and host reporting. Later idle/exit drains may therefore
elide on both success and failure; the next-run SM reset closes the lifecycle.

Skipped incremental release does not need a terminal barrier or bulk slot
closure. The next run clears the entire SM with `memset` and rebuilds flow
control via `init_per_ring` → `fc.init()`, and the orchestrator already
self-cleans each reused slot on submit. There is no functional downstream
reader of terminal-exit `CONSUMED` watermarks between runs. An earlier
barrier / `TerminalClose` layer was removed as redundant.

This optimization is independent of the A5 K=16 progress-publication policy in
the preceding section. K=16 controls how often an already-advanced reclaim
head is copied to shared memory during the run. Terminal release elision avoids
the per-task reference-count and ring-advance work itself after graph sealing;
its deferred-release array still has capacity 256 and does not impose a
16-task release limit.

| Stage | A2/A3 | A5 |
| ----- | ----- | -- |
| Before graph sealing | Complete tasks, defer release, then incrementally call `on_task_release()` | Same |
| After graph sealing | Continue incremental release | Drop deferred release work at existing release boundaries |
| Successful Scheduler exit | Drain every remaining deferred entry | Drop remaining deferred entries; next run resets SM |
| Orchestration failure | Emergency teardown after the existing error checks | Publish `orchestrator_done_`, then emergency teardown; exit/idle drains may elide because no later submission can use reclaimed capacity |
| Scheduler / async failure after orchestration terminates | Emergency teardown | `orchestrator_done_` remains true, so exit/idle drains still elide; SM reset on the next run closes the lifecycle |
| Profiling | Release phases | Release phases only when a real drain runs (no `terminal_close`) |

| File | A5-only terminal-release role |
| ---- | ----------------------------- |
| `runtime/async_wait.h`, `runtime/scheduler/scheduler_completion.cpp` | Sync completion reads the graph seal at deferred-release capacity boundaries; async capacity drains keep exact release |
| `runtime/scheduler/scheduler_dispatch.cpp` | Elide sealed idle/exit backlog drains via shared `drain_or_elide_deferred_releases`; skip DFX `release` when elided |
| `runtime/scheduler/scheduler.h` | Define the shared drain-or-elide helper used by deferred-release sites |
| `runtime/scheduler/scheduler_cold_path.cpp` | Publish `orchestrator_done_` whenever orchestration exits; propagate any orchestration error independently |

The post-removal A/B was run only on the local A5 system against current
`main`. The seven non-Qwen workloads improved by `9.753%` in Effective
geometric mean; Qwen3 changed by `-0.010%`, and no workload regressed by 5% or
more in Effective time. The largest Scheduler gains remain in the
paged-attention-unroll family (`10.550%` to `14.021%` Effective). Full tables
are recorded under the [PR #2070](https://github.com/hw-native-sys/simpler/pull/2070)
benchmark discussion / local `outputs/pr2070_full_bench/` artifacts.

The platform scope follows the observed bottleneck. On A5, the motivating
timelines contain a visible tail after the Orchestrator has finished submitting
tasks: Schedulers continue executing per-task release work even though no new
task can consume the reclaimed capacity. That release interval extends the
execution critical path, which gives seal-and-elide a direct optimization
target. No tail release blocking was found on A2/A3. Its release work did not
appear as the corresponding post-orchestration critical-path interval, so there
is currently no performance evidence that A2/A3 would benefit from the extra
graph-seal observation. A2/A3 therefore keeps the simpler incremental release
protocol, and this experiment does not run an A2/A3 benchmark or port the
implementation there. This is an evidence-based software decision rather than
an A5 hardware requirement; revisit it if a future A2/A3 timeline exposes the
same tail release blocking.

### Fatal Teardown

The A2/A3 scheduler uses a dedicated fatal latch to elect an owner, broadcasts
EXIT to all AICores that completed the handshake, and then joins them in
parallel against a single deadline. A5 uses `completed_` to elect the owner and
deinitializes each core sequentially.

The A2/A3 implementation mitigates failure scenarios involving 48 SDMA remote
streams, but the hardware, PTO-ISA, and platform ABI do not mandate this
algorithm. The current A5 sequential deinitialization does not omit the
acknowledgement wait and is therefore not a known correctness defect. The risk
is that each unresponsive core may consume a one-second timeout, so the
aggregate teardown time can reach or exceed the AICPU operation-execution
timeout.

The current implementations are retained. The total teardown time for N
unresponsive cores should first be measured on A5. If a port is needed, the A5
handshake, EXIT MMIO, shared deadline, and core deinitialization semantics must
then be validated. See
[Issue #1710](https://github.com/hw-native-sys/simpler/issues/1710) for tracking.

| File | Current difference |
| ---- | ------------------ |
| `runtime/scheduler/scheduler_cold_path.cpp` | A2/A3 uses a fatal latch, broadcasts EXIT, and uses a shared deadline; A5 uses a `completed_` owner and deinitializes each core sequentially |
| `runtime/scheduler/scheduler_context.h` | A2/A3 declares fatal-owner election and broadcast/join helpers; A5 retains its existing emergency-shutdown interface |

### Scheduler Trace Attribution

The scheduler implementations expose the same scheduling protocol, but their
DFX trace bookkeeping is not byte-equivalent. In
`runtime/scheduler/scheduler_dispatch.cpp`, A2/A3 advances the scheduler phase
anchor across idle iterations, while A5 leaves idle gaps for post-processing to
reconstruct. The two versions also use different local spellings for some
phase-state references. These differences affect generated diagnostic
timelines, not task scheduling or completion semantics.

Trace output should be compared before converging this code, because a
mechanical copy could change how idle time is attributed in Perfetto without
changing runtime execution.

## Excluded Scope: Examples and Tests

`examples/.../tensormap_and_ringbuffer` and
`tests/st/.../tensormap_and_ringbuffer` are outside the runtime implementation
comparison. Examples and tests unique to either platform primarily reflect
chip-feature validation and test-porting progress; they cannot be used to
infer whether the runtime supports a shared algorithm.

For example, A5 has `urma_deferred_completion_demo`, but this does not mean
that the current build defines `PTO_URMA_SUPPORTED`. Likewise, the absence of a
workload on one platform does not automatically mean that the corresponding
runtime capability is unavailable.
