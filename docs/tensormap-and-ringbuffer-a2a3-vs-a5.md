# `tensormap_and_ringbuffer`: A2/A3 vs. A5

This document describes the audited main-tree differences under
`src/{a2a3,a5}/runtime/tensormap_and_ringbuffer/` and explicitly identified
pending reconciliation work.

> **Maintenance baseline:** Source/history audit on 2026-10-09 at
> `main @ 60167bcc1ea898cc0bf861b2e3aa0aa6f70ba40d`; no new hardware validation.
> Pending retirement work is identified separately as PR #2388, rather than
> delivered main behavior. Its historical performance snapshot is
> `87ce4653bbea29841b188f7a6b4906d073bf9dac`; the proposal is rebased on this main.
> Recompute the counts and contracts whenever the compared trees change.

## Comparison Boundary and Classification

The direct comparison covers tracked files under
`src/{a2a3,a5}/runtime/tensormap_and_ringbuffer/`, matched by relative path.
There are 51 A2/A3 files and 55 A5 files: 51 shared paths and four additional
paths present only on A5. At the main baseline, every file belongs to one of these
categories:

| Category | Count | Definition |
| -------- | ----: | ---------- |
| Byte-identical | 23 | The files at the same relative path have identical bytes |
| Compile-time or non-functional differences | 8 | The file's diff contains only include guards, naming, or comments; dependent platform contracts may still differ |
| Functional differences | 24 | Twenty shared paths and four A5-only paths encode or document differences in behavior, layout, capacity, diagnostics, or supported backends |

With the pending #2388 retirement changes, `runtime/shared/runtime.cpp` has matching code
and only differing comments, so it moves to the non-functional category:
23 byte-identical, 9 non-functional, and 23 functional files. This is source
convergence, not an omitted retirement feature; byte-identical comments are
not required. The lists below classify the pinned main baseline.

A file is classified as functional when any part of its diff changes behavior,
even if the same diff also contains include-order, comment, or formatting
changes. Files outside the direct boundary, such as platform configuration and
PMU collector implementations, are cited only as supporting evidence.

The boundary shrinks when a pair is deduplicated rather than reconciled: three
`host/` paths left it for `src/common/tensormap_and_ringbuffer/host/`, where
there is one file and so nothing to compare.

## Byte-Identical Files

The following 23 files are byte-identical:

```text
build_config.py
common/runtime_status.h
docs/{SCALAR_DATA_ACCESS.md,SUBMIT_BY_CLUSTER.md,device_log_profiling.md,profiling_levels.md}
orchestration/{common.cpp,arg_with_deps.h,orchestration_api.h}
runtime/{common.h,async_kernel_api.h,dep_compute.h,orchestrator.h,
         runtime_core.cpp,runtime_core.h,shared_memory.h,tensor.h,
         tensormap.h,tensor_create_info.h}
runtime/scheduler/{scheduler.cpp,scheduler_types.h}
runtime/shared/{shared_memory.cpp,tensormap.cpp}
```

Byte identity is a textual result only. A shared file may still consume
platform-specific constants or APIs supplied by files outside this comparison
boundary.

The `host/` directory holds only `runtime_maker.cpp` and appears nowhere above:
`dep_gen_replay.{cpp,h}` and `runtime_compile_info.cpp` now exist once, in
`src/common/tensormap_and_ringbuffer/host/`, so they have no pair to compare.

## Compile-Time or Non-Functional Differences

The following 8 matching paths contain no behavioral change in their own diff:

| Files | Difference |
| ----- | ---------- |
| `runtime/constants.h` | Path-derived include-guard macro names only |
| `runtime/backend/sdma/sdma_completion_kernel.h`, `runtime/types.h` | `#pragma once` on A2/A3 versus a path-derived include guard on A5 |
| `runtime/dispatch_payload.h` | Comments describe the platform-specific context fields; the `GlobalContext` layout difference is defined in `common/intrinsic.h` |
| `runtime/aicore_completion_mailbox.h` | Path-derived include guards and comment wording only |
| `runtime/completion_token.h` | Path-derived include guards and an A2/A3-only explanatory comment |
| `runtime/runtime_types.h` | Comments state the corresponding 72- or 108-worker capacity; the mask remains two 64-bit words on both platforms |
| `runtime/submit_types.h` | The launch accessor and backing field are named `block_num` on A2/A3 and `core_num` on A5; both represent the logical SPMD block count |

The first two rows, covering three files, are the strict "compile macro only"
subset. Mechanical include guards, comment order, and function placement are
outside the reconciliation blockers. They can converge when the affected file
is next modified; textual equality is not a closure requirement.

## Files with Functional Differences

The following 20 shared paths have at least one functional difference:

```text
aicore/aicore_executor.cpp
aicpu/aicpu_executor.cpp
common/intrinsic.h
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
runtime/shared/runtime.cpp
runtime/shared/runtime_init.cpp
```

A5 also has four backend files with no A2/A3 counterpart:

```text
runtime/backend/rdma/rdma_completion_kernel.h
runtime/backend/rdma/rdma_completion_scheduler.h
runtime/backend/urma/urma_completion_kernel.h
runtime/backend/urma/urma_completion_scheduler.h
```

The functional differences group into the following themes:

| Difference | Root cause | Must remain platform-specific? | Current decision |
| ---------- | ---------- | ------------------------------ | ---------------- |
| Compute topology | Physical hardware | Yes | Retain each platform's capacity constants and derived layouts |
| AICPU launch plan | Product thread limits, CANN launch ABI, and firmware topology | Yes | Retain the A5 dynamic topology query; do not equate thread limits with physical topology |
| Cache coherence | Producer/consumer visibility contract | Yes | Distinguish Host-DMA coherence from AICore-to-AICPU slab/counter visibility; retain each path's required maintenance |
| AICore context and L2 alias | Driver/device configuration and compiler ABI | Yes | Retain A2/A3's `l2_cache_offset` field/getter; A5's compiler-reserved name substitutions do not imply layout equivalence |
| PMU collection | Hardware PMU and platform collection protocol | Yes | Retain the different counter counts, readers, and FIN submission paths |
| System counter and DMB | Hardware timing and register layout | Yes | Use the constants for each platform |
| URMA/RDMA completion | Platform backend implementation and product capability gates | Yes, for now | Retain the four A5-only files and explicit opt-in workspace gates; source presence is not default capability |
| Next-block prefetch | Platform-sensitive performance optimization | No | Retain on A2/A3; the recorded A5 experiment did not establish stable benefit and regressed short tasks |
| Empty Tier-0 staging | Portable software optimization measured only on A5 (#2105) | No | Benchmark A2/A3 before a port or an explicit keep-as-is decision |
| Profiling queue-tag publication | Software producer contract | No | Reconcile invalid-tag initialization or justify the differing producer/consumer path |
| Scheduler progress publication | AICPU topology and measured publication cost | No | Retain A5's 16-task batching; keep per-advance publication on A2/A3, where the portable implementation showed no significant benefit |
| Terminal task release | Measured end-of-run scheduler cost | No | A5 traces show per-task release blocking the tail after task submission has ended, so successful A5 runs elide deferred release after the graph seal; retain incremental release on A2/A3 because no tail release blocking was found there |
| Normal/fatal retirement | Software reliability protocol | No | Track #2387/#2388 through integration/acceptance; the A5 proposal is not yet delivered to main |
| Joined launch and error status | Cross-run ordering/result ownership (#2491) | No | Retain A5's early-submission contract and A2/A3's serial shared-header fallback as a coupled distinction |
| Async stall diagnostics | Software diagnostic strategy | No | Decide shared COUNTER/SDMA diagnostic scope; retain backend-specific snapshots |
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

Host-DMA coherence does not establish AICore-to-AICPU visibility. Both platforms
retain deferred-slab invalidations after FIN. COUNTER polling also retains
cache maintenance: A2/A3 groups invalidations in the wait-list loop, while A5
invalidates in `counter_poll_op`. Do not remove those operations based on the
Host-DMA distinction.

| File | Current difference |
| ---- | ------------------ |
| `aicpu/aicpu_executor.cpp` | A2/A3 invalidates `runtime->dev`, which Host DMA writes, before teardown; A5 does not require the corresponding operation |
| `runtime/backend/sdma/sdma_completion_scheduler.h` | A2/A3 invalidates the cache line before the acquire load of the completed post ID; A5 performs the acquire load directly; retirement is a no-op on both platforms |
| `runtime/async_wait.h`, `runtime/scheduler/scheduler.h` | A2/A3 invalidates each distinct COUNTER line in the wait-list loop; A5 invalidates in `counter_poll_op` |
| `runtime/scheduler/scheduler_completion.cpp` | Both platforms invalidate the AICore-written deferred slab before consuming its header/entries |

### AICore Context and L2 Alias

`common/intrinsic.h` is functional, not a naming-only difference. A2/A3's
`GlobalContext` includes `uint64_t l2_cache_offset` and
`get_l2_cache_offset(args)`, populated from resident device configuration during
scheduler initialization. The value is the driver-provided nocache-alias
offset; zero leaves an ordinary cached load. A5 has neither this field nor the
getter, so the exposed global-context layout is different.

A5's `s_block_idx`/`s_block_num` names remain a compiler constraint, while the
logical SPMD block-index/count semantics are shared. Classify the whole file by
its functional L2/context difference rather than by those spelling changes.

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

### Optional A5-Specific URMA and RDMA Backends

A5 contains URMA and RDMA request-issue and completion poll/retire backend
pairs; A2/A3 registers COUNTER and SDMA only. The A5 host CMake options
`SIMPLER_ENABLE_PTO_URMA_WORKSPACE` and `SIMPLER_ENABLE_PTO_RDMA_WORKSPACE`
default to `OFF`. Enabling them defines `PTO_URMA_SUPPORTED` or
`PTO_RDMA_SUPPORTED` (plus `PTO_RDMA_BACKEND_HNS_1825_SUPPORTED`) for the host
workspace path. The runtime builder propagates these options; the AICore
kernel compiler separately propagates the RDMA capability definitions. A
successful request also requires the corresponding workspace and toolchain.
Source presence proves neither default availability nor hardware absence on
A2/A3.

Both platforms already share `CompletionToken::backend_cookie`,
`ASYNC_ENGINE_URMA`, the 32-byte `DeferredCompletionEntry`, end-to-end cookie
propagation from the AICore slab into the 64-byte mailbox message, and the
generic completion-backend dispatch. The actual platform divergence is that
A2/A3 lacks the URMA/RDMA completion types, request-issue implementation,
registered backend operations, and CQ polling/retirement paths.

| Path stage | File | Current difference |
| ---------- | ---- | ------------------ |
| Request issue | `runtime/backend/urma/urma_completion_kernel.h` | Present only on A5; it invokes `TGET_ASYNC`/`TPUT_ASYNC` only when `PTO_URMA_SUPPORTED` is defined |
| Request issue | `runtime/backend/rdma/rdma_completion_kernel.h` | Present only on A5; native request issue requires `PTO_RDMA_SUPPORTED` |
| Deferred entry | `runtime/aicore_completion_mailbox_types.h`, `runtime/async_kernel_api.h` | Both platforms use the same 32-byte entry and propagate `backend_cookie`; A5 additionally defines URMA and RDMA completion types |
| FIN forwarding | `runtime/scheduler/scheduler_completion.cpp`, `runtime/aicore_completion_mailbox.h` | Both platforms carry `backend_cookie` into the same 64-byte mailbox message; A5 can populate it with URMA workspace metadata |
| CQ polling/retirement | `runtime/backend/urma/urma_completion_scheduler.h`, `runtime/async_wait.h` | A5 registers URMA operations, polls CQE owner/status, advances the CQ/WQ tail, and updates the doorbell; the scheduler header itself is not guarded by the capability macro |
| CQ polling/retirement | `runtime/backend/rdma/rdma_completion_scheduler.h`, `runtime/async_wait.h` | A5 registers RDMA-specific completion operations and snapshots; A2/A3 has no corresponding backend |

### A2/A3 Next-Block Prefetch

The A2/A3 completion path calls `prefetch_block_dst()` in
`runtime/scheduler/scheduler_completion.cpp`; the corresponding helper is
declared in `runtime/scheduler/scheduler_context.h`. This prefetch reduces the
A2/A3 sync-start drain burst without changing the scheduling protocol.

Prefetch effectiveness depends on platform characteristics such as cache
capacity. The recorded A5 experiment failed to establish stable benefit and
showed short-task regressions, so the current decision is no A5 port.
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
| `runtime/ring_buffer.{h,cpp}` | A5 reclaim consumers request and await exact watermark publication after 10 ms without progress and before structural classification |
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

### Normal and Fatal AICore Retirement

On the main baseline, A2/A3 uses a dedicated fatal latch, per-core retirement
claims, grouped EXIT broadcast, a shared-deadline ACK sweep, register
close/readback/drain, and a GM return gate. A5 still elects emergency shutdown
using `completed_` and deinitializes cores serially on normal and fatal paths.

PR [#2388](https://github.com/hw-native-sys/simpler/pull/2388), tracked by
[#2387](https://github.com/hw-native-sys/simpler/issues/2387), proposes the A5
normal/fatal protocol: exactly-once per-core claims, EXIT broadcast and a
shared deadline, IDLE close/readback/drain before releasing the per-core GM
gate, and fatal publication before completion. It also applies the return-gate
ordering to A5 `host_build_graph`. A5 does not define A2/A3's FAST_PATH register;
the port closes and reads back the A5 dispatch register instead. The proposal
must remain distinct from main until integration and acceptance are complete.

The proposal also retains retirement requests received before core readiness.
The corresponding A2/A3 READY/REQUESTED follow-up is
[#2492](https://github.com/hw-native-sys/simpler/issues/2492) / PR
[#2506](https://github.com/hw-native-sys/simpler/pull/2506), not an A2/A3 change
inside #2388. Broader fatal-timeout work remains under
[#1710](https://github.com/hw-native-sys/simpler/issues/1710). The older #2286
no-net-gain result is superseded as a tracking disposition by #2387, not evidence
that the current proposal has no cost or requires no acceptance.

| File | Main baseline and pending proposal |
| ---- | ---------------------------------- |
| `runtime/scheduler/{scheduler_cold_path.cpp,scheduler_context.h}` | A2/A3 has dedicated fatal/retirement ownership; #2388 proposes A5 per-core ownership, READY/REQUESTED publication, and group retirement |
| `runtime/shared/runtime.cpp`, `runtime/runtime.h` | Main A2/A3 separates the host-initialized prefix from its AICPU-initialized gate tail; #2388 adds the same descriptor lifetime distinction to A5 |
| `aicore/aicore_executor.cpp` | Main A2/A3 waits at the GM gate after ACK; #2388 adds the corresponding A5 return wait |

### Joined Native Launch and Error Status

Main includes [#2491](https://github.com/hw-native-sys/simpler/pull/2491): A5
TMR can accept a successor's native submission before its predecessor completes,
while device execution remains serial. Its AICPU-stream ordering waits for the
predecessor's AICore end before the whole-operator boundary can admit the next
arena reset. Compatible live preparation requires a prebuilt-arena cache hit;
rebuilding/uploading that shared arena beside a live predecessor is refused.

A5 reports failures only from that run's retained published result. A missing
or short record preserves the existing execution error; it cannot fall back to
the shared header, which the successor may already have reset. A2/A3 TMR still
uses serial launch and deliberately retains its shared-header fallback while
the failed run owns the execution claim. A port requires coupled ordering,
per-slot result-region publication, and status handling; toggling the capability
or deleting the fallback alone is invalid. These are newer main contracts;
PR #2388's historical performance snapshot `87ce4653b` predates #2491;
its rebased source includes the main launch/status contract. Historical
performance measurements do not validate that newer cross-run contract.

### Remaining Software and Diagnostic Reconciliation

The [2026-10-08 #1582 audit](https://github.com/hw-native-sys/simpler/issues/1582#issuecomment-6051030154)
keeps these separate from #2388 retirement acceptance:

| Item | Source difference | Required disposition |
| ---- | ----------------- | -------------------- |
| Empty Tier-0 (#2105) | A5 publishes `sync_task_seen` and skips empty sync-start staging; A2/A3 always probes Tier-0 | Benchmark A2/A3 before porting or recording keep-as-is |
| Profiling queue-tag producer | A5's profiled `ChipReadyQueue::push` initializes `task_id_snapshot` to the invalid sentinel; A2/A3's overload does not | Align and add focused coverage, or establish why the tag is irrelevant for that path; no observed misdispatch is claimed |
| Async stall diagnostics | A5 `AsyncWaitList::log_diagnostics` dumps mailbox/wait entries and backend snapshots; A2/A3 lacks the generic dump | Decide shared COUNTER/SDMA coverage without blindly porting RDMA-specific details |

The original July/August migration list is not a current missing-feature
checklist. Its landed items stay completed; these remaining decisions and
retirement acceptance keep #1582 open.

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
