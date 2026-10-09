# `tensormap_and_ringbuffer`: A2/A3 vs. A5

This page compares the current TMR runtime implementations: their hardware
contracts, software policies, supported backends, and diagnostic behavior.

**Source baseline:** default branch `main` at
[`f365ae97ea3d61b4d5c71d496df8a11a397c870c`](https://github.com/hw-native-sys/simpler/tree/f365ae97ea3d61b4d5c71d496df8a11a397c870c),
audited on 2026-10-09. The comparison describes source behavior, not measured
performance. Paths below are relative to
`src/{a2a3,a5}/runtime/tensormap_and_ringbuffer/` unless qualified otherwise.

## Scope and File Inventory

The two runtime trees contain 51 A2/A3 files and 55 A5 files: 51 matching
relative paths and four A5-only backend files. The categories count each
matching path once, plus each platform-only path once.

| Category | Count | Meaning |
| -------- | ----: | ------- |
| Byte-identical | 23 | Matching Git blobs have identical content |
| Non-behavioral source differences | 10 | Include guards/includes, comments, diagnostic assertion text, declaration order, or equivalent API naming |
| Behavioral or capability differences | 22 | 18 matching paths encode or document different behavior; four backend files exist only on A5 |

This is a classification of each file's own diff. Identical declarations can
still use different platform constants or dependent types: for example,
`runtime/runtime.h` derives its worker count from `PLATFORM_MAX_CORES`, and
`runtime/dispatch_payload.h` embeds the platform's `GlobalContext`.
Neither category implies cross-platform binary-layout compatibility.

### Byte-identical paths

```text
build_config.py
common/runtime_status.h
docs/SCALAR_DATA_ACCESS.md
docs/SUBMIT_BY_CLUSTER.md
docs/device_log_profiling.md
docs/profiling_levels.md
orchestration/arg_with_deps.h
orchestration/common.cpp
orchestration/orchestration_api.h
runtime/async_kernel_api.h
runtime/common.h
runtime/dep_compute.h
runtime/orchestrator.h
runtime/runtime_core.cpp
runtime/runtime_core.h
runtime/scheduler/scheduler.cpp
runtime/scheduler/scheduler_types.h
runtime/shared/shared_memory.cpp
runtime/shared/tensormap.cpp
runtime/shared_memory.h
runtime/tensor.h
runtime/tensor_create_info.h
runtime/tensormap.h
```

### Non-behavioral source differences

| Path | Difference |
| ---- | ---------- |
| `runtime/constants.h` | Include-guard names |
| `runtime/backend/sdma/sdma_completion_kernel.h`, `runtime/types.h` | `#pragma once` versus an include guard |
| `runtime/aicore_completion_mailbox.h`, `runtime/completion_token.h` | Include guards and comments |
| `runtime/dispatch_payload.h`, `runtime/runtime_types.h` | Comments about platform context fields and worker capacity |
| `runtime/runtime.h` | Includes/guards, comments, assertion messages, and accessor order; descriptor fields and size expressions match |
| `runtime/shared/runtime.cpp` | Comments; device-copy prefix and extent calculations match |
| `runtime/submit_types.h` | `LaunchSpec::block_num`/`set_block_num` on A2/A3 versus `core_num`/`set_core_num` on A5; both specify the logical SPMD block count |

The `LaunchSpec` spelling difference matters to callers compiling directly
against these headers, even though its stored value and scheduling meaning match.

### Behavioral or capability differences

The 18 matching paths are:

```text
aicore/aicore_executor.cpp
aicpu/aicpu_executor.cpp
common/intrinsic.h
docs/MULTI_RING.md
docs/RUNTIME_LOGIC.md
host/runtime_maker.cpp
runtime/aicore_completion_mailbox_types.h
runtime/async_wait.h
runtime/backend/sdma/sdma_completion_scheduler.h
runtime/orchestrator.cpp
runtime/ring_buffer.cpp
runtime/ring_buffer.h
runtime/scheduler/scheduler.h
runtime/scheduler/scheduler_cold_path.cpp
runtime/scheduler/scheduler_completion.cpp
runtime/scheduler/scheduler_context.h
runtime/scheduler/scheduler_dispatch.cpp
runtime/shared/runtime_init.cpp
```

The four A5-only paths are:

```text
runtime/backend/rdma/rdma_completion_kernel.h
runtime/backend/rdma/rdma_completion_scheduler.h
runtime/backend/urma/urma_completion_kernel.h
runtime/backend/urma/urma_completion_scheduler.h
```

Shared files under `src/common/tensormap_and_ringbuffer/host/`, platform
implementations, examples, and tests are outside this inventory. Supporting
platform and host contracts are linked below. An example's presence on only
one platform does not establish runtime capability or default enablement.

## Hardware and ABI Contracts

| Contract | A2/A3 | A5 | Effect |
| -------- | ----- | -- | ------ |
| Runtime device capacity | 24 clusters: 24 AIC + 48 AIV | 36 clusters: 36 AIC + 72 AIV | Per-worker arrays and diagnostic capacities differ |
| Active AICPU thread limit | 4 | 5 | Bounds active roles, not the number of threads initially launched |
| Launch topology | Platform affinity/launch policy | Dynamic `simpler_aicpu_query_topology` using OCCUPY/FG/PG/SMT | Actual launch count and selected active CPUs are separate from worker capacity |
| System counter | 50 MHz | 1 GHz | Cycle values require platform-specific conversion |
| `DATA_MAIN_BASE` offset | `0xA0` | `0xD0` | Register accesses use platform definitions |
| PMU counters | 8, read by AICPU through MMIO after FIN | 10, staged by AICore before FIN | Collection paths and record layouts differ |

The A2/A3 device view covers one die; an A3 chip exposes its two dies as two
device IDs. The A5 device view spans both dies. Hardware capacity, product
thread limits, and firmware launch topology are distinct constraints.
Sources: platform configuration ([A2/A3][a3-config], [A5][a5-config]) and host
runtime registration ([A2/A3][a3-maker], [A5][a5-maker]).

### Cache visibility

A2/A3 invalidates AICPU cache lines before reading Host-DMA/SDMA-written GM
at the relevant descriptor and completion-record boundaries. A5's corresponding
paths rely on DMA/HBM coherence with AICPU. Both SDMA backends acquire-load a
monotonic completed post ID and leave the shared record intact.

That distinction does not remove AICore-to-AICPU visibility requirements.
Both completion paths invalidate the AICore-written deferred slab after FIN.
COUNTER polling also retains invalidation: A2/A3 groups it by cache line in
the wait-list loop, while A5 performs it in `counter_poll_op`.
See SDMA polling ([A2/A3][a3-sdma], [A5][a5-sdma]), completion
([A2/A3][a3-completion], [A5][a5-completion]), and async polling
([A2/A3][a3-async], [A5][a5-async]).

### Kernel context and profiling

A2/A3's `GlobalContext` contains `l2_cache_offset`, exposed through
`get_l2_cache_offset(args)`. Scheduler initialization copies the driver-provided
nocache-alias offset from resident device configuration. Zero leaves an ordinary
cached address. A5 exposes neither this field nor its getter, so the context
layouts differ. A5 also names local fields `s_block_idx`/`s_block_num` to avoid
compiler built-in name collisions; the logical block semantics are shared.
See `common/intrinsic.h` ([A2/A3][a3-intrinsic], [A5][a5-intrinsic]).

The field is A2/A3-only because its DMA load instruction does not encode the
L2 cache policy as an operand: the kernel can instead select the uncached
mapping by adding the offset, avoiding L2 allocation for streaming operands.
A5's `TLOAD` carries `l2Control` directly, so that path does not consume an
address-alias offset. The offset is a driver-defined, per-device virtual-address
layout value rather than a hardware constant; A2/A3 therefore queries it with
`rtGetL2CacheOffset` instead of hard-coding it. Unsupported devices retain zero
and ordinary cached-load behavior. See the [A2/A3 host query][a3-l2-query].

PMU collection is conditional on profiling enablement. A2/A3 consumes hardware
counters on AICPU after FIN. A5 reads counters on AICore and publishes a per-core
staging record before FIN; AICPU then commits that record. These paths use
different hardware readers and cannot share a raw counter layout.
A5 uses the AICore `ld_dev` MMIO reader, so its snapshot must be staged before
FIN lets AICPU consume it. A2/A3 instead reads the counters directly from AICPU
after FIN. The publication order follows the selected producer/consumer path;
the difference does not by itself establish that another collection path is
impossible on either platform.
See executors ([A2/A3][a3-executor], [A5][a5-executor]) and collectors
([A2/A3][a3-pmu], [A5][a5-pmu]).

## Completion Backends and Capability Gates

Both platforms register COUNTER and SDMA completion operations. A5 additionally
contains URMA/RDMA issue and completion backends; A2/A3 has no corresponding
backend files or registrations. This is a software capability boundary, not
evidence that A2/A3 hardware cannot support those transports.

| Capability | Default and requirements |
| ---------- | ------------------------ |
| COUNTER/SDMA | Registered on both platforms; requests still require valid counters or SDMA workspace/resources |
| A5 URMA workspace | Host CMake option `SIMPLER_ENABLE_PTO_URMA_WORKSPACE=OFF`; enabling it supplies `PTO_URMA_SUPPORTED` for the host workspace path. Kernel request issue is also guarded by `PTO_URMA_SUPPORTED` |
| A5 RDMA workspace | Host CMake option `SIMPLER_ENABLE_PTO_RDMA_WORKSPACE=OFF`; enabled host and kernel paths use `PTO_RDMA_SUPPORTED` and `PTO_RDMA_BACKEND_HNS_1825_SUPPORTED` |

The [runtime builder][builder] propagates workspace options; the
[kernel compiler][compiler] separately propagates the RDMA definitions.
A usable request also needs compatible workspace and toolchain support.
Backend registration or an example alone does not enable a workspace.
See [A5 host options][a5-options] and the [A5 backend sources][a5-backends].

The transport-independent representation is shared: `CompletionToken` carries
`backend_cookie`, a deferred completion entry is 32 bytes, and the 64-byte
AICore mailbox message forwards that cookie. A5 adds URMA/RDMA completion
types and CQ polling/retirement operations; these are dispatched through
`runtime/async_wait.h` ([A2/A3][a3-async], [A5][a5-async]).

## Scheduling and Lifecycle Policies

The following differences are implementation policies, not hardware-mandated
scheduler architectures. Unless a condition is stated, they are part of the
corresponding runtime path rather than additional user opt-ins.

### Progress publication and reclamation

A2/A3 publishes `ring->fc.last_task_alive` after every local advance. A5 batches
non-blocking publication at `PUBLISH_INTERVAL_K=16`; the shared watermark can
lag by at most 15 local advances. This reduces publication traffic, but does not
remove scheduler contention on `advance_lock`.

The selection reflects topology and measured publication cost. With the default
four active roles (one Orchestrator and three Schedulers), A2/A3's affinity
policy prefers a single four-core AICPU cluster when the available CPU pool
permits it. A5's default five roles cannot fit in one two-core cluster even
with two-way SMT, so Scheduler-to-Orchestrator publication can transfer cache
lines across clusters, and potentially dies. Batching reduces those transfers;
it does not imply the same benefit for other thread counts or CPU placements.
The evaluated A2/A3 port did not establish sufficient benefit to justify the
additional reclamation lag, so A2/A3 retains per-advance publication. This is a
software tradeoff for the evaluated configurations, not a hardware requirement.
See [A2/A3 affinity selection][a3-affinity] and [A5 topology][a5-topology].

A5 force-publishes at `current_task_index` and services exact-watermark requests
from reclaim consumers after 10 ms without progress. Scheduler thread 0 services
the request/ack protocol under `advance_lock` in productive and idle iterations.
Structural head-of-line checks wait for acknowledgment so they do not classify
a stale batched watermark as the exact reclaim head. TensorMap pressure requests
publication from all rings; slot, heap, dependency-list, and fanin-spill pressure
use the corresponding ring.

Batching is enabled only when all reclaim request/ack pointers are wired to the
current scheduler. Initialization and reuse reset the publication state;
incomplete wiring falls back to per-advance publication. Sources:
[A2/A3 scheduler][a3-scheduler], [A5 scheduler][a5-scheduler],
[A5 ring allocators][a5-ring], and [A5 initialization][a5-init].

### Deferred release and dispatch preparation

Completion unlocks dependents; release drops retained references and advances
reclamation. Both platforms defer release in arrays of 256 entries.
A2/A3 drains incrementally at capacity, idle cleanup, and exit. A5 can discard
that backlog after `orchestrator_done_` seals further submission, using
`drain_or_elide_deferred_releases` at existing release boundaries. Async
capacity drains still perform exact release. Sealing also occurs on orchestration
failure; error reporting remains independent. The next run resets shared memory
and rebuilds flow control. This policy avoids terminal release work when no
future submission can consume reclaimed capacity; it is separate from K=16
watermark publication. Sources: completion ([A2/A3][a3-completion],
[A5][a5-completion]) and dispatch ([A2/A3][a3-dispatch], [A5][a5-dispatch]).

The A5-only choice targets a post-orchestration release tail observed on the
critical path in the evaluated A5 workloads. The corresponding bottleneck was
not observed in the A2/A3 investigation, so A2/A3 retains incremental release;
this does not establish that future A2/A3 workloads cannot benefit. Elision also
depends on there being no functional consumer of terminal `CONSUMED` watermarks
between runs: host correctness does not depend on those terminal reclaim values.
The next run clears shared memory and rebuilds flow control through
`init_per_ring` and `fc.init()`, closing the lifecycle without a bulk terminal
slot close. See [shared-memory initialization][a5-sm-init].

A2/A3 calls `prefetch_block_dst` while staging sync-start blocks; A5 does not.
This prepares destination cache lines without changing the launch protocol.
The evaluated A5 port did not establish stable benefit and showed regressions
on short sync-start workloads, so prefetch is not adopted on A5 at this source
baseline. This is a performance choice based on that evaluation, not a missing
dispatch capability or a claim that prefetch can never help A5.
A5 also uses `sync_task_seen` to skip empty ready-queue Tier-0 sync-start probing until
orchestration has published a sync-start task. A2/A3 probes that tier directly.
These are cache/dispatch-work policies, not claims of a universal speedup.
The Tier-0 latch avoids six empty queue probes per dispatch iteration for
programs that have not submitted a sync-start task. Its platform scope reflects
validation: this optimization was measured on A5, while the corresponding
A2/A3 path was left unchanged without an A2/A3 performance evaluation. Unlike
the evaluated prefetch and publication-batching ports, that absence is not
evidence of a negative A2/A3 result.
See the completion and dispatch sources above and [A5 orchestration][a5-orch].

### Normal and fatal AICore retirement

Both runtimes use a dedicated fatal latch, per-core retirement ownership,
grouped EXIT signaling with a shared ACK deadline, register close/readback/drain,
and an isolated GM return gate. AICore waits for that gate before returning;
a timed-out, unacknowledged core is not released and requires host recovery.
The AICPU initializes the gate tail before handshake. Host descriptor copies
exclude that tail on both platforms.

The readiness handoff differs at this baseline. A2/A3 checks `reg_addr` and
claims an initialized core with `core_retired_.exchange(true)`; a request for
a zero-address core is skipped. A5 records `REQUESTED` even before readiness,
publishes `READY` after ownership/payload initialization, and assigns retirement
to whichever atomic operation adds the second bit. Initialization failures use
the blocked partition or unassigned fallback, with zero addresses filtered after
claiming. The two-bit handoff retains early requests; a single claim flag does not.
See cold paths ([A2/A3][a3-cold], [A5][a5-cold]).

The platform CLOSE operation also differs: A2/A3 disables its FAST_PATH window;
A5 writes IDLE to `DATA_MAIN_BASE` and reads it back. A5 has no A2/A3 FAST_PATH
register. Both preserve close/drain before gate release. Sources:
platform register helpers ([A2/A3][a3-regs], [A5][a5-regs]).

### Joined native launch and error ownership

A5 TMR advertises `joined_native_launch_supported_impl()`: a compatible
successor can be submitted while its predecessor is live, but whole-operator
device execution stays serial. Ordering waits for the predecessor's AICore end
before the successor may reset the shared arena. Live preparation requires a
prebuilt-arena cache hit; rebuilding/uploading that arena is refused while a
predecessor owns it.

A5 therefore reads failure details only from the run's retained published
result. A missing or short result leaves the execution error intact rather than
reading a shared header that the successor may have reset. A2/A3 uses serial
launch and retains the shared-header fallback while the failed run owns the
execution claim. Submission capability and error ownership form one contract.
The retained result prevents a successor's shared-memory reset from corrupting
the predecessor's reported status. Extending this capability to A2/A3 would
require the corresponding ordering and result-publication contract; changing
only the capability flag or the shared-header fallback would be insufficient.
See host runtime implementations ([A2/A3][a3-maker], [A5][a5-maker]) and
[host execution ordering][runner].

## Diagnostic Differences

| Area | A2/A3 | A5 | Interpretation |
| ---- | ----- | -- | -------------- |
| Profiled `ChipReadyQueue::push` | Does not explicitly initialize `task_id_snapshot` in that overload | Initializes it to `TaskId::invalid()` | Producer bookkeeping differs; this observation alone does not establish misdispatch |
| Async stall details | No generic `AsyncWaitList::log_diagnostics` dump | Dumps mailbox/wait entries and backend snapshots | Diagnostic coverage differs; backend-specific details are not shared capabilities |
| Scheduler phase attribution | Advances the phase anchor over idle iterations | Leaves idle gaps for post-processing | Timelines can attribute idle time differently despite shared scheduling semantics |

Sources: scheduler definitions ([A2/A3][a3-scheduler], [A5][a5-scheduler]),
dispatch ([A2/A3][a3-dispatch], [A5][a5-dispatch]), and cold-path diagnostics
([A2/A3][a3-cold], [A5][a5-cold]).

These source differences are not all established platform-specific design
requirements. The profiled queue-tag initialization difference has no justified
platform restriction in this comparison. A5's async dump helps identify stalled
completion entries and backend state, but shared COUNTER/SDMA diagnostics are
not inherently A5-only; only backend-specific snapshots require those backends.
Phase attribution reflects where idle time is accounted for, so changing it can
alter diagnostic timelines without changing task execution. Source inspection
alone does not establish which platform's diagnostic policy is preferable.

[a3-config]: https://github.com/hw-native-sys/simpler/blob/f365ae97ea3d61b4d5c71d496df8a11a397c870c/src/a2a3/platform/include/common/platform_config.h
[a5-config]: https://github.com/hw-native-sys/simpler/blob/f365ae97ea3d61b4d5c71d496df8a11a397c870c/src/a5/platform/include/common/platform_config.h
[a3-maker]: https://github.com/hw-native-sys/simpler/blob/f365ae97ea3d61b4d5c71d496df8a11a397c870c/src/a2a3/runtime/tensormap_and_ringbuffer/host/runtime_maker.cpp
[a5-maker]: https://github.com/hw-native-sys/simpler/blob/f365ae97ea3d61b4d5c71d496df8a11a397c870c/src/a5/runtime/tensormap_and_ringbuffer/host/runtime_maker.cpp
[a3-intrinsic]: https://github.com/hw-native-sys/simpler/blob/f365ae97ea3d61b4d5c71d496df8a11a397c870c/src/a2a3/runtime/tensormap_and_ringbuffer/common/intrinsic.h
[a5-intrinsic]: https://github.com/hw-native-sys/simpler/blob/f365ae97ea3d61b4d5c71d496df8a11a397c870c/src/a5/runtime/tensormap_and_ringbuffer/common/intrinsic.h
[a3-executor]: https://github.com/hw-native-sys/simpler/blob/f365ae97ea3d61b4d5c71d496df8a11a397c870c/src/a2a3/runtime/tensormap_and_ringbuffer/aicore/aicore_executor.cpp
[a5-executor]: https://github.com/hw-native-sys/simpler/blob/f365ae97ea3d61b4d5c71d496df8a11a397c870c/src/a5/runtime/tensormap_and_ringbuffer/aicore/aicore_executor.cpp
[a3-pmu]: https://github.com/hw-native-sys/simpler/blob/f365ae97ea3d61b4d5c71d496df8a11a397c870c/src/a2a3/platform/shared/aicpu/pmu_collector_aicpu.cpp
[a5-pmu]: https://github.com/hw-native-sys/simpler/blob/f365ae97ea3d61b4d5c71d496df8a11a397c870c/src/a5/platform/shared/aicpu/pmu_collector_aicpu.cpp
[a3-sdma]: https://github.com/hw-native-sys/simpler/blob/f365ae97ea3d61b4d5c71d496df8a11a397c870c/src/a2a3/runtime/tensormap_and_ringbuffer/runtime/backend/sdma/sdma_completion_scheduler.h
[a5-sdma]: https://github.com/hw-native-sys/simpler/blob/f365ae97ea3d61b4d5c71d496df8a11a397c870c/src/a5/runtime/tensormap_and_ringbuffer/runtime/backend/sdma/sdma_completion_scheduler.h
[a3-async]: https://github.com/hw-native-sys/simpler/blob/f365ae97ea3d61b4d5c71d496df8a11a397c870c/src/a2a3/runtime/tensormap_and_ringbuffer/runtime/async_wait.h
[a5-async]: https://github.com/hw-native-sys/simpler/blob/f365ae97ea3d61b4d5c71d496df8a11a397c870c/src/a5/runtime/tensormap_and_ringbuffer/runtime/async_wait.h
[a3-completion]: https://github.com/hw-native-sys/simpler/blob/f365ae97ea3d61b4d5c71d496df8a11a397c870c/src/a2a3/runtime/tensormap_and_ringbuffer/runtime/scheduler/scheduler_completion.cpp
[a5-completion]: https://github.com/hw-native-sys/simpler/blob/f365ae97ea3d61b4d5c71d496df8a11a397c870c/src/a5/runtime/tensormap_and_ringbuffer/runtime/scheduler/scheduler_completion.cpp
[a3-scheduler]: https://github.com/hw-native-sys/simpler/blob/f365ae97ea3d61b4d5c71d496df8a11a397c870c/src/a2a3/runtime/tensormap_and_ringbuffer/runtime/scheduler/scheduler.h
[a5-scheduler]: https://github.com/hw-native-sys/simpler/blob/f365ae97ea3d61b4d5c71d496df8a11a397c870c/src/a5/runtime/tensormap_and_ringbuffer/runtime/scheduler/scheduler.h
[a3-dispatch]: https://github.com/hw-native-sys/simpler/blob/f365ae97ea3d61b4d5c71d496df8a11a397c870c/src/a2a3/runtime/tensormap_and_ringbuffer/runtime/scheduler/scheduler_dispatch.cpp
[a5-dispatch]: https://github.com/hw-native-sys/simpler/blob/f365ae97ea3d61b4d5c71d496df8a11a397c870c/src/a5/runtime/tensormap_and_ringbuffer/runtime/scheduler/scheduler_dispatch.cpp
[a3-cold]: https://github.com/hw-native-sys/simpler/blob/f365ae97ea3d61b4d5c71d496df8a11a397c870c/src/a2a3/runtime/tensormap_and_ringbuffer/runtime/scheduler/scheduler_cold_path.cpp
[a5-cold]: https://github.com/hw-native-sys/simpler/blob/f365ae97ea3d61b4d5c71d496df8a11a397c870c/src/a5/runtime/tensormap_and_ringbuffer/runtime/scheduler/scheduler_cold_path.cpp
[a5-ring]: https://github.com/hw-native-sys/simpler/blob/f365ae97ea3d61b4d5c71d496df8a11a397c870c/src/a5/runtime/tensormap_and_ringbuffer/runtime/ring_buffer.h
[a5-init]: https://github.com/hw-native-sys/simpler/blob/f365ae97ea3d61b4d5c71d496df8a11a397c870c/src/a5/runtime/tensormap_and_ringbuffer/runtime/shared/runtime_init.cpp
[a5-orch]: https://github.com/hw-native-sys/simpler/blob/f365ae97ea3d61b4d5c71d496df8a11a397c870c/src/a5/runtime/tensormap_and_ringbuffer/runtime/orchestrator.cpp
[a3-regs]: https://github.com/hw-native-sys/simpler/blob/f365ae97ea3d61b4d5c71d496df8a11a397c870c/src/a2a3/platform/shared/aicpu/platform_regs.cpp
[a5-regs]: https://github.com/hw-native-sys/simpler/blob/f365ae97ea3d61b4d5c71d496df8a11a397c870c/src/a5/platform/shared/aicpu/platform_regs.cpp
[a5-options]: https://github.com/hw-native-sys/simpler/blob/f365ae97ea3d61b4d5c71d496df8a11a397c870c/src/a5/platform/onboard/host/CMakeLists.txt
[a5-backends]: https://github.com/hw-native-sys/simpler/tree/f365ae97ea3d61b4d5c71d496df8a11a397c870c/src/a5/runtime/tensormap_and_ringbuffer/runtime/backend/
[builder]: https://github.com/hw-native-sys/simpler/blob/f365ae97ea3d61b4d5c71d496df8a11a397c870c/simpler_setup/runtime_builder.py
[compiler]: https://github.com/hw-native-sys/simpler/blob/f365ae97ea3d61b4d5c71d496df8a11a397c870c/simpler_setup/kernel_compiler.py
[runner]: https://github.com/hw-native-sys/simpler/blob/f365ae97ea3d61b4d5c71d496df8a11a397c870c/src/common/platform/onboard/host/device_runner_base.cpp
[a3-affinity]: https://github.com/hw-native-sys/simpler/blob/f365ae97ea3d61b4d5c71d496df8a11a397c870c/src/a2a3/platform/onboard/host/aicpu_affinity_select.cpp
[a5-topology]: https://github.com/hw-native-sys/simpler/blob/f365ae97ea3d61b4d5c71d496df8a11a397c870c/src/a5/platform/onboard/host/aicpu_topology_probe.h
[a5-sm-init]: https://github.com/hw-native-sys/simpler/blob/f365ae97ea3d61b4d5c71d496df8a11a397c870c/src/a5/runtime/tensormap_and_ringbuffer/runtime/shared/shared_memory.cpp
[a3-l2-query]: https://github.com/hw-native-sys/simpler/blob/f365ae97ea3d61b4d5c71d496df8a11a397c870c/src/a2a3/platform/onboard/host/device_runner.cpp
