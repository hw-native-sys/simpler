# Percounter payload preparation

Each AICore lane claims one list entry at a time. The compile-time constant
`PERCOUNTER_PENDING_CAPACITY` controls how many ordinary candidates it retains;
the default is 1, and setting it to 4 enables a four-candidate window without
changing the scheduling code. A candidate can prepare its `DispatchPayload` once its argument
metadata is published, before its producers complete. Preparation reads tensor
descriptors and scalar arguments, not tensor contents or dispatch predicates.
Fanin readiness and predicate evaluation still gate kernel execution.

The host allocates `PERCOUNTER_PENDING_CAPACITY + 2` 512-byte GM payloads per
lane and seeds each payload's context pointers. Ordinary candidates own the
first `PERCOUNTER_PENDING_CAPACITY` slots; the next slot belongs to the lane's
ordinary MIX subtask and the last to its `sync_start` block. Host and device
derive this layout from the same constant and must be rebuilt together when
it changes. Moving a candidate in the local window retains its payload slot.
Slots are written back before the lane prepares or executes other work;
execution invalidates cached input data.

Each cluster's AIV0 announces a claimed ordinary MIX through a separate SSBUF
preparation record, including its task, block and participant mask. Named peers
prepare their own MIX payload without waiting for its producers or partners.
This record does not reserve a core and does not participate in the MIX/cohort
priority checks. A peer can continue ordinary work or execute its cohort while
the announced MIX remains unready.

In CPU simulation, concurrent preparation-field reads use acquire loads and
publication uses release stores.
These accesses remain atomic if a reader overlaps record reuse and rejects its
snapshot; reading a newer field also orders the opening generation before the
reader's closing generation check.
Initialization uses ordinary stores before the cluster-ready word is published.

Only the existing ready rendezvous admits MIX execution. Its fanin check,
cohort-hold checks, staged mask and execution-order checks remain authoritative.
A late peer prepares its payload before publishing arrival. An ordinary MIX
reuses the payload prepared from its announcement; a MIX cohort uses the
separate cohort slot. Different blocks and subtasks never share mutable block
contexts. Preparing a Graph body starts only after the AICPU publishes its
materialized metadata and order entry.

The allocation is bounded by lane count rather than task or SPMD block count.
It does not expand all task payloads on the Host or move payload construction
to the AICPU. The default uses 1536 bytes per lane; capacity 4 uses 3072 bytes.

## Chip swimlanes (L1–L4)

`enable_chip_swimlane` uses the standard profiling levels:

| Level | Recorded data |
| ----- | ------------- |
| 0 | Disabled; the uninstrumented lane implementation is compiled separately |
| 1 | Kernel execution and final input acquisition, streamed through the shared collector |
| 2 | L1 plus per-execution local dispatch and return timestamps, joined by `(worker_id, reg_task_id, run_epoch)` |
| 3 | L2 plus nested percounter function intervals of at least 1 microsecond, individual GM/SSBUF operations, cache issues, barriers and backoff |
| 4 | L3 with every instrumented function interval retained, plus Host orchestration phases |

Percounter dispatch means entry to execution of a ready, prepared payload;
finish means local kernel return after the output-store barrier. It does not
mean the last sibling finished or every consumer observed the task's completion.
Earlier preparation, fanin probes, MIX rendezvous, sync-start admission and
completion-counter publication appear on the L3/L4 Scalar lane.

Each active worker has its own **AICore Scheduler** lane, grouped by cluster
and C/V0/V1 role. Function names are percounter's own; task IDs retain the HBG
layout shared by the two runtimes. The standard converter nests kernel
intervals inside execution/service functions and computes exclusive time without
double-counting parents. Worker View retains the original kernel intervals.

L2 reserves up to 65,536 task-timing records per worker, bounded by the graph's
block count. Each worker partition starts on its own cache line. Overflow fails the timing export instead of silently joining an
incomplete stream. L3/L4 additionally reserve 8 MiB per worker slot. A compact
Scalar row occupies one 8-byte word: a 32-bit start offset from the worker's
window, a 16-bit duration, an 8-bit phase and a 7-bit nesting depth. This retains
the full clock precision. Intervals outside those bounds use five words with
the complete timestamps, task reference, phase/depth and call count. The buffer
holds up to 1,048,576 compact rows or 209,715 extended rows. Only headers are
initialized/uploaded; the Host downloads committed words after device retirement
and derives the row count while decoding. Scalar overflow is exported as
`capture.truncated=true` with a dropped count; the analyzer does not claim a
complete time partition for that worker. L0/L1 allocate neither record tail.
Profiling storage belongs to the run's scheduler-state allocation and is reset
on reuse; each profiled AICore entry resets its local recorder binding.

Memory leaves preserve the protocol's existing accesses and barriers. Returning
atomics and loads end when the result is locally consumable; stores and atomics
whose results are unused record issue time. DCCI issue and DSB completion are
separate events. L2+ task records are staged locally and written after the
business completion signal; Scalar records use private cached storage. A terminal
cache clean and barrier publish the records before retirement. L1 batches cache
publication at full-buffer handoff and final partial-buffer retirement, retaining
the shared collector's bounded buffer rotation.
The detailed stream starts after the startup handshake and topology acquisition;
it ends at record publication, excluding the final retirement wait. Profiling
perturbs execution, so use level 0 for latency measurements.

`enable_chip_swimlane=1` records each executed kernel through the shared
`ChipSwimlaneAicore` collector. The standard `swimlane_converter` accepts
`runtime=percounter` and decodes the shared HBG task identity, including Graph
body tasks. Host runtime selection reports mode 5.

The receive timestamp is entry to execution of an already prepared payload;
start is immediately before the kernel call, and end follows its output-store
barrier. Local setup therefore includes execution-time cache acquisition and
record reservation, but excludes earlier payload preparation and fanin waits.
Predicated-off and dependency-only tasks do not emit kernel records.

Task identity is captured from the selected task when its kernel returns and
retained in the pending timing record. Publication copies that captured identity;
it does not reread a reused task or payload. No async token is stored in the
payload. A self-scheduling core reserves from its cached buffer without
observing the shared head while space remains; only the first reservation and
full-buffer replacement need to observe it. This rule belongs to the
self-scheduled path, whose core alone publishes the full count.

Each buffer holds `PLATFORM_AICORE_BUFFER_SIZE` records. A self-scheduling core
publishes its committed count only when full or when the lane exits with a
partial tail, and waits for replacement when a full buffer has no space.
The shared AICPU collector uses the existing rotation, ready queue, Host drain
and free-buffer refill paths; there is no per-run 1024-record limit. The final
partial buffer uses the same flush path. A replacement wait that exceeds the
platform DFX backpressure budget reports a runtime timeout. L1 does not record
scheduler phases. Profiling disabled uses the same runtime binary.

The common device-timing output also reports `AICore` wall independently of L1:
the earliest post-handshake lane-work start to the latest lane-work end. It
includes AICore scheduling and dependency waits, with two clock reads per lane.
It excludes the handshake and the final retirement wait. See
[Host trace timing](../../../../docs/dfx/host-trace.md) for the metric boundary.

## Deferred completion regression tests

Percounter is an opt-in runtime (`BUILD_BY_DEFAULT = False`). Its deferred
completion scenes are manual, matching the explicit runtime build requirement:

```bash
python -m simpler_setup.build_runtimes --platforms a5sim --runtime percounter
pytest tests/st/a5/percounter/defer_wait --platform a5sim --manual include
```

The same scene cases support `a5`; submit each hardware case in its own device
allocation after preparing its kernels.
