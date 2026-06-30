# Run Latency Optimization Assessment

**Date**: 2026-06-29
**Status**: assessment

## Summary

This note evaluates run-level latency optimizations for the L2
`tensormap_and_ringbuffer` path. The focus is the per-token gap between host
wall time, device wall time, and device-log `Total`.

The main conclusion is:

- Host-only cross-run staging is low risk but probably low value.
- Overlapping the current full bind path is not a planned optimization because
  it needs a second set of per-run device tensor buffers.
- Simple double buffering does not make one run's AICPU init/teardown overlap
  with that same run's `Total`; that ordering is a real dependency.
- Cross-run AICPU init overlap is plausible only with isolated per-run
  scheduler/control state, or with a longer-lived executor that avoids most
  per-run init/teardown.
- The first work item should be measurement: split bind, runtime arena,
  device-copy, validate, and AICPU init/teardown timers before implementing a
  pipeline.

## Scope

This is separate from callable-level prepare overlap. Callable prepare overlap
prepares a future callable while the current callable runs. Run-level work is
the per-invocation work still paid by every `run_prepared()` call:

- bind this run's tensor and scalar arguments;
- allocate or acquire per-run device buffers;
- copy or memset tensor payloads;
- stage runtime args and kernel args;
- launch AICPU/AICore work;
- copy outputs back and free per-run buffers.

The current host call order is:

```text
run_prepared()
  bind callable snapshot
  bind_callable_to_runtime_impl()
  DeviceRunner::run()
  validate_runtime_impl()
```

For TRB, `bind_callable_to_runtime_impl()` already performs device work for
non-child-memory tensors: it calls `device_malloc()`, then H2D copy or device
memset, before `DeviceRunner::run()` launches.

## Timing Terms

The important timing terms are:

- `host_wall`: host-side `steady_clock` around `run_prepared()`. Includes
  bind, launch/sync, and validate.
- `device_wall`: full on-NPU AICPU exec wall. Earliest AICPU entry start to
  latest AICPU entry end.
- `Total`: device-log window from `min(orch_start, sched_start)` to
  `max(orch_end, sched_end)`.
- `Orch`: device-log orchestrator graph-build window.
- `Sched`: device-log scheduler/execute window across scheduler threads.

Useful relationship:

```text
host_wall   = host_pre_launch + launch/sync + host_post_sync
device_wall = AICPU init + Total + AICPU teardown
```

`Total` is the useful AICPU orchestration plus scheduler/execution window. It
does not include host bind/validate, nor AICPU init/teardown.

## Staging Levels

### 1. Host-Only Staging

Host-only staging means preparing run N+1 while all outputs of the preparation
remain in host memory. It does not call device allocation, device copy, device
memset, runtime-arena upload, or AICPU/AICore launch APIs.

Examples:

- parse N+1 `ChipStorageTaskArgs`;
- count tensors/scalars and inspect shape/stride/dtype;
- classify tensor direction and child-memory status;
- decide copy-back policy;
- choose logical double-buffer slots;
- build a host-side bind plan;
- prepare host-side `Runtime` template fields;
- compute block dim, ring sizes, and launch descriptors.

Expected benefit is likely small. Once `device_malloc`, H2D copy, memset,
runtime upload, and validate/free are excluded, the remaining host CPU work is
usually tens of microseconds to a few hundred microseconds, and only likely
reaches low single-digit milliseconds for very large argument sets or
unexpectedly expensive host code.

Use host-only overlap only if measurement proves:

- host-only plan/build time is consistently above 1 ms, or above 5 percent of
  steady TPOT;
- the implementation does not force extra device memory;
- the resulting pipeline does not complicate error handling or ownership.

### 2. Current Full Bind Staging

The current full TRB bind is not host-only. For every non-child-memory tensor
it:

- allocates a device buffer;
- copies input/INOUT tensors to device;
- memsets pure output tensors on device;
- records tensor pairs for later D2H and free.

Therefore, a cross-run pipeline that simply starts today's full bind for run
N+1 while run N is executing needs run N+1 device buffers to coexist with run
N's live buffers. This can hide real `host_pre_launch` time, but it consumes
extra HBM and may contend for allocator locks, H2D bandwidth, and HBM.

Do not make this the default optimization path. Treat it as out of scope unless
there is an explicit product decision to spend the extra HBM for a second
per-run tensor-buffer set.

### 3. Device-Control Staging

Device-control staging means preparing device-side `Runtime`, `KernelArgs`,
PTO2 shared memory, runtime arena image, or even AICPU scheduler state for run
N+1 before run N completes.

Small device-control staging may be cheap, for example an extra `Runtime` and
`KernelArgs` slot. Starting N+1's AICPU scheduler init during N's `Total` is
much stronger. It requires:

- separate `Runtime` / `KernelArgs` slots;
- scheduler state that does not overwrite run N;
- AICore handshake/register state that does not disturb run N's active cores;
- a launch/control protocol that is not serialized behind the current
  `DeviceRunner::run()` stream sync;
- spare AICPU capacity so N+1 init does not slow N's scheduler/orchestrator.

Until those isolation properties exist, prefer reducing per-run init/teardown
or using a persistent executor over trying to overlap AICPU init directly.
If this path stages device-side state for N+1, it may need extra small control
slots, an extra runtime arena image, or an independent PTO2 shared-memory
region. Extra small control slots are conditional; extra arena or shared-memory
regions are larger HBM costs and should be avoided by default.

## Double Buffering And Memory

The useful cross-run shape is:

```text
run N:     live buffer A ---- execution ---- validate/free A
run N+1:          prepare buffer B ---------------- launch B
```

Double buffering does not overwrite run N's live memory. It either uses a
separate run N+1 slot or waits until run N releases its slot.

If a device-side staging path is explicitly accepted, there are two
implementation choices:

- Allocate N+1 buffers while run N is executing. This hides allocation but can
  add allocator variance and fail if free HBM is low.
- Preallocate A/B slots during warmup. This removes allocator latency from the
  hot pipeline, but reserves the extra HBM for the worker's lifetime.

These choices are memory-lifetime mechanisms, not default recommendations for
duplicating tensor buffers, runtime arenas, or PTO2 shared memory.

In both cases, if the staged object is device-side, peak HBM increases:

```text
peak_hbm_with_pipeline
  = steady_hbm_without_pipeline
  + staged_run_N_plus_1_bytes
  + allocator/safety margin
```

This does not "steal" an already-owned run N buffer. It can still reduce the
free HBM available to run N if both runs allocate from the same device memory
pool and run N performs late allocations. A production design should reserve a
pipeline pool up front or gate staging by measured free HBM.

Likely duplicated device memory:

- per-run input/output tensor buffers, if full bind is overlapped; this is not
  planned by default;
- `Runtime` device copy;
- `KernelArgs` device copy;
- PTO2 shared memory, if N and N+1 both need independent staged state;
- runtime arena image, if upload overlaps execution;
- diagnostic/device-wall buffers, if N is read after N+1 starts.

Data that should not be duplicated:

- uploaded callable/kernel buffers;
- AICPU-prewarmed orchestration SO handles;
- model weights;
- stable KV/cache buffers, when update semantics are in-place and ordered;
- long-lived GM heap region, unless two logical runs need independent heap
  cursors at the same time; that concurrent-heap case is not planned by
  default.

Device-buffer duplication classes:

- No duplication: host-only plans, topology metadata, log-level changes,
  resident weights, ordered KV/cache updates, and child-memory pass-through.
- Small control duplication: extra `Runtime`, `KernelArgs`, diagnostic, or
  device-wall slots. This is still device memory, but it is not a second tensor
  buffer set.
- Arena duplication: an extra runtime arena image or independent PTO2
  shared-memory region. These can be MB-scale or larger, so avoid duplicating
  them by default.
- Tensor-buffer duplication: full-bind staging or output double buffering where
  run N data stays live while run N+1 writes another buffer. This is not planned
  by default.

Memory decision list:

- Host-only plan/template: no extra device memory; allowed.
- Runtime/KernelArgs staging: one extra small slot set; conditional.
- Runtime arena upload: one extra runtime arena image; avoid by default.
- PTO2 independent staged state: second shared-memory region; avoid by
  default.
- Output retained while next writes: second output buffer; avoid by default.
- Full bind tensor staging: second per-run tensor buffer set; not planned.
- True concurrent device runs: full per-run isolation; not planned.

This section describes the memory contract only. It does not make double
buffering a first-line optimization; the decision status is captured in
`Cross-Run Pipeline / Double Buffering` below.

## Optimization Candidates

### 1. Add Split Timing First

Current timing is too coarse. `args_malloc_copy` mixes host loops, allocation,
H2D copy, and memset. It cannot prove host-only overlap value.

Add timers for:

- callable snapshot bind;
- host-only bind plan construction;
- tensor device allocation;
- tensor H2D copy or memset;
- runtime arena host build;
- runtime arena device upload;
- `Runtime` device copy;
- `KernelArgs` device copy;
- launch/sync;
- validate status/header read;
- validate tensor D2H copy;
- validate free;
- AICPU init, `Total`, and teardown.

Acceptance:

- timer overhead is below 1 percent in quiet mode;
- timings are reported with a stable naming scheme;
- p50/p90/p99 are collected for steady decode.

### 2. Per-Run Tensor Binding

Current work:

- allocate device memory for each non-child tensor;
- copy input and INOUT tensors H2D;
- memset pure output tensors on device;
- record tensor pairs for later D2H and free.

Optimization directions:

- Keep stable input/output buffers resident across decode steps when shape and
  lifetime are stable.
- Pass device-resident child memory through instead of rematerializing host
  tensors.
- Pool per-shape tensor buffers and reuse them across runs.
- Keep pooling serial by default: reuse one logical slot after the previous
  owner releases it, instead of allocating a second per-run tensor-buffer set.
- Skip H2D for immutable data that is already resident on device, including
  KV cache, weights, and constant inputs.
- Skip D2H for intermediate outputs that immediately feed another device run.
- Use smaller explicit output descriptors when the host only needs a scalar or
  compact final result.

Device-buffer duplication note:

- Resident immutable data should be shared, not duplicated.
- Pooling does not require two device buffers if producer/consumer lifetimes are
  serial.
- A second tensor buffer is needed only when old tensor contents must stay live
  while the next run writes another logical version; avoid this by default.

Potential conditions:

- tensor allocation/copy/memset is consistently above 1 ms;
- tensor shapes are stable across many decode steps;
- large input tensors are copied every run without changing contents;
- KV cache, weights, or constant inputs are repeatedly uploaded from host;
- output tensors are copied back only to feed another device run.

Acceptance:

- `host_wall` drops by at least 5 percent, or at least 1 ms for steady decode;
- `device_wall` and `Total` do not increase by more than 1 percent;
- HBM stays inside a documented per-worker budget;
- repeated runs do not grow live allocation count.

### 3. Validate / Copy-Back / Free

Code reference:

- `src/a5/runtime/tensormap_and_ringbuffer/host/runtime_maker.cpp`:
  `validate_runtime_impl()`.

Current work:

- read PTO2 shared-memory header back to host;
- copy OUTPUT and INOUT tensors D2H;
- free all per-run tensor device allocations;
- clear dispatch-table and tensor-pair state.

Optimization directions:

- Keep outputs on device when the next stage can consume device pointers.
- Copy back only declared final outputs, not every writable tensor.
- Copy only the final small result that must be visible to host; avoid copying
  intermediate large tensors.
- Replace per-run free with a buffer-pool release.
- Reuse or pool output buffers so `device_free` moves from every run to a
  longer-lived owner or lifecycle boundary.
- Batch small D2H copies when correctness allows.
- Split error-status readback from full output validation.
- Make status/header D2H a lighter-weight error snapshot path, separate from
  successful hot-path output materialization.

Device-buffer duplication note:

- Keeping an output on device does not require a second buffer if ownership is
  transferred to the next consumer and the slot is released before reuse.
- It does require output double buffering if run N's output remains live while
  run N+1 writes the same logical output slot. Avoid that for large tensors
  unless explicitly accepted.

Potential conditions:

- D2H output copy or per-run free is visible in `host_post_sync`;
- workload has multi-stage device pipelines where host does not inspect
  intermediate tensors;
- output size is large compared with the host-consumed result.

Acceptance:

- `host_post_sync` drops by at least 1 ms, or D2H bytes per token drop by at
  least 50 percent;
- no output is skipped without an ownership and consumer contract;
- error paths still copy enough state for diagnostics.

### 4. Runtime Arena, Runtime Args, And Kernel Args

Current work:

- derive effective ring sizes;
- reserve and commit a host arena;
- initialize the prebuilt PTO2 runtime image on host;
- upload the image into the pooled runtime arena each run;
- allocate/copy/free `Runtime` and `KernelArgs` device slots.

Optimization directions:

- Cache runtime arena layout metadata and image by `(task_window, heap,
  dep_pool)`.
- Patch only fields that differ for the current run.
- Keep resident `Runtime` and `KernelArgs` slots per runner.
- Use two small `Runtime` / `KernelArgs` slots only if N+1 control staging
  overlaps N.
- Patch and copy only changed bytes if a safe ABI is introduced.

Device-buffer duplication note:

- Resident `Runtime` and `KernelArgs` slots are small control buffers; keeping
  one slot is preferred.
- Two slots are only needed when N+1 device-side staging overlaps N.
- Host-side runtime arena layout/image cache does not need another device
  buffer.
- Overlapping N+1 runtime-arena device upload while N is still running needs a
  second device runtime-arena slot. The runtime arena can be large, so avoid
  this by default; prefer host-side cache and upload only after the current run
  no longer needs the device arena.

Potential conditions:

- runtime arena build/upload is consistently above 0.5 to 1 ms;
- `Runtime` / `KernelArgs` allocation or copy is visible in `host_pre_launch`;
- the workload repeats the same ring sizes and task-window shape.

Acceptance:

- warmed runs show measurable `host_pre_launch` reduction;
- AICPU attach/wire/reset remains correct for every run;
- cached images invalidate cleanly when ABI, ring sizes, or platform config
  changes;
- error recovery and finalize free resident slots exactly once.

### 5. Topology, Block-Dim, And Launch Metadata

Current work:

- resolve block dim;
- on a5, probe AICPU topology and compute allowed CPUs;
- fill launch metadata into `Runtime`;
- derive launch counts from requested AICPU thread count and platform limits.

Optimization directions:

- Cache topology probe results per `(device_id, process, platform)`.
- Cache a5 topology and allowed CPU lists by requested AICPU thread count.
- Avoid repeated block-dim queries when config pins a known value.
- Keep launch metadata templates for repeated decode shapes.

Potential conditions:

- topology probe or block-dim query is above 0.2 to 0.5 ms per run;
- the same worker runs many tokens on one device without changing launch shape;
- launch metadata is identical across steady decode steps.

Acceptance:

- launch metadata matches the uncached path bit-for-bit for the same device;
- cache clears on device reset/finalize;
- wrong-arch or wrong-SKU failures remain fail-fast;
- `host_pre_launch` drops without changing `device_wall` or `Total`.

### 6. Device Wall Init/Teardown

Status: deferred / conditional. Do not treat this as a first-line overlap
optimization.

Decision note:

- Same-run AICPU init, `Total`, and teardown cannot be overlapped by double
  buffering; their order is a real dependency.
- Cross-run N+1 init during N's `Total` is theoretically possible, but only
  after scheduler/control state, AICore register state, and per-run args slots
  are isolated.
- The safer direction is to reduce or persist init/teardown work, not to make
  simple double buffering carry it.

Current work inside `device_wall`:

- AICPU executor init;
- scheduler init;
- AICore handshake and assignment;
- runtime attach/wire/reset;
- scheduler shutdown;
- AICore register deinit;
- executor deinit and runtime destroy.

Optimization directions:

- Measure AICPU init and teardown separately from `Total`.
- Preserve static per-core assignment and metadata across runs.
- Keep AICore worker state alive when queues can be reset safely.
- Convert repeated launch/shutdown into a long-lived executor that receives
  work through a device mailbox or ring.

Gate conditions:

- measured signal remains after logging/profiling noise is controlled:
  `device_wall - Total` is consistently above 3 ms, or init/teardown alone is
  above 1 ms for steady decode;
- the workload runs many homogeneous decode steps on the same device;
- split timing proves this is still a bottleneck after lower-risk host and
  binding optimizations are applied.

Acceptance:

- `device_wall - Total` drops by at least 30 percent, or by at least 1 ms;
- `Total` does not increase by more than 1 percent;
- no increase in AICore op-timeout, AICPU exception, or stream-sync failures;
- emergency shutdown still leaves the device recoverable.

### 7. Cross-Run Pipeline / Double Buffering

Status: deferred / conditional. Use this only after split timing proves there
is enough stageable work.

Decision note:

- Host-only pipeline is feasible but expected to be low value unless host-only
  plan/build time independently measures above 1 ms or 5 percent of TPOT.
- Full-bind pipeline is not planned by default because it requires run N+1's
  tensor buffers to coexist with run N's live buffers.
- Output double buffering is also not planned by default when it keeps run N
  outputs live while run N+1 writes a second output slot.
- Double buffering should not be used as the current plan for AICPU
  init/teardown overlap; that becomes device-control pipelining with much
  stronger isolation requirements.

Optimization directions:

- If only host-only staging is expensive, build a host bind-plan pipeline.
- Do not add A/B tensor slots for full-bind overlap unless the extra device
  buffer set is explicitly accepted.
- Do not add output double buffers for large tensors unless the consumer
  lifetime proves they are necessary and HBM is explicitly budgeted.
- Prefer preallocated small control/arg slots only for accepted device-side
  staging.
- Gate any device-side N+1 staging by HBM headroom.
- Keep launch ordering explicit: N+1 may not consume staged state until its
  staging is complete and N's required resources are released.

Gate conditions:

- host-only plan/build is measured above 1 ms or 5 percent of TPOT;
- run N has enough `Total` time to cover N+1 staging;
- no second per-run tensor or output buffer set is required, unless explicitly
  accepted;
- active-run inflation is less than 20 percent of hidden staging time.

Acceptance:

- TPOT improves by at least 5 percent, or at least 1 ms absolute;
- p99 does not regress by more than 5 percent;
- peak HBM stays inside the documented budget;
- run N `Total` does not materially increase from background copy/allocation;
- error and timeout paths release or quarantine both slots correctly.

### 8. Logging And Diagnostic Overhead

Current work:

- hot paths contain many `LOG_INFO_V0` and `LOG_INFO_V9` records;
- tensor binding and validate paths print per-tensor `LOG_INFO_V0` records;
- external reports indicate some `printf` paths can be multi-ms.

Optimization directions:

- Ensure performance runs use a quiet log level.
- Move per-tensor and per-thread logs behind higher verbosity.
- Reduce `LOG_INFO_V0` in hot paths, especially per-tensor bind and validate
  prints.
- Use counters or compact summaries instead of repeated formatted strings.
- Verify device-log timing collection does not perturb the target workload.

Potential conditions:

- `host_wall`, `device_wall`, or `Total` changes materially when log level
  changes;
- device log contains repeated per-tensor/per-task records in steady decode.

Acceptance:

- quiet mode keeps required error diagnostics;
- performance variance drops;
- removing logs does not change correctness or device synchronization.

## Decision Rules

Recommended default order after measurements:

1. Add split timers.
2. Optimize tensor residency, child-memory use, buffer pooling, and H2D/D2H
   avoidance.
3. Cache runtime arena metadata/images and keep small arg slots resident.
4. Cache topology, block-dim, and launch metadata if they measure visible.
5. If `device_wall - Total` remains a measured bottleneck, consider persistent
   executor or reduced teardown.
6. Do not pursue full-bind or output-double-buffer pipelines unless the second
   device tensor-buffer set is explicitly accepted.
7. Consider host-only cross-run staging only if it independently measures above
   1 ms or 5 percent of TPOT.

Reject or postpone an optimization when:

- the measured component is below 1 ms and below 5 percent of TPOT;
- the change increases run N `Total` by more than 1 percent;
- the change improves average TPOT but worsens p99 by more than 5 percent;
- HBM headroom cannot cover any explicitly accepted device-side staging;
- the design requires a second tensor/output buffer set without explicit
  acceptance;
- ownership on error/timeout paths is unclear.

## Measurement Plan

For any optimization above, collect before/after:

- `host_wall`;
- `device_wall`;
- device-log `Total`, `Orch`, and `Sched`;
- split `host_pre_launch`;
- split `host_post_sync`;
- `device_wall - Total`;
- HBM live allocation high-water mark;
- background-staging active-run inflation;
- p50/p90/p99 over steady decode tokens.

Recommended thresholds:

- Treat changes within +/-2 percent as noise unless repeated across devices.
- Require at least 5 percent TPOT improvement, or at least 1 ms absolute
  improvement for steady decode, before accepting a complexity-increasing
  optimization.
- Reject optimizations that improve average latency but worsen p99 by more than
  5 percent without an explicit serving-policy reason.

## References

- `docs/dfx/l2-timing.md`
- `docs/callable-prepare-overlap-plan.md`
- `src/common/platform/onboard/host/c_api_shared.cpp`
- `src/common/platform/onboard/host/device_runner_base.cpp`
- `src/a5/platform/onboard/host/device_runner.cpp`
- `src/a5/runtime/tensormap_and_ringbuffer/host/runtime_maker.cpp`
- `src/a5/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp`
