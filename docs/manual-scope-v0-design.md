# Manual Scope V0 Design

Date: 2026-04-15

Branch: `manual_scope_v0`

## Goal

Add a lighter manual-scope mode to `a2a3/tensormap_and_ringbuffer` that:

- keeps the same submit API shape as AUTO mode
- does not introduce a separate manual submit API family
- does not support delayed dependency wiring
- publishes tasks at submit time, like AUTO mode
- allows explicit same-scope task ordering without using TensorMap for
  current-manual-scope-local tensors

This is intentionally smaller than the previous manual-scope branch.

## Constraints

The v0 branch must follow these rules:

1. Use the same submit API as AUTO mode.
2. Append explicit dependencies into `Arg`, for example `args.add_dep(task_id)`.
3. Do not support delayed wiring and delayed linking.
4. Publish at submit time, same as AUTO mode.
5. Treat tensor allocation as a task for manual dependency building.
6. Determine whether TensorMap lookup or insert is required from tensor scope
   metadata first, not from ring id.

## Non-goals

- no nested manual scopes in v0
- no post-submit `add_dependency(...)`
- no delayed explicit-edge replay or scope-end linking
- no batch publish barrier at manual `scope_end()`
- no implicit TensorMap fallback for current-manual-scope-local tensors
- no attempt to redesign AUTO mode

## User-Facing API

### Scope

Manual mode remains an explicit scope:

```cpp
PTO2_SCOPE(PTO2ScopeMode::MANUAL) {
    ...
}
```

Nested manual scopes are rejected.

### Submit API

Keep the existing AUTO-mode submit entry points:

```cpp
auto out = pto2_rt_submit_aic_task(FUNC_ID, args);
auto out = pto2_rt_submit_aiv_task(FUNC_ID, args);
auto out = pto2_rt_submit_task(mixed_kernels, args);
```

No `*_manual(...)` or `*_manual_with_deps(...)` APIs in v0.

Each submit result should carry:

- a standalone `task_id`
- zero or more materialized output tensors

The returned task id must not depend on whether the task created any new
`OUTPUT` tensors.

### Explicit dependencies

`Arg` grows explicit dependency support:

```cpp
Arg args;
args.add_input(...);
args.add_dep(task_id);
```

Rules:

- `Arg.add_dep(...)` is valid only inside `PTO2_SCOPE(PTO2ScopeMode::MANUAL)`
- explicit deps must point to tasks created in the current manual scope
- explicit deps are attached to the consumer before submit
- no delayed wiring after submit

### Tensor alloc

`alloc_tensors(...)` stays output-only, but returns a producer task id:

```cpp
auto alloc = alloc_tensors(create_info0, create_info1);
// alloc.task_id()
// alloc.get_ref(...)
```

This allows later manual tasks to depend on the allocation task explicitly.

`alloc_tensors(...)` itself does not accept `Arg.add_dep(...)`.

## Runtime Model

### High-level behavior

Manual mode in v0 is:

- AUTO-style submit and publish
- plus explicit deps from `Arg`
- plus full TensorMap bypass for current-manual-scope-local tensors

There is no hidden manual subgraph and no delayed publish.

### Submit-time flow

Inside manual scope, submit should do this:

1. Allocate the task slot and task id.
2. Read explicit deps from `Arg`.
3. Validate that explicit deps belong to the current manual scope.
4. Turn explicit deps into ordinary fanins immediately.
5. Classify tensor args as current-manual-scope-local or boundary.
6. Do not perform TensorMap lookup or TensorMap insert for any manual-local
   tensor, regardless of whether it is `INPUT`, `INOUT`, or
   `OUTPUT_EXISTING`.
7. Treat explicit task ids as the only dependency source for manual-local
   tensors.
8. Keep normal creator-retention and TensorMap behavior for boundary tensors.
9. Publish the task immediately, same as AUTO mode.

### Scope-end behavior

`scope_end()` should keep only normal scope-lifetime behavior.

It should not do any of the old manual-specific work:

- no deferred explicit-edge linking
- no explicit-edge replay
- no batch publish of manual tasks

## Metadata Model

Ring id is not the primary locality test in v0.

Each produced task/tensor should carry scope metadata:

- `producer_scope_depth`
- `producer_manual_scope_depth`

For v0, nested manual scopes are still rejected, but storing
`producer_manual_scope_depth` now gives a clean upgrade path later for
distinguishing outer-manual-scope tensors.

External tensors use invalid producer scope metadata.

## Tensor Dependency Rule

When a tensor is used inside manual scope:

### Manual-local tensor

Treat a tensor as manual-local only when:

- it was produced in the current manual scope
- its stored producer manual-scope depth matches the current manual-scope depth

Behavior:

- explicit task ids are the only ordering source
- do not perform TensorMap lookup
- do not perform TensorMap insert
- do not rely on TensorMap to rediscover either creators or modifiers
- producer / updater coverage must already be present in `Arg.add_dep(...)`

### Boundary tensor

Treat everything else as boundary:

- external tensors
- tensors from AUTO scope
- tensors from outer scopes
- tensors from outer manual scopes

Behavior:

- keep creator retention
- keep normal TensorMap lookup/insert behavior unless `manual_dep=true`

This is intentionally conservative.

## `manual_dep=true`

`manual_dep=true` keeps its existing meaning:

- skip TensorMap lookup/insert for that tensor
- keep creator retention through task ownership metadata

It is orthogonal to manual scope.

Inside manual scope, `manual_dep=true` is mainly useful for boundary tensors.
Manual-local tensors already bypass TensorMap by scope rule, so they do not
need `manual_dep=true` to get that behavior.

## Representation Change From Previous Design

V0 intentionally narrows manual scope.

Previous heavier direction:

- submit first
- wire later
- link later
- publish later

V0:

- consumer must know explicit deps at submit time
- no post-submit dependency wiring
- no delayed linking
- no delayed publish

This means manual scope in v0 is not a general explicit-graph construction API.
It is a lighter explicit-dependency annotation on top of normal submit.

## Practical Example

```cpp
PTO2_SCOPE(PTO2ScopeMode::MANUAL) {
    Arg qk = make_qk_args(...);
    auto qk_out = pto2_rt_submit_aic_task(FUNC_QK_MATMUL, qk);

    Arg sf = make_sf_args(qk_out.outputs.tensor(0), ...);
    sf.add_dep(qk_out.task_id);
    auto sf_out = pto2_rt_submit_aiv_task(FUNC_SOFTMAX_PREPARE, sf);

    Arg pv = make_pv_args(sf_out.outputs.tensor(0), ...);
    pv.add_dep(sf_out.task_id);
    auto pv_out = pto2_rt_submit_aic_task(FUNC_PV_MATMUL, pv);

    Arg up = make_update_args(...);
    up.add_dep(sf_out.task_id);
    up.add_dep(pv_out.task_id);
    (void)pto2_rt_submit_aiv_task(FUNC_ONLINE_UPDATE, up);
}
```

This keeps the orchestration shape readable while avoiding a separate manual
submit API family.

For repeated zero-output updater chains, orchestration must still carry the
chain explicitly:

```cpp
PTO2TaskId prev_update = PTO2TaskId::invalid();

for (...) {
    Arg up = make_update_args(...);
    if (prev_update.is_valid()) {
        up.add_dep(prev_update);
    }
    TaskOutputTensors update_out = pto2_rt_submit_aiv_task(FUNC_ONLINE_UPDATE, up);
    prev_update = update_out.task_id();
}
```

This is the intended v0 use of the direct submit-result task id after removing
manual-local TensorMap fallback.

## Important Limitation

V0 still rejects nested manual scopes and still requires explicit deps to be
known at submit time.

With a standalone task id in `TaskOutputTensors`, a zero-output updater can
still be named by later `Arg.add_dep(...)` calls. That keeps manual-local
dependency expression explicit without relying on TensorMap fallback.

## Validation Plan

### Unit / behavior

- reject `Arg.add_dep(...)` outside manual scope
- reject invalid task ids in manual scope
- reject explicit deps that are not from the current manual scope
- reject nested manual scopes
- verify manual-local tensors skip both TensorMap lookup and TensorMap insert
- verify boundary tensors still use TensorMap

### Allocation behavior

- verify `alloc_tensors(...)` returns `{task_id, outputs}`
- verify manual tasks can depend on alloc task ids

### Regression / examples

- add explicit-manual variants for the two paged-attention workloads:
  `paged_attention_manual_scope` and `paged_attention_unroll_manual_scope`
- keep the existing AUTO paths as the comparison baseline
- verify `Case1` and `Case2` against golden outputs on real hardware
- benchmark AUTO vs manual-scope with the same 30-round trimmed-average method
  used by `tools/benchmark_rounds.sh`

## Validation Status

This document reflects the cleaned `manual_scope_v0` branch state validated on
2026-04-15. The validation below is the status of the current implementation,
including the latest unit-level alignment for direct submit-result task ids and
manual-local TensorMap bypass, plus the example-side explicit updater chaining
needed once zero-output updater tasks stop relying on TensorMap fallback.

### Automated checks

- C++ UT passed:
  - `test_a2a3_pto2_manual_scope_api`
  - `test_a2a3_pto2_manual_scope_runtime`
- simulation negative test passed:
  - `tests/st/a2a3/tensormap_and_ringbuffer/test_manual_scope_validation.py`
- rerun commands:
  - `ctest --test-dir tests/ut/cpp/build -R 'test_a2a3_pto2_manual_scope_(api|runtime)' --output-on-failure`
  - `python -m pytest tests/st/a2a3/tensormap_and_ringbuffer/test_manual_scope_validation.py --platform a2a3sim --device 0 -q`

### Real-device golden checks

Environment:

- platform: `a2a3`
- device: `9`
- PTO-ISA commit: `d96c8784`
- local package rebuilt with `pip install --no-build-isolation -e .`

Fresh hardware reruns passed for all four kept paged-attention paths:

- `examples/a2a3/tensormap_and_ringbuffer/paged_attention`
  - `Case1`
  - `Case2`
- `examples/a2a3/tensormap_and_ringbuffer/paged_attention_manual_scope`
  - `Case1`
  - `Case2`
- `tests/st/a2a3/tensormap_and_ringbuffer/paged_attention_unroll`
  - `Case1`
  - `Case2`
- `examples/a2a3/tensormap_and_ringbuffer/paged_attention_unroll_manual_scope`
  - `Case1`
  - `Case2`

## Fresh Benchmark Status

### Method

This benchmark block is the authoritative batch for the current aligned branch
state.

It includes:

- direct `TaskOutputTensors::task_id()` storage on submit results
- manual-local TensorMap lookup / insert bypass in the runtime
- explicit `update(i) -> update(i+1)` chaining in the two manual paged-attention
  examples

- base commit before the example-side chaining fix: `3d36370`
- platform: `a2a3`
- device: `9`
- PTO-ISA commit: `d96c8784`
- rounds: `30`
- aggregation: trimmed average, dropping `10` low and `10` high rounds
- runner:
  - scene-test entrypoint for AUTO paths
  - `run_example.py` for manual-scope example paths
- benchmark mode: `--skip-golden`
- timing source: device log parsing with the same `orch_start/orch_end` logic
  as `tools/benchmark_rounds.sh`

Important reading rule:

- compare AUTO vs manual only within the same row
- do not compare `paged_attention` rows directly against
  `paged_attention_unroll` rows as if they were the same workload

### Results

| Example | Case | Auto Elapsed Trim (us) | Auto Orch Trim (us) | Manual Elapsed Trim (us) | Manual Orch Trim (us) | Elapsed Delta | Orch Delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `paged_attention` | `Case1` | 77.5 | 62.4 | 124.0 | 108.6 | +46.5 us (+60.0%) | +46.2 us (+74.0%) |
| `paged_attention` | `Case2` | 96.2 | 74.1 | 146.7 | 124.6 | +50.5 us (+52.5%) | +50.5 us (+68.2%) |
| `paged_attention_unroll` | `Case1` | 1138.4 | 800.7 | 1131.3 | 694.9 | -7.1 us (-0.6%) | -105.8 us (-13.2%) |
| `paged_attention_unroll` | `Case2` | 519.7 | 319.1 | 511.3 | 282.6 | -8.4 us (-1.6%) | -36.5 us (-11.4%) |

### Reading The Batch

- Non-unroll manual scope is still materially slower than AUTO in this fresh
  batch. The regression is almost entirely orchestration time.
- Unroll manual scope is now slightly faster than AUTO in total elapsed time
  and materially lower in orchestration time on both kept cases.
- The current v0 branch is functionally correct on hardware, but the
  non-unroll manual path still does not meet the earlier performance target.

## Why Raw Non-unroll Looks Faster Than Unroll

The raw elapsed columns above are not a fair cross-workload comparison.
`paged_attention` and `paged_attention_unroll` are intentionally very different
problem sizes.

### Case Shape Comparison

| Example | Case | Batch | Num Heads | Head Dim | Block Size | Context Len |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `paged_attention` | `Case1` | 1 | 16 | 16 | 16 | 33 |
| `paged_attention_unroll` | `Case1` | 256 | 16 | 128 | 128 | 8192 |
| `paged_attention` | `Case2` | 1 | 16 | 16 | 16 | 128 |
| `paged_attention_unroll` | `Case2` | 64 | 64 | 128 | 64 | 8192 |

So the unroll example is not a tuned version of the same small test. It is the
production-scale workload.

### Device-log Evidence

From the fresh benchmark logs on device `9`:

- `paged_attention Case1` submitted `13` tasks total
- `paged_attention_unroll Case1` submitted `1280` tasks total

These numbers match the orchestration structure:

- non-unroll `Case1`
  - `batch=1`
  - `bn_this_batch=ceil(33 / 16)=3`
  - `q_loop=1`
  - per batch: `1 alloc + 3 * 4 kernel tasks = 13`
- unroll `Case1`
  - `batch=256`
  - `bn_this_batch=ceil(8192 / 128)=64`
  - `N_UNROLL=64`, so one unrolled group per batch
  - per batch: `1 alloc + 4 grouped kernel tasks = 5`
  - total: `256 * 5 = 1280`

The important result is therefore not:

- "non-unroll is faster than unroll"

The important result is:

- unroll processes a vastly larger workload, yet its runtime does not scale
  remotely in proportion to raw work size
- unroll dramatically reduces orchestration cost per unit of work

One simple normalization is orchestration time per submitted task:

| Example | Case | Orch Trim (us) | Submitted Tasks | Orch per Task (us) |
| --- | --- | ---: | ---: | ---: |
| `paged_attention` | `Case1` | 64.6 | 13 | 4.97 |
| `paged_attention_unroll` | `Case1` | 777.4 | 1280 | 0.61 |

So unroll is not "worse" in the meaningful sense. Its absolute latency is
larger because the workload is massively larger, while its orchestration cost
per submitted task is much lower.

## Fresh TensorMap Profiling

This section is the required profiling checkpoint for the current branch.
Any optimization that touches the manual-scope runtime hot path or the manual
paged-attention orchestration should refresh both this section and the benchmark
table above with new real-device data.

### Method

- local profiling-only rebuild with:
  - `PTO2_ORCH_PROFILING=1`
  - `PTO2_TENSORMAP_PROFILING=1`
- platform: `a2a3`
- device: `9`
- PTO-ISA commit: `d96c8784`
- rounds: `30`
- mode: `--skip-golden`
- AUTO runner:
  - `python examples/a2a3/tensormap_and_ringbuffer/paged_attention/test_paged_attention.py -p a2a3 -d 9 -n 30 --case <Case> --skip-golden`
- manual runner:
  - `python examples/scripts/run_example.py -k examples/a2a3/tensormap_and_ringbuffer/paged_attention_manual_scope/kernels -g examples/a2a3/tensormap_and_ringbuffer/paged_attention_manual_scope/golden.py -p a2a3 -d 9 -c d96c8784 -n 30 --case <Case> --skip-golden`
- parsing:
  - per-round device-log `=== Orchestrator Profiling ===` blocks
  - per-round device-log `=== TensorMap Lookup Stats ===` blocks
  - trimmed average for time fields, mean for lookup / insert counts

### Results

| Case | Mode | Tasks | `lookup+dep` Trim (us) | `tensormap_ins` Trim (us) | TensorMap Lookups Avg | TensorMap Inserts Avg | Profiled Submit Trim (us) | Full Orch Trim (us) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Case1` | AUTO | 13 | 4.132 | 1.842 | 40.0 | 12.0 | 16.422 | 194.508 |
| `Case1` | MANUAL | 13 | 1.944 | 1.414 | 16.0 | 3.0 | 14.458 | 259.318 |
| `Case2` | AUTO | 33 | 6.320 | 2.638 | 105.0 | 32.0 | 24.368 | 210.274 |
| `Case2` | MANUAL | 33 | 2.598 | 1.728 | 41.0 | 8.0 | 21.560 | 285.182 |

### What The Numbers Prove

- The manual-local TensorMap bypass is working.
  - `Case1`: lookups dropped from `40.0` to `16.0` (`-60.0%`), inserts
    dropped from `12.0` to `3.0` (`-75.0%`), and `lookup+dep` time dropped
    from `4.132us` to `1.944us` (`-53.0%`).
  - `Case2`: lookups dropped from `105.0` to `41.0` (`-61.0%`), inserts
    dropped from `32.0` to `8.0` (`-75.0%`), and `lookup+dep` time dropped
    from `6.320us` to `2.598us` (`-58.9%`).
- The manual path still shows non-zero TensorMap traffic because boundary
  tensors still use TensorMap in v0. That is expected.
- The remaining non-unroll regression is no longer explained by TensorMap.
  - The profiled submit buckets are lower in manual mode
    (`-12.0%` in `Case1`, `-11.5%` in `Case2`).
  - But full orchestration time is still much higher
    (`+33.3%` in `Case1`, `+35.6%` in `Case2`).

### Why Full Orch Is Still Worse

The gap has moved out of the TensorMap buckets.

The current profiling points to two more likely hot regions:

1. Orchestration-side explicit-dep construction.
   The manual paged-attention orchestration adds many `Arg.add_dep(...)`
   calls and threads task ids explicitly through the loop body.
2. Runtime explicit-dep validation and dedupe before the first profiled phase.
   `pto2_submit_mixed_task()` validates every explicit dep against the current
   scope and deduplicates it before the first `alloc/sync/lookup/insert` lap is
   recorded.

So the next optimization target is no longer "remove more TensorMap work".
It is "make explicit-dep construction and validation cheaper".

### Code Pointers For The Current Design

- Current-manual-scope-local classification:
  - [pto_orchestrator.cpp](/data/uvxiao/pto-runtime/src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp#L143)
- Manual-local tensors bypass TensorMap lookup / insert here:
  - [pto_orchestrator.cpp](/data/uvxiao/pto-runtime/src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp#L742)
- Explicit deps are validated and deduplicated here:
  - [pto_orchestrator.cpp](/data/uvxiao/pto-runtime/src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp#L627)
- `Arg.add_dep(...)` storage is a simple append here:
  - [pto_types.h](/data/uvxiao/pto-runtime/src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_types.h#L228)
- The manual paged-attention example that exercises this path is here:
  - [paged_attention_orch.cpp](/data/uvxiao/pto-runtime/examples/a2a3/tensormap_and_ringbuffer/paged_attention_manual_scope/kernels/orchestration/paged_attention_orch.cpp#L131)

## Current Implementation Gap

The runtime and manual paged-attention examples now match the core v0 alignment
rules:

- `TaskOutputTensors` carries `task_id` directly, including for zero-output
  updater tasks
- manual-local tensors bypass TensorMap lookup and TensorMap insert in the
  submit path
- manual examples explicitly chain repeated zero-output updater tasks with
  `add_dep(prev_update_task)`

The remaining work is performance-focused:

- reduce non-unroll manual orchestration cost without reintroducing TensorMap
  fallback for manual-local tensors
- focus the next optimization round on explicit-dep construction / validation,
  not on TensorMap lookup / insert removal
- keep the unroll gains while tightening the small-workload path

## Historical Optimization Notes

The current PR intentionally does not carry the broader optimization branch.
Those attempts were removed from the code to keep the PR scope small, but the
history is still useful and should stay documented here as historical context.

These entries are not claims about the current branch state. They record what
was tried on the earlier heavier line of work and whether it was kept there,
reverted there, or later dropped from this PR scope.

### Historical Kept Attempts On The Earlier Branch

| Optimization | Files / Functions Touched | Historical Effect |
| --- | --- | --- |
| O(1) explicit-dep membership check | `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp` `pto2_get_current_scope_task_slot()` `pto2_submit_mixed_task()` `pto2_scope_begin()` `pto2_prepare_task()`; `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.h`; `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_ring_buffer.h` | Large win on the old branch because it removed a linear current-scope scan from each `add_dep(...)` validation. |
| Exact per-scope epoch validation | `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp` `pto2_get_current_scope_task_slot()` `pto2_scope_begin()` `pto2_prepare_task()`; `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.h`; `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_ring_buffer.h`; `tests/ut/cpp/test_a2a3_pto2_manual_scope_runtime.cpp` | Correctness fix that preserved the O(1) validation path while closing the stale-scope hole. |
| Skip redundant creator-retention when owner is already explicit | `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp` `pto2_submit_mixed_task()` `pto2_append_explicit_manual_deps()` | Reduced duplicate submit-path work when explicit deps already named the producer. |
| Canonicalize explicit dep ids once per submit | `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp` `pto2_submit_mixed_task()` | Small hotpath cleanup on the earlier branch. |
| Eager `start_offset` caching, remove hot-path recompute from payload init | `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/tensor.h` `Tensor::init()` `Tensor::init_with_view()` `Tensor::init_from_create_info()`; `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_runtime2_types.h` `PTO2TaskPayload::init()`; `tests/ut/cpp/test_a2a3_pto2_manual_scope_runtime.cpp` | Reduced the earlier branch's `param_copy` hotspot substantially, but this was later dropped from the PR to keep the diff tightly coupled to manual-scope v0 itself. |
| Cached allocator-state fast path before shared reload | `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_ring_buffer.h` `PTO2TaskAllocator::alloc()` `has_task_slot_capacity()` `try_alloc_with_last_alive()` | Reduced allocator overhead on the earlier branch, but this was also dropped from the PR because it is a general runtime optimization, not a manual-scope-v0-specific change. |

### Historical Reverted / Rejected Attempts

| Attempt | Files / Functions Touched | Historical Result |
| --- | --- | --- |
| Fold output owner metadata into `payload->init()` to save the post-init output loop | `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/tensor.h`; `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_runtime2_types.h`; `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp` | Reverted on the earlier branch. End-to-end gains were not reliable and the code became harder to reason about. |
| Change `out_view` in paged-attention update from `INOUT` to `NO_DEP` | earlier `paged_attention` orchestration files on the heavy branch | Reverted. Real-device golden failed. |
| Change `out_view` in paged-attention update from `INOUT` to `OUTPUT_EXISTING` | earlier `paged_attention` orchestration files on the heavy branch | Reverted. Real-device golden also failed. |

## Risks

1. Manual scope still relies on orchestration authors to supply producer-flow
   explicit deps correctly.
2. Wrong scope metadata would still cause incorrect locality classification.
3. The current non-unroll benchmark gap is large enough that performance should
   not be described as improved relative to AUTO.

## Recommendation

Keep v0 narrow and explicit:

- same submit APIs as AUTO
- `Arg.add_dep(task_id)` only inside manual scope
- no delayed wiring
- no delayed linking
- immediate publish
- alloc returns `task_id`
- submit results carry `task_id` directly, independent of outputs
- scope-depth metadata decides locality
- manual-local tensors bypass TensorMap entirely

This is the smallest v0 target design that keeps the API narrow. The current
branch is functionally correct on the current hardware batch, but it is not yet
fully aligned with the stricter "manual-local tensors bypass TensorMap
entirely" rule above. The next implementation work should close that gap first,
then focus on the non-unroll orchestration regression without widening the API
again.
