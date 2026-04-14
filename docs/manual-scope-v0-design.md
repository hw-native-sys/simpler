# Manual Scope V0 Design

Date: 2026-04-14

Branch: `manual_scope_v0`

Base: `upstream/main` at `617add6`

## Goal

Add a lighter manual-scope mode to `a2a3/tensormap_and_ringbuffer` that:

- keeps the same submit API shape as AUTO mode
- does not introduce a separate manual submit API family
- does not support delayed dependency wiring
- publishes tasks at submit time, like AUTO mode
- allows explicit same-scope task ordering without relying entirely on
  TensorMap rediscovery

This is intentionally smaller than the previous manual-scope branch.

## Constraints

The v0 branch must follow these rules:

1. Use the same submit API as AUTO mode.
2. Append explicit dependencies into `Arg`, for example `args.add_dep(task_id)`.
3. Do not support delayed wiring and delayed linking.
4. Publish at submit time, same as AUTO mode.
5. Treat tensor allocation as a task for manual dependency building.
6. Determine whether TensorMap lookup is required from tensor scope metadata
   first, not from ring id.

## Non-goals

- no nested manual scopes in v0
- no post-submit `add_dependency(...)`
- no delayed explicit-edge replay or scope-end linking
- no batch publish barrier at manual `scope_end()`
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
// alloc.task_id
// alloc.outputs
```

This allows later manual tasks to depend on the allocation task explicitly.

`alloc_tensors(...)` itself does not accept `Arg.add_dep(...)`.

## Runtime Model

### High-level behavior

Manual mode in v0 is:

- AUTO-style submit and publish
- plus explicit deps from `Arg`
- plus reduced TensorMap work for current-manual-scope-local tensors

There is no hidden manual subgraph and no delayed publish.

### Submit-time flow

Inside manual scope, submit should do this:

1. Allocate the task slot and task id.
2. Read explicit deps from `Arg`.
3. Validate that explicit deps belong to the current manual scope.
4. Turn explicit deps into ordinary fanins immediately.
5. Classify tensor args as current-manual-scope-local or boundary.
6. Skip TensorMap lookup/insert only for current-manual-scope-local cases
   where explicit ordering is already provided by task ids.
7. Keep normal creator-retention and TensorMap behavior for boundary tensors.
8. Publish the task immediately, same as AUTO mode.

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

## Tensor Lookup Rule

When a tensor is used inside manual scope:

### Manual-local tensor

Treat a tensor as manual-local only when:

- it was produced in the current manual scope
- its stored producer manual-scope depth matches the current manual-scope depth

Behavior:

- explicit task ids are the ordering source
- skip TensorMap lookup for same-scope ordering
- skip TensorMap insert for same-scope local update cases where the dependency
  stays entirely inside the current manual scope

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

## Validation Plan

### Unit / behavior

- reject `Arg.add_dep(...)` outside manual scope
- reject invalid task ids in manual scope
- reject explicit deps that are not from the current manual scope
- reject nested manual scopes
- verify manual-local tensors skip TensorMap lookup when explicit deps are
  present
- verify boundary tensors still use TensorMap

### Allocation behavior

- verify `alloc_tensors(...)` returns `{task_id, outputs}`
- verify manual tasks can depend on alloc task ids

### Regression / examples

- paged attention partial-manual example rewritten to `Arg.add_dep(...)`
- compare against AUTO and `aicpu_build_graph`
- verify correctness against golden outputs

## Validation Status

The supported v0 scope is currently functionally correct for the cases we
intend to support now.

Verified on `manual_scope_v0`:

- C++ UT passed:
  - `test_a2a3_pto2_manual_scope_api`
  - `test_a2a3_pto2_manual_scope_runtime`
  - `test_a2a3_pto2_fatal`
- Simulation ST passed:
  - `tests/st/a2a3/tensormap_and_ringbuffer/test_manual_scope_validation.py`
- Real-device golden checks passed:
  - `tests/st/a2a3/tensormap_and_ringbuffer/paged_attention` `Case1`
  - `tests/st/a2a3/tensormap_and_ringbuffer/paged_attention` `Case2`
  - `examples/a2a3/tensormap_and_ringbuffer/paged_attention` `Case1`
  - `examples/a2a3/tensormap_and_ringbuffer/paged_attention` `CaseVarSeq2`
- Unsupported shapes now fail fast with runtime invalid-args instead of
  silently reaching a bad golden comparison:
  - `tests/st/a2a3/tensormap_and_ringbuffer/test_paged_attention_validation.py`

This status does not mean the branch is performance-ready.

## Benchmark Status

### Authoritative Current vs Baselines

This is the current branch-status table. Only same-method, same-device,
same-round-count reruns go here.

### Method For The Authoritative Table

- Platform: `a2a3` real device
- Device: `10`
- Workload: supported production `paged_attention` cases only
  - `Case1`: `batch=256, num_heads=16, head_dim=128, block_size=128`
  - `Case2`: `batch=64, num_heads=64, head_dim=128, block_size=64`
- Rounds: `10`
- Aggregation: simple average from `tmp/measure_case.py`
- Current TMR measured commit: `371349f`
- Baseline TMR measured commit: merge-base `617add6`
- ABG measured source: current branch `examples/a2a3/aicpu_build_graph/paged_attention`
- All three runs used the same local `PTO_ISA_ROOT` checkout so the runtime
  delta is measured against the same kernel dependency tree

| Case | Runtime | Commit / Source | Elapsed Avg (us) | Orch Avg (us) | Elapsed Delta vs Base TMR | Elapsed Delta vs ABG |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Case1 | TMR merge-base | `617add6` | 32579.9 | 32442.7 | baseline | +162.8 (+0.5%) |
| Case1 | TMR optimized current | `371349f` | 32288.2 | 32275.2 | -291.7 (-0.9%) | -128.9 (-0.4%) |
| Case1 | ABG current | current branch | 32417.1 | 32127.0 (`orch_func_cost`) | -162.8 (-0.5%) | baseline |
| Case2 | TMR merge-base | `617add6` | 18749.7 | 18305.1 | baseline | +1920.6 (+11.4%) |
| Case2 | TMR optimized current | `371349f` | 16548.1 | 16482.1 | -2201.6 (-11.7%) | -281.0 (-1.7%) |
| Case2 | ABG current | current branch | 16829.1 | 16246.0 (`orch_func_cost`) | -1920.6 (-10.2%) | baseline |

Notes:

- `371349f` is the first measured state where current TMR beats both merge-base
  TMR and ABG on elapsed time for the two supported non-unroll paged-attention
  cases.
- `Case1` is still nearly tied. The elapsed win is real in this batch, but it
  is small enough that more tuning is still needed before calling it robust.
- ABG's `orch` column comes from `orch_func_cost`, so it is informative but not
  identical to the TMR orchestration-span metric.

### Historical Pre-tuning Snapshot

These numbers explain where tuning started. They are not the current
branch-status table.

- Platform: `a2a3` real device
- Device: `5`
- Rounds: `30`
- Aggregation: trimmed average, dropping `10` low and `10` high rounds
- Current TMR measured commit: `a682ccc`

| Case | Runtime | Elapsed Trimmed Avg (us) | Orch Trimmed Avg (us) | Delta vs Base | Delta vs ABG |
| --- | --- | ---: | ---: | ---: | ---: |
| Case1 | TMR merge-base | 30206.8 | 30161.7 | baseline | -2823.2 |
| Case1 | TMR current | 49628.2 | 49627.9 | +19421.4 (+64.3%) | +16598.2 (+50.3%) |
| Case1 | ABG current | 33030.0 | 32723.6 (`orch_func_cost`) | +2823.2 (+9.3%) vs base TMR | baseline |
| Case2 | TMR merge-base | 15576.6 | 15339.3 | baseline | -2009.4 |
| Case2 | TMR current | 33248.6 | 33248.1 | +17672.0 (+113.5%) | +15662.6 (+89.1%) |
| Case2 | ABG current | 17586.0 | 17018.7 (`orch_func_cost`) | +2009.4 (+12.9%) vs base TMR | baseline |

### Reading The Tables

- The current tuning direction has recovered the branch from clearly slower
  than both baselines to slightly faster on elapsed time for the two supported
  cases.
- The remaining gap is not "manual scope is broken". It is now about making the
  small `Case1` win more stable and reducing the still-visible orchestration
  cost.
- The heavy regression was in orchestration, and the recovery also came from
  orchestration-path changes rather than worker-side math changes.

## Optimization Effects

The entries below record the follow-up optimizations that were applied after
`a682ccc`. These are iterative tuning measurements, mainly with:

- platform: `a2a3`
- device: `10`
- workload: `tests/st/a2a3/tensormap_and_ringbuffer/paged_attention`
- rounds: `10`
- metric: simple average from `tmp/measure_case.py`

These numbers are still useful for attribution, but they are noisier than the
30-round trimmed-average table above.

### Kept Optimizations

| Optimization | Files / Functions Touched | Why It Helps | Measured Effect |
| --- | --- | --- | --- |
| Cached allocator-state fast path before shared reload | `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_ring_buffer.h` `PTO2TaskAllocator::alloc()` `has_task_slot_capacity()` `try_alloc_with_last_alive()` | The allocator used to pay an acquire-load of `last_task_alive` and a heap-tail refresh before every submit, even when the cached window and heap state were already sufficient. The new path first tries the monotonic cached state, then falls back to the shared refresh only when the cached view says "not enough room". | Measured commit `371349f`, compared against the previous best-known spot snapshot at `eeaa456`: `Case1 34105.5 -> 32288.2us`, `Case2 18503.3 -> 16548.1us`. The same-batch current-vs-baseline table above shows this was enough to move current TMR slightly ahead of both baselines on elapsed time. |
| O(1) explicit-dep membership check | `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp` `pto2_get_current_scope_task_slot()` `pto2_submit_mixed_task()` `pto2_scope_begin()` `pto2_prepare_task()`; `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.h`; `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_ring_buffer.h` | Removed the per-dependency linear scan over current-scope tasks. Manual paged-attention emits many `add_dep(...)` edges inside the inner loop, so the old validation shape was wrong for this workload. | First spot run after replacing the scan: `Case1 49628.2 -> 36016.3us`, `Case2 33248.6 -> 17906.5us`. |
| Exact per-scope epoch validation | `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp` `pto2_get_current_scope_task_slot()` `pto2_scope_begin()` `pto2_prepare_task()`; `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.h`; `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_ring_buffer.h`; `tests/ut/cpp/test_a2a3_pto2_manual_scope_runtime.cpp` | The first O(1) version used only `scope_depth`, which was fast but could accept a task from an older closed scope reopened at the same depth. The epoch fix keeps the O(1) path while restoring exact scope isolation. | Correctness fix. No meaningful speed target by itself; it preserves the O(1) win and closes the stale-scope hole. |
| Skip redundant creator-retention when owner is already explicit | `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp` `pto2_submit_mixed_task()` `pto2_append_explicit_manual_deps()` | For manual-local tensors, if the producer task is already present in `Arg.add_dep(...)`, the submit path does not need to look up the producer slot again just to add the same fanin and let `fanin_builder` deduplicate it. | Spot run during tuning: `Case1 36016.3 -> 33170.4us`; `Case2` was roughly flat/noisy (`17906.5 -> 18304.6us`). |
| Canonicalize explicit dep ids once per submit | `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp` `pto2_submit_mixed_task()` | Stores validated dep ids / slot pointers once, then reuses them in later branches instead of repeatedly consulting `Arg`. This is a small submit-path cleanup, not a large standalone win. | Small cleanup contribution. No isolated large gain observed beyond the creator-retention optimization above. |
| Eager `start_offset` caching, remove hot-path recompute from payload init | `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/tensor.h` `Tensor::init()` `Tensor::init_with_view()` `Tensor::init_from_create_info()`; `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_runtime2_types.h` `PTO2TaskPayload::init()`; `tests/ut/cpp/test_a2a3_pto2_manual_scope_runtime.cpp` | `payload->init()` used to recompute `start_offset` for every tensor arg on every submit. For paged-attention this fell into the large `param_copy` bucket. Moving the work to tensor creation / view construction takes it out of the submit hot path. | AICPU phase profiling on `Case1` showed `param_copy` drop from about `9879us` to `4629us`. Spot wall-time after this change was about `Case1 34105.5us`, `Case2 18503.3us`. |

### Reverted / Rejected Optimizations

| Attempt | Files / Functions Touched | Result |
| --- | --- | --- |
| Fold output owner metadata into `payload->init()` to save the post-init output loop | `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/tensor.h`; `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_runtime2_types.h`; `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp` | Reverted. It did not improve end-to-end time reliably and made the code harder to reason about. |
| Change `out_view` in paged-attention update from `INOUT` to `NO_DEP` | `tests/st/a2a3/tensormap_and_ringbuffer/paged_attention/kernels/orchestration/paged_attention_orch.cpp`; `examples/a2a3/tensormap_and_ringbuffer/paged_attention/kernels/orchestration/paged_attention_orch.cpp` | Reverted. Real-device golden failed badly, so `NO_DEP` was too strong for this tensor. |
| Change `out_view` in paged-attention update from `INOUT` to `OUTPUT_EXISTING` | `tests/st/a2a3/tensormap_and_ringbuffer/paged_attention/kernels/orchestration/paged_attention_orch.cpp`; `examples/a2a3/tensormap_and_ringbuffer/paged_attention/kernels/orchestration/paged_attention_orch.cpp` | Reverted. Real-device golden also failed, so the current runtime still needs `INOUT` semantics there. |

### Current Hotspots

After `371349f`, the branch has crossed the elapsed-time target in the current
spot batch, but the `Case1` margin is still thin. The remaining large buckets
from the latest phase breakdown before the allocator fast path were:

- `task+heap_alloc`
- `lookup+dep`
- `fanin+ready`

`param_copy` was reduced substantially by eager `start_offset` caching and is
no longer the largest problem.

The next optimization pass should keep focusing on:

1. More submit-path cost reduction in allocator / fanin materialization.
2. Scheduler wiring fast paths for small `fanin` counts.
3. Avoiding work that only protects AUTO-style lookup paths when the manual
   scope already supplied explicit ordering.

## Mandatory Update Checklist

After every optimization try, update this document before moving on:

1. Add one entry under either `Kept Optimizations` or `Reverted / Rejected`.
2. Record the attempt name and outcome: kept, reverted, or rejected.
3. Record the measured commit and the exact compared-against commit.
4. Record the touched files and functions only, not a vague subsystem label.
5. Record the hypothesis: which cost bucket or branch the change was meant to
   reduce.
6. Record the exact workload, platform, device, runner, round count, and
   aggregation method.
7. Record whether the same `PTO_ISA_ROOT` and runtime environment were used.
8. Record before/after elapsed numbers per measured case.
9. Record before/after orch numbers per measured case when available.
10. Record the correctness checks that were run, or what failed.
11. If the branch best-known state changed, refresh the `Authoritative Current
    vs Baselines` table immediately.
12. If a number is only a noisy spot result, keep it in the optimization log
    and do not promote it into the authoritative table.
13. If the measurement method changes, update the method section in the same
    edit so later readers do not compare mismatched numbers.

## Risks

1. If explicit deps are missing, current-manual-scope-local tensors may be
   under-constrained.
2. Treating a tensor as local from wrong metadata would cause missing TensorMap
   ordering.
3. Allocation tasks need clear ownership metadata so downstream explicit deps
   behave like normal produced tensors.
4. The lighter API is less expressive than the previous delayed-wiring design.

## Recommendation

Implement v0 as the minimal, explicit, submit-time-only manual scope:

- same submit APIs as AUTO
- `Arg.add_dep(task_id)` only inside manual scope
- no delayed wiring
- no delayed linking
- immediate publish
- alloc returns `task_id`
- scope-metadata-based tensor locality check

This keeps the PR small, aligns with maintainer feedback, and preserves the
useful part of manual scope for the current examples.

The implementation direction still looks right, but the current measured state
shows we need more runtime/orchestration optimization before this branch is
ready to present as a performance improvement.
