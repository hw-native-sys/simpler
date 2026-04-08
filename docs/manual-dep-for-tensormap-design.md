# Manual Dependency For TensorMap Runtime

## Goal

Add a scoped manual-dependency mode to `tensormap_and_ringbuffer` without
regressing the default automatic path:

- `PTO2_SCOPE()` stays in automatic mode
- `PTO2_SCOPE(PTO2ScopeMode::MANUAL)` enables scoped manual dependency wiring
- same-manual-scope edges use explicit `pto2_rt_add_dependency(...)`
- cross-scope edges still use `owner_task_id` and TensorMap discovery

This is a hybrid model, not a port of `aicpu_build_graph`.

## API Surface

The orchestration-facing API is:

```cpp
enum class PTO2ScopeMode : uint8_t {
    AUTO = 0,
    MANUAL = 1,
};

PTO2_SCOPE() {
    // default: AUTO
}

PTO2_SCOPE(PTO2ScopeMode::MANUAL) {
    auto qk = pto2_rt_submit_aic_task_manual(...);
    auto sf = pto2_rt_submit_aiv_task_manual(...);
    pto2_rt_add_dependency(qk.task_id, sf.task_id);
}
```

Current restrictions:

- manual submit APIs are only valid inside
  `PTO2_SCOPE(PTO2ScopeMode::MANUAL)`
- `pto2_rt_add_dependency(...)` requires both tasks to belong to the current
  manual scope
- nested scope inside manual scope is rejected in v1
- blocking tensor access helpers are rejected inside manual scope

## Dependency Semantics

### Tensor origin matters first

Each tensor argument is classified at submit time:

- `manual-local`: the tensor owner was created inside the current manual scope
- `boundary`: anything else, including external tensors and tensors produced by
  tasks outside the current manual scope

Manual-local tensors skip TensorMap entirely. Boundary tensors stay on the
normal TensorMap path unless `manual_dep=true`.

### `INPUT`, `OUTPUT`, `INOUT`, and friends

`TensorArgType` behavior in the runtime:

| Arg kind | Meaning | Incoming dependency work | Outgoing frontier work |
| --- | --- | --- | --- |
| `INPUT` | existing tensor, read-only | creator retention, plus TensorMap lookup unless skipped | none |
| `OUTPUT` | fresh runtime-allocated tensor | none | no TensorMap insert at creation; `owner_task_id` is stamped on the produced tensor |
| `INOUT` | existing tensor, read + write | creator retention, plus TensorMap lookup unless skipped | TensorMap insert unless skipped |
| `OUTPUT_EXISTING` | existing tensor, write-only | creator retention only | TensorMap insert unless skipped |
| `NO_DEP` | existing tensor, creator-retention-only | creator retention only | none |

### Manual-local vs boundary behavior

| Arg kind | Manual-local tensor | Boundary tensor |
| --- | --- | --- |
| `INPUT` | no TensorMap lookup, requires explicit manual edge | creator retention; TensorMap lookup unless `manual_dep=true` |
| `OUTPUT` | fresh local tensor; later same-scope uses rely on explicit manual edges | not applicable |
| `INOUT` | no TensorMap lookup/insert, requires explicit manual edge | creator retention; TensorMap lookup for incoming state; TensorMap insert for outgoing state unless `manual_dep=true` |
| `OUTPUT_EXISTING` | no TensorMap insert, requires explicit manual edge if later reused in scope | creator retention; TensorMap insert for outgoing state unless `manual_dep=true` |
| `NO_DEP` | creator-only object passing, no publish | same |

### `manual_dep=true`

`Tensor::manual_dep` keeps its existing meaning:

- skip TensorMap lookup/insert
- keep creator-only retention via `owner_task_id`

It is a per-tensor optimization hint. It is not the core manual-scope
mechanism.

## Runtime Model

### High-level flow

```text
PTO2_SCOPE(MANUAL)
        |
        v
  submit_*_manual()
        |
        +-- classify tensor args
        |     |- manual-local -> no TensorMap
        |     `- boundary     -> owner retention + optional TensorMap
        |
        +-- allocate slot / payload / outputs
        |
        +-- wire boundary producers immediately
        |     `- keep one extra fanin publish barrier
        |
        `-- return { task_id, outputs }
                  |
                  v
      pto2_rt_add_dependency()
                  |
                  `-- wire same-scope producer -> consumer immediately

scope_end()
        |
        +-- validate fanin bounds
        +-- repair monotonic dep_pool_mark prefix
        +-- release publish barrier and batch-publish tasks
        `-- do normal scope lifetime release
```

### What manual submit iterates

Current implementation is in
`src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp`.

For a manual submit:

1. allocate the task slot, payload, and task id immediately
2. classify each tensor arg as manual-local or boundary
3. build `manual_local_mask` for same-scope tensors
4. decide whether TensorMap sync is needed at all
   - if every relevant arg is manual-local or `manual_dep=true`, skip sync
   - otherwise run the normal TensorMap sync
5. for each non-`OUTPUT` arg that is not manual-local
   - always do creator retention from `owner_task_id`
   - for `INPUT` and `INOUT`, do TensorMap lookup unless `manual_dep=true`
6. for `INOUT` and `OUTPUT_EXISTING` boundary args
   - update TensorMap frontier unless `manual_dep=true`
7. initialize scheduler state, but keep the task unpublished behind a deferred
   publish barrier

Important consequence:

- cross-scope dependency discovery is still paid at submit time
- same-scope dependency discovery is no longer replayed from tensors later

### What `pto2_rt_add_dependency(...)` does now

This is the key difference from the older design draft.

`pto2_rt_add_dependency(...)` no longer records an edge for replay at
`scope_end()`. It validates both task ids belong to the current manual scope,
dedups against the consumer payload, ensures dep-pool space, and wires the edge
immediately:

- increments producer `fanout_count`
- prepends the consumer into the producer fanout list
- appends the producer slot state into `payload->fanin_slot_states[]`
- increments consumer `fanin_count`
- updates consumer `dep_pool_mark`

That removes the old replay-heavy finalize path.

### What `scope_end()` does now

Manual `scope_end()` is now intentionally small and TensorMap-free.

It only:

1. validates `fanin_actual_count`
2. repairs a monotonic `dep_pool_mark` prefix
3. calls `publish_manual_scope_tasks_and_end_scope(...)`
4. performs the normal scope lifetime release

There is no explicit-edge replay at `scope_end()` anymore.

## Why This Split Is Correct

### Cross-scope correctness

Cross-scope tensors still need TensorMap because the runtime must preserve:

- latest-writer frontier tracking
- overlap-based modifier discovery
- boundary ordering across scopes

If manual scope disabled TensorMap globally, outer reads and writes would
become incorrect.

### Same-scope performance

Manual-local tensors are exactly where TensorMap is unnecessary work:

- the producer is already known from the current manual scope
- the ordering can be expressed directly by `pto2_rt_add_dependency(...)`
- replaying those edges at `scope_end()` added serial overhead without adding
  correctness

### Zero-overhead AUTO path

The manual-scope extension must not slow down the normal AUTO runtime.

Fresh measurements below show the current AUTO runtime stays within roughly
`±1%` end-to-end of the unmodified baseline on the two paged-attention scenes,
which is the intended zero-overhead result.

## Example Requirements

Manual mode only helps when the example exposes a real same-scope
producer/consumer chain that TensorMap would otherwise rediscover.

For paged attention, the profitable chain is:

```text
qk_matmul -> softmax_prepare -> pv_matmul -> online_update
```

Inside a manual scope:

- intermediate tensors in that chain should stay manual-local
- explicit edges should connect those tasks directly
- outer tensors such as the external KV cache and the final output still keep
  boundary semantics

If an example keeps using boundary tensors everywhere, manual mode cannot
remove much runtime work.

## Benchmark Enablement

Current branch benchmark entrypoints:

```bash
./tools/benchmark_rounds.sh -d 6 -n 5 -c 6622890 -r aicpu_build_graph --build
./tools/benchmark_rounds.sh -d 6 -n 5 -c 6622890 -r tensormap_and_ringbuffer --build
./tools/benchmark_rounds.sh -d 6 -n 5 -c 6622890 -r tensormap_and_ringbuffer_partial_manual --build
```

`tensormap_and_ringbuffer_partial_manual` is a selector in
`tools/benchmark_rounds.sh`. The example `kernel_config.py` files still use
`RUNTIME_CONFIG["runtime"] = "tensormap_and_ringbuffer"`. The selector only
switches the scene directories to:

- `tests/st/a2a3/tensormap_and_ringbuffer/paged_attention_partial_manual`
- `tests/st/a2a3/tensormap_and_ringbuffer/paged_attention_unroll_partial_manual`

The old unmodified runtime is intentionally not kept on this branch. To rerun
it side-by-side:

```bash
git worktree add tmp/worktree_unmodified a71ba16
(
  cd tmp/worktree_unmodified
  ./tools/benchmark_rounds.sh -d 6 -n 5 -c 6622890 \
    -r tensormap_and_ringbuffer_unmodified --build
)
```

For this document, a serial `run_example.py` pass was used instead of the
wrapper so every run used one uncontended process on one device.

Fresh result CSV:

- `tmp/bench_matrix_20260409_0006_direct/results.csv`

## Fresh Hardware Results

Fresh rerun settings:

- date: `2026-04-09`
- platform: `a2a3`
- device: `6`
- rounds: `5`
- PTO-ISA commit: `6622890`

Units below are `elapsed_us (orch_us)`. `aicpu_build_graph` does not emit the
same orch timing lines, so only elapsed time is shown there.

### `paged_attention`

| Case | `aicpu_build_graph` | `tensormap_and_ringbuffer_unmodified` | `tensormap_and_ringbuffer` | `tensormap_and_ringbuffer_partial_manual` |
| --- | ---: | ---: | ---: | ---: |
| `Case1` | `31037.8` | `36992.8 (36991.9)` | `36791.2 (36790.5)` | `31563.9 (31407.2)` |
| `Case2` | `16719.2` | `18753.6 (18752.8)` | `18615.9 (18615.1)` | `16757.6 (16343.9)` |

### `paged_attention_unroll`

| Case | `aicpu_build_graph` | `tensormap_and_ringbuffer_unmodified` | `tensormap_and_ringbuffer` | `tensormap_and_ringbuffer_partial_manual` |
| --- | ---: | ---: | ---: | ---: |
| `Case1` | `1421.2` | `1320.0 (853.6)` | `1322.5 (820.0)` | `1327.0 (835.5)` |
| `Case2` | `707.8` | `632.5 (383.5)` | `635.9 (391.8)` | `633.7 (365.5)` |

## Feature / Optimization -> Gain

### 1. AUTO stays effectively zero-overhead

The current AUTO runtime is flat versus the unmodified baseline:

- `paged_attention/Case1`: `36791.2 us` vs `36992.8 us` (`-0.5%`)
- `paged_attention/Case2`: `18615.9 us` vs `18753.6 us` (`-0.7%`)
- `paged_attention_unroll/Case1`: `1322.5 us` vs `1320.0 us` (`+0.2%`)
- `paged_attention_unroll/Case2`: `635.9 us` vs `632.5 us` (`+0.5%`)

This is the zero-overhead result we needed on the normal tensormap path.

### 2. Partial-manual removes the non-unroll gap

Against the current AUTO runtime, partial-manual improves the non-unroll scene
substantially:

- `paged_attention/Case1`
  - elapsed: `36791.2 us -> 31563.9 us` (`-14.2%`)
  - orch: `36790.5 us -> 31407.2 us` (`-14.6%`)
- `paged_attention/Case2`
  - elapsed: `18615.9 us -> 16757.6 us` (`-10.0%`)
  - orch: `18615.1 us -> 16343.9 us` (`-12.2%`)

Against `aicpu_build_graph`, the remaining end-to-end gap on non-unroll is now
small:

- `Case1`: `31563.9 us` vs `31037.8 us` (`+1.7%`)
- `Case2`: `16757.6 us` vs `16719.2 us` (`+0.2%`)

This is the target workload. Partial-manual is now effectively in the same
performance band as `aicpu_build_graph` there.

### 3. Unroll already amortizes most of the cost

On `paged_attention_unroll`, the AUTO tensormap path was already strong, so
partial-manual brings little extra value:

- `Case1`: `1322.5 us -> 1327.0 us` elapsed (`+0.3%`)
- `Case2`: `635.9 us -> 633.7 us` elapsed (`-0.3%`)

That is expected. The unroll example already amortizes dependency-construction
overhead, so partial-manual mainly matters for the non-unroll shape.

### 4. What specifically helped

The important runtime-side wins were:

- classify manual-local tensors from `owner_task_id`
- skip TensorMap work for those manual-local tensors
- wire explicit same-scope edges immediately in `pto2_rt_add_dependency(...)`
- keep `scope_end()` down to publish-barrier release plus `dep_pool_mark`
  fixup

The important example-side win was using manual scope only where the
non-unroll paged-attention orchestration still had repeated same-scope
dependency work to remove.

## Current Risks

1. `manual_dep=true` can still be abused.
   - It suppresses TensorMap lookup/insert for that tensor.
   - It is only safe when ordering/frontier requirements are already covered by
     other logic.

2. Nested scope inside manual scope is still unsupported.
   - This is a current implementation restriction, not a theoretical property.

3. `pto2_rt_add_dependency(...)` now spends dep-pool entries on the submit path.
   - That is intentional, but it means dep-pool pressure moved from the old
     replay path into explicit-edge wiring.

4. Manual publish still relies on `dep_pool_mark` prefix repair at `scope_end()`.
   - This is required because explicit edges can touch older consumers after
     newer tasks were already submitted.

## Recommendation Summary

Keep the design as:

- AUTO mode by default
- explicit MANUAL mode through `PTO2ScopeMode`
- TensorMap kept only for cross-scope correctness
- explicit immediate wiring for same-scope manual edges
- `scope_end()` reduced to publish-barrier release and normal lifetime work

That gives the required feature coverage while keeping the AUTO path
effectively zero-overhead and bringing non-unroll partial-manual paged
attention to within `~0-2%` of `aicpu_build_graph`.
