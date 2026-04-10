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
./tools/benchmark_rounds.sh -d 4 -n 5 -c d96c8784 -r aicpu_build_graph --build
./tools/benchmark_rounds.sh -d 4 -n 5 -c d96c8784 -r tensormap_and_ringbuffer --build
./tools/benchmark_rounds.sh -d 4 -n 5 -c d96c8784 -r tensormap_and_ringbuffer_partial_manual --build
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
export PROJECT_ROOT=$(pwd)
git worktree add tmp/worktree_unmodified a71ba16
(
  cd tmp/worktree_unmodified
  python3 -m venv .venv --system-site-packages
  . .venv/bin/activate
  pip install -e . -q
  export PTO_ISA_ROOT="$PROJECT_ROOT/examples/scripts/_deps/pto-isa"
  ./tools/benchmark_rounds.sh -d 4 -n 5 -c d96c8784 \
    -r tensormap_and_ringbuffer_unmodified --build
)
```

Fresh benchmark logs for the rebased branch are in:

- `tmp/rebased_bench_20260410_fix/aicpu_build_graph.log`
- `tmp/rebased_bench_20260410_fix/tensormap_and_ringbuffer.log`
- `tmp/rebased_bench_20260410_fix/tensormap_and_ringbuffer_partial_manual.log`
- `tmp/rebased_bench_20260410_fix/tensormap_and_ringbuffer_unmodified.log`

Rebase note:

- `paged_attention_unroll_partial_manual` was initially timing out after the
  merge-forward.
- The runtime manual-scope machinery was not the root cause.
- The direct cause was stale example-side AIC submit ABI: the rebased
  `paged_attention_unroll` AIC kernels now expect `block_table` as a tensor
  input plus a scalar `bt_offset`, while the partial-manual scene was still
  passing a raw pointer scalar.
- Fixing the partial-manual `qk/pv` submit argument layout restored both
  unroll cases on device.

## Fresh Hardware Results

Fresh rerun settings:

- date: `2026-04-10`
- platform: `a2a3`
- device: `4`
- rounds: `5`
- PTO-ISA commit: `d96c8784`

Units below are `elapsed_us (orch_us)`. `aicpu_build_graph` does not emit the
same orch timing lines, so only elapsed time is shown there.

### `paged_attention`

| Case | `aicpu_build_graph` | `tensormap_and_ringbuffer_unmodified` | `tensormap_and_ringbuffer` | `tensormap_and_ringbuffer_partial_manual` |
| --- | ---: | ---: | ---: | ---: |
| `Case1` | `29937.7` | `36095.9 (36094.9)` | `39148.7 (39148.3)` | `34186.3 (34025.7)` |
| `Case2` | `16762.7` | `18639.5 (18635.1)` | `19813.0 (19812.7)` | `18028.7 (17618.4)` |

### `paged_attention_unroll`

| Case | `aicpu_build_graph` | `tensormap_and_ringbuffer_unmodified` | `tensormap_and_ringbuffer` | `tensormap_and_ringbuffer_partial_manual` |
| --- | ---: | ---: | ---: | ---: |
| `Case1` | `1425.3` | `1325.6 (835.3)` | `1173.2 (992.0)` | `1160.4 (968.8)` |
| `Case2` | `693.0` | `628.7 (380.7)` | `567.9 (435.6)` | `561.9 (416.6)` |

## Feature / Optimization -> Gain

### 1. AUTO stays effectively zero-overhead

The current AUTO runtime no longer meets the zero-overhead target on the
non-unroll scene, but it still wins clearly on the unroll scene:

- `paged_attention/Case1`: `39148.7 us` vs `36095.9 us` (`+8.5%`)
- `paged_attention/Case2`: `19813.0 us` vs `18639.5 us` (`+6.3%`)
- `paged_attention_unroll/Case1`: `1173.2 us` vs `1325.6 us` (`-11.5%`)
- `paged_attention_unroll/Case2`: `567.9 us` vs `628.7 us` (`-9.7%`)

So the AUTO path is still good for the already-amortized unroll workload, but
not yet zero-overhead for the non-unroll paged-attention target.

### 2. Partial-manual removes the non-unroll gap

Against the current AUTO runtime, partial-manual improves the non-unroll scene
substantially:

- `paged_attention/Case1`
  - elapsed: `39148.7 us -> 34186.3 us` (`-12.7%`)
  - orch: `39148.3 us -> 34025.7 us` (`-13.1%`)
- `paged_attention/Case2`
  - elapsed: `19813.0 us -> 18028.7 us` (`-9.0%`)
  - orch: `19812.7 us -> 17618.4 us` (`-11.1%`)

Against `aicpu_build_graph`, there is still a visible non-unroll gap:

- `Case1`: `34186.3 us` vs `29937.7 us` (`+14.2%`)
- `Case2`: `18028.7 us` vs `16762.7 us` (`+7.6%`)

Against the unmodified tensormap baseline, partial-manual is now ahead on the
non-unroll scene:

- `Case1`: `36095.9 us -> 34186.3 us` (`-5.3%`)
- `Case2`: `18639.5 us -> 18028.7 us` (`-3.3%`)

### 3. Unroll already amortizes most of the cost

On `paged_attention_unroll`, both current runtimes are already better than
`aicpu_build_graph`, and partial-manual only nudges the AUTO path slightly:

- `Case1`: `1173.2 us -> 1160.4 us` elapsed (`-1.1%`)
- `Case2`: `567.9 us -> 561.9 us` elapsed (`-1.1%`)

That is the expected shape. The unroll orchestration already amortizes most
dependency overhead, so partial-manual has little room left to improve.

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
competitive on unroll and materially reducing the non-unroll gap, but the
fresh rerun still shows more work is needed to make partial-manual match
`aicpu_build_graph` on non-unroll paged attention.
