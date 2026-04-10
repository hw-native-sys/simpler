# Manual Dependency For TensorMap Runtime

## Goal

Add a scoped manual-dependency mode to `tensormap_and_ringbuffer` without
changing the default API shape:

- `PTO2_SCOPE()` stays `AUTO` by default
- `PTO2_SCOPE(PTO2ScopeMode::MANUAL)` enables scoped manual dependency wiring
- cross-scope correctness still relies on `owner_task_id` and TensorMap
- same-manual-scope ordering is expressed explicitly, not rediscovered from
  tensors

This is a hybrid runtime model. It is not a port of `aicpu_build_graph`.

## API Surface

### Scope mode

```cpp
enum class PTO2ScopeMode : uint8_t {
    AUTO = 0,
    MANUAL = 1,
};

PTO2_SCOPE() {
    // default: AUTO
}

PTO2_SCOPE(PTO2ScopeMode::MANUAL) {
    // manual mode
}
```

### Manual submit APIs

Manual submit is valid only inside `PTO2_SCOPE(PTO2ScopeMode::MANUAL)`.

```cpp
auto qk = pto2_rt_submit_aic_task_manual(FUNC_QK_MATMUL, params_qk);
auto sf = pto2_rt_submit_aiv_task_manual_with_deps(
    FUNC_SOFTMAX_PREPARE, params_sf, {qk.task_id}
);
auto pv = pto2_rt_submit_aic_task_manual_with_deps(
    FUNC_PV_MATMUL, params_pv, {sf.task_id}
);
auto up = pto2_rt_submit_aiv_task_manual_with_deps(
    FUNC_ONLINE_UPDATE, params_up, {sf.task_id, pv.task_id}
);
```

`pto2_rt_add_dependency(producer, consumer)` is still supported in manual
scope. The new `*_manual_with_deps(...)` helpers exist to fold explicit edges
into submit and remove hot orchestration-side `add_dependency(...)` overhead.

### Current restrictions

- manual submit APIs are only valid inside
  `PTO2_SCOPE(PTO2ScopeMode::MANUAL)`
- `pto2_rt_add_dependency(...)` requires both task ids to belong to the current
  manual scope
- nested manual scope is rejected in v1
- blocking tensor access helpers are rejected inside manual scope

## Dependency Semantics

### First split: manual-local vs boundary

At manual submit time, each tensor argument is classified as:

- `manual-local`
  - the tensor owner was created inside the current manual scope
- `boundary`
  - everything else: external tensors, tensors from outer scopes, or tensors
    produced by already-published tasks

Manual-local tensors skip TensorMap. Boundary tensors keep the normal
cross-scope correctness path unless `manual_dep=true` suppresses TensorMap
lookup/insert for that tensor.

### `INPUT`, `OUTPUT`, `INOUT`, `OUTPUT_EXISTING`, `NO_DEP`

| Arg kind | Meaning | Incoming work | Outgoing work |
| --- | --- | --- | --- |
| `INPUT` | existing tensor, read-only | creator retention, plus TensorMap lookup unless skipped | none |
| `OUTPUT` | fresh runtime-allocated tensor | none | no TensorMap insert at creation; the produced tensor gets `owner_task_id` |
| `INOUT` | existing tensor, read + write | creator retention, plus TensorMap lookup unless skipped | TensorMap insert unless skipped |
| `OUTPUT_EXISTING` | existing tensor, write-only | creator retention only | TensorMap insert unless skipped |
| `NO_DEP` | creator-retention-only handle passing | creator retention only | none |

### Manual-local vs boundary behavior

| Arg kind | Manual-local tensor | Boundary tensor |
| --- | --- | --- |
| `INPUT` | no TensorMap lookup; ordering must come from explicit manual edges | creator retention; TensorMap lookup unless `manual_dep=true` |
| `OUTPUT` | fresh local tensor; later same-scope consumers rely on explicit manual edges | not applicable |
| `INOUT` | no TensorMap lookup/insert; ordering must come from explicit manual edges | creator retention; TensorMap lookup for incoming state; TensorMap insert for outgoing state unless `manual_dep=true` |
| `OUTPUT_EXISTING` | no TensorMap insert; later same-scope reuse needs explicit manual edges | creator retention; TensorMap insert unless `manual_dep=true` |
| `NO_DEP` | creator-only handle passing | same |

### `manual_dep=true`

`Tensor::manual_dep` keeps its original meaning:

- skip TensorMap lookup/insert for that tensor
- still preserve creator retention via `owner_task_id`

It is a per-tensor optimization hint. It is not the manual-scope feature by
itself.

## Runtime Model

### High-level flow

```text
PTO2_SCOPE(MANUAL)
        |
        v
  submit_*_manual[_with_deps]()
        |
        +-- classify tensor args
        |     |- manual-local -> no TensorMap
        |     `- boundary     -> owner retention + optional TensorMap
        |
        +-- allocate slot / payload / outputs
        |
        +-- append explicit same-scope producer slot states
        |     into the consumer payload tail
        |     (no dep-pool fanout publish yet)
        |
        `-- return { task_id, outputs }
                  |
                  +-- optional pto2_rt_add_dependency(...)
                  |      appends more explicit producer slot states to
                  |      the same cached payload tail range
                  |
                  v
scope_end()
        |
        +-- link only the cached explicit tail range into dep-pool fanout lists
        +-- repair monotonic dep_pool_mark prefix
        +-- release publish barrier and batch-publish tasks
        `-- do normal scope lifetime release
```

### Why explicit manual edges are not linked into dep-pool during submit

This was attempted and rejected.

Manual-scope tasks are unpublished until `scope_end()`. If explicit manual
edges are materialized into dep-pool fanout lists during submit:

- producers are still invisible to the scheduler
- `last_task_alive` cannot advance far enough for safe dep-pool reclamation
- manual scopes can deadlock the dep-pool

So the current design intentionally splits the work:

- submit time:
  - append explicit producer slot states into the consumer payload
  - increment `fanin_count`
  - remember where the explicit tail range starts
- `scope_end()`:
  - materialize only that cached explicit tail range into dep-pool fanout lists
  - publish the scope in one batch

That keeps the semantics correct without reviving the large old replay scan.

### What manual submit iterates

Current implementation lives in
`src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp`.

For manual submit:

1. allocate the task slot, payload, and task id immediately
2. classify each tensor arg as manual-local or boundary
3. build `manual_local_mask` for same-scope tensors
4. decide whether TensorMap sync is needed at all
   - if all relevant tensors are manual-local or `manual_dep=true`, skip sync
   - otherwise run normal TensorMap sync
5. for each non-`OUTPUT` boundary arg
   - always do creator retention from `owner_task_id`
   - for `INPUT` and `INOUT`, do TensorMap lookup unless `manual_dep=true`
6. for `INOUT` and `OUTPUT_EXISTING` boundary args
   - update TensorMap frontier unless `manual_dep=true`
7. initialize scheduler state, but keep the task unpublished behind the manual
   publish barrier
8. append explicit same-scope producers into the consumer payload tail if the
   caller used `*_manual_with_deps(...)`

Important consequence:

- cross-scope correctness is still paid at submit time
- same-scope explicit dependencies no longer pay TensorMap lookup
- explicit same-scope edges also no longer pay hot orchestration-side
  `add_dependency(...)` calls when the submit helper is used

### What `pto2_rt_add_dependency(...)` does now

`pto2_rt_add_dependency(...)` is still valid in manual scope, but it no longer
builds dep-pool fanout links immediately.

It now:

1. validates producer and consumer belong to the current manual scope
2. deduplicates against the consumer payload fanins
3. appends the producer slot state into the consumer payload
4. extends the cached explicit tail range metadata
5. increments the consumer `fanin_count`

This preserves the same manual-scope semantics as the helper APIs.

### What `scope_end()` does now

Manual `scope_end()` remains TensorMap-free.

It now:

1. validates `fanin_actual_count`
2. sums cached explicit tail sizes across the scope
3. ensures dep-pool space once for the whole manual scope
4. links only the cached explicit tail ranges into producer fanout lists
5. repairs a monotonic `dep_pool_mark` prefix
6. calls `publish_manual_scope_tasks_and_end_scope(...)`
7. performs normal scope lifetime release

The important detail is step 4:

- older versions had to rescan every consumer fanin and test whether each
  producer belonged to the current manual scope
- current code links only the cached explicit manual tail range
- that is the optimization in commit `6dc2e1e`

## Why This Split Is Correct

### Cross-scope correctness

Boundary tensors still need TensorMap because the runtime must preserve:

- latest-writer frontier tracking
- overlap-based modifier discovery
- ordering across scopes

If manual scope disabled TensorMap globally, outer reads and writes would
become incorrect.

### Same-scope performance

Manual-local tensors are exactly where TensorMap is unnecessary:

- the producer is already known inside the current manual scope
- the ordering is explicitly available from task ids
- same-scope fanins do not need overlap discovery

### Zero-overhead AUTO path

The manual-scope feature must not add work to `AUTO`.

The committed runtime changes stay on the manual submit/scope-end path. The
remaining zero-overhead concern is still the non-unroll AUTO paged-attention
scene, which is a separate runtime/orch optimization problem.

## Example Requirements

Manual mode only helps when the orchestration exposes a real same-scope
producer/consumer chain that TensorMap would otherwise rediscover.

For paged attention, the profitable chain is:

```text
qk_matmul -> softmax_prepare -> pv_matmul -> online_update
```

Inside a manual scope:

- intermediate tensors in that chain stay manual-local
- explicit task edges express the chain directly
- boundary tensors such as external KV cache or outer outputs keep
  cross-scope semantics

If the example keeps boundary tensors everywhere, partial-manual cannot remove
much runtime work.

## Benchmark Enablement

### Current selectors

```bash
./tools/benchmark_rounds.sh -d 5 -n 5 -c d96c8784 -r aicpu_build_graph --build
./tools/benchmark_rounds.sh -d 5 -n 5 -c d96c8784 -r tensormap_and_ringbuffer --build
./tools/benchmark_rounds.sh -d 5 -n 5 -c d96c8784 -r tensormap_and_ringbuffer_partial_manual --build
```

`tensormap_and_ringbuffer_partial_manual` is a benchmark selector in
`tools/benchmark_rounds.sh`. It switches the scene directories to:

- `tests/st/a2a3/tensormap_and_ringbuffer/paged_attention_partial_manual`
- `tests/st/a2a3/tensormap_and_ringbuffer/paged_attention_unroll_partial_manual`

The partial-manual scenes enable the new runtime behavior by:

- entering `PTO2_SCOPE(PTO2ScopeMode::MANUAL)`
- using `pto2_rt_submit_*_manual_with_deps(...)` for hot explicit chains

### Unmodified runtime

`tensormap_and_ringbuffer_unmodified` exists only in a separate worktree, not
on this branch.

Example rerun flow:

```bash
git worktree add tmp/worktree_unmodified a71ba16
(
  cd tmp/worktree_unmodified
  python3 -m venv .venv --system-site-packages
  . .venv/bin/activate
  pip install -e . -q
  export PTO_ISA_ROOT="$PROJECT_ROOT/examples/scripts/_deps/pto-isa"
  ./tools/benchmark_rounds.sh -d 5 -n 5 -c d96c8784 \
    -r tensormap_and_ringbuffer_unmodified --build
)
```

## Fresh Hardware Results

Fresh rerun settings:

- date: `2026-04-10`
- platform: `a2a3`
- device: `5`
- rounds: `5`
- PTO-ISA commit: `d96c8784`

Units below are elapsed microseconds. These are fresh reruns only.

### `paged_attention`

| Case | `aicpu_build_graph` | `tensormap_and_ringbuffer_unmodified` | `tensormap_and_ringbuffer` | `tensormap_and_ringbuffer_partial_manual` |
| --- | ---: | ---: | ---: | ---: |
| `Case1` | `30943.4` | `36722.9` | `38296.0` | `32008.9` |
| `Case2` | `16189.9` | `18682.5` | `19904.9` | `16605.3` |

### `paged_attention_unroll`

| Case | `aicpu_build_graph` | `tensormap_and_ringbuffer_unmodified` | `tensormap_and_ringbuffer` | `tensormap_and_ringbuffer_partial_manual` |
| --- | ---: | ---: | ---: | ---: |
| `Case1` | `1392.4` | `1325.9` | `1151.3` | `1159.1` |
| `Case2` | `678.1` | `638.1` | `563.3` | `568.0` |

## Feature / Optimization -> Gain

### 1. Folding explicit edges into submit removed hot `add_dependency(...)` cost

Hot non-unroll partial-manual profiling before the helper change showed:

- `add_dependency`: about `4.6-4.8 ms` on `paged_attention/Case1`
- `submit_manual`: about `26.4 ms`
- `scope_close`: about `4.8-5.0 ms`

After switching the hot path to `*_manual_with_deps(...)`:

- `add_dependency`: `0`
- the same explicit edges are still represented correctly
- the orchestration hot path no longer pays separate explicit-edge calls

This was the big submit-side win.

### 2. Caching explicit fanin ranges cut manual `scope_end()` overhead again

Commit `6dc2e1e` changed manual payloads to cache the explicit manual tail
range and changed manual `scope_end()` to link only that range.

Measured effect on device `5`:

- non-unroll `Case1`
  - elapsed: about `32.76 ms -> 32.01 ms`
  - `scope_close`: about `4.8-5.1 ms -> 3.5-3.8 ms`
- non-unroll `Case2`
  - elapsed: about `17.33 ms -> 16.61 ms`
  - `scope_close`: about `2.4-2.7 ms -> 1.9-2.0 ms`

This confirms the remaining manual scope-end cost was still meaningful, but it
is no longer the main gap.

### 3. Current non-unroll ranking is now close to the intended shape

For `paged_attention`:

- `aicpu_build_graph` is still the fastest
- `tensormap_and_ringbuffer_partial_manual` is now much closer to it
- `tensormap_and_ringbuffer_partial_manual` is clearly better than both
  current AUTO and the unmodified tensormap runtime

Concrete non-unroll deltas:

- vs `aicpu_build_graph`
  - `Case1`: `32008.9 us` vs `30943.4 us` (`+3.4%`)
  - `Case2`: `16605.3 us` vs `16189.9 us` (`+2.6%`)
- vs `tensormap_and_ringbuffer_unmodified`
  - `Case1`: `36722.9 us -> 32008.9 us` (`-12.8%`)
  - `Case2`: `18682.5 us -> 16605.3 us` (`-11.1%`)
- vs current AUTO
  - `Case1`: `38296.0 us -> 32008.9 us` (`-16.4%`)
  - `Case2`: `19904.9 us -> 16605.3 us` (`-16.6%`)

### 4. Unroll remains a low-return target for partial-manual

For `paged_attention_unroll`, AUTO already amortizes most of the dependency
cost, so partial-manual stays nearly flat:

- `Case1`: `1151.3 us -> 1159.1 us`
- `Case2`: `563.3 us -> 568.0 us`

This is expected. The profitable target for partial-manual is the non-unroll
scene.

### 5. Remaining gap is now mostly submit-time, not `scope_end()`

Latest non-unroll partial-manual profiling after `6dc2e1e`:

- `submit_manual`: about `27.5-28.0 ms`
- `scope_close`: about `3.5-3.8 ms`
- `add_dependency`: `0`

So the next worthwhile optimization target is manual submit itself, especially
the runtime/orch path for boundary tensors inside manual scopes.

## Current Risks

1. Submit-time dep-pool linking for explicit manual edges is still unsafe.
   - Because manual-scope tasks are unpublished until `scope_end()`, doing that
     work at submit time can deadlock dep-pool reclamation.

2. `manual_dep=true` can still be abused.
   - It suppresses TensorMap lookup/insert for that tensor.
   - It is only safe when ordering/frontier requirements are covered elsewhere.

3. Nested manual scope is still unsupported.
   - This is an implementation restriction of the current design.

4. The remaining non-unroll gap is no longer dominated by `scope_end()`.
   - More `scope_end()` work is unlikely to close the gap.
   - The next gains must come from the manual submit path.

5. AUTO non-unroll still needs work if zero-overhead is required there too.
   - The partial-manual feature should not be used as an excuse to leave AUTO
     slow on that scene.

## Recommendation Summary

Keep the design as:

- `AUTO` by default
- explicit `MANUAL` mode through `PTO2ScopeMode`
- TensorMap retained for cross-scope correctness
- explicit same-scope edges appended at submit time
- dep-pool fanout materialization deferred to manual `scope_end()`
- manual `scope_end()` linking only the cached explicit tail range

This gives the required feature coverage, keeps the methodology sound, and has
already moved partial-manual non-unroll paged attention close to
`aicpu_build_graph` without changing the fundamental runtime model.

The next optimization pass should focus on reducing manual submit overhead for
boundary tensors that still pay runtime/orch bookkeeping inside manual scopes.
