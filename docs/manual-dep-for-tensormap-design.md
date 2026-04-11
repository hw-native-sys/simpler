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

- date: `2026-04-11`
- platform: `a2a3`
- device: `5`
- rounds: `7`
- PTO-ISA commit: `d96c8784`

Units below are elapsed microseconds. These are fresh reruns only. The
comparison uses median round time because outliers were observed in the latest
run (`aicpu_build_graph/Case1` and `tensormap_and_ringbuffer/Case2` each had
one large outlier).

### `paged_attention`

| Case | `aicpu_build_graph` | `tensormap_and_ringbuffer` | `tensormap_and_ringbuffer_partial_manual` |
| --- | ---: | ---: | ---: |
| `Case1` | `30026.2` | `37047.9` | `28303.4` |
| `Case2` | `15945.8` | `18757.3` | `15640.3` |

Notes:

- the table above is from the newest reruns after commits `a65894a` and
  `6d33941`
- `tensormap_and_ringbuffer_unmodified` was not rerun in this update, so it is
  intentionally omitted rather than mixed with old results
- `paged_attention_unroll` was also not rerun in this update, so it is omitted
  for the same reason

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

### 3. Skipping no-op manual TensorMap bookkeeping gave a small win

Commit `a65894a` replaced broad manual-arg rescans with exact work masks for:

- creator-retention work
- TensorMap lookup work
- TensorMap insert work

Measured effect on partial-manual device-log medians:

- `Case1`: `31808.4 us -> 31412.1 us` (`-1.2%`)
- `Case2`: `31058.6 us -> 31019.8 us` (`-0.1%`)

This was worth keeping, but it was not the main gap-closer.

### 4. Caching `Tensor::start_offset` at creation was the major recent win

Commit `6d33941` changed TMR tensors so that:

- external tensors cache `start_offset = 0` at construction
- view tensors cache the flattened offset when the view is created
- fresh runtime-created outputs stay at zero
- payload materialization no longer recomputes `start_offset` for every tensor

A focused C++ unit test was added to lock this down:

- `tests/ut/cpp/test_a2a3_tmr_tensor_offsets.cpp`

Measured effect on partial-manual device-log medians before the final rerun:

- `Case1`: `31808.4 us -> 29144.0 us` (`-8.4%`)
- `Case2`: `31058.6 us -> 29514.6 us` (`-5.0%`)

The final rerun shows a larger end-to-end gain because it avoided the earlier
bad `Case2` measurement window:

- `Case1`: `30026.2 us` (`aicpu_build_graph`) vs `28303.4 us`
  (`partial_manual`)
- `Case2`: `15945.8 us` (`aicpu_build_graph`) vs `15640.3 us`
  (`partial_manual`)

### 5. Current non-unroll ranking now matches the intended shape

For `paged_attention` after the newest reruns:

- vs `aicpu_build_graph`
  - `Case1`: `30026.2 us -> 28303.4 us` (`-5.7%`)
  - `Case2`: `15945.8 us -> 15640.3 us` (`-1.9%`)
- vs current AUTO
  - `Case1`: `37047.9 us -> 28303.4 us` (`-23.6%`)
  - `Case2`: `18757.3 us -> 15640.3 us` (`-16.6%`)

This is the intended non-unroll shape:

- `partial_manual` keeps the feature coverage of scoped manual dependencies
- it avoids the heavy TensorMap path for same-manual-scope dependencies
- it now matches or slightly beats `aicpu_build_graph` on the two measured
  non-unroll cases

### 6. Remaining work is still submit-path robustness

Latest non-unroll partial-manual profiling after `6d33941`:

- `Case1`
  - `submit_manual`: about `24.0 ms`
  - `scope_close`: about `3.8 ms`
  - `scope_close(x256)`
- `Case2`
  - `submit_manual`: about `13.0 ms`
  - `scope_close`: about `2.0 ms`
  - `scope_close(x64)`
- `add_dependency`: `0`

The earlier suspicious `Case2` run showed a `scope_close(x256)` shape, which
does not match `Case2`. The rerun shows the expected `scope_close(x64)` shape
and removes the apparent large ABG gap. Future benchmark updates should keep
the device log and round summary paired together to avoid this kind of
wrong-log or contaminated-window interpretation.

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
