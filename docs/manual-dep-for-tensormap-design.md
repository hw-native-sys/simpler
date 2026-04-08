# Manual Dependency For TensorMap Runtime

## Goal

Add a scoped manual-dependency mode to `tensormap_and_ringbuffer` without
regressing the default automatic path:

- `PTO2_SCOPE()` keeps the existing automatic mode
- `PTO2_SCOPE(PTO2ScopeMode::MANUAL)` enables scoped manual dependency wiring
- same-manual-scope edges use explicit `pto2_rt_add_dependency(...)`
- cross-scope edges still use `owner_task_id` and TensorMap discovery

This is a hybrid model, not a port of `aicpu_build_graph`.

## API Surface

The orchestration API now uses an enum instead of the old boolean-style scope
switch:

```cpp
PTO2_SCOPE() {
    // default: PTO2ScopeMode::AUTO
}

PTO2_SCOPE(PTO2ScopeMode::MANUAL) {
    auto qk = pto2_rt_submit_aic_task_manual(...);
    auto sf = pto2_rt_submit_aiv_task_manual(...);
    pto2_rt_add_dependency(qk.task_id, sf.task_id);
}
```

Current modes:

- `PTO2ScopeMode::AUTO`
- `PTO2ScopeMode::MANUAL`

Current restrictions:

- manual scope cannot be nested inside another manual scope
- manual submit APIs are only valid inside `PTO2_SCOPE(PTO2ScopeMode::MANUAL)`
- `pto2_rt_add_dependency(...)` requires both tasks to belong to the current
  manual scope

## Current Design

### High-level rule

Manual mode only changes same-scope dependency discovery.

- in-scope tensors: use explicit task edges
- cross-scope tensors: keep TensorMap semantics
- scope-local lifetime model: unchanged
- scheduler execution model after publish: unchanged

### Why this split exists

We need two properties at the same time:

1. Manual scopes must be able to skip TensorMap work for same-scope chains.
2. Outer producers and outer consumers must still see the correct frontier.

If we disabled TensorMap for everything inside a manual scope, cross-scope
reads and writes would become incorrect. If we kept TensorMap for everything,
manual mode would not remove the overhead we care about.

The chosen split is:

- same-scope manual-local traffic: explicit edges only
- boundary traffic: existing creator-retention + TensorMap lookup/insert

## Dependency Semantics

### Tensor origin matters first

Each tensor argument is classified at submit time:

- `manual-local`: the tensor owner was created inside the current manual scope
- `boundary`: anything else, including external tensors and tensors produced by
  tasks outside the current manual scope

Manual-local tensors skip TensorMap entirely. Boundary tensors stay on the
normal TensorMap path unless `manual_dep=true`.

### What `INPUT`, `OUTPUT`, `INOUT`, and friends mean

`TensorArgType` in the runtime:

- `INPUT`: read-only existing tensor
- `OUTPUT`: fresh runtime-allocated tensor
- `INOUT`: read existing state and publish a new state
- `OUTPUT_EXISTING`: write-only existing tensor
- `NO_DEP`: existing tensor with no TensorMap dependency work and no publish

### Behavior matrix

| Arg kind | Manual-local tensor | Boundary tensor |
| --- | --- | --- |
| `INPUT` | no creator retention, no TensorMap lookup, requires explicit manual edge | creator retention; TensorMap lookup unless `manual_dep=true` |
| `OUTPUT` | fresh local tensor; later same-scope uses rely on explicit manual edges | not applicable |
| `INOUT` | no TensorMap lookup/insert, requires explicit manual edge | creator retention; TensorMap lookup for incoming state; TensorMap insert for outgoing state unless `manual_dep=true` |
| `OUTPUT_EXISTING` | no TensorMap insert, requires explicit manual edge if later reused in scope | creator retention; TensorMap insert for outgoing state unless `manual_dep=true` |
| `NO_DEP` | creator-only object passing, no publish | same |

### `manual_dep=true` still matters

`Tensor::manual_dep` keeps its existing meaning:

- skip TensorMap lookup/insert
- keep creator-only retention via `owner_task_id`

This is orthogonal to manual scope mode. It is a per-tensor override, not a
replacement for scoped manual dependency wiring.

## Submit-Time Algorithm

Current implementation is in
`src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp`.

For a manual submit:

1. Allocate the task slot, payload, and task id immediately.
2. Classify each tensor arg as manual-local or boundary.
3. Build `manual_local_mask` for same-scope tensors.
4. Decide whether TensorMap sync is needed at all:
   - if every relevant arg is manual-local or `manual_dep=true`, skip sync
   - otherwise run the normal TensorMap sync
5. For each non-`OUTPUT` arg that is not manual-local:
   - always do creator retention from `owner_task_id`
   - for `INPUT` and `INOUT`, do TensorMap lookup unless `manual_dep=true`
6. For `INOUT` and `OUTPUT_EXISTING` boundary args:
   - update TensorMap frontier unless `manual_dep=true`
7. Initialize scheduler state, but keep the task unpublished behind a deferred
   publish barrier.

Important consequence:

- cross-scope dependency discovery is paid at submit time
- same-scope dependency discovery is not replayed from tensors later

## `scope_end` Algorithm

Manual `scope_end` is now intentionally small.

It does not replay explicit same-scope edges from a separate side buffer
anymore. Those edges are already realized when `pto2_rt_add_dependency(...)`
is called.

Current manual `scope_end` does:

1. validate `fanin_actual_count`
2. repair a monotonic `dep_pool_mark` prefix
3. batch-publish the scope tasks to the scheduler
4. perform the normal scope lifetime release

That is the key change from the older draft design. The old replay-heavy model
is gone.

## What Is Maintained

Current manual mode keeps only the state that is still needed:

- `scope_tasks[]`: ordered list of tasks in the current scope
- `manual_scope_active`: current scope mode
- per-task `fanin_slot_states[]` / `fanin_actual_count`
- normal scheduler `fanin_count`, `fanin_refcount`, `fanout_head`
- `dep_pool_mark` for tail reclamation

Removed from the active design:

- manual edge replay buffers
- manual task metadata used only for finalize-time dependency reconstruction
- manual-scope dependency re-materialization at `scope_end`

## Why Partial-Manual Was Slow Before

The bad version paid two costs at once:

1. It still did TensorMap-like work for the same-scope region.
2. It also paid a serial `scope_end` replay barrier to rebuild dependencies and
   publish tasks.

That was the worst possible combination: extra submit cost plus extra finalize
cost.

The current design removes that double payment:

- same-scope edges are explicit and immediate
- boundary discovery stays on the TensorMap path
- manual `scope_end` is only a publish barrier, not a dependency replay pass

## Zero-Overhead Target

The zero-overhead target here means:

- no extra cost on `PTO2ScopeMode::AUTO`
- no extra TensorMap work for manual-local traffic
- no second dependency engine after publish

What manual mode is allowed to cost:

- explicit dependency calls that the example asked for
- one deferred publish barrier at `scope_end`
- boundary TensorMap work only when the task actually crosses scope boundaries

## Example Requirements

Manual mode only helps when the example exposes a same-scope producer/consumer
chain that TensorMap would otherwise rediscover.

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
./tools/benchmark_rounds.sh -d 4 -n 5 -c 6622890 -r aicpu_build_graph
./tools/benchmark_rounds.sh -d 4 -n 5 -c 6622890 -r tensormap_and_ringbuffer
./tools/benchmark_rounds.sh -d 4 -n 5 -c 6622890 -r tensormap_and_ringbuffer_partial_manual
```

The old unmodified runtime is intentionally not kept on this branch. To rerun
it side-by-side:

```bash
git worktree add tmp/worktree_unmodified a71ba16
cd tmp/worktree_unmodified
PTO_ISA_ROOT=../../examples/scripts/_deps/pto-isa \
  ./tools/benchmark_rounds.sh -d 4 -n 5 -c 6622890 -r tensormap_and_ringbuffer_unmodified
```

In this work, direct `run_example.py` reruns were more reliable than the
wrapper for collecting fresh device logs, especially for the old runtime and
for cases where the wrapper suppressed useful failure output.

## Fresh Performance Snapshot

Device and settings used for the rerun set:

- device: `4`
- rounds: `5`
- ISA commit: `6622890`

### End-to-end / scheduler-side comparison

These numbers are the freshest rerun values used in this document.

| Example | Case | `aicpu_build_graph` | old `tensormap*` | new `tensormap*` | `tensormap* + partial_manual` |
| --- | --- | ---: | ---: | ---: | ---: |
| `paged_attention` | `Case1` | `31312.6 us` | `37061.0 us` | `36585.4 us` | `31814.5 us` |
| `paged_attention` | `Case2` | `16474.4 us` | `18589.4 us` | `19348.8 us` | `16221.1 us` |
| `paged_attention_unroll` | `Case1` | `1426.8 us` | `1383.6 us`* | `1322.1 us` | `1327.9 us` |
| `paged_attention_unroll` | `Case2` | `728.5 us` | `668.6 us`* | `623.8 us` | `639.1 us` |

`*` The old unmodified unroll runtime logs use the older
`Scheduler summary: total_time=...` format instead of the newer full elapsed
markers. Those two rows therefore use scheduler-summary averages for the old
baseline.

### Orchestrator comparison

| Example | Case | old `tensormap*` | new `tensormap*` | `tensormap* + partial_manual` |
| --- | --- | ---: | ---: | ---: |
| `paged_attention` | `Case1` | `37060.3 us` | `36584.7 us` | `31657.9 us` |
| `paged_attention` | `Case2` | `18588.6 us` | `19348.2 us` | `15799.4 us` |
| `paged_attention_unroll` | `Case1` | `716.2 us`* | `826.3 us` | `826.0 us` |
| `paged_attention_unroll` | `Case2` | `336.7 us`* | `368.8 us` | `387.6 us` |

`*` Old unmodified unroll uses `orch_func_cost`, not the newer
`orch_end - orch_start` marker. It is still useful for side-by-side direction,
but it is not byte-for-byte the same logging mode.

Old-baseline rerun logs used in this table:

- `device-1533354_20260409000304311.log`
- `device-1546617_20260409000317313.log`
- `device-1536746_20260409000312312.log`
- `device-1568129_20260409000326314.log`

## What Helped and What Mattered

### 1. Collapse manual `scope_end` into publish-only work

This is the main fix for non-unroll paged attention.

Current effect against the new automatic TensorMap runtime:

- `paged_attention Case1`: `36584.7 us -> 31657.9 us` orch
- `paged_attention Case2`: `19348.2 us -> 15799.4 us` orch

This is the difference between “manual mode is the worst case” and
“manual mode is in the same band as `aicpu_build_graph`”.

### 2. Skip TensorMap sync/lookup/insert for fully manual-local traffic

Manual mode now checks whether a submit actually touches a boundary tensor.
If not, it skips TensorMap sync entirely.

This matters most when the example keeps intermediates local to the manual
scope. In the unrolled example, both automatic and partial-manual runtimes are
already in the sub-millisecond range, which shows that the remaining cost is no
longer dominated by boundary TensorMap work.

### 3. Keep boundary correctness on the normal path

Boundary reads and writes still use:

- creator retention
- TensorMap overlap lookup
- TensorMap frontier publish

This does not make manual mode faster by itself. It is the correctness guard
that prevents stale external state and wrong cross-scope dependencies.

### 4. Example structure still dominates the ceiling

`paged_attention_unroll` already reduces submit pressure aggressively. Because
that example exposes less repeated same-scope dependency work, partial-manual
does not beat the best automatic/unmodified paths there.

The no-unroll `paged_attention` case is where partial-manual matters most, and
that is also the target case where it now tracks `aicpu_build_graph`.

## Clear Conclusions

1. Manual scope correctness is preserved by keeping cross-scope tensors on the
   normal TensorMap path.
2. Manual scope performance only improves when the example exposes a real
   same-scope chain that can stay manual-local.
3. The replay-heavy finalize model was the wrong design. The current
   submit-time wiring plus publish-only `scope_end` is the right direction.
4. On non-unroll paged attention, `partial_manual` now matches the intended
   performance band: close to `aicpu_build_graph`, clearly better than both old
   and current automatic TensorMap runtimes.

## Remaining Risks

- manual scopes are still single-level only
- old unmodified unroll comparisons rely on older log markers
- explicit dependency misuse is still a fatal orchestration error by design
- the runtime still depends on the example to choose good manual-scope
  boundaries; bad example structure can erase the benefit
