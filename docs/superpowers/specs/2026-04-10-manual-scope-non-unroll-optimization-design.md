# Manual Scope Non-Unroll Optimization Design

## Goal

Reduce `tensormap_and_ringbuffer_partial_manual` orchestration overhead on
non-unroll `paged_attention` so it moves closer to
`aicpu_build_graph`, without regressing AUTO mode or weakening
cross-scope correctness.

This pass stays runtime-first. Example reshaping is allowed later only if the
runtime path stalls.

## Current Situation

On the rebased branch, the remaining non-unroll gap is concentrated in the
manual runtime/orchestrator path:

- boundary TensorMap lookup/insert still contributes substantial submit cost
- manual `scope_end()` still performs a full per-task `dep_pool_mark` repair
  scan before publish
- `pto2_rt_add_dependency(...)` still resolves producer and consumer by two
  linear scans over the current manual scope

The current design note and profiling already show that the old explicit-edge
replay model is gone. The next step is to remove the manual-only bookkeeping
that is still left in the common case.

## Scope

This optimization pass changes only the manual-scope runtime/orchestrator path
in:

- `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.h`
- `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp`

Tests and docs may be updated as needed to cover the new invariants.

This pass does not change:

- AUTO-mode dependency behavior
- TensorMap semantics for cross-scope tensors
- the user-facing manual-scope API shape
- the partial-manual example methodology

## Design

### 1. Common-case manual `scope_end()` should skip dep-pool repair

Today manual `scope_end()` always walks every task in the scope and repairs a
monotonic `dep_pool_mark` prefix before publish. That work is only needed when
an explicit dependency is added to an older consumer after newer tasks have
already advanced the dep pool.

The common orchestration pattern is different:

```text
submit producer
submit consumer
add_dependency(producer, consumer)
submit next tasks
...
scope_end
```

In that common case, the consumer is the current tail task when the dependency
is added, so its `dep_pool_mark` can already be correct at edge insertion time.

The runtime will add manual-scope state to distinguish the two cases:

- `manual_scope_needs_dep_pool_repair = false` at manual `scope_begin()`
- keep it `false` when `add_dependency()` targets the current tail consumer
- set it `true` only when `add_dependency()` patches an older consumer after
  later tasks already exist

Then:

- if the flag is `false`, manual `scope_end()` skips the repair loop and goes
  straight to publish + scope release
- if the flag is `true`, manual `scope_end()` keeps the existing repair logic
  as the correctness fallback

This preserves correctness while removing the serial repair scan in the common
manual pattern.

### 2. Manual dependency lookup should be O(1)

Today `pto2_add_dependency()` resolves both task ids by scanning
`scope_tasks[]` twice. In large non-unroll scopes that adds avoidable
orchestrator work.

The runtime will maintain a manual-scope task-id to scope-index map for the
currently active manual scope:

- insert a mapping when each manual task is submitted
- clear/reset the mapping at manual scope end
- use the mapping in `pto2_add_dependency()` instead of the current linear
  search helper

The mapping is only for the active manual scope, not for nested scopes or
global lifetime tracking.

### 3. Keep TensorMap semantics unchanged in this pass

This pass intentionally does not change boundary TensorMap semantics.

Manual-local tensors already skip TensorMap. Boundary tensors still rely on:

- `owner_task_id` creator retention
- TensorMap lookup for boundary read state
- TensorMap insert for boundary write frontier

Those semantics remain unchanged in this pass because they protect the
cross-scope cases that AUTO and MANUAL both still need.

If more performance work is needed after this pass, the next candidate is a
more selective boundary fast path. That is explicitly out of scope here.

## Correctness Invariants

The implementation must preserve all of the following:

1. AUTO mode remains unchanged.
2. Manual-local tensors still require explicit same-scope dependencies.
3. Boundary tensors still use creator retention and TensorMap exactly as they do
   today.
4. Manual publish remains correct when orchestration adds a dependency to an
   older consumer after newer tasks have already advanced the dep pool.
5. Manual-scope task-id lookup is valid only for tasks in the current active
   manual scope.

## Risks

### Risk 1: silent dep-pool reclamation bug

If the runtime wrongly skips repair for a scope that actually needed it,
dep-pool tail reclamation can advance too early.

Mitigation:

- use an explicit fallback flag, not inference at `scope_end()`
- set the flag conservatively whenever a retroactive edge is observed
- keep the existing repair path intact as the fallback path

### Risk 2: stale manual lookup map entries

If the manual task-id lookup map is not reset correctly, later scopes could
resolve old tasks by mistake.

Mitigation:

- initialize/reset the map at manual scope boundaries
- validate mapped ids still belong to the current manual scope before use
- keep current invalid-args guards

### Risk 3: optimization helps only add-dependency, not total orch time

The non-unroll gap may still be dominated by boundary TensorMap work.

Mitigation:

- benchmark with orchestration breakdown after this pass
- treat this as the first runtime cleanup step, not the final answer

## Validation Plan

### Correctness

- run existing manual guard coverage
- run existing manual boundary coverage
- add regression coverage for:
  - common-case tail-consumer dependency path
  - retroactive older-consumer dependency path

### Real device

- rerun non-unroll `paged_attention_partial_manual`
- rerun AUTO and unroll sanity cases to catch regressions

### Benchmark

Rerun the 4-way comparison on the two paged-attention scenes:

- `aicpu_build_graph`
- `tensormap_and_ringbuffer_unmodified`
- `tensormap_and_ringbuffer`
- `tensormap_and_ringbuffer_partial_manual`

Success for this pass means:

- lower non-unroll partial-manual orch time
- no AUTO regression
- no correctness regression

## Expected Outcome

This pass should remove manual-only bookkeeping overhead in the common explicit
dependency pattern while keeping the current hybrid model intact:

- manual-local same-scope ordering stays explicit
- cross-scope correctness still stays on TensorMap + creator retention
- manual `scope_end()` becomes cheaper when no retroactive repair is needed
