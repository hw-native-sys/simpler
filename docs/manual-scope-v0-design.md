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
- scope-depth-based tensor locality check

This keeps the PR small, aligns with maintainer feedback, and preserves the
useful part of manual scope for the current examples.
