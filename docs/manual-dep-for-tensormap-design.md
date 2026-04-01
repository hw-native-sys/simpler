# Manual Dependency For TensorMap Runtime

## Goal

Bring the human-created dependency workflow from `aicpu_build_graph` into `tensormap_and_ringbuffer` in a scoped way:

- `PTO_SCOPE(manual_dep=1) { ... }`
- Tensors crossing scope boundaries use TensorMap semantics
- Tensors used entirely inside the manual scope use explicit `add_dependency`

This is not a port of `aicpu_build_graph`'s fully-explicit runtime model. The target is a hybrid model inside `tensormap_and_ringbuffer`:

- same-scope dependency tracking: explicit
- cross-scope dependency tracking: TensorMap
- scope-local lifetime management: unchanged ring/scope ownership model

## Confirmed Decisions

These decisions are already aligned with the requested direction:

1. `tensormap` scope may contain a manual scope.
2. Manual scope may not contain another manual scope.
3. The design must not simplify away multi-write cases.
4. For an outer-scope tensor written inside a manual scope, readiness is the writer task completion time, not `scope_end`.
5. Therefore, a task inside a manual scope that writes an outer-scope tensor must still publish that tensor to TensorMap.

## Non-Goals

- Do not replace `tensormap_and_ringbuffer` with a fully explicit runtime.
- Do not require explicit export/import APIs at scope boundaries.
- Do not constrain v1 to single-writer exported tensors.
- Do not change the existing rule that inner-scope temporary tensors do not outlive their owning scope unless already represented by an outer-scope tensor.

## Current Runtime Behavior Relevant To This Design

## Scope lifetime

In `tensormap_and_ringbuffer`, each submitted task starts with one scope-held fanout reference. On `scope_end`, the scheduler releases that reference. When fanout is otherwise exhausted, the task can become `CONSUMED` and its slot/buffer can be reclaimed.

This means:

- outer-scope tensors may flow into inner scopes
- inner-scope temporaries are scope-local by default
- `scope_end` affects lifetime ownership, not semantic readiness of a cross-scope tensor write

## Current dependency model

Today the runtime derives dependencies in `pto_orchestrator.cpp` using:

- creator retention through `owner_task_id`
- modifier lookup through TensorMap overlap search
- TensorMap insert for `INOUT` and `OUTPUT_EXISTING`

There is already a `Tensor::manual_dep` bit, but in current code it is effectively a per-tensor escape hatch that skips TensorMap lookup/insert. That is not sufficient for scoped hybrid semantics because the scope, not the tensor alone, decides whether a use is same-scope or cross-scope.

## Problem Statement

If we simply copy `aicpu_build_graph` semantics into `tensormap_and_ringbuffer`, we get a wrong boundary model:

- suppressing TensorMap for all tensors inside `PTO_SCOPE(manual_dep=1)` is incorrect
- delaying publication of an outer tensor until `scope_end` is incorrect

The reason is that cross-scope tensors must become visible at the actual writer frontier. Outside consumers should depend on the task that really produced the latest visible state, not on scope closure.

So the correct split is:

- same-scope tensor relations inside the manual scope: explicit edges only
- cross-scope tensor relations: preserve TensorMap behavior

## Required Semantics

## Core rule

`PTO_SCOPE(manual_dep=1)` means:

- if both producer and consumer are inside this manual scope, the dependency must be established by explicit `add_dependency`
- if a tensor use crosses the scope boundary, dependency tracking still uses TensorMap/owner metadata

This rule applies per tensor use site, not as a global on/off switch for the whole submit.

## Tensor categories

For a task submitted inside a manual scope, every tensor argument falls into one of these categories:

1. Outer-scope tensor, read only
2. Outer-scope tensor, written in place
3. Tensor created inside this manual scope, used again inside this manual scope
4. Tensor created inside this manual scope, then used through an outer-scope tensor alias/view
5. External tensor with no owner task

The runtime must classify behavior from ownership and current scope, not only from argument tag.

## Expected behavior by category

### 1. Outer-scope tensor, read only

- The first internal consumer still needs dependency seeding from existing producer state.
- This must still use creator retention and TensorMap lookup as appropriate.
- Manual scope does not remove the need to wait for the outer producer frontier.

### 2. Outer-scope tensor, written in place

- The internal writer must still publish to TensorMap.
- Readiness of the written tensor is the completion of that writer task.
- Multiple writes inside the same manual scope are allowed.
- TensorMap should continue tracking the latest producer frontier exactly as in normal scope.

### 3. Tensor created inside this manual scope and reused only inside this manual scope

- No TensorMap lookup/insert.
- No automatic same-scope dependency derivation.
- Orchestration must call `add_dependency` explicitly for correctness.

### 4. Tensor created inside this manual scope, then used through an outer-scope alias/view

This case must be handled by ownership classification, not by raw pointer equality.

If the tensor instance still belongs to the manual scope, it remains same-scope and should stay explicit.

If orchestration is mutating an outer-scope tensor through a view that inherits the outer owner/scope identity, that is cross-scope and should keep TensorMap behavior.

### 5. External tensor with no owner task

- There is no creator dependency.
- Reads need no dependency unless TensorMap contains a producer entry.
- Writes to such a tensor should still publish to TensorMap if the tensor is cross-scope visible.

## Recommended API Shape

## Orchestration API

Add explicit edge wiring to `tensormap_and_ringbuffer` orchestration API, mirroring `aicpu_build_graph`:

```cpp
void pto2_rt_add_dependency(PTO2TaskId producer, PTO2TaskId consumer);
```

Add scoped manual mode:

```cpp
PTO_SCOPE(manual_dep = 1) {
    ...
}
```

For C++ implementation, this should compile down to a guard with scope metadata, not a dynamic stringly API.

## Runtime API

Add runtime ops support:

```cpp
void (*add_dependency)(PTO2Runtime* rt, PTO2TaskId producer, PTO2TaskId consumer);
```

Add manual-scope entry/exit plumbing by extending the existing scope API with a mode flag:

```cpp
void pto2_rt_scope_begin(PTO2Runtime* rt, bool manual_dep);
```

Recommendation: extend scope state with a mode flag and keep one scope stack. Avoid separate manual/non-manual stacks and avoid introducing a second scope API family.

## Internal Design

## Scope state

Each scope frame needs:

- `begin_index` into `scope_tasks`
- scope mode: normal or manual
- unique scope id

The unique scope id is required because same-scope vs cross-scope classification must be relative to the current manual scope, not only to nested depth.

Recommendation:

- assign `scope_id` on every `scope_begin`
- store current producing `scope_id` on runtime-created tensors
- views inherit the source tensor's producing `scope_id`

## Tensor metadata

Current `Tensor` already stores:

- `owner_task_id`
- `manual_dep`

For this design, the critical missing concept is producing scope identity. We need enough metadata to answer:

- was this tensor produced in the current manual scope?
- or is it owned by an outer scope and therefore boundary-visible?

Recommendation:

- add `owner_scope_id` to `Tensor`
- initialize runtime-created outputs with the current scope id
- inherit `owner_scope_id` through `view`, `reshape`, and `transpose`

`manual_dep` should no longer be the primary mechanism for scope semantics. It may remain as a per-tensor override, but the scoped design should be driven by:

- current scope mode
- tensor owner scope id
- tensor owner task id

## Submit-time classification

In `pto_orchestrator.cpp`, dependency behavior should be classified per tensor argument.

Pseudo-rule for a task submitted inside a manual scope:

```cpp
same_scope_tensor =
    tensor.owner_task_id.is_valid() &&
    tensor.owner_scope_id == current_manual_scope_id;

if (!in_manual_scope) {
    use existing tensormap behavior;
} else if (same_scope_tensor) {
    skip TensorMap lookup/insert;
    rely on explicit add_dependency;
} else {
    use cross-scope TensorMap/owner behavior;
}
```

This should be applied separately to:

- creator retention
- modifier lookup
- TensorMap insertion for writes

Important nuance:

- same-scope tensors should still retain creator lifetime through explicit dependencies, not through automatic creator retention
- cross-scope tensors should still retain creator lifetime automatically

## Explicit edge wiring

`pto2_add_dependency` from `aicpu_build_graph` can be reused conceptually, but the implementation must match `tensormap_and_ringbuffer` scheduler semantics:

- increment consumer `fanin_count`
- record producer in consumer payload for release traversal
- wire producer fanout list under `fanout_lock`
- handle early-completed producer case

No scope-end batch publish behavior should be imported. `tensormap_and_ringbuffer` tasks are already submit-visible before scope end, and changing that would be a separate design.

## Scope-end behavior

Manual scope does not change lifetime release semantics:

- `scope_end` still releases the owning-scope fanout reference
- `scope_end` is not a publication barrier for cross-scope tensors
- cross-scope visibility must already reflect task completion frontier

This is the main semantic difference from `aicpu_build_graph`.

## Multiple Writes To Outer Tensors

This case must be supported in v1.

Example:

```cpp
PTO_SCOPE(manual_dep=1) {
    t1 writes outer C
    t2 writes outer C
    add_dependency(t1, t2)
}
outside task reads C
```

Correct behavior:

- `t1` publishes `C` to TensorMap
- `t2` publishes `C` again to TensorMap
- outside reader should see `t2` as the latest producer frontier
- because `t1 -> t2` is explicit, `t2` completion is a valid readiness frontier for the final visible state

Potential invalid user pattern:

```cpp
PTO_SCOPE(manual_dep=1) {
    t1 writes outer C
    t2 also writes outer C
    // missing add_dependency(t1, t2)
}
```

This is a user error. The runtime should not try to reconstruct same-scope writer ordering automatically in manual mode.

## Reads Of Outer Tensors Inside Manual Scope

Outer tensors read inside manual scope must still seed internal dependencies from existing producer state.

Otherwise:

- a task inside manual scope may run before the outer producer of its input
- explicit edges inside the scope are insufficient to protect the outer-to-inner boundary

So manual mode disables only same-scope auto-derivation, not boundary seeding.

## Nesting Rules

Supported:

- normal scope contains manual scope
- normal scope contains normal scope

Not supported in v1:

- manual scope contains manual scope

Reason:

- the same-scope vs cross-scope rule is already relative to the current manual frame
- nested manual scopes add little value and complicate classification and diagnostics

Recommendation:

- detect this at `scope_begin`
- fail fast with a clear orchestrator error

## Diagnostics

The runtime should detect and report:

1. nested manual scope not supported
2. `add_dependency` used with invalid task ids
3. dependency overflow from explicit wiring
4. obvious cross-scope/manual mismatch where possible

Nice-to-have diagnostics:

- count of explicit edges added in manual scope
- count of cross-scope TensorMap lookups/inserts preserved inside manual scope

These are not required for correctness, but will make profiling and debugging practical.

## Testing Strategy

Add focused coverage before broad workload migration.

### Unit-style runtime cases

1. Manual scope diamond on scope-local outputs
- all same-scope edges explicit
- no TensorMap dependence required

2. Manual scope reads outer tensor
- internal first task waits on outer producer frontier

3. Manual scope writes outer tensor once
- outside consumer waits on inner writer completion, not `scope_end`

4. Manual scope writes outer tensor multiple times
- latest writer becomes TensorMap frontier
- correctness depends on explicit same-scope edge wiring

5. Normal scope containing manual scope
- outer to inner and inner to outer boundary cases both work

6. Nested manual scope
- rejected with deterministic error

### Example-level migration

Use a small example first, such as vector-style or BGEMM-style, to validate:

- scope-local temp tensors use explicit edges
- outer tensors still behave through TensorMap

Only then move to more complex orchestration such as paged attention.

## Main Risks

1. Treating manual scope as a global TensorMap disable switch.
- This breaks cross-scope correctness.

2. Using `Tensor::manual_dep` as the only signal.
- Scope semantics are relational and need owner scope identity.

3. Letting cross-scope writes publish only at `scope_end`.
- This delays readiness incorrectly.

4. Accidentally preserving creator retention for same-scope tensors in manual mode.
- This reintroduces hidden dependencies and weakens the mental model.

5. Missing alias/view inheritance of scope ownership.
- This causes wrong same-scope vs cross-scope classification.

## Recommended Implementation Order

1. Add API surface for `add_dependency` and manual scope mode.
2. Add scope-frame mode and `scope_id`.
3. Add tensor ownership metadata needed for classification.
4. Implement explicit edge wiring in tensormap runtime.
5. Refactor submit-time dependency logic to branch on:
   - current scope mode
   - tensor owner scope id
   - tensor owner task id
6. Add fail-fast nested-manual-scope check.
7. Add targeted tests for boundary semantics.
8. Migrate one example and validate.

## Open Question Resolved

This design intentionally resolves the central ambiguity:

- `scope_end` controls lifetime release
- task completion controls semantic readiness

For outer tensors written inside manual scope, TensorMap publication must stay aligned with task completion frontier, not with scope closure.

## File Areas Expected To Change

- `src/a2a3/runtime/tensormap_and_ringbuffer/orchestration/pto_orchestration_api.h`
- `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_runtime2.h`
- `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_runtime2.cpp`
- `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp`
- `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/tensor.h`
- `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_runtime2_types.h`
- docs and examples/tests needed to demonstrate the new scoped behavior

## Recommendation Summary

Implement manual dependency as a scope-local override inside `tensormap_and_ringbuffer`, not as a runtime-wide replacement of TensorMap:

- same manual scope: explicit `add_dependency`
- crossing the manual scope boundary: TensorMap
- write visibility: writer completion
- lifetime release: `scope_end`

That is the smallest design that satisfies the requested model without breaking the core tensormap runtime semantics.
