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
6. For an outer-scope tensor read inside a manual scope, the dependency must still be forced by TensorMap/owner-based boundary seeding.

## Change Control Requirements

The implementation PR must follow these rules:

- Keep the change strictly scoped to manual dependency support in `tensormap_and_ringbuffer`.
- Do not refactor unrelated runtime behavior while doing this work.
- Do not change existing normal-scope TensorMap semantics.
- Do not change scope lifetime semantics.
- Prefer the smallest invasive write set that cleanly supports the feature.
- Preserve existing examples/tests unless a targeted update is required to cover the new feature.
- Any behavior change outside manual-scope execution must be treated as a regression.

## Repository Rule Requirements

The implementation must carefully follow the repository's coding rules and conventions:

- obey `CLAUDE.md` directory ownership and workflow rules
- obey `.claude/rules/architecture.md`
- obey `.claude/rules/codestyle.md`
- keep platform-isolation preprocessor ordering consistent with repo rules
- avoid comment styles that encode plan phases or temporary implementation notes
- preserve current behavior unless this spec explicitly requires otherwise
- avoid adding new tensor metadata unless it is strictly necessary for correctness
- prefer provenance on task-side state over changing hot-path `Tensor` layout

## Tooling Requirements

The implementation and follow-up PRs must also respect the current repository tooling state:

- PR #424 has already aligned C and C++ sources with `clang-format`.
- Local development should use `clang-format` `v21.1.0`, matching `.pre-commit-config.yaml`.
- Developers should configure local save-time auto-formatting with that exact `clang-format` version to avoid unnecessary AI-driven formatting churn.
- The feature PR should not include unrelated bulk reformatting.
- `.clang-tidy` is now part of the repository toolchain, but many checks are still intentionally disabled in the config file.
- This feature PR must satisfy the currently active `clang-tidy` expectations for touched code.
- Gradually enabling additional `clang-tidy` checks and fixing old violations is a separate ongoing stream of work, not something this feature should broaden into unless directly required for touched code.

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
4. Outer-scope tensor accessed through a derived view/reshape/transpose inside the manual scope
5. External tensor with no owner task

The runtime must classify behavior from ownership and current scope, not only from argument tag.

## Expected behavior by category

### 1. Outer-scope tensor, read only

- The first internal consumer must still get its dependency from TensorMap/owner-based boundary seeding.
- This is not optional and must not be delegated to explicit manual edges inside the scope.
- Manual scope does not remove the need to wait for the outer producer frontier.
- In other words, outer-read boundary correctness is still forced by TensorMap-side logic.

### 2. Outer-scope tensor, written in place

- The internal writer must still publish to TensorMap.
- Readiness of the written tensor is the completion of that writer task.
- Multiple writes inside the same manual scope are allowed.
- TensorMap should continue tracking the latest producer frontier exactly as in normal scope.

### 3. Tensor created inside this manual scope and reused only inside this manual scope

- No TensorMap lookup/insert.
- No automatic same-scope dependency derivation.
- Orchestration must call `add_dependency` explicitly for correctness.

### 4. Outer-scope tensor accessed through a derived view/reshape/transpose inside the manual scope

This is the real aliasing case that matters for the design. It must be handled by ownership classification, not by raw pointer equality.

An outer-scope tensor may be sliced or reshaped inside the manual scope, but it is still outer-scope.

If orchestration is reading or mutating an outer-scope tensor through a derived view that inherits the outer owner/scope identity, that is still cross-scope and should keep TensorMap behavior.

A tensor created inside the manual scope should not later become an outer-scope alias. That would violate the existing scope lifetime model rather than define a supported boundary case.

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
- store current producing `scope_id` on task-side provenance
- use `owner_task_id` on `Tensor` to reach producing task provenance

## Tensor metadata

Current `Tensor` already stores:

- `owner_task_id`
- `manual_dep`

Recommendation: do not add new tensor metadata in v1.

The critical missing concept is producing scope identity, but that provenance should live on the producer task side if possible, not on `Tensor`.

We need enough information to answer:

- was this tensor produced in the current manual scope?
- or is it owned by an outer scope and therefore boundary-visible?

Preferred approach:

- keep `Tensor` layout unchanged
- use `tensor.owner_task_id` as the provenance pointer
- record `owner_scope_id` on producer task-side metadata such as task descriptor or scheduler/orchestrator slot state
- classify same-scope versus cross-scope through `owner_task_id -> producer provenance -> scope_id`

`manual_dep` should no longer be the primary mechanism for scope semantics. It may remain as a per-tensor override, but the scoped design should be driven by:

- current scope mode
- tensor owner task id
- producer task scope provenance

## Submit-time classification

In `pto_orchestrator.cpp`, dependency behavior should be classified per tensor argument.

Pseudo-rule for a task submitted inside a manual scope:

```cpp
same_scope_tensor =
    tensor.owner_task_id.is_valid() &&
    producer_scope_id(tensor.owner_task_id) == current_manual_scope_id;

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
- cross-scope outer reads must still execute the existing TensorMap/owner dependency path even when the current scope is manual

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

Outer tensors read inside manual scope must still seed internal dependencies from existing producer state through TensorMap/owner logic.

Otherwise:

- a task inside manual scope may run before the outer producer of its input
- explicit edges inside the scope are insufficient to protect the outer-to-inner boundary

So manual mode disables only same-scope auto-derivation, not boundary seeding.

This is a strict requirement:

- outer read boundary dependency is forced by TensorMap/owner metadata
- orchestration code inside the manual scope must not be required to recreate that outer dependency manually

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

Required error text quality:

- the message must explicitly say that `manual scope inside manual scope is not supported`
- the message must identify the offending operation as nested `PTO_SCOPE(manual_dep=1)`
- the message must not use vague wording such as only `invalid scope state`

## Diagnostics

The runtime should detect and report:

1. nested manual scope not supported, with an explicit error message
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

3. Failing to force outer-scope reads through TensorMap/owner dependency seeding.
- This allows manual-scope tasks to read before the outer producer frontier is ready.

4. Letting cross-scope writes publish only at `scope_end`.
- This delays readiness incorrectly.

5. Accidentally preserving creator retention for same-scope tensors in manual mode.
- This reintroduces hidden dependencies and weakens the mental model.

6. Missing alias/view inheritance of scope ownership.
- This causes wrong same-scope vs cross-scope classification.

7. Turning this feature into a broad runtime refactor.
- This increases regression risk and violates the required change scope.

## Recommended Implementation Order

1. Add API surface for `add_dependency` and manual scope mode.
2. Add scope-frame mode and `scope_id`.
3. Add tensor ownership metadata needed for classification.
4. Implement explicit edge wiring in tensormap runtime.
5. Refactor submit-time dependency logic to branch on:
   - current scope mode
   - tensor owner task id
   - producer task scope provenance
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
