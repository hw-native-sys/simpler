# Manual Dependency For TensorMap Runtime

## Goal

Bring the human-created dependency workflow from `aicpu_build_graph` into `tensormap_and_ringbuffer` in a scoped way:

- `PTO2_SCOPE(PTO2ScopeMode::MANUAL) { ... }`
- Tensors crossing scope boundaries use TensorMap semantics
- Tensors used entirely inside the manual scope use explicit `add_dependency`

This is not a port of `aicpu_build_graph`'s fully-explicit runtime model. The target is a hybrid model inside `tensormap_and_ringbuffer`:

- same-scope dependency tracking: explicit
- cross-scope dependency tracking: TensorMap
- scope-local lifetime management: unchanged ring/scope ownership model

## Code-Checked Baseline

This draft is reviewed against the current implementations in:

- `src/a2a3/runtime/aicpu_build_graph/runtime/pto_orchestrator.{h,cpp}`
- `src/a2a3/runtime/aicpu_build_graph/orchestration/pto_orchestration_api.h`
- `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.{h,cpp}`
- `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_runtime2.{h,cpp}`
- `src/a2a3/runtime/tensormap_and_ringbuffer/orchestration/pto_orchestration_api.h`
- `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/tensor.h`

The important current-code facts are:

- `aicpu_build_graph` already has explicit `add_dependency`, returns `SubmitResult { task_id, outputs }`, and batch-publishes tasks at `scope_end`.
- `tensormap_and_ringbuffer` currently derives dependencies during submit, returns only `TaskOutputTensors`, and uses `scope_end` only for lifetime release.
- `tensormap_and_ringbuffer` orchestration is TLS-based today: `PTO2_SCOPE()` and `pto2_rt_submit_*()` do not take an explicit `PTO2Runtime*`.
- In `tensormap_and_ringbuffer`, `Tensor::manual_dep` is creator-retention-only mode: it skips OverlapMap lookup/insert, but `owner_task_id` retention still applies.

## Confirmed Decisions

These decisions are already aligned with the requested direction:

1. `tensormap` scope may contain a manual scope.
2. Manual scope may not contain another manual scope.
3. The design must not simplify away multi-write cases.
4. For an outer-scope tensor written inside a manual scope, readiness is the writer task completion time, not `scope_end`.
5. Therefore, a task inside a manual scope that writes an outer-scope tensor must still publish that tensor frontier through TensorMap before later submissions depend on it.
6. For an outer-scope tensor read inside a manual scope, the dependency must still be forced by TensorMap/owner-based boundary seeding during manual submit.
7. Tasks created inside a manual scope should be batch-published to the scheduler at `scope_end`, matching `aicpu_build_graph` semantics for explicit dependency closure inside the scope.

## Change Control Requirements

The implementation PR must follow these rules:

- Keep the change strictly scoped to manual dependency support in `tensormap_and_ringbuffer`.
- Do not refactor unrelated runtime behavior while doing this work.
- Do not change existing auto-scope TensorMap semantics.
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

There is already a `Tensor::manual_dep` bit, but in current code it is only a per-tensor creator-retention mode: it skips TensorMap overlap lookup/insert while still keeping `owner_task_id` retention. That is not sufficient for scoped hybrid semantics because the scope, not the tensor alone, decides whether a use is same-scope or cross-scope.

## Discovery vs Execution Separation

This distinction is central to the frozen design.

TensorMap is not the execution-time dependency engine. It is only a producer-discovery mechanism.

The scheduler's fanin/fanout graph is the execution-time dependency engine.

In current `tensormap_and_ringbuffer`, submit does two different things:

1. Discover producers from tensors.
- creator retention from `owner_task_id` in `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp`
- overlap lookup from TensorMap in `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp`

2. Convert discovered producers into scheduler edges.
- accumulate unique producer slot states in local `fanin_states[]`
- wire producer `fanout_head`, consumer `fanin_count`, and consumer `fanin_refcount`

After that conversion, execution no longer cares how the dependency was found.

The scheduler only sees:

- producer fanout list
- consumer fanin counters

This is why manual dependency integration should work as follows:

- do not put manual dependencies into TensorMap
- do not bind manual dependencies to tensors
- at manual `scope_end`, realize manual dependencies directly as normal producer-consumer scheduler edges

So for a task inside manual scope:

1. Start a local dedup buffer such as `fanin_states[]`.
2. During submit, add producers from outer-tensor creator retention and TensorMap lookup.
3. Cache that deduped external producer set in the task payload.
4. At `scope_end`, add producers from recorded manual edges.
5. Dedup both sources together before wiring the scheduler edges.
6. Run the normal wiring path into:
   - `payload->fanin_slot_states[]`
   - `fanin_count`
   - producer `fanout_head`

Then after publish:

- manual deps and TensorMap-derived deps are indistinguishable
- both are handled by the existing scheduler readiness and completion fanout path

Concise conclusion:

- TensorMap discovers boundary tensor-related dependencies during manual submit
- manual deps bypass discovery and are replayed only at manual `scope_end`
- both become the same scheduler edges before publish
- execution uses only the scheduler edge machinery, not TensorMap

## Implemented Manual-Scope Algorithm

The current implementation is a submit/scope-end split:

- manual submit still allocates task ids, task slots, and payloads immediately
- manual submit still does boundary producer discovery for cross-scope tensors
- manual submit still updates TensorMap frontier for cross-scope writes
- manual submit does not publish tasks to the scheduler ready graph
- manual `scope_end` replays only explicit same-scope edges plus cached external fanins, then batch-publishes tasks

### How tensor arguments are handled

The runtime decision is per tensor argument, not per scope:

| Tensor use in a manual-scope task | How dependency is found | Uses TensorMap? | What must be maintained |
| --- | --- | --- | --- |
| tensor created in the current manual scope, then reused in the current manual scope | explicit `add_dependency` only | no | recorded manual edge in scope-local edge buffer |
| outer/external `INPUT` | creator retention plus overlap lookup | yes, at manual submit unless `manual_dep=true` | cached external producer set in task payload |
| outer/external `INOUT` | creator retention plus overlap lookup for incoming state | yes, at manual submit unless `manual_dep=true` | cached external producer set plus updated writer frontier |
| outer/external `OUTPUT_EXISTING` | creator retention only for incoming owner, no overlap lookup | yes for outgoing frontier update unless `manual_dep=true` | updated writer frontier |
| runtime-created `OUTPUT` inside manual scope | no incoming dependency | no immediate lookup | `owner_task_id` on produced tensor so later users can classify it as manual-local |

`TensorArgType` matters here:

- `INPUT`: read-only; needs incoming producer discovery, no outgoing frontier update
- `INOUT`: read old value and write new value; needs both incoming producer discovery and outgoing frontier update
- `OUTPUT_EXISTING`: overwrite an existing outer buffer; does not need overlap lookup for an old modifier chain, but still needs outgoing frontier update
- `OUTPUT`: fresh runtime allocation; has no incoming dependency and becomes manual-local to later tasks in the same manual scope

### What manual submit iterates

For each submitted task in a manual scope, the runtime iterates tensor args in submit order:

1. Allocate task id, slot state, and payload immediately.
2. For each tensor arg, classify it as manual-local or outer/external from `owner_task_id` plus current manual-scope ownership.
3. For manual-local tensors:
   - skip creator-retention wiring
   - skip TensorMap lookup/insert
   - rely on explicit `add_dependency`
4. For outer/external tensors:
   - keep creator-retention from `owner_task_id`
   - run TensorMap overlap lookup for `INPUT` and `INOUT` unless `manual_dep=true`
   - cache the deduped external producer set in the task payload
   - update TensorMap frontier for outer writes in original submit order unless `manual_dep=true`
5. Leave the task unpublished behind one deferred publish barrier.

### What manual scope_end iterates

At manual `scope_end`, the runtime iterates tasks in the current manual scope in original submit order:

1. Read the cached external producer set from each task payload.
2. Replay explicit same-scope edges recorded by `add_dependency`.
3. Merge and dedup:
   - cached external producers
   - explicit manual producers
4. Realize the final producer set into the normal scheduler fanin/fanout structures.
5. Release the deferred publish barrier and batch-publish the manual-scope tasks.
6. Release the usual scope-held lifetime reference.

This is why manual `scope_end` is still expensive on non-unroll paged attention:

- it walks every manual-scope task
- it merges cached external fanins with explicit same-scope edges
- it mutates scheduler fanin/fanout state in one serial finalize step

### What state manual scope maintains

The runtime keeps a small amount of scope-local metadata instead of a second execution engine:

- `scope_tasks[]`: task order for the current scope
- `manual_task_meta[]`: per-task metadata for manual finalize
- `manual_edges[]`: explicit same-scope producer-consumer edges
- `payload->fanin_slot_states[]`: cached external producers discovered at manual submit
- `fanin_count` plus one deferred publish barrier

This split was chosen because it preserves the normal scheduler after publish:

- no second execution-time dependency engine
- no change to the ready queue model
- no change to worker dispatch or completion handling
- only boundary discovery stays on the TensorMap path
- only same-scope replay is deferred to manual `scope_end`

## Problem Statement

If we simply copy `aicpu_build_graph` semantics into `tensormap_and_ringbuffer`, we get a wrong boundary model:

- suppressing TensorMap for all tensors inside `PTO2_SCOPE(PTO2ScopeMode::MANUAL)` is incorrect
- delaying publication of an outer tensor until `scope_end` is incorrect

The reason is that cross-scope tensors must become visible at the actual writer frontier. Outside consumers should depend on the task that really produced the latest visible state, not on scope closure.

So the correct split is:

- same-scope tensor relations inside the manual scope: explicit edges only
- cross-scope tensor relations: preserve TensorMap behavior

## Required Semantics

## Core rule

`PTO2_SCOPE(PTO2ScopeMode::MANUAL)` means:

- if a tensor was created inside this manual scope and is reused inside this manual scope, the dependency must be established by explicit `add_dependency`
- all outer-scope tensors still use existing TensorMap/owner metadata
- tasks submitted inside the manual scope remain invisible to the scheduler until `scope_end`

This rule applies per tensor use site, not as a global on/off switch for the whole submit.

## Two Different Publication Semantics

The design must distinguish two different kinds of publication:

1. Scheduler publication
2. TensorMap boundary publication

### Scheduler publication

For tasks inside `PTO2_SCOPE(PTO2ScopeMode::MANUAL)`:

- submit builds the internal task records and records explicit dependencies
- those tasks are not yet published as executable scheduler work
- `scope_end` batch-publishes them to the scheduler

This is required so all same-scope explicit edges are fully wired before any task in the manual scope can start execution.

### TensorMap boundary publication

For cross-scope tensors touched by tasks inside `PTO2_SCOPE(PTO2ScopeMode::MANUAL)`:

- outside tasks submitted after a writer task is submitted must still be able to discover that writer frontier through TensorMap
- therefore the producer frontier for an external tensor written inside the manual scope must be updated during manual submit
- however the tensor is still not semantically ready until that producer task actually completes

So:

- scheduler visibility of the task is controlled by manual `scope_end`
- dependency readiness for later consumers is still enforced by waiting on producer task completion

The document must not conflate these two mechanisms.

More precisely:

- before manual `scope_end`, the task record already exists and TensorMap boundary discovery/publication has already happened
- before manual `scope_end`, the task is still invisible to the executable scheduler graph
- after manual `scope_end`, the task becomes part of the executable published graph
- once published, the task may enter `READY` immediately or remain `PENDING` depending on whether its dependencies are already satisfied

## Discussion Guardrails

The following clarifications are recorded to reduce implementation drift and hallucination risk:

1. Deferred publish does not mean deferred task allocation.
- Manual tasks still allocate task ids, slot state, and payload at submit time.
- What is deferred is explicit same-scope edge realization and ready-queue publication.

2. Manual dependencies are not tracked by TensorMap.
- TensorMap is only used for tensor-related producer discovery.
- Manual dependencies are explicit producer-consumer edges recorded by orchestration.
- At manual `scope_end`, both kinds of dependencies are converted into the same scheduler fanin/fanout structures.

3. After manual `scope_end`, there is no special execution-time manual mechanism.
- The runtime should not keep a second dependency engine alive after publish.
- Once the scope is finalized, all dependencies are handled only by the existing scheduler fanin/fanout path.

4. Submit-time boundary wiring does not change tensor readiness semantics.
- Outer writes become TensorMap-visible at manual submit.
- Their semantic readiness is still producer-task completion.

## Tensor categories

For a task submitted inside a manual scope, every tensor argument falls into one of these categories:

1. Outer-scope tensor, read only
2. Outer-scope tensor, written in place
3. Tensor created inside this manual scope, used again inside this manual scope
4. Outer-scope tensor accessed through a derived view/reshape/transpose inside the manual scope
5. External tensor with no owner task

The runtime only needs one special classification for v1:

- tensor created in the current manual scope

Everything else stays on the existing TensorMap path.

## Expected behavior by category

### 1. Outer-scope tensor, read only

- The first internal consumer must still get its dependency from TensorMap/owner-based boundary seeding.
- This is not optional and must not be delegated to explicit manual edges inside the scope.
- Manual scope does not remove the need to wait for the outer producer frontier.
- In other words, outer-read boundary correctness is still forced by TensorMap-side logic.

### 2. Outer-scope tensor, written in place

- The internal writer must still publish its producer frontier for TensorMap boundary tracking.
- That boundary frontier must become visible at manual submit, in original submit order, so later submissions can attach to the correct writer task immediately.
- Readiness of the written tensor is the completion of that writer task.
- Multiple writes inside the same manual scope are allowed.
- TensorMap should continue tracking the latest producer frontier exactly as in auto scope while the scope is still unpublished to the scheduler.

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

Keep the existing `tensormap_and_ringbuffer` orchestration style: TLS-based helpers with no explicit runtime argument in user orchestration code. Do not make the public surface look like `aicpu_build_graph`'s `PTO2_SCOPE(rt)` family just to add manual mode.

Add explicit edge wiring to `tensormap_and_ringbuffer` orchestration API:

```cpp
void pto2_rt_add_dependency(PTO2TaskId producer, PTO2TaskId consumer);
```

Use an explicit scope mode enum for the scoped API:

```cpp
enum class PTO2ScopeMode : uint8_t {
    AUTO = 0,
    MANUAL = 1,
};

PTO2_SCOPE(PTO2ScopeMode::MANUAL) {
    ...
}
```

`PTO2_SCOPE()` remains the auto-scope form by default. `PTO2_SCOPE(PTO2ScopeMode::MANUAL)` enters manual mode explicitly.

Do not change `TaskOutputTensors`.

Add new manual-submit APIs with `_manual` suffix so orchestration code can get task ids without changing existing normal submit call sites. This mirrors the role of `aicpu_build_graph`'s `SubmitResult`, but keeps the existing `tensormap_and_ringbuffer` submit APIs intact:

```cpp
struct PTO2ManualSubmitResult {
    PTO2TaskId task_id;
    TaskOutputTensors outputs;
};

PTO2ManualSubmitResult pto2_rt_submit_task_manual(const MixedKernels& mixed_kernels, const Arg& args);
PTO2ManualSubmitResult pto2_rt_submit_aic_task_manual(int32_t kernel_id, const Arg& args);
PTO2ManualSubmitResult pto2_rt_submit_aiv_task_manual(int32_t kernel_id, const Arg& args);
```

These APIs are intended for use inside `PTO2_SCOPE(PTO2ScopeMode::MANUAL)` where explicit dependency wiring is required.

This design intentionally splits task APIs, not tensor storage APIs:

- auto scope uses existing `pto2_rt_submit_*` task APIs
- manual scope uses `pto2_rt_submit_*_manual` task APIs
- both modes continue using the same `Tensor`, `Arg`, and `TensorArgType` model

So manual mode changes how tasks are recorded and finalized, not how tensors are represented.

## Rejected API Alternatives

The following alternatives were considered and rejected for v1:

1. Add a new user-facing “external tensor” API for manual scope.
- Rejected because manual mode only needs to identify manual-local tensors.
- Everything else can be treated as outer/external and handled by the existing TensorMap path.
- Adding a second tensor annotation API would increase surface area without adding necessary information.

2. Change `TaskOutputTensors` to carry task ids.
- Rejected to avoid broad churn in existing orchestration code.
- Manual mode gets separate `_manual` submit APIs instead.

3. Create a second tensor representation for manual mode.
- Rejected because payload already stores the copied tensor/scalar data needed for deferred finalize.
- The task API split is enough; tensor storage stays unified.

## Runtime API

Add runtime ops support:

```cpp
void (*add_dependency)(PTO2Runtime* rt, PTO2TaskId producer, PTO2TaskId consumer);
void (*scope_begin)(PTO2Runtime* rt, PTO2ScopeMode mode);
```

The orchestration-facing helper can stay TLS-style and hide the runtime pointer, for example by plumbing the mode through the existing `pto2_rt_scope_begin()` / `PTO2ScopeGuard` path.

Add manual-scope entry/exit plumbing by extending the existing runtime entry point with a mode flag:

```cpp
void pto2_rt_scope_begin(PTO2Runtime* rt, PTO2ScopeMode mode = PTO2ScopeMode::AUTO);
```

Recommendation: extend scope state with a mode flag and keep one scope stack. Avoid separate manual/non-manual stacks.

## Internal Design

## Scope state

Each scope frame needs:

- `begin_index` into `scope_tasks`
- scope mode: normal or manual
- `begin_index` into a manual-edge buffer when the scope is manual
- `begin_index` into a manual-task-meta buffer when the scope is manual

Manual scope needs a local edge registry because `add_dependency` should record edges during orchestration but should not mutate scheduler fanin/fanout state until manual `scope_end`.

Manual scope also needs a compact per-task metadata stream so `scope_end` can replay the deferred dependency logic without copying full `Arg` objects.

## Tensor metadata

Current `Tensor` already stores:

- `owner_task_id`
- `manual_dep`

Recommendation: do not add new tensor metadata in v1.

The narrowed v1 rule only needs to identify tensors created in the current manual scope. That can be derived from:

- `tensor.owner_task_id`
- the set of task ids created in the current manual scope

So the preferred approach is:

- keep `Tensor` layout unchanged
- keep `owner_task_id` as the provenance pointer
- track the current manual scope's owned task membership in scope-local orchestrator state

`manual_dep` should not become the primary mechanism for scoped semantics. It may remain as a per-tensor escape hatch for existing behavior, but the manual-scope design should be driven by:

- current scope mode
- tensor owner task id
- whether that owner belongs to the current manual scope

## Shared Tensor Path, Split Task APIs

The design should keep one shared tensor recording path across normal and manual scope:

- `Arg` remains the user-facing container for tensor refs, tensor create-info, scalars, and `TensorArgType`
- `PTO2TaskPayload` remains the destination for copied tensors and scalars
- runtime-created outputs still receive `owner_task_id` during submit

What changes in manual scope is only the task API and the time when dependency logic runs:

- normal submit APIs perform dependency lookup and TensorMap insert immediately
- manual submit APIs only allocate the task, copy payload data, and record compact finalize metadata
- manual `scope_end` replays only explicit same-scope edges and combines them with the external producer set already cached at submit

This keeps normal-mode APIs unchanged while avoiding a second tensor representation for manual mode.

## Classification rule

In manual scope, the runtime only needs one special classification:

```cpp
manual_local_tensor =
    tensor.owner_task_id.is_valid() &&
    current_manual_scope_owns(tensor.owner_task_id);
```

Then:

```cpp
if (!in_manual_scope) {
    use existing tensormap behavior;
} else if (manual_local_tensor) {
    use explicit add_dependency only;
} else {
    use existing TensorMap/owner behavior;
}
```

Important nuance:

- tensors created in the current manual scope use explicit same-scope dependencies
- all outer tensors stay on the existing TensorMap path, even if two tasks inside the manual scope both access them
- this means outer tensors may still create implicit same-scope edges through TensorMap inside a manual scope
- this is an accepted v1 decision and should be documented in the PR description as a deliberate tradeoff

This is why a separate user-facing “external tensor” API is not required for v1:

- manual mode only needs to identify manual-local tensors
- everything else is treated as outer/external and goes through the existing TensorMap path
- that decision can be derived from `owner_task_id` plus the current manual scope's owned-task membership check

## Scheduler-Safe Hybrid Design

The scheduler changes should be localized and should not disturb existing auto-scope behavior.

### Design principle

Keep two execution paths:

- auto scope path: existing `tensormap_and_ringbuffer` behavior
- manual scope path: deferred dependency realization and deferred scheduler publication

The auto path should remain unchanged as much as possible.

### What a manual-scope task must count as dependencies

For a task inside `PTO2_SCOPE(PTO2ScopeMode::MANUAL)`, total fanin is:

- explicit manual dependencies added by `add_dependency`
- external dependencies derived from TensorMap/owner logic for outer-scope reads
- one extra publish barrier released only at manual `scope_end`

In other words:

```cpp
fanin_count =
    manual_dep_edges +
    external_tensor_deps +
    1;  // publish barrier
```

This is the key mechanism that lets the scheduler ignore manual-local TensorMap lookup while still respecting out-of-scope dependencies.

### What submit should do in manual scope

For a task submitted inside manual scope:

1. Allocate slot and payload exactly as today.
2. Initialize `task_state = PENDING`.
3. Initialize `fanin_count = 1` and `fanin_refcount = 0` for deferred publication.
4. Return a stable `task_id` immediately so orchestration can call `add_dependency`.
5. Do not realize explicit manual edges into scheduler fanin/fanout yet.
6. Realize outer-boundary producer discovery immediately:
   - creator retention from `owner_task_id`
   - TensorMap lookup for outer `INPUT` / `INOUT`
   - covered-entry removal for outer `INOUT`
7. Publish outer writes to TensorMap immediately for outer `INOUT` / `OUTPUT_EXISTING`.
8. Do not push the task into ready queues during submit.
9. Retain every cached external producer strongly enough that it cannot be reclaimed or slot-reused before manual `scope_end`.
10. Cache the deduped external producer set in the task payload so manual `scope_end` can realize the scheduler edges without touching TensorMap.
11. Preserve enough scope-local information so manual `scope_end` can realize explicit same-scope edges before publish.

Submit-time task records are still required even though execution is deferred:

- manual submit APIs must return stable task ids immediately
- runtime-created outputs need `owner_task_id` immediately so later scope-local tensors and their derived views can be recognized
- the scheduler only sees these tasks after manual `scope_end`

Manual mode should also record a compact per-task finalize descriptor rather than a second full copy of `Arg`.

Recommended shape:

```cpp
struct PTO2ManualTaskMeta {
    uint64_t packed_tags;   // compact encoding of TensorArgType for this task
    uint16_t tensor_count;
    uint16_t edge_begin;    // range in manual_edges[]
    uint16_t edge_count;
    uint16_t _pad;
};

struct PTO2ManualEdge {
    uint16_t producer_idx;  // index in current manual scope's task slice
    uint16_t consumer_idx;
};
```

Why this is low-overhead:

- tensor values are already copied into `PTO2TaskPayload`
- scalars are already copied into `PTO2TaskPayload`
- tags are much smaller than copying `Arg` again
- the edge list is dense, append-only, and scope-local

The design should prefer a packed tag stream plus a dense edge stream over storing duplicated tensor refs or explicit user-marked external tensors.

That gives a manual pre-publish state:

- task records and task ids already exist
- explicit edges are only recorded, not yet wired into scheduler fanin/fanout
- external TensorMap-derived producers are already discovered and cached
- cached external producers are retained so deferred publish cannot attach to a reused slot
- outer writes are already reflected in TensorMap frontier state
- the task is still unpublished as executable scheduler work because the publish barrier is not yet released

### What scope_end should do in manual scope

Manual `scope_end` needs one additional finalize-and-publish step before the existing lifetime-release step completes.

Recommended sequence:

1. For every task directly owned by this manual scope:
   - realize recorded explicit `add_dependency` edges into scheduler fanin/fanout state
   - start from the external producer set cached during submit
   - dedup explicit same-scope edges against those cached external producers
   - realize the final deduped producer set into scheduler fanin/fanout state
2. After all dependency realization is complete for the scope:
   - release the publish barrier by incrementing `fanin_refcount`
   - if `fanin_refcount == fanin_count`, transition to `READY` and push to ready queue
   - otherwise keep the task in published `PENDING` state so later producer completion can resolve it
3. Release the scope lifetime reference exactly as current `on_scope_end` does

This can be implemented as a manual-scope finalize path in the orchestrator plus a small scheduler helper for the publish-barrier release.

Example helper shape:

```cpp
void publish_manual_scope_tasks(PTO2TaskSlotState** task_slot_states, int32_t count);
```

This helper should reuse the existing ready-transition logic as much as possible.

### How external dependency replay works

Manual submit should discover external dependencies in original submit order, using:

- `scope_tasks[]` for task order
- `manual_task_meta[]` for packed tags and edge ranges
- `PTO2TaskPayload::tensors[]` for actual tensor values

For each task in that order during submit:

1. Decode tensor tags from `packed_tags`.
2. For each tensor arg:
   - if `owner_task_id` belongs to the current manual scope's owned task set, treat it as manual-local and skip TensorMap logic
   - otherwise treat it as outer/external
3. For outer/external tensors:
   - apply creator-retention logic from `owner_task_id`
   - apply existing TensorMap overlap lookup for `INPUT` / `INOUT`
4. Cache the deduped external producer set in the task payload.
5. After lookup for this task:
   - apply normal TensorMap insertion for outer writes (`INOUT` / `OUTPUT_EXISTING`)

This submit order matters:

- it preserves current tensormap behavior for multiple writes to outer tensors
- earlier outer writes from the same manual scope become visible to later tasks in the same manual scope during submit
- that matches the accepted v1 tradeoff that outer tensors may still induce implicit same-scope TensorMap edges
- it requires the same TensorMap validity synchronization that normal auto submit uses before lookup/insert

The split must not be implemented as:

- deferring all lookups and inserts to `scope_end`
- wiring scheduler fanout during manual submit
- counting cached external producers and explicit manual edges independently without one dedup pass at publish time

Those variants would diverge from current tensormap semantics and are considered incorrect for this design.

### Important case: external dependency already produced before manual publish

For a manual-scope task that reads an outer-scope tensor:

- if the external producer task has already completed when scheduler realization happens at manual `scope_end`, that edge should immediately contribute to `fanin_refcount`
- then manual `scope_end` releases only the publish barrier, and the task may become `READY` immediately

If the external producer has only published its TensorMap frontier but not yet completed:

- the manual-scope consumer has already cached that producer at submit time and is published at manual `scope_end`
- but it remains in published `PENDING`
- later producer completion notifies fanout and increments `fanin_refcount`
- once `fanin_refcount == fanin_count`, the consumer transitions to `READY`

This is the desired hybrid behavior:

- explicit same-scope dependency replay happens at manual `scope_end`, before publish
- cross-scope dependency discovery already happened at manual submit
- dependency satisfaction is still handled by the normal runtime execution path after publish

### Why this is low-risk

- no change to ready queue implementation
- no change to worker dispatch loop
- no change to normal TensorMap scope behavior
- no need for a new scheduler task state
- reuse the existing `fanin_count` / `fanin_refcount` / `PENDING -> READY` transition model

The main new behavior is submit-time boundary discovery plus deferred release of explicit same-scope publish for manual-scope tasks.

## Current-Manual-Scope Ownership Without Tensor Changes

To decide whether a tensor is manual-local or outer-visible, the orchestrator only needs to know whether its `owner_task_id` belongs to the current manual scope.

Recommended minimal design:

- keep `Tensor` unchanged
- use `Tensor.owner_task_id` as the provenance link
- keep a scope-local registry of task ids created in the current manual scope

A good low-risk implementation is to reuse the existing flat `scope_tasks` buffer plus a parallel manual-edge buffer, rather than widening hot structs unnecessarily.

Classification then becomes:

```cpp
if (!tensor.owner_task_id.is_valid()) {
    // external tensor with no producer task
} else {
    manual_local = current_manual_scope_owns(tensor.owner_task_id);
}
```

## Lifecycle Clarification

This design needs precise task-lifecycle terms:

- `COMPLETED`: task execution has finished; produced tensor data is semantically ready
- `CONSUMED` / reclaimed: all fanout references and the owning-scope reference have been released, so the task slot may be reused and `last_task_alive` may advance
- tensor readiness: data-level concept, typically tied to producer task completion

This matters for deferred manual `scope_end` wiring:

- an outer-scope producer task may already be `COMPLETED` before the inner manual scope ends
- that is fine, and the manual finalize path should treat it as an already-satisfied dependency
- what must remain true is that the producer task has not yet reached the reclaimed / slot-reusable state before the inner manual `scope_end`

Why this is expected to hold:

- tasks created in the current manual scope are still protected by the current manual scope reference until manual `scope_end`
- tasks created in an outer still-active scope may complete early, but the outer scope still holds their scope reference until that outer scope ends
- therefore an inner manual scope can still rely on the producer state already discovered and retained during manual submit when it finalizes

This does not mean the producer task is still runnable or incomplete.
It may already be `COMPLETED`; the manual finalize path should then treat it as an already-satisfied dependency.

So the safety argument is not "outer producers cannot complete early". The correct statement is:

- outer producers may complete before inner manual `scope_end`
- they should still remain alive enough to be discoverable until the deferred boundary wiring for that inner manual scope has finished

## External Dependency Publication In Manual Scope

The spec needs two explicit rules here.

### External reads

A task inside manual scope that reads an outer-scope tensor:

- must still collect the external producer through TensorMap/owner logic
- must cache that dependency during manual submit
- must include that cached dependency in its fanin during manual `scope_end`, before manual batch publish
- must not require the user to restate that outer dependency manually

### External writes

A task inside manual scope that writes an outer-scope tensor:

- must publish its producer frontier to TensorMap during manual submit
- must not publish same-scope temporary tensors into TensorMap
- may still be `PENDING` and unpublished to the scheduler until manual `scope_end`

This is safe because later outside submissions only need to identify the producer task and wire dependency to it. Actual execution readiness is still controlled by task completion and the scheduler's normal completion path.

## Manual Dependencies And External Dependencies On The Same Task

A single task inside manual scope may simultaneously depend on:

- explicit same-scope manual producers
- external TensorMap-derived producers

This is supported by the same fanin accounting model.

Example:

```cpp
PTO2_SCOPE(PTO2ScopeMode::MANUAL) {
    t0 = in-scope producer of tmp
    t1 = consumer of tmp and outer tensor X
    add_dependency(t0, t1)
}
```

At manual `scope_end`, for `t1`:

- `t0 -> t1` contributes one explicit manual fanin edge
- outer tensor `X` contributes boundary-derived external fanin edges
- publish barrier contributes one extra deferred fanin unit

`t1` becomes READY only after:

- explicit in-scope producers complete
- external producers complete
- manual `scope_end` releases the publish barrier

That is the intended scheduler behavior.

## Explicit edge wiring

`pto2_add_dependency` from `aicpu_build_graph` can be reused conceptually, but manual scope should not wire scheduler fanin/fanout immediately.

Recommended behavior inside manual scope:

- validate that both task ids belong to the current manual scope
- record the edge in a scope-local manual-edge buffer
- do not increment `fanin_count` yet
- do not mutate producer `fanout_head` yet

Then at manual `scope_end`:

- realize each recorded edge into the scheduler's existing fanin/fanout structures
- increment `fanin_count`
- record producer in consumer payload for release traversal
- handle the already-completed producer case exactly once, during realization

This avoids racing live external completion against partially built manual dependency state.

Important discussion note:

- the deduped producer set for one consumer must include all sources together:
  - explicit manual edges
  - creator retention from `owner_task_id`
  - TensorMap overlap lookup

The implementation must not count these sources independently and then wire fanout multiple times for the same producer-consumer pair.

## Scope-end behavior

Manual scope changes scheduler publication semantics for tasks inside that scope:

- tasks in manual scope are batch-published to the scheduler at `scope_end`
- same-scope explicit edges must be fully wired before that publish happens

Manual scope does not change lifetime release semantics:

- `scope_end` still releases the owning-scope fanout reference

Manual scope also does not change cross-scope readiness semantics:

- external tensor readiness is still producer-task completion, not `scope_end`
- but external-writer frontier information must already be visible to later TensorMap lookups at manual submit

This manual-scope behavior intentionally combines:

- `aicpu_build_graph`-style scope-end batch publish for explicit same-scope dependencies
- `tensormap_and_ringbuffer`-style TensorMap boundary tracking for cross-scope tensors

## Multiple Writes To Outer Tensors

This case must be supported in v1.

Example:

```cpp
PTO2_SCOPE(PTO2ScopeMode::MANUAL) {
    t1 writes outer C
    t2 writes outer C
    add_dependency(t1, t2)
}
outside task reads C
```

Correct behavior:

- at manual submit, `t1` publishes `C` to TensorMap
- at manual submit, `t2` publishes `C` again to TensorMap
- outside reader should see `t2` as the latest producer frontier
- because `t1 -> t2` is explicit, `t2` completion is a valid readiness frontier for the final visible state
- outer tensors may still create implicit same-scope TensorMap edges inside the manual scope; this is an accepted v1 tradeoff and should be called out in the PR description

Potential invalid user pattern:

```cpp
PTO2_SCOPE(PTO2ScopeMode::MANUAL) {
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
- even though the consumer task itself is only batch-published to the scheduler at manual `scope_end`, its fanin accounting must include the external TensorMap-derived dependency discovered at submit time

## Nesting Rules

Supported:

- auto scope contains manual scope
- auto scope contains auto scope

Not supported in v1:

- manual scope contains manual scope
- manual scope contains any nested scope with its own publish boundary

Reason:

- current ring selection depends on scope depth
- the top scope frame is also the publication and lifetime-release boundary
- allowing a child scope inside `PTO2_SCOPE(PTO2ScopeMode::MANUAL)` would split one manual region across multiple scope/ring boundaries unless extra machinery is added
- rejecting nested scopes inside manual mode keeps `current_manual_scope_owns(...)` a simple membership check over one manual frame

Recommendation:

- detect this at `scope_begin`
- fail fast with a clear orchestrator error

Required error text quality:

- the message must explicitly say that nested scope inside `PTO2_SCOPE(PTO2ScopeMode::MANUAL)` is not supported in v1
- the message must explicitly say that `manual scope inside manual scope is not supported`
- the message must identify the offending operation as nested `PTO2_SCOPE(PTO2ScopeMode::MANUAL)`
- the message must not use vague wording such as only `invalid scope state`

## Blocking Cross-Layer Tensor Access

`get_tensor_data` and `set_tensor_data` are blocking cross-layer access APIs. Their current contract assumes producer state is already published through TensorMap/owner metadata.

That assumption does not hold inside manual scope because tasks remain unpublished until manual `scope_end`.

So v1 should fail fast:

- `get_tensor_data` inside `PTO2_SCOPE(PTO2ScopeMode::MANUAL)` is an error
- `set_tensor_data` inside `PTO2_SCOPE(PTO2ScopeMode::MANUAL)` is an error

Required error text quality:

- the message must explicitly say that blocking tensor data access is not supported inside `PTO2_SCOPE(PTO2ScopeMode::MANUAL)`
- the message should tell the user to exit the manual scope first

## Diagnostics

The runtime should detect and report:

1. nested scope inside manual mode not supported, with an explicit error message
2. `add_dependency` used with invalid task ids
3. dependency overflow from explicit wiring
4. `get_tensor_data` or `set_tensor_data` called inside manual scope

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
- accepted implicit TensorMap edges on outer tensors are documented

5. Normal scope containing manual scope
- outer to inner and inner to outer boundary cases both work

6. Nested scope inside manual mode
- rejected with deterministic error

### Example-level migration

Use a small example first, such as vector-style or BGEMM-style, to validate:

- scope-local temp tensors use explicit edges
- outer tensors still behave through TensorMap

Only then move to more complex orchestration such as paged attention.

## Fresh Hardware Benchmark

Fresh benchmark data was rerun on real hardware on 2026-04-08 with:

- platform: `a2a3`
- device: `3`
- rounds: `10`
- pinned PTO-ISA commit: `6622890`
- runner: `tools/benchmark_rounds.sh`

The four compared variants are:

- `aicpu_build_graph`
- `tensormap_and_ringbuffer_unmodified`
- `tensormap_and_ringbuffer`
- `tensormap_and_ringbuffer_partial_manual`

`tensormap_and_ringbuffer` is the current/new AUTO-path runtime under evaluation.
`tensormap_and_ringbuffer_partial_manual` is the same runtime tree, but benchmarked
through the `_partial_manual` paged-attention scenes.

### Benchmark Script Selectors

The benchmark wrapper enables the variants as follows:

- `./tools/benchmark_rounds.sh -d 3 -n 10 -r aicpu_build_graph -c 6622890`
  - benchmarks `tests/st/a2a3/aicpu_build_graph/paged_attention`
  - benchmarks `tests/st/a2a3/aicpu_build_graph/paged_attention_unroll`
- `./tools/benchmark_rounds.sh -d 3 -n 10 -r tensormap_and_ringbuffer_unmodified -c 6622890`
  - benchmarks `tests/st/a2a3/tensormap_and_ringbuffer_unmodified/paged_attention`
  - benchmarks `tests/st/a2a3/tensormap_and_ringbuffer_unmodified/paged_attention_unroll`
- `./tools/benchmark_rounds.sh -d 3 -n 10 -r tensormap_and_ringbuffer -c 6622890`
  - benchmarks `tests/st/a2a3/tensormap_and_ringbuffer/paged_attention`
  - benchmarks `tests/st/a2a3/tensormap_and_ringbuffer/paged_attention_unroll`
- `./tools/benchmark_rounds.sh -d 3 -n 10 -r tensormap_and_ringbuffer_partial_manual -c 6622890`
  - uses the same ST root as `tensormap_and_ringbuffer`
  - benchmarks `tests/st/a2a3/tensormap_and_ringbuffer/paged_attention_partial_manual`
  - benchmarks `tests/st/a2a3/tensormap_and_ringbuffer/paged_attention_unroll_partial_manual`

There is no separate runtime named `partial_manual` in the example `kernel_config.py`.
The partial-manual scenes still declare `RUNTIME_CONFIG["runtime"] =
"tensormap_and_ringbuffer"`, and the benchmark wrapper switches to the
`*_partial_manual` scene directories when `-r tensormap_and_ringbuffer_partial_manual`
is selected.

Similarly, the current/new AUTO-path runtime is enabled directly by `-r
tensormap_and_ringbuffer`, while the copied side-by-side baseline is enabled by `-r
tensormap_and_ringbuffer_unmodified`.

### Fresh Results

Units below are `elapsed_us (orch_us)`.

| Workload | Case | `aicpu_build_graph` | `tensormap_and_ringbuffer_unmodified` | `tensormap_and_ringbuffer` | `tensormap_and_ringbuffer_partial_manual` |
| --- | --- | --- | --- | --- | --- |
| `paged_attention` | `Case1` | `31318.9 (-)` | `35367.3 (35366.7)` | `36996.3 (36995.6)` | `35187.6 (35030.2)` |
| `paged_attention` | `Case2` | `16844.5 (-)` | `19739.8 (19736.1)` | `19861.8 (19856.8)` | `18685.5 (18274.5)` |
| `paged_attention_unroll` | `Case1` | `1412.7 (-)` | `1321.7 (841.6)` | `1323.9 (831.3)` | `1321.3 (884.4)` |
| `paged_attention_unroll` | `Case2` | `705.5 (-)` | `628.1 (381.6)` | `632.5 (378.9)` | `637.5 (406.4)` |

### Feature-To-Gain Mapping

The most important question is which change actually moved performance.

The rerun below isolates the non-unroll partial-manual example on device `3`. All
three rows already include the same rewritten orchestration structure
(`one manual scope per q_idx`, `AIV_HUB` moved inside manual scope, and explicit
`prev_update_task -> up_outs.task_id` chaining). The only thing changing between rows
is boundary annotation.

| Optimization step | What changed | `Case1` orch delta | `Case2` orch delta | What it means |
| --- | --- | --- | --- | --- |
| Baseline after rewrite | no `manual_dep` boundary hints | baseline `36791.9 us` | baseline `19792.7 us` | structural rewrite alone is the starting point |
| Add input-side hints | `query`, `key_cache`, `value_cache`, and their views use `manual_dep=true` | `-39.6 us` (`-0.1%`) | `-496.0 us` (`-2.5%`) | minor benefit; input-side TensorMap work is not the main bottleneck |
| Add output-side hints on top | `out` and `out_view` also use `manual_dep=true` | `-1683.6 us` (`-4.6%`) | `-1628.5 us` (`-8.4%`) | major gain; repeated external-output overlap tracking was expensive |
| Full boundary hints vs no hints | inputs + output boundary hints together | `-1723.2 us` (`-4.7%`) | `-2124.5 us` (`-10.7%`) | this is the measurable win that was worth keeping |

Two conclusions matter:

1. The kept optimization is not “use `manual_dep` everywhere”.
   - The measurable gain came mostly from suppressing repeated external-output
     TensorMap work on `out` / `out_view`, where same-scope write ordering is already
     carried by explicit manual edges.
2. Input-side `manual_dep=true` alone is not enough.
   - It helps a little on `Case2`, but almost nothing on `Case1`.

### Benchmark Takeaways

1. The non-unroll target is still not met.
   - target cell: `paged_attention/Case1`
   - `aicpu_build_graph`: `31318.9 us`
   - `partial_manual`: `35187.6 us`
   - remaining gap: about `+12.4%`

2. Partial-manual now improves the modified/current tensormap AUTO runtime on both
   non-unroll paged-attention cases, but it does not beat `aicpu_build_graph`.
   - `paged_attention/Case1`: about `-4.9%` elapsed, about `-5.3%` orch vs current/new
   - `paged_attention/Case2`: about `-5.9%` elapsed, about `-8.0%` orch vs current/new

3. Partial-manual is not yet consistently better than the copied unmodified baseline.
   - `paged_attention/Case1`: about `-0.5%` elapsed, about `-1.0%` orch vs unmodified
   - `paged_attention/Case2`: about `+5.2%` elapsed, about `+7.8%` orch vs unmodified

4. The current/new AUTO runtime is still slower than the copied unmodified runtime on
   the non-unroll scene.
   - `paged_attention/Case1`: about `+4.6%` elapsed, about `+4.6%` orch
   - `paged_attention/Case2`: about `+0.6%` elapsed, about `+0.6%` orch

5. On the unroll scene, all three tensormap-family runtimes remain faster than
   `aicpu_build_graph` end-to-end, but partial-manual stays slightly worse than both
   AUTO tensormap variants in orch time.
   - `paged_attention_unroll/Case1`: `884.4 us` orch vs `841.6 us` unmodified and `831.3 us` current/new
   - `paged_attention_unroll/Case2`: `406.4 us` orch vs `381.6 us` unmodified and `378.9 us` current/new

6. The remaining performance problem is still concentrated in the non-unroll
   partial-manual path, especially the replay/publish cost paid at manual `scope_end`.

### Boundary Annotation Note

There is still no explicit “scope arguments” API in
`src/a2a3/runtime/tensormap_and_ringbuffer/orchestration/pto_orchestration_api.h`.

The closest current mechanism is per-tensor `manual_dep=true` on
`make_tensor_external(...)` and derived `view(...)` objects. That mechanism is not a
good substitute for scope-boundary declaration:

- it is tensor-local, not scope-local
- it suppresses TensorMap lookup/insert for that tensor
- if used carelessly, it can hide an output frontier that boundary semantics still need

For the committed non-unroll partial-manual paged-attention example, the stable
improvement came from two pieces together:

- the orchestration rewrite already present in the example:
  - one manual scope per `q_idx`
  - move `AIV_HUB` creation into the manual scope
  - add an explicit `prev_update_task -> up_outs.task_id` chain
- explicit `manual_dep=true` boundary hints on the external inputs and external output
  views in the example itself

Fresh device-3 measurements for the non-unroll partial-manual example were:

| Variant | `paged_attention/Case1` orch | `paged_attention/Case2` orch |
| --- | --- | --- |
| rewrite only, no boundary hints | `36791.9 us` | `19792.7 us` |
| rewrite + input-side hints | `36752.3 us` | `19296.7 us` |
| rewrite + input/output hints | `35068.7 us` | `17668.2 us` |

So the important hint is not just the external producers (`query`, `key_cache`,
`value_cache`), but also the external consumer path through `out` / `out_view`.
That is consistent with the current runtime behavior:

- `manual_dep=true` skips TensorMap overlap lookup/insert but still keeps creator
  retention through `owner_task_id`
- the explicit `prev_update_task` chain already serializes same-scope `ONLINE_UPDATE`
  writes
- marking `out` / `out_view` `manual_dep=true` avoids paying repeated external-output
  overlap tracking on every block update when that ordering is already explicit

Why this was kept even though `manual_dep` is not the core semantics:

- manual scope still uses TensorMap for general cross-scope correctness
- this example already has explicit same-scope write ordering for `ONLINE_UPDATE`
- there is no same-scope external consumer that needs `out` / `out_view` to stay on
  TensorMap before manual `scope_end`
- so suppressing repeated output overlap tracking is a valid example-level
  optimization, not a change to the runtime's semantic model

This is still not a general “scope arguments” API. It is an example-local optimization
that is only safe when the manual scope already carries the same-scope write ordering
explicitly and there is no same-scope external consumer that depends on TensorMap
publication before manual `scope_end`.

## Main Risks

1. Treating manual scope as a global TensorMap disable switch.
- This breaks cross-scope correctness.

2. Using `Tensor::manual_dep` as the only signal.
- Scoped semantics should be driven by current manual-scope ownership, not by the tensor flag alone.

3. Failing to force outer-scope reads through TensorMap/owner dependency seeding.
- This allows manual-scope tasks to read before the outer producer frontier is ready.

4. Confusing scheduler batch publication with tensor readiness semantics.
- Manual-scope tasks should be scheduler-visible at `scope_end`, but external tensor readiness is still producer completion.

5. Letting cross-scope writer frontier become visible only after producer completion.
- This is too late for later outside submissions made after manual `scope_end`.

6. Wiring external producers into scheduler fanout during manual submit.
- This can let unpublished tasks become runnable before `scope_end`.

7. Publishing external writer frontier later than manual submit.
- This makes later boundary lookups see stale producer state and diverges from current tensormap semantics for multiple writes.

8. Missing a final dedup pass between cached external producers and explicit manual edges.
- This double-counts fanin and can over-release dependencies.

9. Missing alias/view inheritance of scope ownership.
- This causes wrong same-scope vs cross-scope classification.

10. Turning this feature into a broad runtime refactor.
- This increases regression risk and violates the required change scope.

11. Allowing blocking cross-layer tensor access inside manual scope.
- `get_tensor_data` and `set_tensor_data` assume published producer state and should fail fast in manual scope.

12. Replacing the existing scheduler edge machinery with a separate manual execution path.
- This would duplicate fanin/fanout handling, completion notification, and release traversal.
- The design requires one unified post-publish scheduler mechanism.

13. Using `manual_dep=true` as a blanket scope-boundary annotation.
- This can suppress TensorMap work that is still required for cross-scope correctness.
- It is only safe as a narrowly-scoped example optimization when the same-scope
  ordering is already explicit and no early external consumer needs TensorMap
  publication for that tensor.

## Dangerous Risks For The Submit/Scope-End Split

The implementation should explicitly guard the following failure modes before any
performance tuning claims are accepted.

1. Early-ready bug from submit-time scheduler mutation.
- Manual submit may discover external producers early, but it must not mutate
  producer `fanout_head` or consumer ready state early.
- Required safeguard: manual submit may cache producer slot states only.

2. Stale frontier bug for outer writes.
- If outer `INOUT` / `OUTPUT_EXISTING` writes stay deferred until `scope_end`,
  later submissions can miss the newest writer frontier.
- Required safeguard: publish TensorMap frontier at manual submit in original
  task order.

3. Double-accounting bug across cached external fanins and explicit manual edges.
- One producer may be found both through boundary discovery and through an
  explicit edge.
- Required safeguard: publish-time fanin construction must run one dedup pass
  over both sources before incrementing `fanin_count` or wiring fanout.

4. Completed-before-publish bug.
- An external producer may already be `COMPLETED` when the manual scope reaches
  `scope_end`.
- Required safeguard: publish-time scheduler wiring must detect already-finished
  producers and credit `fanin_refcount` exactly once.

5. Producer-lifetime bug for cached external fanins.
- A cached producer that is not retained may reach `CONSUMED` and have its slot
  reused before the manual scope publishes.
- Required safeguard: manual submit must take a real retained reference on each
  unique cached producer, and consumer release must drop that same reference.

6. Scope-abort visibility bug for submit-time outer writes.
- If manual submit mutates TensorMap for outer writes and the scope later fails,
  global TensorMap state can point at unpublished internal writers.
- Required safeguard: treat post-submit fatal paths as terminal for the runtime,
  and keep the implementation free of late scope validation after submit-time
  TensorMap mutation.

7. Wrong manual-local classification for aliases and views.
- Boundary discovery must skip TensorMap only for tensors whose
  `owner_task_id` belongs to the current manual scope, including derived views.
- Required safeguard: keep classification on task provenance, not on a new
  tensor-side mode bit.

## Recommended Implementation Order

1. Add API surface for `add_dependency` and manual scope mode.
2. Add manual-submit APIs with `_manual` suffix returning task ids plus outputs.
3. Add scope-frame mode plus scope-local manual-edge storage.
4. Implement submit-time outer-tensor TensorMap lookup/insert with cached external fanins.
5. Keep manual `scope_end` TensorMap-free and realize only explicit same-scope edges plus scheduler publish.
6. Implement manual-local tensor classification from `owner_task_id` plus current manual-scope ownership.
7. Add fail-fast nested-scope-in-manual check and block `get_tensor_data` / `set_tensor_data` in manual scope.
8. Add targeted tests for boundary semantics.
9. Migrate one example and validate.

## Open Question Resolved

This design intentionally resolves the central ambiguity:

- `scope_end` controls lifetime release
- task completion controls semantic readiness

For outer tensors written inside manual scope, TensorMap frontier publication happens at
manual submit, while semantic readiness is still producer-task completion.

## File Areas Expected To Change

- `src/a2a3/runtime/tensormap_and_ringbuffer/orchestration/pto_orchestration_api.h`
- `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_runtime2.h`
- `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_runtime2.cpp`
- `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp`
- `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_runtime2_types.h`
- `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_scheduler.h`
- docs and examples/tests needed to demonstrate the new scoped behavior

## Recommendation Summary

Implement manual dependency as a scope-local override inside `tensormap_and_ringbuffer`, not as a runtime-wide replacement of TensorMap:

- tensors created in the current manual scope: explicit `add_dependency`
- outer tensors: existing TensorMap path
- TensorMap boundary realization for manual scopes: manual submit
- semantic readiness of outer writes: writer completion
- lifetime release: `scope_end`

That is the smallest design that satisfies the requested model without breaking the core tensormap runtime semantics.
