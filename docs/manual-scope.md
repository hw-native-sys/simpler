# Manual Scope V0 Design

## Goal

Keep manual scope small and explicit:

- same submit API family as AUTO mode
- explicit task-to-task deps via `Arg.add_dep(task_id)`
- publish at submit time, same as AUTO mode
- support allocation-as-task through `alloc_tensors(...).task_id()`

## Constraints

The v0 design keeps these rules:

1. `PTO2_SCOPE(PTO2ScopeMode::MANUAL)` opts a scope into manual mode.
2. `MANUAL` nested inside active `MANUAL` is allowed.
3. `AUTO` nested inside active `MANUAL` is rejected.
4. Manual deps are attached before submit through `Arg.add_dep(...)`.
5. No post-submit `add_dependency(...)` API exists.
6. `alloc_tensors(...)` remains output-only and returns a task id.
7. Scope handling stays close to upstream AUTO mode.

## User-Facing API

### Scope

```cpp
PTO2_SCOPE(PTO2ScopeMode::MANUAL) {
    ...
}
```

### Submit

Manual mode still uses the normal submit calls:

```cpp
auto out = rt_submit_aic_task(FUNC_ID, args);
auto out = rt_submit_aiv_task(FUNC_ID, args);
auto out = rt_submit_task(mixed_kernels, args);
```

### Explicit deps

```cpp
Arg args;
args.add_input(tensor);
args.add_dep(prev_task_id);
```

Rules:

- `add_dep(...)` must refer to an earlier producer task id
- if that producer task is already retired when the consumer is submitted, the
  runtime skips the edge without reporting an error
- explicit deps become ordinary fanins immediately at submit time
- this applies uniformly in AUTO and MANUAL submits; the task id already carries
  the ring needed for slot lookup

### Allocation task ids

```cpp
auto alloc = alloc_tensors(ci0, ci1, ci2);
PTO2TaskId alloc_tid = alloc.task_id();
const Tensor &tmp = alloc.get_ref(0);
```

This is required for manual-local tensors because the alloc task itself may be
the producer that later tasks must explicitly depend on.

## Runtime Model

### High-Level Behavior

Manual scope v0 is:

- AUTO-style submit and publish
- plus explicit fanins from `Arg.add_dep(...)`
- plus full TensorMap lookup / insert bypass for tasks submitted inside manual
  scope

### Scope State

The runtime keeps two manual-scope-specific pieces of state:

- `manual_begin_depth`
- the current scope task list (`scope_tasks[...]`)

`manual_begin_depth` decides whether the current submit is in manual mode.
`scope_tasks[...]` remains scope-lifetime bookkeeping for `scope_end()`.
`manual_begin_depth` is explicitly initialized in `PTO2OrchestratorState::init()` and
reset in `mark_done()` so a reused orchestrator starts cleanly on
the next run.

The depth model treats `manual_begin_depth` as the depth where the outermost
active manual scope began:

- outer `MANUAL` begin sets `manual_begin_depth`
- nested `MANUAL` begin does not overwrite it
- nested `AUTO` begin is rejected
- `scope_end()` clears `manual_begin_depth` only when the outermost manual
  scope exits

### How Manual Scope Decides TensorMap Use

The rule is submit-scoped, not tensor-scoped:

- inside manual scope: skip the entire TensorMap lookup section
- inside manual scope: skip the entire TensorMap insert section
- outside manual scope: keep normal AUTO TensorMap behavior

### Dependency Handling

For a submitted task:

- explicit `add_dep(...)` edges are always checked first
- retired explicit-dep producers are skipped as already satisfied
- if the task is inside manual scope, TensorMap lookup / insert is skipped
- if the task is outside manual scope, normal AUTO TensorMap behavior is used

## Example Pattern

```cpp
PTO2_SCOPE(PTO2ScopeMode::MANUAL) {
    auto alloc = alloc_tensors(tmp_ci);

    Arg qk;
    qk.add_input(qi, kj);
    qk.add_output(sij_ci);
    qk.add_dep(alloc.task_id());
    auto qk_out = rt_submit_aic_task(FUNC_QK, qk);

    Arg sf;
    sf.add_input(qk_out.get_ref(0));
    sf.add_output(pij_ci, li_ci, mi_ci);
    sf.add_dep(qk_out.task_id());
    auto sf_out = rt_submit_aiv_task(FUNC_SF, sf);

    Arg up;
    up.add_input(sf_out.get_ref(1), sf_out.get_ref(2), pv_out.get_ref(0));
    up.add_inout(mi, li, out_view, tmp);
    up.add_dep(sf_out.task_id());
    up.add_dep(pv_out.task_id());
    auto up_out = rt_submit_aiv_task(FUNC_UP, up);
}
```

Repeated zero-output updater chains must explicitly thread the returned task id:

```cpp
PTO2TaskId prev_update = PTO2TaskId::invalid();
for (...) {
    Arg up = ...;
    if (prev_update.is_valid()) {
        up.add_dep(prev_update);
    }
    prev_update = rt_submit_aiv_task(FUNC_UP, up).task_id();
}
```

## A5 Port Scope

The a5 port uses the same manual-scope v0 runtime model as a2a3:

- same `manual_begin_depth` state model
- same `MANUAL`-inside-`MANUAL` behavior
- same rejection of `AUTO` inside active `MANUAL`
- same submit-time explicit dependency model through `Arg.add_dep(...)`
- same submit-time TensorMap bypass for tasks submitted inside manual scope
