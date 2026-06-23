# AICPU Callable Prewarm Plan

**Status**: implementation plan updated for origin/main after PR #1061
**Date**: 2026-06-18
**Last updated**: 2026-06-23

## Problem

For the `tensormap_and_ringbuffer` runtime, `prepare_callable()` already
removes the host-side kernel upload and orchestration-SO H2D copy from repeated
`run_prepared()` calls. The remaining first-run cost is on the AICPU side:
the first real run for a `callable_id` sets `register_new_callable_id_`, and
`AicpuExecutor::run()` then materializes the device orchestration SO, calls
`dlopen`, resolves entry/config/bind symbols with `dlsym`, and stores the
result in `orch_so_table_[callable_id]`.

Subsequent runs of the same `callable_id` reuse the cached
`orch_so_table_` entry, so this cost is a first-call latency issue rather than
a steady-state issue.

This cost should not sit on the critical path of the first real task.

## Current Main Context

Current `origin/main` already includes PR #1061 (`8db7fc62`), which removed the
old `simpler_aicpu_init` kernel. The host no longer registers, resolves, or
launches that entry. The existing AICPU loader contract has
`simpler_aicpu_exec` as the run entry, and this plan adds
`simpler_aicpu_prewarm_callable` as a new peer entry for prewarming.

The implementation must not restore or depend on `simpler_aicpu_init`.

## Measurement Gate

Before implementing the prewarm path, first quantify the current first-run
cost on hardware. The implementation should proceed only if the AICPU SO-load
cost is a meaningful part of first-task latency for representative
`tensormap_and_ringbuffer` callables.

The baseline should record, at minimum:

- `prepare_callable()` wall time.
- First `run_prepared()` wall time and device wall time.
- Second and later `run_prepared()` wall time and device wall time for the same
  `callable_id`.
- `aicpu_dlopen_count` before prepare, after prepare, after the first run, and
  after a repeated run.
- An AICPU-side signal that the first run executed the SO-load branch, such as a
  device log marker around the load helper.

If the measured delta between first and repeated runs is small or dominated by
noise outside AICPU SO loading, do not implement the prewarm path. Record the
measurement and the "do not implement" decision under `docs/investigations/`
instead.

## Current Flow

`Worker.register()` / `ChipWorker.prepare_callable()`:

- Uploads child kernel code and records `func_id` to device-address mappings.
- Copies the orchestration SO bytes into a device-resident buffer, with
  Build-ID based deduplication.
- Does not cause AICPU to `dlopen` the orchestration SO.

First `run_prepared(callable_id, ...)`:

- Host restores the prepared kernel mapping into a fresh `Runtime`.
- Host marks the callable as first sighting for AICPU.
- AICPU sees `register_new_callable_id_ == true`.
- AICPU writes the orchestration SO to a temporary file, calls `dlopen`, calls
  `dlsym`, and fills `orch_so_table_[callable_id]`.
- The real task then proceeds.

Later `run_prepared(callable_id, ...)` calls:

- AICPU sees `register_new_callable_id_ == false`.
- AICPU reuses `orch_so_table_[callable_id]`.
- No AICPU-side SO load is paid.

## Desired Change

Add an AICPU-side callable prewarm path that can run after host-side
`prepare_callable()` but before the first real task. The prewarm should:

- Materialize and `dlopen` the orchestration SO on AICPU.
- Resolve the entry/config/bind symbols.
- Populate `orch_so_table_[callable_id]`.
- Mark host-side callable state so the first real run passes
  `register_new_callable_id_ == false`.
- Avoid invoking the orchestration entry function or submitting runtime tasks.
- Avoid launching AICore work.
- Avoid requiring real tensor inputs.

This is not a host-side `dlopen` migration. Host-side `dlopen` would produce
host-process function pointers, which are not valid in the AICPU address
space. The work still needs to happen on AICPU; the goal is to move it earlier
than the first real task.

## Candidate Design

Do not add a new public C or Python API for the current recommended change.
Fold the AICPU prewarm into the existing `prepare_callable()` contract for
device orchestration callables.

This keeps the change scoped: every current L2 and L3 prewarm path already
funnels through `ChipWorker.prepare_callable()` or
`prepare_callable_from_blob()`. Adding a separate public
`prewarm_callable(DeviceContextHandle ctx, int32_t callable_id)` entry would
expand the full exported ABI, `ChipWorker` `dlsym` surface, nanobind bindings,
and Python wrappers before there is a concrete caller that needs an explicit
prewarm operation independent of prepare.

For `host_build_graph`, or any prepared callable whose
`host_dlopen_handle != nullptr`, the internal prewarm step is a no-op success.
That runtime has already resolved orchestration on the host during
`prepare_callable()`, so there is no AICPU `orch_so_table_` entry to populate
and no `aicpu_seen_callable_ids_` / `aicpu_dlopen_total_` state to update.

On onboard hardware, add a third runtime SO AICPU entry point, for example:

```text
simpler_aicpu_prewarm_callable
```

`LoadAicpuOp` must register it in the generated AICPU op JSON and resolve its
`rtFuncHandle` alongside the existing `simpler_aicpu_exec` entry. It must not
reference the removed `simpler_aicpu_init` entry. Prewarm must not reuse
`simpler_aicpu_exec`: the exec path initializes scheduler state, binds a
per-run runtime arena, can wait on scheduler threads, invokes the orchestration
entry, and shuts down AICore-facing state. A prewarm entry should only run the
pure orchestration-SO load operation.

The prewarm entry should launch as a single AICPU instance. On a5, it must not
enter the normal exec affinity gate, which requires
`runtime.aicpu_allowed_cpu_count` / `runtime.aicpu_launch_count` and is designed
for scheduler/orchestrator role assignment. The prewarm entry only needs to
publish log config, set the device id used for orchestration-SO temp-file
naming, and call the SO-load helper.

On simulator platforms, do not make `tensormap_and_ringbuffer` prewarm a no-op.
The sim host runner loads the AICPU executor SO directly and already preserves
its static `g_aicpu_executor` across runs. Add a simulator-only dlsym path for a
pure prewarm function, or equivalent private entry, and call the same SO-load
helper against that preserved executor state. The simulator should therefore
exercise the same `orch_so_table_[callable_id]` state transition as onboard,
without using `LoadAicpuOp`.

The implementation would:

1. Let `prepare_callable()` complete the existing host-side registration.
2. For host-orchestration callables, return success without AICPU work.
3. For device-orchestration callables, build a minimal `Runtime` carrying the
   active `callable_id`, device orch SO address/size, and symbol names.
4. Launch the AICPU prewarm entry on the AICPU stream and wait for completion.
5. On AICPU, call a factored helper that materializes the SO, calls `dlopen`,
   resolves entry/config/bind symbols, and fills `orch_so_table_[callable_id]`.
6. Return success only after the AICPU table entry is valid.
7. Mark host `DeviceRunner` state so a following real run is treated as an
   AICPU cache hit.

If the AICPU prewarm fails, `prepare_callable()` should fail and roll back the
host-side callable registration it just created. That keeps the public contract
simple: a successful prepare means the callable is both host-prepared and
AICPU-prewarmed for `tensormap_and_ringbuffer`; a failed prepare leaves no
prepared slot behind.

The implementation still needs a private helper that:

1. Verifies the callable has already been host-prepared.
2. Stamps a minimal `Runtime` with the active `callable_id`, device orch SO
   address/size, and symbol names.
3. Runs the platform-specific AICPU prewarm launch or simulator call.
4. Commits host seen/counting state only after the device-side helper succeeds.

The helper factored out of `AicpuExecutor::run()` should cover only the SO-load
branch. It must stop before:

- calling the config function with real args;
- binding or creating a `PTO2Runtime`;
- `rt_scope_begin()` / `rt_scope_end()`;
- invoking `aicpu_orchestration_entry`;
- submitting scheduler or AICore tasks.

The helper must preserve today's symbol requirements: the dlopen handle and
entry function are required for a valid cache entry; the config function and
`framework_bind_runtime` are optional and may remain null if `dlsym` cannot
resolve them, matching the current run path.

Strictly speaking, `dlopen` can execute ELF/C++ static constructors. The
prewarm contract is therefore not "no code in the SO ever runs"; it is "do not
explicitly invoke the orchestration entry function or submit runtime tasks."

No new L3 control command is needed. The existing L3 `_CTRL_PREPARE` path calls
child-side `prepare_callable()`, and the post-init L3 `_CTRL_REGISTER` path
calls `prepare_callable_from_blob()`. Once those prepare entries include AICPU
prewarm internally, startup registration, dynamic registration, and L2 direct
registration all receive the same behavior without extra Python mailbox
plumbing.

The recommended implementation should be synchronous and serialized against
normal `run_prepared()` work on the same `DeviceRunner` / child mailbox. It
should not attempt to overlap AICPU prewarm with an active run until a separate
measurement shows that such overlap does not extend the active run's stream
sync or contend with scheduler/orchestrator AICPU threads.

## State Consistency

The implementation must keep these states aligned:

- Host registered-callable table.
- Host `aicpu_seen_callable_ids_`.
- AICPU `orch_so_table_[callable_id]`.
- L3 child-process `prepared` set.

The host state update must be a success-after-commit transition. The existing
`prepare_orch_so()` helper inserts into `aicpu_seen_callable_ids_` before the
AICPU has actually completed `dlopen`; prewarm must not use that helper as-is.
Split the logic into:

- a metadata-stamp step that copies callable SO address/size/symbol names into
  the temporary `Runtime` without marking the cid seen;
- a host commit step that inserts into `aicpu_seen_callable_ids_` and increments
  `aicpu_dlopen_total_` only after the AICPU prewarm entry returns success.

If AICPU prewarm succeeds but host state is not marked seen, the first real run
will reload the same SO again. If host state is marked seen but AICPU state was
not populated, the first real run will fail or hit an empty cached handle.

If AICPU prewarm fails, host seen state must remain uncommitted. In the first
implementation the callable registration is rolled back. If a future fallback
leaves the callable host-prepared but not prewarmed, the next real run must use
`register_new_callable_id_ == true` and retry the full load.

The same commit-after-success rule should apply to the first real run when it
falls back to doing the AICPU load. If the initial real-run load fails after the
host has already marked the cid seen, a later retry can incorrectly advertise an
AICPU cache hit and fail on an empty cached handle. Implementations should
either move `aicpu_seen_callable_ids_` insertion to after successful AICPU load
for both prewarm and real-run first load, or explicitly document and test the
remaining failure-retry limitation.

`unregister_callable()` and callable-id reuse need explicit handling. Reusing a
slot should either invalidate the AICPU `orch_so_table_` entry or force the next
prewarm/real run to replace it.

For L3, `_CTRL_PREPARE` and `_CTRL_REGISTER` should report failure if
`prepare_callable()` fails during AICPU prewarm. The child-process `prepared`
set should be updated only after the full prepare call returns success. If a
future implementation chooses to allow "prepared but not prewarmed" as a
fallback, that state must be explicit and the first real task must force a
reload instead of advertising an AICPU cache hit.

## Validation

Add tests or hardware checks that prove:

- The measurement gate records the current first-run cost before implementation,
  and the implementation is skipped if the measured benefit is not material.
- `prepare_callable()` plus prewarm completes the AICPU-side load before any
  real `run_prepared()`.
- `aicpu_dlopen_count` increments during successful
  `tensormap_and_ringbuffer` `prepare_callable()`, not during the first real
  `run_prepared()`.
- The first real `run_prepared()` for that callable leaves
  `aicpu_dlopen_count` unchanged.
- Repeated real runs continue to reuse the cached AICPU handle.
- L3 `_CTRL_PREPARE` performs both child-side host prepare and AICPU prewarm via
  the existing prepare call.
- L3 post-init `_CTRL_REGISTER` performs host prepare and AICPU prewarm via
  `prepare_callable_from_blob()` before the first task using that callable.
- The `lazy-prepare` warning remains absent on normal L3 fast paths.
- A prewarm failure leaves host seen state uncommitted, and a later real run
  retries the AICPU load instead of failing with an empty cached handle.
- Prewarm does not invoke the orchestration entry function and does not launch
  AICore work.
- `host_build_graph` callables treat the internal prewarm step as no-op success
  and do not change AICPU dlopen accounting.
- Simulator `tensormap_and_ringbuffer` callables populate the same preserved
  `orch_so_table_` state as onboard callables, instead of treating prewarm as a
  no-op.
- Unregister and re-register of the same slot forces a fresh AICPU load.

`aicpu_dlopen_count` alone is not proof that AICPU `dlopen` succeeded, because
the counter is host-maintained. Validation should also assert an AICPU-side
signal, such as a dedicated prewarm success record, a device log marker from the
SO-load helper, or a follow-up real run that proves `reload=false` can reuse a
non-empty `orch_so_table_[callable_id]`.

For performance validation, compare both first real `run_prepared()` latency
after prepare and combined `prepare_callable()` plus first-run latency while
keeping device compute constant. The expected improvement is the removal of
AICPU-side SO materialization, `dlopen`, and `dlsym` from the first real task.
The combined prepare-plus-first-run latency may stay similar or increase
slightly because the same work is intentionally moved into prepare.

## References

- `src/a2a3/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp`
- `src/a5/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp`
- `src/common/platform/onboard/host/device_runner_base.cpp`
- `src/common/platform/onboard/host/c_api_shared.cpp`
- `src/common/platform/sim/host/device_runner_base.cpp`
- `src/common/platform/sim/host/c_api_shared.cpp`
- `src/common/aicpu_loader/host/load_aicpu_op.*`
- `python/simpler/worker.py`
- GitHub issue #545: runtime performance optimization tracking.
- GitHub PR #1061: removal of `simpler_aicpu_init` and per-run AICPU init
  launch.
