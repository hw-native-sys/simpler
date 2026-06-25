# Polling PR — Rebase Reconciliation Notes

State of `polling-pr-minimal` (HEAD `188be7e4`) vs `upstream/main`
(currently `ecfb1663`, 14 commits ahead of the PR's base `fcc33bcb`).

## TL;DR

- **`git rebase upstream/main` produces 15 file conflicts.** Mechanical
  resolution (take "theirs" for files we rewrote, take "ours" for files
  with upstream feature additions, rename `Arg→L0TaskArgs` and `.ptr→.ref()`
  throughout) gets the tree to **compile clean**.
- **Compile-clean tree still hangs at runtime** with the now-familiar
  507018 AICore op-timeout. The hang is a **protocol-level mismatch**
  between upstream's evolved init/dispatch handshake and the polling-side
  SHM/scheduler — not a few-more-renames-away fix. Estimated 1-2 days of
  targeted protocol alignment + re-benchmarking to land cleanly.
- Decision (24 Jun 2026): pause the rebase; reviewer/maintainer to either
  rebase as part of merge or this PR will be rebased in a future session.

## Upstream commits since `fcc33bcb`

| Commit | Touches polling design? | Why |
|---|---|---|
| `10a7680b` `Refactor: tensormap L0/L2TaskArgs arg hierarchy` | **Yes — heavy** | `Arg` → `Arg<MaxT,MaxS>` template; `L0TaskArgs = Arg<32,16>` for core submit; `TensorRef::ptr` → `.ref()`/`.create_info()` accessors. Renames propagate through every submit signature. |
| `c6354842` `feat(runtime): unify runtime_env ring sizing` | **Yes — heavy** | `PTO2RuntimeArenaLayout` gains `task_window_sizes[PTO2_MAX_RING_DEPTH]`/`heap_sizes[]`/`dep_pool_capacities[]` arrays. New `init_per_ring` on `PTO2SharedMemoryHandle`. `runtime_init_data_from_layout` per-ring overload. |
| `6dd8a5dc` `consolidate profiling init into SchedulerContext::init()` | Yes — medium | `SchedulerContext::init()` signature changed; `l2_swimlane_level` moved from `PTO2OrchestratorState` to `SchedulerContext`. `runtime_destroy(rt, arena)` 2-arg signature. |
| `6c3a9e49` `consumed/reuse deadlock fix` | **No** | Fixes interaction between `fanout_refcount` / `fanout_count` / `task_state=CONSUMED` / `scope_end` producer-release — all four mechanisms removed by the polling design. |
| `11f0bf40` `AICPU callable prewarm` | Yes — light | Adds `aicpu_prewarm_callable` C entry to `aicpu_executor.cpp`. |
| `4725ef7b` `dispatcher fresh-process retry` | Yes — light | Adds retry path in `device_runner.cpp`. |
| `78b123e7` `rename init-claim flag to init_claimed_` | Trivial | Field rename in scheduler. |
| `ae59a8e9` `in-place card recovery` | No | `device_runner.cpp` only. |
| `3aa94a99` `close unpublished sim host orchestration handles` | No | Sim host only. |
| `e2112e9f` `restore SDMA async completion demo` | No | Example. |
| Others (`ecfb1663`, `cce30871`, `2f77399a`, `e583b8a0`) | Trivial | CI / docs / examples. |

## Per-file conflict matrix

After `git rebase upstream/main`, 15 files conflict. Recommended
resolution + work needed:

| File | Recommended action | Status |
|---|---|---|
| `runtime/pto_runtime2_types.h` | take theirs (polling) | ✓ compile-fixed |
| `runtime/pto_runtime2.h` | take theirs + add per-ring overloads | ✓ compile-fixed (added `runtime_reserve_layout` and `runtime_init_data_from_layout` per-ring overloads; added `runtime_destroy(rt, arena)` overload) |
| `runtime/pto_runtime2.cpp` | take theirs (stub) | ✓ |
| `runtime/pto_orchestrator.cpp` | take theirs (stub) | ✓ |
| `runtime/pto_orchestrator.h` | take theirs + rename `Arg → L0TaskArgs` + `.create_info →`→`.create_info()` + `.ptr → &.ref()` + add `l2_swimlane_level` field | ✓ compile-fixed |
| `runtime/pto_dep_compute.h` | take theirs + `inputs.tensors[i].ptr → &inputs.tensors[i].ref()` | ✓ compile-fixed |
| `runtime/scheduler/pto_scheduler.h` | take theirs (polling) | ✓ |
| `runtime/scheduler/scheduler_context.h` | take theirs + add `thread_idx` to `on_orchestration_done` signature | ✓ compile-fixed |
| `runtime/scheduler/scheduler_cold_path.cpp` | take theirs (stub) | ✓ |
| `runtime/scheduler/scheduler_dispatch.cpp` | take theirs (stub) | ✓ |
| `runtime/pto_shared_memory.h` | take theirs (polling) + add `init_per_ring` method (broadcast to scalar init) | ✓ compile-fixed |
| `runtime/runtime.h` | add `needs_copy_back` to `TensorPair` (upstream-API compat) | ✓ compile-fixed |
| `aicpu/aicpu_executor.cpp` | take ours (upstream — has prewarm, profiling consolidation, deadlock-fix-related changes) | ✓ compile-fixed via signature adapters above |
| `host/runtime_maker.cpp` | take ours (upstream — has per-ring env parsing #1128) | ✓ compile-fixed |
| `orchestration/pto_arg_with_deps.h` | take ours (upstream) | ✓ trivial |
| `orchestration/pto_orchestration_api.h` | take ours (upstream) | ✓ trivial |
| `docs/MULTI_RING.md` | take theirs (updated for polling) | ✓ trivial |

## Runtime hang — root cause hypothesis

After the compile-clean tree above runs `paged_attention` Case1, AICore
times out at 507018 with no orchestration log past the `simpler-dispatcher`
init. Suspect chain:

1. **`init_per_ring` is a stub**. My implementation broadcasts
   `task_window_sizes[0]` to the old scalar `init_header` /
   `setup_pointers`. If upstream's `aicpu_executor` writes
   `prebuilt_layout.task_window_sizes[r]` for r > 0 with different values
   than [0], the SHM layout's per-ring offsets diverge from what the
   AICPU expects → wrong pointers → silent corruption or hang.
2. **`PTO2OrchestratorState::l2_swimlane_level`** is back as a field, but
   upstream's `SchedulerContext::init` may now own that state. Adding
   the field in two places creates a tearing concern only if both writers
   actually fire — unlikely to be the hang root cause but worth checking.
3. **`runtime_destroy(rt, arena)`**: my overload calls the 1-arg form,
   but upstream's `arena` parameter may be used for staged teardown
   (e.g., scope finalize). The polling design's destroy doesn't need it
   but the *order* of teardown might matter for upstream's aicpu_executor
   loop. Not the boot-time hang, but a leak/reset issue downstream.
4. **AICPU dispatch handshake**: upstream's aicpu_executor may have
   ordering expectations around when the polling design's wiring queue
   is initialized vs when the AICore handshake fires. The polling
   scheduler initializes wiring lazily in `init_data_from_layout`; if
   upstream's executor handshakes AICore *before* the wiring queue is
   ready, AICore spins for tasks that never arrive.

The fix path: thread true per-ring sizes through `PTO2SharedMemoryHandle`
(currently the polling code uses a uniform per-ring layout — needs to
honor the array), then add a runtime trace point at the boundary
between aicpu_executor's `init_per_ring` call and the scheduler's first
`drain_wiring_queue` to confirm where the AICore handshake is firing
vs when the wiring becomes ready.

## What to do next session

1. `git rebase upstream/main`, apply the resolutions above (the order is
   mechanical now that this doc records them).
2. Build (should compile clean as documented).
3. Run `paged_attention` Case1 to confirm the runtime hang reproduces.
4. Add device-side `LOG_INFO_V0` traces at:
   - `PTO2SharedMemoryHandle::init_per_ring` entry/exit (per ring)
   - `AicpuExecutor::run` immediately before / after the first scheduler
     `drain_wiring_queue` call
   - `SchedulerContext::on_orchestration_done` entry
5. Diagnose the gap revealed by the traces; align the polling SHM /
   wiring init order with upstream's handshake.
6. Re-run the 26-test benchmark sweep (the one in `PR_NOTES.md`) and
   confirm parity with the pre-rebase result.

## Quick repro recipe

```bash
git checkout polling-pr-minimal             # HEAD = 188be7e4
git rebase upstream/main                    # 15 conflicts

# Take theirs (polling) for files we rewrote:
git checkout --theirs \
  src/a2a3/runtime/tensormap_and_ringbuffer/runtime/{pto_runtime2_types.h,pto_runtime2.cpp,pto_runtime2.h,pto_orchestrator.cpp,pto_orchestrator.h,pto_dep_compute.h,scheduler/pto_scheduler.h,scheduler/scheduler_context.h,scheduler/scheduler_cold_path.cpp,scheduler/scheduler_dispatch.cpp} \
  src/a2a3/runtime/tensormap_and_ringbuffer/docs/MULTI_RING.md

# Take ours (upstream) for files where upstream adds features:
git checkout --ours \
  src/a2a3/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp \
  src/a2a3/runtime/tensormap_and_ringbuffer/host/runtime_maker.cpp \
  src/a2a3/runtime/tensormap_and_ringbuffer/orchestration/{pto_arg_with_deps.h,pto_orchestration_api.h}

git add -u src/

# Apply compile-fixes (see "Per-file conflict matrix" for details).
# Build is clean after these. Runtime hangs — see "Runtime hang — root
# cause hypothesis" above for the next investigation steps.
```
