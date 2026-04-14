# Hierarchical Runtime — Roadmap

The six per-component docs (`orchestrator.md`, `scheduler.md`,
`worker-manager.md`, `task-flow.md`, `chip-level-arch.md`,
`distributed_level_runtime.md`) describe the **target** design of the
hierarchical runtime. This page tracks what has already landed vs. what is
still in flight, so readers can tell which bits of the design are running
today and which are planned.

If you only read one file to understand "what will this look like when
it's done", read the per-component doc. If you want to know "what do I
get if I pip install `main` today", this page.

---

## Landed

### Schedule engine shape

- **Component split** — `Orchestrator` (DAG builder) / `Scheduler` (DAG
  executor) / `WorkerManager` + `WorkerThread` (execution layer) — lives
  in `src/common/distributed/`.
- **Level model** — L0–L6 as described in
  [distributed_level_runtime.md](distributed_level_runtime.md) §1. L2
  (single-chip) and L3 (composite over ChipWorker + SubWorker) are
  implemented; L4+ recursion is not (see below).

### User-facing API

- **Unified `TaskArgs`** — vector-backed builder with per-tensor
  `TensorArgType` tags (`INPUT` / `OUTPUT` / `INOUT` / `OUTPUT_EXISTING`
  / `NO_DEP`). Replaces separate `TaggedTaskArgs` / `DynamicTaskArgs`.
- **Tag-driven `submit_*` on `Orchestrator`** —
  `submit_next_level` / `submit_next_level_group` / `submit_sub` /
  `submit_sub_group`. No `inputs=`/`outputs=` kwargs; tags inside the
  `TaskArgs` drive `tensormap.lookup`/`insert` automatically.
- **`SubmitResult = {slot_id}`** — downstream consumers reference output
  tensors by their own data pointers.
- **`Worker` has no `submit`/`scope`/`drain`** — those concepts belong
  to `Orchestrator` (accessed via `worker.get_orchestrator()`).
  `Orchestrator._scope_begin` / `_scope_end` / `_drain` are invoked by
  the Python `Worker.run` facade only.
- **`orch.alloc(shape, dtype)`** — runtime-owned intermediate buffer
  backed by `mmap(MAP_SHARED | MAP_ANONYMOUS)`. Lifetime follows a
  synthetic task slot so the buffer is freed once all downstream
  consumers have completed (see
  [orchestrator.md](orchestrator.md) §8b).

### Dispatch internals

- `Scheduler` dispatches via a single ready queue into `WorkerManager`
  pools (next-level + sub). Slot stores `chip_storage_list` (one
  `ChipStorageTaskArgs` per group worker) that dispatch passes through
  a `WorkerPayload` handed to `IWorker::run`.
- `DistChipProcess` / `DistSubWorker` are separate classes today;
  unified `WorkerThread` with `THREAD | PROCESS` modes is not yet
  implemented.

---

## In flight / not yet landed

### PR-H: HeapRing + `OUTPUT` auto-alloc

- Replace the current per-call `mmap` in `orch.alloc` with a single
  pre-allocated `MAP_SHARED` region at `Worker.init()` (default 1 GB),
  bump-allocated with FIFO reclamation (mirrors L2's
  `PTO2TaskAllocator`).
- `OUTPUT` tag will auto-allocate from the ring;
  `OUTPUT_EXISTING` keeps the "user-provided buffer" path.
- Merge slot ring + heap ring into one allocator
  (matches L2-consistency audit Strict-2).
- Fork-safety hygiene at `Worker.init()` (`setenv
  OMP_NUM_THREADS=1` / `pthread_atfork` on runtime-owned locks).

### PR-C: drop `WorkerPayload`, new `IWorker::run` signature

- `IWorker::run(callable, TaskArgsView, config)` — no `WorkerPayload`
  wrapper; mailbox encodes a length-prefixed blob of `callable +
  config + args` at dispatch.
- Slot drops `chip_storage_list` and stores the `TaskArgs` itself.
  Child assembles `ChipStorageTaskArgs` from the view at the L2 ABI
  edge only.
- Strict-1 (per-scope rings, 4 depth) lands here.

### PR-D: WorkerThread unification + per-shape ready queues

- Fold `DistChipProcess` / `DistSubWorker` into `WorkerThread` with
  `Mode = THREAD | PROCESS`.
- Strict-4: 3 ready queues (AIC / AIV / MIX) instead of a single queue.

### PR-E: uniform `Worker.run` + callable registry unification

- Python `Worker.run` drops the `if level==2` branch.
- Callable registry moves fully into C++
  (`unordered_map<uint64_t, nb::object>` owned by `Worker`) so
  `ChipCallable` and Python `sub` callables share one lookup path.
  This unblocks L4+ recursion.

### PR-F: C++ `Worker::run(Task)` for L4+ recursion

- C++ `Task { OrchFn orch; TaskArgs task_args; CallConfig config; }`
  so a higher-level `Worker` can register a lower-level `Worker` as a
  next-level child and dispatch via `IWorker::run`.

### PR-G: drop the `Dist` prefix

- Final rename sweep: `DistOrchestrator` → `Orchestrator`, files
  `dist_*.{h,cpp}` → `*.{h,cpp}`.

---

## Behavioural notes on the current implementation

- **`DistOrchestrator::release_ref` threshold is `>= total + 1`** (not
  `>= total`). This matches `DistScheduler::try_consume` — the
  `+1` accounts for the slot's own self-release contribution. Alloc
  slots (synthetic, never dispatched) pre-bump `fanout_released` to
  `1` in `alloc()` so this threshold math works for them too.
  `on_consumed` uses a CAS on state to remain idempotent across the two
  call paths (`release_ref` and `try_consume`).
- **scene_test has two helper functions** —
  `_build_chip_task_args` returns `ChipStorageTaskArgs` (POD, for the
  current L2 path: `ChipWorker.run(callable, POD, config)`) and
  `_build_l3_task_args` returns a tagged `TaskArgs` (for
  `orch.submit_next_level`). PR-C will collapse these into one helper
  when `ChipWorker::run` takes a `TaskArgsView`.
