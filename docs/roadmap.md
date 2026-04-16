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
  carved out of the Worker's HeapRing (per-scope ring chosen by the
  caller's scope depth; see "Per-scope HeapRing" below). Lifetime
  follows a synthetic task slot; the slab is reclaimed implicitly by
  the allocator once all downstream consumers have completed and the
  ring's `last_alive` sweeps over it (see
  [orchestrator.md](orchestrator.md) §8b).
- **`OUTPUT` auto-allocation** — `OUTPUT`-tagged tensors submitted with
  `data == 0` are auto-allocated from the same HeapRing as part of the
  allocator call that claims the slot (1024-byte aligned). `OUTPUT`
  tensors with a pre-set `data` pointer are passed through untouched —
  pure overwrite with no WaW dep on the prior owner. Matching L2
  semantics, only `INPUT` and `INOUT` do a tensormap lookup in
  `infer_deps`; user code that writes into an `orch.alloc()` buffer
  must tag it `INOUT` so the alloc-slot stays live as a WaW producer
  (see [orchestrator.md](orchestrator.md) §8b "Tag semantics for
  write-after-write"). `OUTPUT_EXISTING` is never auto-allocated.
- **`heap_ring_size` knob** — `Worker(level=3, heap_ring_size=...)`
  selects the **per-ring** HeapRing size (default 1 GiB; total VA
  reservation is `heap_ring_size * DIST_MAX_RING_DEPTH`). The
  underlying `DistWorker(level, heap_ring_size)` ctor also installs
  fork hygiene (setenv of `OMP/BLIS/OPENBLAS/MKL_NUM_THREADS=1`, plus
  `KMP_DUPLICATE_LIB_OK=TRUE` on macOS, and a `pthread_atfork` landing
  pad).
- **Per-scope HeapRing (Strict-1) + user-facing nested scope** —
  `DistRing` owns `DIST_MAX_RING_DEPTH = 4` independent HeapRing
  instances, each its own `mmap(MAP_SHARED | MAP_ANONYMOUS)` taken
  before fork. Ring selection is driven by scope depth
  (`min(scope_depth, DIST_MAX_RING_DEPTH - 1)`); every ring has its own
  `mu` / `cv` / `last_alive`, so inner-scope tasks reclaim independently
  of outer-scope tasks. `Orchestrator::scope_begin` / `scope_end` are
  now user-facing (bound on the nanobind `DistOrchestrator`); the
  Python facade adds a `with orch.scope():` context manager. Outermost
  scope is still opened by `Worker::run`. Max user nesting is
  `DIST_MAX_SCOPE_DEPTH = 64`; scopes deeper than the ring depth share
  the innermost ring.

### Dispatch internals

- `IWorker::run(uint64_t callable, TaskArgsView args, ChipCallConfig cfg)`
  is the dispatch surface — no `WorkerPayload` carrier. Each
  `WorkerThread` reads `callable` / `task_args` / `config` from
  `ring->slot_state(task_slot)` at dispatch time and builds a
  `TaskArgsView` on demand (`slot.args_view(i)` for THREAD;
  `write_blob` / `read_blob` for PROCESS). `ChipWorker::run` assembles
  the L2 ABI `ChipStorageTaskArgs` POD from the view right before
  `pto2_run_runtime` — the slot itself stores only the tagged
  `TaskArgs` (single) or `task_args_list` (group).
- `Scheduler` dispatches slot ids via **per-worker-type ready queues**
  (Strict-4; one `DistReadyQueue` for `NEXT_LEVEL`, one for `SUB`) into
  `WorkerManager` pools (next-level + sub). `dispatch_ready` drains each
  queue with its own head-of-line break, so a saturated pool of one
  type cannot stall dispatch for the other. For group slots it pushes
  a `WorkerDispatch { slot, group_index }` per member onto N idle
  threads.
- `DistChipProcess` / `DistSubWorker` are separate classes today;
  unified `WorkerThread` with `THREAD | PROCESS` modes is not yet
  implemented (lands in PR-D).
- `DistRing` owns the task-id counter, the heap slab, **and** the
  per-slot state (`std::deque<std::unique_ptr<DistTaskSlotState>>`).
  Slot ids are monotonic within a run (no fixed window, no modulo
  wrap); `std::deque::push_back` keeps existing pointers valid across
  concurrent allocs, so `ring.slot_state(id)` hands out a stable
  pointer for every live slot. Heap bytes are the only back-pressure
  source: `alloc()` throws `std::runtime_error` on timeout if the heap
  is full. At end of `Worker.run()`, `drain()` calls
  `ring.reset_to_empty()` to drop all slot state and zero counters,
  so per-run memory stays bounded (see plan Allowed Exception #6).

---

## In flight / not yet landed

### PR-D-2: WorkerThread unification (PROCESS mode)

- Fold `DistChipProcess` / `DistSubWorker` into `WorkerThread` with
  `Mode = THREAD | PROCESS` (no separate fork-proxy classes). Strict-4
  per-worker-type ready queues already landed in PR-D-1.

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
  L2 `ChipWorker.run(callable, POD, config)` overload still used by
  inline callers) and `_build_l3_task_args` returns a tagged
  `TaskArgs` (for `orch.submit_next_level`). The
  `ChipWorker::run(uint64_t, TaskArgsView, ChipCallConfig)` IWorker
  entry now accepts a view, so these helpers can collapse into one —
  the POD overload is retained as an internal convenience for the
  Python-bound `_ChipWorker.run_raw` path.
