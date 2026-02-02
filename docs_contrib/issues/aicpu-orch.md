# Issue: Add `aicpu_build_graph` runtime (AICPU builds + AICPU schedules)

## Summary

Today, this repo builds the task graph on the **host** (via a host orchestration `.so` executed during `Runtime.initialize()`), then runs an AICPU scheduler with `N` AICPU instances to dispatch tasks to AICore workers.

Requested change (translated): build the task graph on **AICPU** using **1** AICPU instance, then schedule/execute using **3** AICPU instances (total `4`).

This issue proposes a new runtime implementation named `aicpu_build_graph` under `src/runtime/`, plus a copied example, and documents the required design changes and risks.

## Background (current behavior, with references)

### Host orchestration builds the graph today

- The example orchestration function `build_example_graph()` constructs tensors *and* builds tasks/edges on the host: `examples/host_build_graph_example/kernels/orchestration/example_orch.cpp:20`.
- It is compiled as a **host `.so`** by `PTOCompiler.compile_orchestration()` in the runner: `examples/scripts/code_runner.py:481`.
- It is loaded and executed by `dlopen()` + `dlsym()` inside `init_runtime_impl()`: `src/runtime/host_build_graph/host/runtime_maker.cpp:96`.

### AICPU runs a scheduler loop (graph is assumed already built)

- AICPU entrypoint: `aicpu_execute(Runtime*)`: `src/runtime/host_build_graph/aicpu/aicpu_executor.cpp:633`.
- It snapshots `total_tasks` and initial ready tasks during init: `src/runtime/host_build_graph/aicpu/aicpu_executor.cpp:110`, `src/runtime/host_build_graph/aicpu/aicpu_executor.cpp:114`.

### Host assigns kernel addresses to tasks before launching

- Host sets `task->function_bin_addr` for all tasks before copying runtime to device: `src/platform/a2a3/host/device_runner.cpp:303`.
- Kernels are registered into GM memory and mapped by `func_id`: `src/platform/a2a3/host/device_runner.cpp:499`, `src/platform/a2a3/host/device_runner.cpp:541`.

## Problem statement

We want a runtime where the **graph-building program is executed on AICPU** (not on host), while preserving:

- the existing Python runner flow (compile runtime + compile orchestration + register kernels + initialize + launch),
- the existing AICore execution contract (tasks run by calling a function pointer from `Task::function_bin_addr`),
- required support for *concurrent* build and schedule.

The blockers are captured as four challenges (A–D).

## Goals

- Add a runtime `src/runtime/aicpu_build_graph/` discoverable by `RuntimeBuilder` (`python/runtime_builder.py:34`).
- Make graph building execute on AICPU via a programmable function (not restricted to a fixed “struct schema”).
- Support two build/schedule modes:
  - **Concurrent** build∥schedule (required).
  - **Sequential** build→schedule (supported baseline; useful for debugging).
- Fixed thread split for this feature: `1` builder + `3` scheduler threads (total `4`).
- Keep simulation (`a2a3sim`) as the primary dev/test path.

## Non-goals (for the first iteration)

- Implementing a device-side dynamic loader equivalent to host `dlopen()`.
- Redesigning the global C API surface unless strictly necessary.
- Making `host_build_graph` behavior more complex; changes should be gated so it remains stable.

## Proposed approach (high level)

1. **New runtime**: `src/runtime/aicpu_build_graph/` with its own `runtime/runtime.h`.
2. **Programmable AICPU builder**: introduce a function like `extern "C" int build_graph_aicpu(Runtime* runtime)` that runs on AICPU and builds tasks/edges.
3. **Host orchestration becomes “prepare + marshal”**:
   - it allocates/copies tensors on host (unchanged),
   - it writes a **generic `uint64_t[]` payload** into `Runtime` (e.g. `orch_args[]`/`orch_argc`) for the AICPU builder to interpret,
   - it does *not* call `add_task()` / `add_successor()`.
4. **Kernel address binding**: host writes `func_id -> function_bin_addr` into a runtime-visible table so AICPU-built tasks can set `Task::function_bin_addr`.

## Challenges (A–D)

### Challenge A — Kernel addresses for tasks created on AICPU

If tasks are created on AICPU, host cannot pre-fill `task->function_bin_addr` by iterating tasks (because tasks do not exist yet). Without this field, AICore will skip execution (`src/runtime/host_build_graph/aicore/aicore_executor.cpp:37`).

Proposed fix (runtime-local):

- Add `uint64_t kernel_addrs[MAX_FUNC_ID]` to `src/runtime/aicpu_build_graph/runtime/runtime.h`.
- Host already knows `func_id -> addr` via `DeviceRunner::register_kernel()` bookkeeping (`src/platform/a2a3/host/device_runner.cpp:543`, `src/platform/a2a3/host/device_runner.cpp:550`).
- In `DeviceRunner::run()`, before copying runtime to device / launching AICPU, write:
  - `runtime.kernel_addrs[func_id] = get_function_bin_addr(func_id)`.
- In `build_graph_aicpu()`, set `task->function_bin_addr = runtime->kernel_addrs[task->func_id]` at task creation time.

### Challenge B — Programmable graph building on AICPU (not host-only orchestration)

The existing orchestration function cannot be “moved to AICPU” as-is:

- It calls host-only hooks via `runtime->host_api.*` (`examples/host_build_graph_example/kernels/orchestration/example_orch.cpp:44`), which are initialized on host (`src/platform/a2a3/host/pto_runtime_c_api.cpp:64`).
- It receives host pointers in `args[]` (`examples/host_build_graph_example/kernels/orchestration/example_orch.cpp:29`).
- It is executed via host `dlopen()` (`src/runtime/host_build_graph/host/runtime_maker.cpp:96`), which has no device-side equivalent in this repo.

Proposed programming model:

- Keep host orchestration as a host `.so`, but restrict it to:
  - allocating/copying tensors (`runtime->host_api.*`),
  - recording output tensors (`runtime->record_tensor_pair()`),
  - writing a generic payload into `Runtime`:
    - `int orch_argc`
    - `uint64_t orch_args[MAX_ORCH_ARGS]`

This preserves flexibility: `orch_args[]` is just a word array; it can represent arbitrary builder inputs (device pointers, sizes, scalar values, flags, offsets, small tables).

Then implement the actual graph build logic as an AICPU function:

- `extern "C" int build_graph_aicpu(Runtime* runtime);`
- It interprets `runtime->orch_args[]` however it wants and calls `add_task()` / `add_successor()`.

Execution mechanism (important): **link-time inclusion into the AICPU binary**, not device-side `dlopen`.

- The AICPU binary is already built from source using `CUSTOM_SOURCE_DIRS` (`src/platform/a2a3/aicpu/CMakeLists.txt:20`).
- `aicpu_build_graph/build_config.py` can include a copied example directory containing the builder implementation as a `source_dir` for the AICPU target (developer experience similar to “host builds `.so` from example source”, but implemented as “AICPU binary includes example builder source”).

### Challenge C — Sequential vs concurrent build/schedule (and concrete algorithms)

Status: **there is no existing implementation of concurrent build∥schedule in the repo**. Current AICPU scheduling assumes the graph is fully built and immutable during scheduling.

We require concurrent build∥schedule, and we also support sequential build→schedule as a baseline.

#### Mode 1: Sequential build→schedule (supported baseline)

Design:

- Total AICPU instances launched is `aicpu_thread_num` (current C API).
- Runtime adds fields:
  - `int build_thread_num` (default 1)
  - `int schedule_thread_num` (default `aicpu_thread_num - build_thread_num`)
  - `std::atomic<int> build_done` (0/1)
  - `int build_mode` (0=sequential, 1=concurrent)
- Role assignment:
  - Each AICPU instance obtains `thread_idx` as today (`src/runtime/host_build_graph/aicpu/aicpu_executor.cpp:498`).
  - Builder threads are those with `thread_idx < build_thread_num`.
  - Scheduler threads are the remaining threads.
- Barrier:
  - Builder thread runs `build_graph_aicpu(runtime)` once, then sets `build_done=1`.
  - Scheduler threads spin-wait on `build_done` before starting scheduling.

Core assignment implication:

- Scheduler threads manage AICore workers; builder threads should manage **zero** worker cores.
- Any existing “even distribution across threads” logic must be parameterized on `schedule_thread_num` (not on total AICPU instances), otherwise builder threads “steal” cores they never use.

Termination:

- Use the existing termination assumptions from the scheduler loop: scheduling ends when all tasks complete and all cores are idle (current code has “double verification” when counters indicate done but cores are still busy: `src/runtime/host_build_graph/aicpu/aicpu_executor.cpp:333`).

#### Mode 2: Concurrent build∥schedule (required)

Correct-first strategy (recommended initial approach): **global graph mutex**.

New executor state:

- Add `std::mutex graph_mutex_` (similar to existing ready-queue mutexes: `src/runtime/host_build_graph/aicpu/aicpu_executor.cpp:40`).
- Add `std::atomic<int> published_task_count` and reuse `build_done`.

Publication rule:

- Builder must fully initialize a task (including setting `function_bin_addr`) and any edges it adds, while holding `graph_mutex_`, then increment `published_task_count` before releasing the lock.

Scheduler behavior changes:

- When a core completes a task, the scheduler thread:
  - acquires `graph_mutex_`,
  - reads the completed task’s `fanout[]`,
  - decrements successor `fanin`,
  - pushes newly-ready successors into the appropriate ready queue,
  - releases `graph_mutex_`.

Termination condition (must include build progress):

Schedulers may exit only when all of the following are true:

- `build_done == 1`
- `completed_tasks == published_task_count`
- ready queues are empty
- all managed cores are idle (retain the existing “core idle verification” idea; see `src/runtime/host_build_graph/aicpu/aicpu_executor.cpp:333`)

Why this is “no redundancy”: it builds on the existing scheduler structure (ready queues, polling, verification), and adds only a lock + a publication counter.

Faster alternative (deferred): publish-only DAG append (requires data-structure redesign away from inline `fanout[]`).

### Challenge D — Thread configuration (fixed 1+3)

Constraints today:

- Platform max AICPU threads is `PLATFORM_MAX_AICPU_THREADS = 4` (`src/platform/include/common/platform_config.h:41`).
- Host validates `block_dim % launch_aicpu_num == 0` (`src/platform/a2a3/host/device_runner.cpp:262`), treating all launched AICPU instances as scheduling participants.

For `aicpu_build_graph`:

- Use `aicpu_thread_num=4` with `build_thread_num=1` and `schedule_thread_num=3`.
- Keep `block_dim` a multiple of `aicpu_thread_num` initially to avoid touching host-side validation (`src/platform/a2a3/host/device_runner.cpp:262`).

## Acceptance criteria

- A new runtime name `aicpu_build_graph` appears in `RuntimeBuilder.list_runtimes()` (see test style in `tests/test_runtime_builder.py:23`).
- Running the copied example on `a2a3sim` succeeds with default settings (total 4 AICPU instances).
- Tasks built on AICPU execute correctly on AICore (no `function_bin_addr == 0` cases for real tasks).
- Concurrent build∥schedule terminates correctly and does not deadlock/hang in simulation.
- Sequential build→schedule also works as a baseline (useful for debugging).

## Test plan

- Simulation:
  - Run `python examples/scripts/run_example.py -k examples/<new_example>/kernels -g examples/<new_example>/golden.py -p a2a3sim`.
  - Run both modes: sequential build→schedule and concurrent build∥schedule (how the mode is selected is part of the runtime design; e.g. `runtime.build_mode`).
- Unit tests:
  - Extend pytest to assert discovery of `aicpu_build_graph`.
- Diagnostics:
  - Use existing AICPU timeout diagnostics patterns as needed (see `src/runtime/host_build_graph/aicpu/aicpu_executor.cpp:545`).

## References (most relevant files)

- Host orchestration example: `examples/host_build_graph_example/kernels/orchestration/example_orch.cpp:20`
- Orchestration compilation in runner: `examples/scripts/code_runner.py:481`
- Host `dlopen` orchestration: `src/runtime/host_build_graph/host/runtime_maker.cpp:96`
- Host runtime launch/param injection: `src/platform/a2a3/host/device_runner.cpp:241`
- Kernel registration mapping: `src/platform/a2a3/host/device_runner.cpp:499`
- AICPU scheduler entry: `src/runtime/host_build_graph/aicpu/aicpu_executor.cpp:633`
- AICore dispatch from `function_bin_addr`: `src/runtime/host_build_graph/aicore/aicore_executor.cpp:43`
