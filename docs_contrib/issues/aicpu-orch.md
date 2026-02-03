# Issue: Add `aicpu_build_graph` runtime (AICPU builds + AICPU schedules)

## Summary

Today, this repo builds the task graph on the **host** (via a host orchestration `.so` executed during `Runtime.initialize()`), then runs an AICPU scheduler with `N` AICPU instances to dispatch tasks to AICore workers.

Requested change (translated): build the task graph on **AICPU** using **1** AICPU instance, then schedule/execute using **3** AICPU instances (total `4`).

This issue proposes a new runtime implementation named `aicpu_build_graph` under `src/runtime/`, plus a copied example, and documents the required design changes and risks.

## Background (current behavior, with references)

### Host orchestration builds the graph today

- The example orchestration function `build_example_graph()` constructs tensors *and* builds tasks/edges on the host: `examples/host_build_graph_example/kernels/orchestration/example_orch.cpp:20`.
- It is compiled as a **host `.so`** by `PTOCompiler.compile_orchestration()` in the runner: `examples/scripts/code_runner.py:545`.
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
- It interprets `runtime->orch_args[]` however it wants and calls the runtime-provided AICPU build APIs:
  - `aicpu_runtime_add_task()`
  - `aicpu_runtime_add_successor_conditional()`
  - `aicpu_runtime_publish_task()`

Execution mechanism (important): **link-time inclusion into the AICPU binary**, not device-side `dlopen`.

- The AICPU binary is already built from source using `CUSTOM_SOURCE_DIRS` (`src/platform/a2a3/aicpu/CMakeLists.txt:20`).
- The example provides the builder program under:
  - `examples/<example>/kernels/aicpu/build_graph_aicpu.cpp`
- The runtime build config (`src/runtime/aicpu_build_graph/build_config.py`) automatically adds `PTO_KERNELS_DIR/aicpu`
  into AICPU include/source dirs when present. The runner sets `PTO_KERNELS_DIR`:
  - `examples/scripts/code_runner.py:270`
- For runtime behavior knobs (like build/schedule mode), the example sets `RUNTIME_ENV` and the runner applies it
  while calling `Runtime.initialize()`: `examples/scripts/code_runner.py:599`

### Challenge C — Sequential vs concurrent build/schedule (and concrete algorithms)

Status: **implemented in `aicpu_build_graph`** (concurrent build∥schedule + sequential build→schedule). `host_build_graph` still assumes the graph is fully built and immutable during scheduling.

We require concurrent build∥schedule, and we also support sequential build→schedule as a baseline.

#### Mode 1: Sequential build→schedule (supported baseline)

Implementation:

- Builder/scheduler split is fixed in the executor:
  - `BUILDER_THREAD_NUM = 1`: `src/runtime/aicpu_build_graph/aicpu/aicpu_executor.cpp:13`
  - Scheduler threads = `thread_num - BUILDER_THREAD_NUM`: `src/runtime/aicpu_build_graph/aicpu/aicpu_executor.cpp:100`
- Mode is selected via `Runtime::build_mode` (0=sequential, 1=concurrent):
  - `src/runtime/aicpu_build_graph/runtime/runtime.h:243`
  - Set at init time by `PTO_AICPU_BUILD_GRAPH_BUILD_MODE`: `src/runtime/aicpu_build_graph/host/runtime_maker.cpp:149`
- Barrier:
  - Builder thread runs `build_graph_aicpu(runtime)`, then sets `build_done_`: `src/runtime/aicpu_build_graph/aicpu/aicpu_executor.cpp:557`
  - Scheduler threads wait for `build_done_` when `build_mode==0`: `src/runtime/aicpu_build_graph/aicpu/aicpu_executor.cpp:569`

Core assignment implication:

- Scheduler threads manage AICore workers; the builder thread manages **zero** worker cores:
  - `src/runtime/aicpu_build_graph/aicpu/aicpu_executor.cpp:225`

Termination:

- Scheduling ends when:
  - builder is done,
  - `completed_tasks >= published_tasks`,
  - ready queues are empty,
  - all managed cores are idle (with a bounded re-check loop to avoid tearing): `src/runtime/aicpu_build_graph/aicpu/aicpu_executor.cpp:416`.

#### Mode 2: Concurrent build∥schedule (required)

Correct-first strategy (recommended initial approach): **global graph mutex**.

Executor state (implemented):

- `std::mutex graph_mutex_` to guard task/edge publication: `src/runtime/aicpu_build_graph/aicpu/aicpu_executor.cpp:48`
- `published_tasks_` counter + `build_done_`: `src/runtime/aicpu_build_graph/aicpu/aicpu_executor.cpp:61`

Publication rule:

- Builder publishes tasks via the C ABI helpers:
  - `aicpu_runtime_add_task()`: sets `Task::function_bin_addr` (supports `function_bin_addr==0` → `kernel_addrs[func_id]`) under the mutex: `src/runtime/aicpu_build_graph/aicpu/aicpu_executor.cpp:762`
  - `aicpu_runtime_add_successor_conditional()`: appends fanout and conditionally increments fanin under the mutex: `src/runtime/aicpu_build_graph/aicpu/aicpu_executor.cpp:808`
  - `aicpu_runtime_publish_task()`: marks published, increments `published_tasks_`, and enqueues if `fanin==0`: `src/runtime/aicpu_build_graph/aicpu/aicpu_executor.cpp:831`

Scheduler behavior changes:

- When a core completes a task, the scheduler thread acquires `graph_mutex_` before walking `fanout[]` and decrementing successor `fanin` so the builder can append edges safely: `src/runtime/aicpu_build_graph/aicpu/aicpu_executor.cpp:463`.

Termination condition (must include build progress):

Schedulers may exit only when all of the following are true:

- `build_done == 1`
- `completed_tasks >= published_tasks`
- ready queues are empty
- all managed cores are idle (retain the existing “core idle verification” idea; see `src/runtime/host_build_graph/aicpu/aicpu_executor.cpp:333`)

Why this is “no redundancy”: it builds on the existing scheduler structure (ready queues, polling, verification), and adds only a lock + a publication counter.

Faster alternative (deferred): publish-only DAG append (requires data-structure redesign away from inline `fanout[]`).

### Challenge D — Thread configuration (fixed 1+3)

Constraints today:

- Platform max AICPU threads is `PLATFORM_MAX_AICPU_THREADS = 4` (`src/platform/include/common/platform_config.h:41`).
- Host validates `block_dim % launch_aicpu_num == 0` (`src/platform/a2a3/host/device_runner.cpp:262`), treating all launched AICPU instances as scheduling participants.

For `aicpu_build_graph`:

- Use `aicpu_thread_num=4` (1 builder + 3 schedulers). In this repo:
  - The runner picks 4 threads for `aicpu_build_graph`: `examples/scripts/code_runner.py:372`
  - Thread 0 is the builder (`BUILDER_THREAD_NUM=1`): `src/runtime/aicpu_build_graph/aicpu/aicpu_executor.cpp:13`
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
  - Run both modes by setting `RUNTIME_ENV["PTO_AICPU_BUILD_GRAPH_BUILD_MODE"]` in the example’s `kernel_config.py`:
    - `"1"` = concurrent build∥schedule (required default)
    - `"0"` = sequential build→schedule (debug baseline)
    The env is applied during `Runtime.initialize()` so `init_runtime_impl()` can set `Runtime::build_mode`.
- Unit tests:
  - Extend pytest to assert discovery of `aicpu_build_graph`.
- Diagnostics:
  - Use existing AICPU timeout diagnostics patterns as needed (see `src/runtime/host_build_graph/aicpu/aicpu_executor.cpp:545`).

## Implementation status (as of current branch)

- Done:
  - Runtime skeleton: `src/runtime/aicpu_build_graph/` (host/aicpu/aicore + `runtime/runtime.h`)
  - Example builder moved out of runtime:
    - `examples/aicpu_build_graph_example/kernels/aicpu/build_graph_aicpu.cpp`
    - Runtime provides a weak default that errors if the example forgets it: `src/runtime/aicpu_build_graph/aicpu/build_graph_default.cpp`
  - Concurrency-safe AICPU build APIs + logs:
    - `aicpu_runtime_add_task`, `aicpu_runtime_add_successor_conditional`, `aicpu_runtime_publish_task`
  - Mode selection wired for simulation runs:
    - `RUNTIME_ENV["PTO_AICPU_BUILD_GRAPH_BUILD_MODE"]` → `Runtime::build_mode`
- Pending verification:
  - None (validated via `a2a3sim` for `host_build_graph_example` and `aicpu_build_graph_example` in both concurrent and sequential modes).

## References (most relevant files)

- Host orchestration example: `examples/host_build_graph_example/kernels/orchestration/example_orch.cpp:20`
- Orchestration compilation in runner: `examples/scripts/code_runner.py:545`
- Host `dlopen` orchestration: `src/runtime/host_build_graph/host/runtime_maker.cpp:96`
- Host runtime launch/param injection: `src/platform/a2a3/host/device_runner.cpp:241`
- Kernel registration mapping: `src/platform/a2a3/host/device_runner.cpp:499`
- AICPU scheduler entry: `src/runtime/host_build_graph/aicpu/aicpu_executor.cpp:633`
- AICore dispatch from `function_bin_addr`: `src/runtime/host_build_graph/aicore/aicore_executor.cpp:43`
