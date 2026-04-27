# Unit Test Suite Reference

Comprehensive reference for the unit tests under `tests/ut/`. For build
commands, hardware classification, and CI integration see
[testing.md](testing.md) and [ci.md](ci.md).

## Directory Layout

```text
tests/ut/
├── cpp/                              # C++ GoogleTest binaries (CMake)
│   ├── CMakeLists.txt                # Build orchestration and helper functions
│   ├── test_helpers.h                # Shared test utilities
│   ├── stubs/
│   │   └── test_stubs.cpp            # Platform-abstraction stubs (logging, asserts)
│   ├── hierarchical/                 # Host-side hierarchical runtime (L0-L6)
│   │   ├── test_tensormap.cpp
│   │   ├── test_ring.cpp
│   │   ├── test_scope.cpp
│   │   ├── test_orchestrator.cpp
│   │   ├── test_scheduler.cpp
│   │   └── test_worker_manager.cpp
│   ├── platform/                     # Platform abstraction layer (sim variant)
│   │   ├── test_platform_memory_allocator.cpp
│   │   └── test_platform_host_log.cpp
│   ├── types/                        # Cross-cutting ABI-contract types
│   │   ├── test_child_memory.cpp
│   │   ├── test_pto_types.cpp
│   │   └── test_tensor.cpp
│   ├── pto2_a2a3/                    # PTO2 on-chip runtime (A2A3 architecture)
│   │   ├── test_a2a3_pto2_fatal.cpp
│   │   ├── test_core_types.cpp
│   │   ├── test_dispatch_payload.cpp
│   │   ├── test_handshake.cpp
│   │   ├── test_submit_types.cpp
│   │   ├── test_ring_buffer.cpp
│   │   ├── test_ring_buffer_edge.cpp
│   │   ├── test_tensormap_edge.cpp
│   │   ├── test_ready_queue.cpp
│   │   ├── test_scheduler_state.cpp
│   │   ├── test_scheduler_edge.cpp
│   │   ├── test_shared_memory.cpp
│   │   ├── test_boundary_edge.cpp
│   │   ├── test_coupling.cpp
│   │   ├── test_coupling_stub.cpp
│   │   ├── test_runtime_graph.cpp
│   │   ├── test_runtime_lifecycle.cpp
│   │   ├── test_runtime_status.cpp
│   │   ├── test_orchestrator_submit.cpp
│   │   └── test_orchestrator_fatal.cpp
│   ├── pto2_a5/                      # PTO2 on-chip runtime (A5 architecture)
│   │   └── test_a5_pto2_fatal.cpp
│   └── hardware/                     # Hardware-gated tests (CANN required)
│       └── test_hccl_comm.cpp
└── py/                               # Python pytest-based tests
    ├── conftest.py                   # Fixtures, sys.path setup
    ├── test_elf_parser.py
    ├── test_env_manager.py
    ├── test_kernel_compiler.py
    ├── test_runtime_compiler.py
    ├── test_toolchain.py
    ├── test_toolchain_setup.py
    ├── test_task_interface.py
    ├── test_runtime_builder.py
    ├── test_chip_worker.py
    ├── test_hostsub_fork_shm.py
    ├── test_worker/                  # Worker subsystem tests
    │   ├── test_host_worker.py
    │   ├── test_bootstrap_channel.py
    │   ├── test_bootstrap_context_hw.py
    │   ├── test_bootstrap_context_sim.py
    │   ├── test_error_propagation.py
    │   ├── test_group_task.py
    │   ├── test_l4_recursive.py
    │   ├── test_mailbox_atomics.py
    │   ├── test_multi_worker.py
    │   ├── test_platform_comm.py
    │   ├── test_worker_distributed_hw.py
    │   └── test_worker_distributed_sim.py
```

## Organization Principles

### Subdirectory-per-component

C++ tests are grouped by the source component they exercise:

| Subdirectory | Source under test | CMake helper |
| ------------ | ----------------- | ------------ |
| `hierarchical/` | `src/common/hierarchical/` | `add_hierarchical_test` |
| `platform/` | `src/a2a3/platform/` (sim variant) | inline targets |
| `types/` | `src/common/task_interface/` | `add_task_interface_test` |
| `pto2_a2a3/` | `src/a2a3/runtime/tensormap_and_ringbuffer/` | `add_a2a3_pto2_test` / `add_a2a3_pto2_runtime_test` |
| `pto2_a5/` | `src/a5/runtime/tensormap_and_ringbuffer/` | `add_a5_pto2_test` |
| `hardware/` | HCCL comm backend (needs CANN) | `add_comm_api_test` |

Python tests are grouped by functional area: build infrastructure
(compilers, toolchain, ELF parsing), nanobind bindings, and the worker
subsystem.

### Header-only vs runtime-linked

PTO2 tests come in two flavors:

- **Header-only** (`add_a2a3_pto2_test`): compile against orchestration/
  runtime headers only. No `.cpp` from the runtime is linked. Used for
  type-layout, constant, and API-contract tests.
- **Runtime-linked** (`add_a2a3_pto2_runtime_test`): link the real
  `pto_ring_buffer.cpp`, `pto_shared_memory.cpp`, `pto_scheduler.cpp`,
  `pto_tensormap.cpp` (and optionally `pto_orchestrator.cpp`,
  `pto_runtime2.cpp`). Used for behavioral and integration tests.

### Hardware gating

All tests default to `no_hardware` (runnable on standard CI runners). Tests
that need Ascend hardware are gated by:

- **C++**: `SIMPLER_ENABLE_HARDWARE_TESTS` CMake option + ctest labels
  (`requires_hardware_a2a3`).
- **Python**: `@pytest.mark.requires_hardware` / `requires_hardware("a2a3")`
  markers.

### Test-design conventions

- **AAA pattern**: Arrange-Act-Assert structure in each test.
- **Fixtures over globals**: GoogleTest fixtures (`TEST_F`) manage per-test
  state; pytest fixtures handle setup/teardown.
- **Stubs for platform isolation**: `stubs/test_stubs.cpp` provides logging,
  assertion, and timer stubs so on-chip runtime code compiles on x86/macOS
  without CANN dependencies.
- **Edge-case files**: Files named `*_edge.cpp` focus on boundary conditions,
  concurrency stress, and design-contract verification.

## Test Design Philosophy

The suite targets three goals:

1. **ABI contract verification** — `sizeof`, `alignof`, field offsets, and
   enum values are checked with `static_assert` and runtime assertions.
   This catches silent layout drift when headers change.

2. **Component isolation** — each test exercises one module with minimal
   dependencies. Coupling tests (`test_coupling.cpp`,
   `test_coupling_stub.cpp`) explicitly measure and document inter-component
   dependencies.

3. **Bug-candidate documentation** — edge-case tests encode known defects
   and design tradeoffs as executable tests. When a test documents a real
   src defect, it is preserved as a regression barrier. When a test
   documents intentional design (e.g., LIFO dispatch order), it serves as
   a contract anchor.

## Coverage Map

### C++ — Hierarchical Runtime (`hierarchical/`)

Source: `src/common/hierarchical/`

| File | Tests | What it covers |
| ---- | ----- | -------------- |
| `test_tensormap.cpp` | 4 | Insert, lookup, overwrite, erase by task ID. Compound keys (pointer + worker ID). |
| `test_ring.cpp` | 5+ | Slot allocation monotonicity, heap slab alignment, FIFO reclamation, allocation bounds, back-pressure with small heap (8 KiB). |
| `test_scope.cpp` | 5 | Scope depth tracking, begin/end pairing, nested scopes, task registration and release callbacks, empty scope handling. |
| `test_orchestrator.cpp` | 1+ | Wiring TensorMap + Ring + Scope + ReadyQueues into a full Orchestrator. Independent-task readiness detection. |
| `test_scheduler.cpp` | 2+ | MockWorker-based dispatch verification. Single-task and task-group dispatch through Scheduler + WorkerManager integration. |
| `test_worker_manager.cpp` | 4+ | Worker pool lifecycle (THREAD mode), idle worker selection, dispatch, group dispatch. CountingWorker tracks run() calls. |

### C++ — Platform Abstraction (`platform/`)

Source: `src/a2a3/platform/`

| File | Tests | What it covers |
| ---- | ----- | -------------- |
| `test_platform_memory_allocator.cpp` | 4 | Sim memory allocator: allocation tracking, multi-allocation, nullptr safety, untracked-pointer handling. |
| `test_platform_host_log.cpp` | 3+ | HostLogger singleton: level filtering (`is_enabled`), env-var parsing (`PTO_LOG_LEVEL`), `reinitialize()` behavior. |

### C++ — Cross-cutting Types (`types/`)

Source: `src/common/task_interface/`

| File | Tests | What it covers |
| ---- | ----- | -------------- |
| `test_child_memory.cpp` | 3 | `ContinuousTensor` ABI layout (`sizeof == 40`), `child_memory` bit field, blob serialization roundtrip. |
| `test_pto_types.cpp` | 5+ | `TaskOutputTensors` init/materialize/get_ref/max-outputs, `Arg` tensor/scalar storage, `add_scalars_i32` zero-extension, `copy_scalars_from`. |
| `test_tensor.cpp` | 5+ | Segment intersection logic (overlapping, touching, disjoint, zero-length), `make_tensor_external()` factory, cache-line layout coupling. |

### C++ — PTO2 A2A3 On-chip Runtime (`pto2_a2a3/`)

Source: `src/a2a3/runtime/tensormap_and_ringbuffer/`

#### API and type contracts (header-only)

| File | Tests | What it covers |
| ---- | ----- | -------------- |
| `test_a2a3_pto2_fatal.cpp` | 3+ | Fatal-path reporting through `pto2_orchestration_api.h`. Fake runtime + va_list formatting. |
| `test_core_types.cpp` | 5 | `PTO2TaskId` encode/extract (ring in upper 32, local in lower 32), roundtrip, `PTO2TaskSlotState` size (64 bytes), `PTO2_ALIGN_UP` macro. |
| `test_dispatch_payload.cpp` | 5+ | `PTO2DispatchPayload` 64-byte alignment, SPMD context index constants, `LocalContext`/`GlobalContext` field read/write. |
| `test_handshake.cpp` | 4+ | Handshake protocol macros: `MAKE_ACK_VALUE`/`MAKE_FIN_VALUE`, `EXTRACT_TASK_ID`/`EXTRACT_TASK_STATE`, bit-31 state encoding, reserved task IDs. |
| `test_submit_types.cpp` | 3+ | `pto2_subtask_active()` bitmask (AIC, AIV0, AIV1), `pto2_active_mask_to_shape()`, `pto2_mixed_kernels_to_active_mask()`. |
| `test_runtime_status.cpp` | 9 | `pto2_runtime_status()`: zero codes, single-error negation, precedence rules (orch > sched), pass-through for already-negative codes, range non-overlap. |

#### Ring buffer and memory allocation (runtime-linked)

| File | Tests | What it covers |
| ---- | ----- | -------------- |
| `test_ring_buffer.cpp` | 10+ | `PTO2TaskAllocator` init, state queries, window size, heap allocation, FIFO reclamation, wrap-guard boundary. |
| `test_ring_buffer_edge.cpp` | 10+ | Edge cases: wrap-guard at `tail==alloc_size`, fragmentation reporting (`max` not `sum`), zero-size allocation, exact-heap-size allocation, oversized allocation, window saturation, slot mapping, task ID near INT32_MAX. `DepListPool` edge cases: contract violation, prepend chain, high-water mark, overflow error code. |

#### TensorMap (runtime-linked)

| File | Tests | What it covers |
| ---- | ----- | -------------- |
| `test_tensormap_edge.cpp` | 15+ | Bug-candidate documentation: `check_overlap()` dimension mismatch, lookup saturation (16-producer limit), pool exhaustion, ABA in `cleanup_retired()`, `copy_from_tensor` zero-padding. Edge cases: 0-dim tensors, max-dim tensors, zero-length shapes. |

#### Scheduler and ready queue (runtime-linked)

| File | Tests | What it covers |
| ---- | ----- | -------------- |
| `test_ready_queue.cpp` | 17 | `ReadyQueue` MPMC: empty pop, single push/pop, FIFO ordering, capacity limit, slot reuse, batch push/pop, size accuracy. Multi-threaded: 2P/2C and 1P/4C stress. `LocalReadyBuffer` LIFO: reset, ordering, overflow. |
| `test_scheduler_state.cpp` | 5+ | `init_slot()` helper, `check_and_handle_consumed` transitions (COMPLETED to CONSUMED), fanin/fanout reference counting. |
| `test_scheduler_edge.cpp` | 25+ | `ReadyQueue` edge cases: interleaved push/pop, exact-capacity fill/drain, relaxed-ordering size guard, high-contention stress (4P/4C, 5000 items). `LocalReadyBuffer` LIFO dispatch order, overflow, null backing. `SharedMem` edge: zero window size, corruption detection, undersized buffer, region non-overlap, header alignment. `TaskState` lifecycle: PENDING to CONSUMED, simultaneous subtask completion, fanin/fanout exactly-once semantics, invalid transitions. |

#### Shared memory (runtime-linked)

| File | Tests | What it covers |
| ---- | ----- | -------------- |
| `test_shared_memory.cpp` | 6+ | `PTO2SharedMemoryHandle` create/destroy, ownership, header init values, per-ring independence, pointer alignment (`PTO2_ALIGN_SIZE`), `calculate_size()`. |

#### Boundary and stress tests (runtime-linked)

| File | Tests | What it covers |
| ---- | ----- | -------------- |
| `test_boundary_edge.cpp` | 17+ | `ReadyQueue` stress: 8P/8C, rapid fill/drain cycles, batch contention. `TaskAllocator` re-init: reset counter, heap, error state, multi-cycle, stale `last_alive`. Sequence wrap near INT64_MAX: single, fill/drain, interleaved, batch, concurrent. `SharedMemory` concurrency: per-ring isolation, atomic increment, `orchestrator_done` race, monotonic advancement, validate after concurrent writes. |

#### Coupling analysis (runtime-linked)

| File | Tests | What it covers |
| ---- | ----- | -------------- |
| `test_coupling.cpp` | 4+ | Architectural coupling detection: whether components can operate in isolation. `TMRSystem` full init/destroy measuring dependency graph. |
| `test_coupling_stub.cpp` | 14 | `DepPool` stub isolation: reclaim below/at interval. Scheduler without orchestrator: init/destroy, standalone `ReadyQueue`, fanin release, non-profiling path, mixed-task completion. `TensorMap` link decoupling: builds without `orchestrator.cpp`, orchestrator pointer never dereferenced in hot path. Compile-time include coupling: `RingBuffer` to `Scheduler`, duplicated slot-mask formula, `PTO2_MAX_RING_DEPTH` in 4 components, transitive includes. Profiling behavior: CAS guard in profiling vs atomicity in non-profiling. |

#### Orchestrator (runtime-linked)

| File | Tests | What it covers |
| ---- | ----- | -------------- |
| `test_orchestrator_submit.cpp` | 12 | `set_scheduler`, `alloc_tensors` validation (empty/scalar/input args mark fatal), output-only materialization, post-fatal short-circuit, submit with error args, pure-input submit, output materialization, `orchestrator_done` idempotency. |
| `test_orchestrator_fatal.cpp` | 11 | Fatal error latching: initial state, `report_fatal` sets local flag + shared code, second report does not overwrite, `ERROR_NONE` does not latch, all 9 error codes latch correctly, null/empty/varargs format strings, status helper reads latched code. |

#### Runtime lifecycle (runtime-linked)

| File | Tests | What it covers |
| ---- | ----- | -------------- |
| `test_runtime_lifecycle.cpp` | 12 | `pto2_runtime_create_custom` initialization, orchestrator-to-scheduler connection, default creation, null SM handle, caller-allocated buffers, null-safe destroy, heap release, `set_mode`, ops table population, `is_fatal` / `report_fatal`. |

#### Runtime graph — host_build_graph (runtime-linked)

| File | Tests | What it covers |
| ---- | ----- | -------------- |
| `test_runtime_graph.cpp` | 10 | `RuntimeGraph`: monotonic task IDs, field storage, successor updates (fanout/fanin), ready-task detection, diamond DAG, linear chain, fanout/fanin consistency, max-task limit, tensor-pair management, function binary address mapping. |

### C++ — PTO2 A5 On-chip Runtime (`pto2_a5/`)

Source: `src/a5/runtime/tensormap_and_ringbuffer/`

| File | Tests | What it covers |
| ---- | ----- | -------------- |
| `test_a5_pto2_fatal.cpp` | 3 | API short-circuit after fatal, explicit fatal routing through ops table, `alloc_tensor` overflow reports invalid args instead of asserting. |

### C++ — Hardware Tests (`hardware/`)

Gated by `SIMPLER_ENABLE_HARDWARE_TESTS=ON`. Labeled
`requires_hardware_a2a3`.

| File | Tests | What it covers |
| ---- | ----- | -------------- |
| `test_hccl_comm.cpp` | 3+ | HCCL backend lifecycle: `dlopen(libhost_runtime.so)`, comm init/alloc/query/destroy. CTest resource allocation for 2-device tests. |

### Python — Build Infrastructure (`py/`)

| File | Tests | What it covers |
| ---- | ----- | -------------- |
| `test_elf_parser.py` | 3+ | ELF64 and Mach-O `.text` section extraction from raw struct-packed binaries. `_extract_cstring`, `extract_text_section`. |
| `test_env_manager.py` | 5+ | `env_manager.get()`, `ensure()`, caching behavior, error on unset/empty vars. Uses `monkeypatch` for env isolation. |
| `test_kernel_compiler.py` | 4+ | Platform include dirs (a2a3 vs a5), orchestration include dirs. Mock `ASCEND_HOME_PATH` fixture. |
| `test_runtime_compiler.py` | 4+ | `BuildTarget` CMake arg generation, `root_dir` absoluteness, `binary_name`, `RuntimeCompiler` singleton reset. |
| `test_toolchain.py` | 5+ | `_parse_compiler_env()` for conda flags, `GxxToolchainCmakeArgs` (plain/conda env, quoted paths, CMAKE_C/CXX_FLAGS). |
| `test_toolchain_setup.py` | 18 | CCEC toolchain compile flags (a2a3/a5, aic/aiv), unknown platform, missing compiler. Gxx15 toolchain (`__DAV_VEC__`/`__DAV_CUBE__` defines, `__CPU_SIM`). Gxx/Aarch64Gxx cmake args, env vars, cross-compile. `ToolchainType` enum values. |
| `test_runtime_builder.py` | 16 | Runtime discovery (real project tree), config resolution, missing/empty dirs, sorted output. `get_binaries()` error handling, compiler invocation count, path resolution, error propagation. Integration: real compilation produces non-empty `.so` files. |

### Python — Nanobind and Type Contracts (`py/`)

| File | Tests | What it covers |
| ---- | ----- | -------------- |
| `test_task_interface.py` | 10+ | `DataType` enum ABI values (FLOAT32, FLOAT16, INT32, ...), `get_element_size()` parametrized, nanobind `_task_interface` extension (`ContinuousTensor`, `TaskArgs`, `ChipStorageTaskArgs`), torch integration. |

### Python — ChipWorker and Fork/SHM (`py/`)

| File | Tests | What it covers |
| ---- | ----- | -------------- |
| `test_chip_worker.py` | 11 | `ChipCallConfig` defaults/setters/repr. `ChipWorker` state machine: uninitialized state, run-before-set-device, set-device-before-init, reset/finalize idempotency, init-after-finalize, nonexistent lib. Python import verification. |
| `test_hostsub_fork_shm.py` | 6 | `SharedMemory` cross-fork access. `torch.share_memory_()` mutations across fork. Callable registry in forked child. Mailbox state machine (IDLE/TASK_READY/TASK_DONE cycling). Parallel wall-time verification (3 SubWorkers). Threading after fork. |

### Python — Worker Subsystem (`py/test_worker/`)

| File | Tests | What it covers |
| ---- | ----- | -------------- |
| `test_host_worker.py` | 18 | Worker lifecycle (init/close, context manager, register-after-init). Single sub-task execution and multiple runs. `submit_sub()` return type. Scope management (run-managed, user-nested, 3-deep nesting). `alloc()` tensor validity, dependency wiring, unused-freed, no-leak across runs. Sub-callable receives tensor metadata, scalar, empty args. |
| `test_bootstrap_channel.py` | 7 | `BootstrapChannel` state machine: fresh=IDLE, write success/error fields, reset transition, cross-process fork, buffer-ptr overflow, error message truncation. |
| `test_bootstrap_context_hw.py` | 1 | 2-rank hardware smoke: `ChipWorker.bootstrap_context` populates device_ctx, window_base, window_size, buffer_ptrs. |
| `test_bootstrap_context_sim.py` | 4 | 2-rank sim bootstrap, `load_from_host` roundtrip, channel SUCCESS fields, invalid-placement error publishing. |
| `test_error_propagation.py` | 5 | Sub-worker exception surfacing (type/message preserved), missing callable_id, failure-does-not-wedge (next run succeeds), post-failure submit re-raises, L4-chained failure surfaces with layer prefixes. |
| `test_group_task.py` | 3 | `submit_sub_group` with 2 args dispatches to 2 SubWorkers, single-arg group, group-then-dependent-task ordering. |
| `test_l4_recursive.py` | 13 | L4 lifecycle (no children, with L3 child, context manager). Validation (level check, add-after-init, initialized-child). L4-to-L3 dispatch (single, triple, with own subs). Multiple runs no-leak. L3 child with multiple subs. L3 own orchestrator. Generalized `_Worker` level parameter. |
| `test_mailbox_atomics.py` | 6 | `_mailbox_store_i32`/`load_i32` roundtrip (positive, negative, offset). Cross-process visibility via `MAP_SHARED`. Release/acquire ordering: payload visible when state observed. L3 sub-worker dispatch roundtrip. |
| `test_multi_worker.py` | 3 | Two-worker parallel execution with thread-local isolation. Sequential task stress (20 tasks, 1 SubWorker). 20 tasks across 2 SubWorkers, all complete exactly once. |
| `test_platform_comm.py` | 1 | 2-rank hardware smoke: `comm_init` to `comm_destroy` lifecycle (barrier failure tolerated per HCCL 507018). |
| `test_worker_distributed_hw.py` | 1 | 2-rank hardware smoke: `Worker(chip_bootstrap_configs=...)` populates `chip_contexts` with device_ctx, window_base, buffer_ptrs per rank. No `comm_barrier`. |
| `test_worker_distributed_sim.py` | 5 | Worker-level chip bootstrap on sim: happy-path `chip_contexts` population + `/dev/shm` leak check, pre-init access rejection, invalid placement error path + cleanup, level-below-3 rejection, config/device_ids length mismatch. |

## Test Counts Summary

| Category | Files | Approx. test cases |
| -------- | ----- | ------------------ |
| C++ hierarchical | 6 | 20+ |
| C++ platform | 2 | 7+ |
| C++ types | 3 | 13+ |
| C++ PTO2 A2A3 | 19 | 180+ |
| C++ PTO2 A5 | 1 | 3 |
| C++ hardware | 1 | 3+ |
| Python build infra | 6 | 50+ |
| Python nanobind | 1 | 10+ |
| Python ChipWorker/fork | 2 | 17 |
| Python worker subsystem | 12 | 67+ |
| **Total** | **53** | **371+** |

## Infrastructure

### CMake Helper Functions

| Function | Linker scope | Use for |
| -------- | ------------ | ------- |
| `add_hierarchical_test(name src)` | Full hierarchical runtime sources | Tests under `hierarchical/` |
| `add_task_interface_test(name src)` | Header-only (`task_interface/`) | ABI-contract tests under `types/` |
| `add_a2a3_pto2_test(name src)` | Header-only (orchestration + runtime headers) | PTO2 type/constant tests |
| `add_a2a3_pto2_runtime_test(name SOURCES ... EXTRA_SOURCES ...)` | Stubs + selected runtime `.cpp` files | Behavioral PTO2 tests |
| `add_a5_pto2_test(name src)` | Header-only (A5 orchestration + runtime) | A5-specific tests |
| `add_comm_api_test(name src)` | CANN `libascendcl` + `dlopen` | Hardware-gated HCCL tests |

### Platform Stubs (`stubs/test_stubs.cpp`)

Provides userspace implementations for symbols that on-chip runtime code
expects from the AICPU environment:

- `unified_log_{error,warn,info,debug,always}` — logging (stderr)
- `get_sys_cnt_aicpu()` — timer stub (returns 0)
- `get_stacktrace()` — stack trace (returns empty string)
- `assert_impl()` — assertion handler (throws `AssertionError`)

This allows the full runtime `.cpp` files to compile and link on
x86_64/aarch64/macOS without CANN.

### Python conftest (`py/conftest.py`)

- Adds `PROJECT_ROOT` to `sys.path` for `import simpler_setup`
- Adds `python/` for `from simpler import env_manager`
- Adds `python/simpler/` for legacy `import env_manager` compatibility
- Provides `project_root` fixture returning the `PROJECT_ROOT` `Path`

### Test Helpers (`test_helpers.h`)

- `test_ready_queue_init()` — initialize a `ReadyQueue` with
  caller-provided buffer and arbitrary start sequence number
