# Testing

## Design Principles

Three axioms govern hardware classification across all test categories:

1. **sim = no hardware**: `a2a3sim`, `a5sim`, and no `--platform` are all equivalent — no hardware required.
2. **a2a3 / a5 are distinct platforms**: Tests may support one, both, or neither.
3. **`requires_hardware` has two levels**:
   - `requires_hardware` (no argument) — needs *any* hardware (a2a3 or a5)
   - `requires_hardware("a2a3")` — needs *specifically* a2a3

These principles apply uniformly to ut-py (pytest markers), ut-cpp (ctest labels), and st (`@scene_test(platforms=[...])`).

## Quick Reference

```bash
# Python unit tests, no hardware (sim or github-hosted)
pytest tests/ut

# Python unit tests, a2a3 hardware
pytest tests/ut --platform a2a3

# C++ unit tests, no hardware
cmake -B tests/ut/cpp/build -S tests/ut/cpp && cmake --build tests/ut/cpp/build
ctest --test-dir tests/ut/cpp/build -LE requires_hardware --output-on-failure

# C++ unit tests, a2a3 hardware (only hw + a2a3-specific tests)
ctest --test-dir tests/ut/cpp/build -L "^requires_hardware(_a2a3)?$" --output-on-failure

# Scene tests (pytest, @scene_test classes)
pytest examples tests/st                          # all sim platforms (auto-parametrized)
pytest examples tests/st --platform a2a3sim       # specific sim
pytest examples tests/st --platform a2a3          # hardware
pytest examples tests/st --platform a2a3 --device 4-7  # hardware with device pool

# Single scene test (standalone)
python examples/a2a3/tensormap_and_ringbuffer/vector_example/test_vector_example.py -p a2a3sim

# Standalone with build-from-source
python examples/a2a3/tensormap_and_ringbuffer/vector_example/test_vector_example.py -p a2a3sim --build

# Benchmark mode (100 rounds, skip golden comparison)
python examples/a2a3/tensormap_and_ringbuffer/vector_example/test_vector_example.py \
    -p a2a3 -d 0 --rounds 100 --skip-golden

# Profiling (first round only)
python examples/a2a3/tensormap_and_ringbuffer/vector_example/test_vector_example.py \
    -p a2a3 --enable-profiling

# Tensor dump
python tests/st/a2a3/tensormap_and_ringbuffer/alternating_matmul_add/test_alternating_matmul_add.py \
    -p a2a3 -d 11 --dump-tensor
```

## Test Organization

Three test categories:

| Category | Abbrev | Location | Runner | Description |
| -------- | ------ | -------- | ------ | ----------- |
| System tests | st | `examples/`, `tests/st/` | pytest (`@scene_test`) or standalone `python test_*.py` | Full end-to-end cases (compile + run + validate) |
| Python unit tests | ut-py | `tests/ut/` | pytest | Unit tests for nanobind-exposed and Python modules |
| C++ unit tests | ut-cpp | `tests/ut/cpp/` | ctest (GoogleTest) | Unit tests for pure C++ modules |

### Choosing ut-py vs ut-cpp

If a module is exposed via nanobind (used by both C++ and Python), test in **ut-py** (`tests/ut/`).
If a module is pure C++ with no Python binding, test in **ut-cpp** (`tests/ut/cpp/`).

## Scene Test CLI Options

Scene tests support advanced CLI options for benchmarking, profiling, and runtime control. These work identically in both pytest and standalone mode.

### pytest

```bash
pytest --platform a2a3sim                                        # default: 1 round + golden
pytest --platform a2a3 --rounds 100 --skip-golden                # benchmark mode
pytest --platform a2a3 --enable-profiling                        # profiling (first round)
pytest --platform a2a3sim --build                                # compile runtime from source
pytest --platform a2a3sim --log-level debug                        # verbose C++ logging
```

### Standalone (test_*.py)

```bash
python test_xxx.py -p a2a3sim                                    # default: 1 round + golden
python test_xxx.py -p a2a3 -d 0 --rounds 100 --skip-golden       # benchmark mode
python test_xxx.py -p a2a3 --enable-profiling                    # profiling (first round)
python test_xxx.py -p a2a3 --dump-tensor                         # dump per-task tensor I/O
python test_xxx.py -p a2a3sim --build                            # compile runtime from source
python test_xxx.py -p a2a3sim --log-level debug                  # verbose C++ logging
```

### Option Reference

| Option | Short | Default | Description |
| ------ | ----- | ------- | ----------- |
| `--rounds N` | | 1 | Run each case N times (reuses the same Worker across rounds) |
| `--device IDS` | `-d` | `0` | Single id (`0`), range (`0-7`), or list (`0,2,5`). Sets the device-id pool for L3 cases and the available slots for L2 fanout. |
| `--max-parallel N` | | `auto` | Max in-flight subprocesses (make-style). `auto` = `min(nproc, len(--device))` on sim, `len(--device)` on hardware. Decouples device-id pool size from parallelism; use to throttle sim on a CPU-constrained runner. |
| `--runtime NAME` | | (all) | Restrict to one runtime (also used internally as the child-mode marker) |
| `--level {2,3}` | | (all) | Restrict to one SceneTestCase level (also the child-mode marker) |
| `--case SEL` | | (all) | Case selector, repeatable: `Foo`, `ClassA::Foo`, `ClassA::` |
| `--manual` | | `exclude` | `exclude`/`include`/`only` for manual cases |
| `--skip-golden` | | false | Skip golden comparison (for benchmarking) |
| `--enable-profiling` | | false | Enable profiling on first round only. Blocked when `--max-parallel > 1`. |
| `--dump-tensor` | | false | Dump per-task tensor I/O during runtime execution |
| `--build` | | false | Compile runtime from source (not pre-built) |
| `--exitfirst` | `-x` | false | Stop on first failing test (fail-fast, primarily for CI) |
| `--log-level LEVEL` | | (none) | Set `PTO_LOG_LEVEL` env var (`error`/`warn`/`info`/`debug`) |

Profiling is enabled only on the first round to avoid overhead on subsequent iterations. Output tensors are reset to their initial values between rounds.

## CLI Design Principles

The same set of flags must work in both pytest and standalone (`python test_*.py`). Two rules govern which short forms we register:

1. **Mirror pytest.** If pytest (or a pytest plugin the project depends on) exposes a flag with a particular short form, standalone must register the same short for the same meaning. Users routinely copy commands between the two entry points; letter-level semantic drift is the fastest way to create confusing bugs.
2. **Never create a collision with pytest's ecosystem.** If pytest (or one of its plugins) already uses a short letter for a *different* concept, standalone must not reuse that letter for anything else, even if pytest doesn't use that letter for the flag we're adding. The goal is that `-X` in both worlds either does the same thing or is unused.

Worked examples:

| Flag | Long form | Short | Why |
| ---- | --------- | ----- | --- |
| `--platform` | both | `-p` | No pytest collision; high-frequency user flag. |
| `--device` | both | `-d` | Same. |
| `--exitfirst` | both | `-x` | pytest ships `-x` → standalone mirrors it so the flag behaves identically across entry points. |
| `--rounds` | both | **(none)** | pytest-xdist already uses `-n` for worker count. Standalone originally had `-n` for `--rounds`, creating a letter-level collision whenever a user switched between pytest (`-n 8` = 8 workers) and standalone (`-n 8` = 8 rounds). Removed in [#574](https://github.com/hw-native-sys/simpler/pull/574); do not reintroduce. |
| `--max-parallel` | both | **(none)** | `-j` would be the natural make-style short, but pytest reserves all lowercase single letters (`parser.addoption` rejects lowercase shorts). Standalone mirrors this to keep both CLIs identical — no short in either, always spell out `--max-parallel`. |
| `--runtime` / `--level` | both | **(none)** | Internal child-mode markers; users rarely type them. No short keeps them distinctive. |
| `--build`, `--skip-golden`, `--enable-profiling`, `--dump-tensor`, `--manual`, `--case`, `--log-level` | both | **(none)** | Low-frequency; long form reads better in scripts and docs. Not worth reserving letters. |

Practical guidance when adding a new CLI option:

- First decide the long name and semantics. Register it on **both** `conftest.py::pytest_addoption` and `simpler_setup/scene_test.py::run_module`.
- Check pytest's built-in shorts (`pytest --help`) and loaded plugins (`pytest-xdist`, `pytest-timeout`, etc.) for any letter you're considering.
  - If pytest uses the letter for a matching concept → register the same short in standalone.
  - If pytest uses the letter for anything else → pick a different short, or drop the short entirely.
  - If the letter is free and the flag is high-frequency user-facing → a short is OK; otherwise skip it.

## Parallel Test Execution and Resource Reuse

Tests are dispatched through an **orchestrator** that pipelines work along two complementary axes:

1. **Resource reuse** — every `ChipWorker` init costs three `dlopen`s plus a device-context acquire. The orchestrator keeps one worker alive for the lifetime of a test group so every class, case, and round on that device reuses the same worker.
2. **Parallelism** — when `--device` names more than one id, independent work is spread across subprocesses so N devices do N-way work.

Both pytest and standalone (`python test_*.py`) walk the same 6-layer hierarchy:

```text
Layer 1  Level axis
│
├─ L3 phase (runs first)
│   One isolated subprocess per case; scheduled by device_count, bin-packed
│   against the --device pool. CANN isolation is automatic (each case is its
│   own process). Cross-runtime L3 cases can overlap when their devices don't.
│
└─ L2 phase (runs after L3 drains)
    │
    ├─ Layer 2  Runtime — serial subprocess per runtime (CANN isolation)
    │   └─ Layer 3  Device — parallel subprocess per device (xdist for pytest,
    │               custom fanout for standalone). Capacity bounded by --device size
    │       └─ Layer 4  Class — one ChipWorker per (runtime, device), reused
    │                   across every class assigned to that device
    │           └─ Layer 5  Case — serial within a class
    │               └─ Layer 6  Rounds — `--rounds N` loop, reuses Worker
```

### Quick examples

```bash
# Serial, single device — identical to pre-parallel behavior
pytest tests/st --platform a2a3sim --device 0
python test_foo.py -p a2a3sim -d 0

# 8-device box: L3 cases bin-pack; L2 fanned out across 8 xdist workers
# (hardware default: --max-parallel auto = 8)
pytest tests/st --platform a2a3 --device 0-7
python test_foo.py -p a2a3 -d 0-7

# Non-contiguous devices (e.g. devices 0, 2, 5 free)
pytest tests/st --platform a2a3 --device 0,2,5
python test_foo.py -p a2a3 -d 0,2,5

# CPU-constrained sim: pool of 16 virtual ids (needed by an L3 case with
# device_count=8), but only 2 subprocesses running at once to avoid thrashing.
# --max-parallel auto would pick this automatically on a 2-core CI runner.
pytest tests/st --platform a2a3sim --device 0-15 --max-parallel 2
python test_foo.py -p a2a3sim -d 0-15 --max-parallel 2

# Fail-fast for CI: stop at the first failing case
pytest tests/st --platform a2a3 --device 0-7 -x
python test_foo.py -p a2a3 -d 0-7 -x

# Narrow run: one level, one runtime, one case
pytest tests/st --platform a2a3sim --level 2 --runtime tensormap_and_ringbuffer --case TestFoo::default
python test_foo.py -p a2a3sim --level 2 --runtime tensormap_and_ringbuffer --case TestFoo::default
```

### Fail-fast (`-x` / `--exitfirst`)

- Default: all cases run; the orchestrator summarizes pass/fail at the end. This is the right mode for local development — you want every failure surfaced at once.
- With `-x` / `--exitfirst`: first failure cancels the pending queue, sends `SIGTERM` to running children, and **skips the L2 phase if L3 failed**. Intended for CI. The short form mirrors pytest's built-in `-x` so the flag behaves identically across pytest and standalone.

### Device-count constraints

If any L3 case declares `device_count > len(--device pool)` the orchestrator fails the whole batch up front rather than deadlocking the scheduler. Either widen `--device` or reduce the case's `device_count`. When `device_count` exceeds the *currently free* pool (but fits within the total), the case waits for an in-flight job to finish and then claims its slot.

### `--device` vs `--max-parallel` (two separate knobs)

On hardware these two concepts collapse: one device = one subprocess's worth of host CPU, so `--device 0-7` naturally implies 8-way parallelism. On sim they diverge:

| `--device` controls | `--max-parallel` controls |
| ------------------- | ------------------------- |
| Size of the virtual device-id pool | Max simultaneous `subprocess.Popen` |
| Must be ≥ any L3 case's `device_count` | Should be ~nproc on sim (CPU-bound) |

This matters on CPU-constrained CI runners. Example: an L3 case needs `device_count=8` but the runner has 2 CPUs.

- `--device 0-1` — can't run the L3 case; static check fails.
- `--device 0-15` (no `--max-parallel`) — L3 case fits, but the orchestrator would also spawn 15 concurrent L2 xdist workers and potentially multiple concurrent L3 cases, thrashing the 2 CPUs.
- `--device 0-15 --max-parallel 2` — L3 case fits in the pool; at most 2 subprocesses run at a time. This is what the `auto` default computes on a 2-core runner.

`--max-parallel` counts top-level subprocesses, like `make -j`. A single L3 case subprocess internally forks N chip-processes for its `device_count`; those forks do **not** count toward `--max-parallel`. One case = one unit regardless of how many devices it uses.

### Mixed L2 + L3 in one file

A single file can declare both L2 and L3 classes; they're grouped by `(runtime, level)` internally. L3 classes run in the L3 phase (subprocess-per-case), L2 classes run in the L2 phase (shared Worker per device).

### Profiling and parallelism are mutually exclusive

`--enable-profiling` writes process-global files (`outputs/perf_swimlane_*.json` and CANN device logs) that collide under parallel runs. The orchestrator errors out up front when `--max-parallel > 1`. Profile on a single device:

```bash
pytest tests/st --platform a2a3 --device 0 --enable-profiling
# Or widen the pool but force serial execution:
pytest tests/st --platform a2a3 --device 0-7 --max-parallel 1 --enable-profiling
```

### Orchestrator skip conditions (normal pytest runs)

The orchestrator only takes over when there's actual work to parallelize or isolate. It falls through to plain pytest when:

- `--collect-only`
- Only one runtime is present and no L3 cases are collected (single L2 batch)
- Both `--runtime` and `--level` are set (child-mode marker — used internally by spawned subprocesses)

## Hardware Classification

The `--platform` flag is the single source of truth for hardware availability. No separate `-m` flags are needed.

### Helper function

```python
def is_device(platform: str | None) -> bool:
    """sim and None are both no-hardware."""
    return platform is not None and not platform.endswith("sim")
```

### ut-py: `@pytest.mark.requires_hardware[(platform)]`

| Declaration | no-hw runner | a2a3 runner | a5 runner |
| ----------- | ------------ | ----------- | --------- |
| *(no marker)* | run | skip | skip |
| `@pytest.mark.requires_hardware` | skip | run | run |
| `@pytest.mark.requires_hardware("a2a3")` | skip | run | skip |
| `@pytest.mark.requires_hardware("a5")` | skip | skip | run |

Skip logic (conftest.py):

```python
marker = item.get_closest_marker("requires_hardware")
on_device = is_device(platform)
if marker is None:
    if on_device:
        skip("no-hardware test, runs in no-hw job")
elif marker.args:
    if platform != marker.args[0]:
        skip(f"requires --platform {marker.args[0]}")
else:
    if not on_device:
        skip("requires hardware")
```

### ut-cpp: ctest labels

| Declaration (CMakeLists.txt) | no-hw runner | a2a3 runner | a5 runner |
| ---------------------------- | ------------ | ----------- | --------- |
| *(no label)* | run | skip | skip |
| `LABELS "requires_hardware"` | skip | run | run |
| `LABELS "requires_hardware_a2a3"` | skip | run | skip |
| `LABELS "requires_hardware_a5"` | skip | skip | run |

Selection:

| Runner | Command |
| ------ | ------- |
| No hardware | `ctest -LE requires_hardware` |
| a2a3 | `ctest -L "^requires_hardware(_a2a3)?$"` |
| a5 | `ctest -L "^requires_hardware(_a5)?$"` |

`-LE` (exclude regex) on no-hw runner: `requires_hardware` matches all three label variants, so only unlabeled tests run.
`-L` (include regex) on device runners: only labeled tests run, unlabeled ones are excluded.

### st: `@scene_test(platforms=[...])`

`platforms` lists all supported platform names (both sim and device).

| `--platform` | Behavior |
| ------------ | -------- |
| `a2a3sim` | Run if `"a2a3sim"` in `platforms`, else skip |
| `a2a3` | Run if `"a2a3"` in `platforms`, else skip |
| *(none)* | Auto-parametrize over all `*sim` entries in `platforms`. Skip if no sim platform declared |

Auto-parametrization logic (conftest.py `pytest_generate_tests`):

```python
def pytest_generate_tests(metafunc):
    cls = metafunc.cls
    if not (cls and hasattr(cls, "_scene_platforms")):
        return
    platform = metafunc.config.getoption("--platform")
    if platform is None:
        sims = [p for p in cls._scene_platforms if p.endswith("sim")]
        if sims:
            metafunc.parametrize("st_platform", sims, indirect=True)
```

Examples:

```python
@scene_test(level=2, platforms=["a2a3sim", "a2a3", "a5sim", "a5"], ...)
class TestFoo(SceneTestCase): ...
# --platform a2a3sim  → TestFoo[a2a3sim]
# --platform a2a3     → TestFoo[a2a3]
# (none)              → TestFoo[a2a3sim] + TestFoo[a5sim]

@scene_test(level=2, platforms=["a2a3"], ...)
class TestHwOnly(SceneTestCase): ...
# --platform a2a3     → TestHwOnly[a2a3]
# (none)              → skip (no sim in platforms)
```

No separate `st` or `requires_hardware` marker — `platforms` is the sole declaration.

## Test Directory Structure

```text
tests/
  conftest.py          # pytest configuration (markers, fixtures, parametrization)
  ut/                  # Python unit tests (ut-py)
    test_task_interface.py
    test_runtime_builder.py
    cpp/               # C++ unit tests (ut-cpp, GoogleTest)
  st/                  # Scene tests
    setup/             # Compilation toolchain (KernelCompiler, RuntimeBuilder, etc.)
    conftest.py        # ST-specific sys.path setup
    test_worker_api.py # L3 distributed worker tests
    a2a3/              # @scene_test classes organized by {arch}/{runtime}/{name}/
    a5/

examples/              # Small examples (sim + onboard)
  a2a3/
    tensormap_and_ringbuffer/
      vector_example/
        test_vector_example.py   # @scene_test class + __main__ entry
        kernels/
          orchestration/*.cpp
          aic/*.cpp               # optional
          aiv/*.cpp               # optional
  a5/...

conftest.py            # Root: --platform/--device options, ST fixtures
```

## Test Types

### C++ Unit Tests (`tests/ut/cpp/`)

GoogleTest-based tests for shared components (`src/common/task_interface/` and `src/{arch}/runtime/common/`):

- `test_data_type.cpp` — DataType enum, get_element_size(), get_dtype_name()

```bash
cmake -B tests/ut/cpp/build -S tests/ut/cpp
cmake --build tests/ut/cpp/build
ctest --test-dir tests/ut/cpp/build --output-on-failure
```

### Python Unit Tests (`tests/ut/`)

Tests for the nanobind extension and the Python build pipeline:

- `test_task_interface.py` — DataType, ContinuousTensor, ChipStorageTaskArgs, torch integration
- `test_runtime_builder.py` — RuntimeBuilder discovery, error handling, build logic (mocked), and real compilation integration tests

```bash
# No-hardware runner (hw tests auto-skip, no-hw tests run)
pytest tests/ut

# a2a3 hardware runner (no-hw tests skip, hw + a2a3-specific tests run)
pytest tests/ut --platform a2a3
```

### Examples (`examples/{arch}/`)

Small, fast examples that run on both simulation and real hardware. Organized by runtime:

- `host_build_graph/` — HBG examples
- `aicpu_build_graph/` — ABG examples
- `tensormap_and_ringbuffer/` — TMR examples

Each example has a `golden.py` with `generate_inputs()` and `compute_golden()` for result validation.

### Device Scene Tests (`tests/st/{arch}/`)

Hardware-only scene tests for large-scale and feature-rich scenarios that are too slow or unsupported on simulation. Organized by runtime. Same structure as examples but focused on testing specific runtime behaviors and edge cases.

| Attribute | `examples/` | `tests/st/` |
| --------- | ----------- | ----------- |
| Runs on sim | Yes | No |
| Runs on device | Yes | Yes |
| Scale | Small, fast | Large, thorough |
| Purpose | Examples + basic regression | Deep functionality/performance |

## Writing New Tests

### New C++ Unit Test

Add a new test file to `tests/ut/cpp/` and register it in `tests/ut/cpp/CMakeLists.txt`:

```cmake
add_executable(test_my_component
    test_my_component.cpp
    test_stubs.cpp
)
target_include_directories(test_my_component PRIVATE ${COMMON_DIR} ${TMR_RUNTIME_DIR} ${PLATFORM_INCLUDE_DIR})
target_link_libraries(test_my_component gtest_main)
add_test(NAME test_my_component COMMAND test_my_component)

# If hardware required:
# set_tests_properties(test_my_component PROPERTIES LABELS "requires_hardware")
# If specific platform required:
# set_tests_properties(test_my_component PROPERTIES LABELS "requires_hardware_a2a3")
```

### New Scene Test

Create a `test_*.py` file using the `@scene_test` decorator:

```python
import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestMyKernel(SceneTestCase):
    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/my_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.OUT],
        },
        "incores": [
            {"func_id": 0, "source": "kernels/aiv/my_kernel.cpp", "core_type": "aiv"},
        ],
    }

    CASES = [
        {
            "name": "default",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 4, "block_dim": 3},
            "params": {},
        },
    ]

    def generate_args(self, params):
        return TaskArgsBuilder(
            Tensor("x", torch.ones(1024, dtype=torch.float32)),
            Tensor("y", torch.zeros(1024, dtype=torch.float32)),
        )

    def compute_golden(self, args, params):
        args.y[:] = args.x + 1

if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
```

Run it:

```bash
# Via pytest (batch, ChipWorker reuse across tests)
pytest examples tests/st --platform a2a3sim

# Standalone (single case)
python test_my_kernel.py -p a2a3sim

# On hardware
pytest examples tests/st --platform a2a3
```

Key fields:

- `level`: 2 = single ChipWorker, 3 = distributed Worker (future)
- `CASES[].platforms`: which platforms each case supports (sim names end in "sim")
- `runtime`: which runtime to use
- `CALLABLE.orchestration.source` / `CALLABLE.incores[].source`: paths relative to the test file

### Case Selection and Manual Cases

`--case` is repeatable and accepts compound forms (useful when a test file declares multiple `SceneTestCase` classes):

| Form | Meaning |
| ---- | ------- |
| `--case Foo` | case named `Foo` in any class |
| `--case ClassA::Foo` | `Foo` in `ClassA` only |
| `--case ClassA::` | all cases in `ClassA` |
| `--case A::x --case B::y` | multiple selectors (repeatable) |

`--manual exclude` (default) skips `manual: True` cases; `--manual include` runs them alongside normal cases; `--manual only` runs only manual cases. These compose orthogonally with `--case`: explicit selectors still respect the manual filter — to run a manual case by name, pass `--manual include`.

### Sharing an Example Between examples/ and tests/st/

If similar coverage exists in both `examples/` and `tests/st/`, collapse it into a single `test_*.py`: small cases get `platforms: ["a2a3sim", "a2a3"]`; large benchmark cases get `platforms: ["a2a3"], "manual": True`.

## CI Pipeline

See [ci.md](ci.md) for the full CI pipeline documentation, including the job matrix, runner constraints, and marker scheme.

## Runtime Isolation Constraint (Onboard)

**One device can only run one runtime per process.** Switching runtimes on the same device within a single process causes AICPU kernel hangs.

### Root Cause

CANN's AICPU dispatch uses a framework SO (`libaicpu_extend_kernels.so`) with a global singleton `BackendServerHandleManager` that:

1. **`SaveSoFile`**: Writes the user AICPU .so to disk on first call, then sets `firstCreatSo_ = true` to skip all subsequent writes.
2. **`SetTileFwkKernelMap`**: `dlopen`s the .so and caches function pointers on first call, then sets `firstLoadSo_ = true` to skip all subsequent loads.

When a second runtime launches on the same device (same CANN process context), the Init kernel call hits the cached flags — the new AICPU .so is never written or loaded. The Exec kernel then calls function pointers from the first runtime's .so, which operates on incompatible data structures and hangs.

### Impact

| Scenario | Result |
| -------- | ------ |
| Same runtime, same device, sequential | Works (same .so, cached pointers valid) |
| Different runtime, same device, sequential | **Hangs** (stale .so, wrong function pointers) |
| Different runtime, different device | Works (separate CANN context per device) |
| Different runtime, different process, same device | Works (`rtDeviceReset` between processes clears context) |

### Mitigation

The orchestrator spawns a separate subprocess per runtime for L2 work and a separate subprocess per case for L3 work. Every subprocess starts from a clean CANN state, so the stale-`.so` hang is structurally impossible.

## Device Allocation (Orchestrator + xdist)

When running `pytest --platform a2a3 --device 8-11`, the orchestrator does this:

### L3 phase — device bin-packing

For every collected L3 case, the scheduler (`simpler_setup/parallel_scheduler.py`) maintains a free-device set starting at `[8, 9, 10, 11]`. It pops the next queued case and, if the free set can cover its `device_count`, grabs that many ids and spawns:

```text
pytest <nodeid> --runtime <rt> --level 3 --case <Class::case> --device <alloc-range>
```

When a subprocess completes, its devices return to the free set and the queue is re-tried. Cases that need more devices than currently free **wait**; cases that need more than the whole pool **fail the batch up front**.

### L2 phase — xdist fanout per device

After L3 drains, one subprocess is spawned per runtime:

```text
pytest --runtime <rt> --level 2 --device 8-11 -n 4 --dist loadfile
```

`pytest-xdist` starts 4 workers (`gw0`..`gw3`). Each worker's `pytest_configure` slices `--device 8-11` down to a single id (`gw0` → `8`, `gw1` → `9`, ...), and `st_worker` is session-scoped, so the worker initializes exactly one `ChipWorker(device=N)` and reuses it for every L2 class routed to it. `--dist loadfile` keeps all cases from one test file on the same worker, amortizing any file-level setup cost.

### L2 phase — standalone fanout

Standalone (`python test_*.py -d 8-11`) uses the same scheduler module: classes are round-robin assigned to `len(device_ids)` chunks, one subprocess per chunk launched with a single device and explicit `--case ClassName::` selectors.

### Sim platforms

On sim (`a2a3sim`, `a5sim`), device IDs are virtual — no hardware state, no isolation constraint. All tests share a single virtual pool with auto-incrementing IDs. The same orchestrator + xdist path is used; speedup on sim comes from actual CPU parallelism, not hardware parallelism.

## Per-Case Device Filtering

The `@scene_test(platforms=[...])` decorator provides per-case platform filtering. A single test class declares which platforms it supports:

```python
@scene_test(level=2, platforms=["a2a3sim", "a2a3"], runtime="tensormap_and_ringbuffer")
class TestSmallCase(SceneTestCase):
    ...  # runs on sim and a2a3 hardware

@scene_test(level=2, platforms=["a2a3"], runtime="tensormap_and_ringbuffer")
class TestLargeCase(SceneTestCase):
    ...  # hardware only (too slow for sim)
```

This eliminates the need for separate `examples/` (sim) and `tests/st/` (device) directories when only scale differs. Both cases can live in the same file.

### When separate directories are still needed

When kernels themselves differ (e.g., templated tile sizes tuned for device), separate test files remain the correct approach.
