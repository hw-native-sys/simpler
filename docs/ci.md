# CI Pipeline

## Overview

The CI pipeline maps test categories (st, ut-py, ut-cpp) × hardware tiers to GitHub Actions jobs. See [testing.md](testing.md) for full test organization and hardware classification.

Design principles:

1. **Separate jobs per test category** — st, ut-py, and ut-cpp run as independent jobs for parallelism and clear dashboard visibility.
2. **Runner matches hardware tier** — no-hardware tests run on `ubuntu-latest`; platform-specific tests run on self-hosted runners with the matching label (`a2a3`, `a5`).
3. **`--platform` is the only filter** — pytest uses `--platform` + the `requires_hardware` marker; ctest uses label `-LE` exclusion. No `-m st`, no `-m "not requires_hardware"`.
4. **sim = no hardware** — `a2a3sim`/`a5sim` jobs run on github-hosted runners alongside unit tests.

## Full Job Matrix

The complete test-type × hardware-tier matrix. Empty cells have no tests yet; only non-empty jobs exist in `ci.yml`.

| Category | github-hosted (no hardware) | a2a3 runner | a5 runner |
| -------- | --------------------------- | ----------- | --------- |
| **ut-py** | `ut-py` | `ut-py-a2a3` | `ut-py-a5` |
| **ut-cpp** | `ut-cpp` | `ut-cpp-a2a3` | `ut-cpp-a5` |
| **st** | `st-sim-a2a3`, `st-sim-a5` | `st-a2a3` | `st-a5` |

## GitHub Actions Jobs

Currently active jobs (a5 jobs commented out — no runner yet):

```text
PullRequest
  ├── ut-py                (ubuntu-latest)
  ├── ut-cpp               (ubuntu-latest)
  ├── st-sim-a2a3          (ubuntu + macOS)
  ├── st-sim-a5            (ubuntu + macOS)
  ├── ut-py-a2a3           (a2a3 self-hosted)
  ├── ut-cpp-a2a3          (a2a3 self-hosted)
  ├── st-a2a3              (a2a3 self-hosted)
  ├── ut-py-a5             (a5 self-hosted, commented out)
  ├── ut-cpp-a5            (a5 self-hosted, commented out)
  └── st-a5                (a5 self-hosted, commented out)
```

| Job | Runner | What it runs |
| --- | ------ | ------------ |
| `ut-py` | `ubuntu-latest` | `pytest tests/ut` |
| `ut-cpp` | `ubuntu-latest` | `ctest --test-dir tests/ut/cpp/build -LE requires_hardware` |
| `st-sim-a2a3` | `ubuntu-latest`, `macos-latest` | `pytest examples tests/st --platform a2a3sim` + `ci.py -p a2a3sim` |
| `st-sim-a5` | `ubuntu-latest`, `macos-latest` | `pytest examples tests/st --platform a5sim` + `ci.py -p a5sim` |
| `ut-py-a2a3` | a2a3 self-hosted | `pytest tests/ut --platform a2a3` |
| `ut-cpp-a2a3` | a2a3 self-hosted | `ctest --test-dir tests/ut/cpp/build -L "^requires_hardware(_a2a3)?$"` |
| `st-a2a3` | a2a3 self-hosted | `pytest examples tests/st --platform a2a3` + `ci.py -p a2a3 -d ...` |
| `ut-py-a5` | a5 self-hosted | `pytest tests/ut --platform a5` |
| `ut-cpp-a5` | a5 self-hosted | `ctest --test-dir tests/ut/cpp/build -L "^requires_hardware(_a5)?$"` |
| `st-a5` | a5 self-hosted | `pytest examples tests/st --platform a5` + `ci.py -p a5 -d ...` |

### Parallel ST runs on hardware

For self-hosted jobs with multiple NPUs, pass a `--device` range (and
optionally pytest's `-x` for fail-fast) to get the full orchestrator
benefit — device bin-packing for L3, xdist fanout for L2, and a shared
`ChipWorker` per `(runtime, device)`:

```bash
# Recommended CI invocation
pytest examples tests/st --platform a2a3 --device 4-7 -x

# Same for a5
pytest examples tests/st --platform a5 --device 0-7 -x
```

`-x` (`--exitfirst`) is appropriate for CI, where aborting on first
failure saves runner minutes. Local development usually wants the opposite
(let every failure surface) — just drop the flag. The short form is the
same in both pytest and standalone on purpose; see
[testing.md §CLI Design Principles](testing.md#cli-design-principles).

`pytest-xdist` is pulled in via the `test` extra. See
[testing.md §Parallel Test Execution](testing.md#parallel-test-execution-and-resource-reuse)
for the full hierarchy, fail-fast semantics, and the
profiling-vs-parallelism trade-off.

### Sim jobs on CPU-constrained runners

Sim jobs (`st-sim-a2a3`, `st-sim-a5`) run on `ubuntu-latest`, which typically
has 2 vCPUs. `--device 0-15` is still the right choice for the **pool size**
(some L3 cases need several virtual ids), but the default `--max-parallel auto`
caps the in-flight subprocess count to `min(nproc, len(--device))` — on a
2-core runner that becomes `2`, avoiding CPU thrashing:

```bash
# Sim: --max-parallel auto resolves to 2 on ubuntu-latest
pytest examples tests/st --platform a2a3sim --device 0-15

# Or pin explicitly if your runner has a different CPU count
pytest examples tests/st --platform a2a3sim --device 0-15 --max-parallel 2
```

On hardware jobs the `auto` default is `len(--device)` because each subprocess
is device-bound (host CPU mostly waits on the NPU), so hardware runners do
not need `--max-parallel` manually.

### Scheduling constraints

- Sim scene tests and no-hardware unit tests run on github-hosted runners (no hardware).
- `a2a3` tests (st + ut-py + ut-cpp) only run on the `a2a3` self-hosted machine.
- `a5` tests (st + ut-py + ut-cpp) only run on the `a5` self-hosted machine.

## Hardware Classification

Three hardware tiers, applied to all test categories. See [testing.md](testing.md#hardware-classification) for the full table including per-category mechanisms (pytest markers, ctest labels, folder structure).

| Tier | CI Runner | Job examples |
| ---- | --------- | ------------ |
| No hardware | `ubuntu-latest` | `ut-py`, `ut-cpp`, `st-sim-*` |
| Platform-specific (a2a3) | `[self-hosted, a2a3]` | `ut-py-a2a3`, `ut-cpp-a2a3`, `st-a2a3` |
| Platform-specific (a5) | `[self-hosted, a5]` | `ut-py-a5`, `ut-cpp-a5`, `st-a5` |

## Test Sources

### `tests/ut/` — Python unit tests (ut-py)

Python unit tests. Run via pytest, filtered by `--platform` + `requires_hardware` marker.

| File | Content | Hardware? |
| ---- | ------- | --------- |
| `test_task_interface.py` | nanobind extension API tests | No |
| `test_runtime_builder.py` (mocked classes) | RuntimeBuilder discovery, error handling, build logic | No |
| `test_runtime_builder.py::TestRuntimeBuilderIntegration` | Real compilation across platform × runtime | Yes (`@pytest.mark.requires_hardware`) |

### `tests/ut/cpp/` — C++ unit tests (ut-cpp)

GoogleTest-based tests for pure C++ modules. Run via ctest, filtered by label `-LE` exclusion.

| Runner | Command |
| ------ | ------- |
| No hardware | `ctest --test-dir tests/ut/cpp/build -LE requires_hardware` |
| a2a3 | `ctest --test-dir tests/ut/cpp/build -L "^requires_hardware(_a2a3)?$"` |
| a5 | `ctest --test-dir tests/ut/cpp/build -L "^requires_hardware(_a5)?$"` |

### `examples/` — Small examples (sim + onboard)

Small, fast examples that run on both simulation and real hardware. Organized as `examples/{arch}/{runtime}/{name}/`. Discovered and executed by `ci.py` (legacy golden.py format) or pytest (`@scene_test` format).

### `tests/st/` — Scene tests (onboard-biased)

Large-scale, feature-rich hardware tests. Too slow or using instructions unsupported by the simulator. Organized as `tests/st/{arch}/{runtime}/{name}/`. Platform compatibility is declared per test via `@scene_test(platforms=[...])`.

### Shared structure

Both `examples/` and `tests/st/` cases follow the same layout:

```text
{name}/
  golden.py                      # generate_inputs() + compute_golden()
  kernels/
    kernel_config.py             # KERNELS, ORCHESTRATION, RUNTIME_CONFIG
    orchestration/*.cpp
    aic/*.cpp                    # optional
    aiv/*.cpp                    # optional
```

A legacy case is discoverable by `ci.py` when both `golden.py` and `kernels/kernel_config.py` exist. `@scene_test` cases are discovered by pytest via `test_*.py` files.

## Selection Scheme

A single `--platform` flag controls hardware/non-hardware splitting across all three categories.

### ut-py (pytest marker)

```python
@pytest.mark.requires_hardware                  # any hardware
class TestRuntimeBuilderIntegration:
    ...

@pytest.mark.requires_hardware("a2a3")          # a2a3 specifically
class TestA2A3Feature:
    ...
```

Selection:

```bash
# No hardware (no-hw tests run, requires_hardware tests skip)
pytest tests/ut

# Hardware (no-hw tests skip, hw + platform-specific tests run)
pytest tests/ut --platform a2a3
```

### ut-cpp (ctest label)

```cmake
# any hardware
set_tests_properties(test_runtime_integration PROPERTIES LABELS "requires_hardware")
# a2a3-specific
set_tests_properties(test_a2a3_feature PROPERTIES LABELS "requires_hardware_a2a3")
```

Selection uses `-LE` (label exclude) on no-hw runner and `-L` (label include) on device runners:

```bash
ctest -LE requires_hardware                 # no-hardware runner: only unlabeled
ctest -L "^requires_hardware(_a2a3)?$"      # a2a3 runner: hw + a2a3-specific
ctest -L "^requires_hardware(_a5)?$"        # a5 runner: hw + a5-specific
```

### st (`@scene_test`)

```python
@scene_test(level=2, platforms=["a2a3sim", "a2a3"], runtime="tensormap_and_ringbuffer")
class TestVectorExample(SceneTestCase):
    ...
```

| `--platform` | Behavior |
| ------------ | -------- |
| `a2a3sim` | Run if `"a2a3sim"` in `platforms` |
| `a2a3` | Run if `"a2a3"` in `platforms` |
| *(none)* | Auto-parametrize over all `*sim` entries in `platforms` |

No `--platform` means "run all sims" — tests with no sim in their `platforms` list are skipped. No additional markers are used.

## Discovery Layer (`tools/test_catalog.py`)

Single source of truth for platform, runtime, and test case discovery. Used by `tests/conftest.py` (via import) and available as a CLI for scripting.

### Python API

```python
from test_catalog import (
    discover_platforms,           # -> ["a2a3", "a2a3sim", "a5", "a5sim"]
    discover_runtimes_for_arch,   # -> ["host_build_graph", "aicpu_build_graph", ...]
    discover_test_cases,          # -> [TestCase(name, dir, arch, runtime, source), ...]
    arch_from_platform,           # "a2a3sim" -> "a2a3"
)
```

### CLI

```bash
python tools/test_catalog.py platforms
python tools/test_catalog.py runtimes --arch a2a3
python tools/test_catalog.py cases --platform a2a3sim --source example
python tools/test_catalog.py cases --platform a2a3 --source st --format json
```

## `ci.py` — Scene Test Runner (Legacy)

`ci.py` handles scene test execution for golden.py-based tests (examples + st). New tests should use `@scene_test` and run via pytest. `ci.py` is retained for backward compatibility during the migration.

### Key features

- **ChipWorker reuse**: Tasks sharing the same runtime reuse a single ChipWorker within their subprocess, avoiding repeated device init/teardown.
- **Subprocess isolation**: Different runtimes run in separate subprocesses (the host `.so` cannot be unloaded within a single process).
- **Device queue**: Hardware tasks are distributed across devices specified by `-d`. Workers pop tasks from a shared queue via threads.
- **Retry**: Failed tasks are retried up to 3 times. Hardware workers quarantine a device after a failure.
- **PTO-ISA pinning**: `-c <commit>` pins the PTO-ISA dependency. On first failure, re-runs failed tasks with the pinned commit.
- **Watchdog**: `-t <seconds>` sets a timeout. The entire run is aborted if it exceeds the limit.
- **Summary table**: After all tasks complete, a formatted results table is printed with pass/fail status, timing, device, and attempt count.

### Usage

```bash
# All sim platforms (no -p: auto-discovers a2a3sim, a5sim, etc.)
python ci.py -t 600

# Single sim platform
python ci.py -p a2a3sim -c 6622890 -t 600

# Hardware with device range
python ci.py -p a2a3 -d 4-7 -c 6622890 -t 600

# Filter by runtime
python ci.py -p a2a3sim -r tensormap_and_ringbuffer
```

### Platform notes

- **macOS libomp collision**: on macOS, `ci.py` sets `KMP_DUPLICATE_LIB_OK=TRUE` at the top of the file to work around a duplicate-libomp abort triggered by homebrew numpy and pip torch coexisting in one process. Do not reorder the imports or remove this workaround without reading [macos-libomp-collision.md](macos-libomp-collision.md) first.

### Task discovery

`ci.py` scans two directories:

1. `examples/` — included for both sim and onboard platforms.
2. `tests/st/` — included only for onboard platforms (non-sim).

For each directory, it walks subdirectories looking for `kernels/kernel_config.py` + `golden.py`. The arch and runtime are extracted from the path: `{root}/{arch}/{runtime}/{case_name}/`.

### Execution flow

```text
1. Parse arguments (-p, -d, -r, -c, -t)
2. If no -p: auto-discover all sim platforms and run each
3. For each platform:
   a. Discover tasks from examples/ and tests/st/
   b. Run tasks (subprocess per runtime group for sim, device queue for hw)
      └── On failure + -c flag: pin PTO-ISA, retry failed tasks
4. Print combined summary table
5. Exit 0 if all passed, 1 otherwise
```
