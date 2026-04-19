# CI Pipeline

## Overview

The CI pipeline maps test categories (st, ut-py, ut-cpp) × hardware tiers to GitHub Actions jobs. See [testing.md](testing.md) for full test organization and hardware classification.

Design principles:

1. **Merge by runner, not by language** — Python and C++ unit tests share setup cost and run as steps within a single job per runner tier (`ut`, `ut-a2a3`, `ut-a5`).
2. **Runner matches hardware tier** — no-hardware tests run on `ubuntu-latest`; platform-specific tests run on self-hosted runners with the matching label (`a2a3`, `a5`).
3. **`--platform` is the only filter** — pytest uses `--platform` + the `requires_hardware` marker; ctest uses label `-LE` exclusion. No `-m st`, no `-m "not requires_hardware"`.
4. **sim = no hardware** — `a2a3sim`/`a5sim` jobs run on github-hosted runners alongside unit tests.
5. **Skip irrelevant platforms** — `detect-changes` gates hardware jobs so pure a5 PRs skip a2a3 runners and vice versa.

## Full Job Matrix

The complete test-type × hardware-tier matrix. Empty cells have no tests yet; only non-empty jobs exist in `ci.yml`.

| Category | github-hosted (no hardware) | a2a3 runner | a5 runner |
| -------- | --------------------------- | ----------- | --------- |
| **ut** (py + cpp) | `ut` | `ut-a2a3` | `ut-a5` |
| **st** | `st-sim-a2a3`, `st-sim-a5` | `st-onboard-a2a3` | `st-onboard-a5` |

## GitHub Actions Jobs

```text
PullRequest
  ├── pre-commit             (ubuntu-latest)
  ├── packaging-matrix       (ubuntu + macOS)
  ├── ut                     (ubuntu + macOS)        — Python + C++ UT, no hardware
  ├── st-sim-a2a3            (ubuntu + macOS)
  ├── st-sim-a5              (ubuntu + macOS)
  ├── detect-changes         (ubuntu-latest)         — gates a2a3 + a5 hw jobs
  ├── ut-a2a3                (a2a3 self-hosted)      — Python + C++ UT, a2a3 hardware
  ├── st-onboard-a2a3        (a2a3 self-hosted)
  ├── ut-a5                  (a5 self-hosted)        — Python + C++ UT, a5 hardware
  └── st-onboard-a5          (a5 self-hosted)
```

| Job | Runner | What it runs |
| --- | ------ | ------------ |
| `ut` | `ubuntu-latest`, `macos-latest` | `pytest tests/ut` + `ctest -LE requires_hardware` |
| `st-sim-a2a3` | `ubuntu-latest`, `macos-latest` | `pytest examples tests/st --platform a2a3sim` |
| `st-sim-a5` | `ubuntu-latest`, `macos-latest` | `pytest examples tests/st --platform a5sim` |
| `ut-a2a3` | a2a3 self-hosted | `pytest tests/ut --platform a2a3` + `ctest -L "^requires_hardware(_a2a3)?$" --resource-spec-file ...` |
| `st-onboard-a2a3` | a2a3 self-hosted | `pytest examples tests/st --platform a2a3 --device ...` |
| `ut-a5` | a5 self-hosted | `pytest tests/ut --platform a5` + `ctest -L "^requires_hardware(_a5)?$"` |
| `st-onboard-a5` | a5 self-hosted | `pytest examples tests/st --platform a5 --device ...` |

### Parallel ST runs on hardware

For self-hosted jobs with multiple NPUs, pass a `--device` range (and
optionally pytest's `-x` for fail-fast) to get the full dispatcher
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
- `detect-changes` gates all hardware jobs: pure a5 PRs skip a2a3 runners and vice versa.
- a2a3 tests (st + ut) only run on the `a2a3` self-hosted machine when a2a3-relevant files change.
- a5 tests (st + ut) only run on the `a5` self-hosted machine when a5-relevant files change.

## Hardware Classification

Three hardware tiers, applied to all test categories. See [testing.md](testing.md#hardware-classification) for the full table including per-category mechanisms (pytest markers, ctest labels, folder structure).

| Tier | CI Runner | Job examples |
| ---- | --------- | ------------ |
| No hardware | `ubuntu-latest` | `ut`, `st-sim-*` |
| Platform-specific (a2a3) | `[self-hosted, a2a3]` | `ut-a2a3`, `st-onboard-a2a3` |
| Platform-specific (a5) | `[self-hosted, a5]` | `ut-a5`, `st-onboard-a5` |

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

Small, fast examples that run on both simulation and real hardware. Organized as `examples/{arch}/{runtime}/{name}/`. Discovered and executed by pytest via each example's `test_*.py` (`@scene_test` format).

### `tests/st/` — Scene tests (onboard-biased)

Large-scale, feature-rich hardware tests. Too slow or using instructions unsupported by the simulator. Organized as `tests/st/{arch}/{runtime}/{name}/`. Platform compatibility is declared per test via `@scene_test(platforms=[...])`.

### Shared structure

Both `examples/` and `tests/st/` cases follow the same layout:

```text
{name}/
  test_{name}.py                 # @scene_test class (generate_args, compute_golden)
  kernels/
    orchestration/*.cpp
    aic/*.cpp                    # optional
    aiv/*.cpp                    # optional
```

Cases are discovered by pytest via `test_*.py` files. Each test module ends with `if __name__ == "__main__": SceneTestCase.run_module(__name__)` so it can also run standalone as `python test_*.py -p <platform>`.

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

## Platform notes

- **macOS libomp collision**: on macOS, the root `conftest.py` sets `KMP_DUPLICATE_LIB_OK=TRUE` before `import pytest` to work around a duplicate-libomp abort triggered by homebrew numpy and pip torch coexisting in one Python process (see [macos-libomp-collision.md](macos-libomp-collision.md)). Standalone `python test_*.py` bypasses conftest — rely on the env var being exported by the shell or `tools/verify_packaging.sh`.
