# CI Pipeline

## Overview

The CI pipeline maps test categories (st, ut-py, ut-cpp) × hardware tiers to GitHub Actions jobs. See [testing.md](testing.md) for full test organization and hardware classification.

Design principles:

1. **Separate jobs per test category** — st, ut-py, and ut-cpp run as independent jobs for parallelism and clear dashboard visibility.
2. **Runner matches hardware tier** — no-hardware tests run on `ubuntu-latest`; platform-specific tests run on self-hosted runners with the matching label (`a2a3`, `a5`).
3. **sim vs onboard for st** — scene tests split into sim jobs (github-hosted, `ci.sh -p a2a3sim`) and onboard jobs (self-hosted, `ci.sh -p a2a3`).

## Full Job Matrix

The complete test-type × hardware-tier matrix. Empty cells have no tests yet; only non-empty jobs exist in `ci.yml`.

| | github-hosted | a2a3 runner | a5 runner |
|---|---|---|---|
| **ut-py** | `ut-py` | `ut-py-a2a3` | `ut-py-a5` |
| **ut-cpp** | *(empty)* | *(empty)* | *(empty)* |
| **st** | `st-sim-a2a3`, `st-sim-a5` | `st-onboard-a2a3` | `st-onboard-a5` |

## GitHub Actions Jobs

Currently active jobs (a5 jobs commented out — no runner yet):

```
PullRequest
  ├── ut-py                (ubuntu-latest)
  ├── st-sim-a2a3          (ubuntu + macOS)
  ├── st-sim-a5            (ubuntu + macOS)
  ├── ut-py-a2a3           (a2a3 self-hosted)
  ├── st-onboard-a2a3      (a2a3 self-hosted)
  ├── ut-py-a5             (a5 self-hosted, commented out)
  └── st-onboard-a5        (a5 self-hosted, commented out)
```

| Job | Runner | What it runs |
|-----|--------|-------------|
| `ut-py` | `ubuntu-latest` | `pytest tests -m "not requires_hardware" -v` |
| `st-sim-a2a3` | `ubuntu-latest`, `macos-latest` | `ci.sh -p a2a3sim` |
| `st-sim-a5` | `ubuntu-latest`, `macos-latest` | `ci.sh -p a5sim` |
| `ut-py-a2a3` | a2a3 self-hosted | `pytest tests -m requires_hardware --platform a2a3 -v` |
| `st-onboard-a2a3` | a2a3 self-hosted | `ci.sh -p a2a3 -d ... --parallel` |
| `ut-py-a5` | a5 self-hosted | `pytest tests -m requires_hardware --platform a5 -v` |
| `st-onboard-a5` | a5 self-hosted | `ci.sh -p a5 -d ... --parallel` |

### Scheduling constraints

- Sim scene tests and no-hardware unit tests run on github-hosted runners (no hardware).
- `a2a3` tests (st + ut-py) only run on the `a2a3` self-hosted machine.
- `a5` tests (st + ut-py) only run on the `a5` self-hosted machine.

## Hardware Classification

Three hardware tiers, applied to all test categories. See [testing.md](testing.md#hardware-classification) for the full table including per-category mechanisms (pytest markers, ctest labels, folder structure).

| Tier | CI Runner | Job examples |
|------|-----------|-------------|
| No hardware | `ubuntu-latest` | `ut-py`, `st-sim-*` |
| Platform-specific (a2a3) | `[self-hosted, a2a3]` | `ut-py-a2a3`, `st-onboard-a2a3` |
| Platform-specific (a5) | `[self-hosted, a5]` | `ut-py-a5`, `st-onboard-a5` |

## Test Sources

### `tests/ut/` — Python unit tests (pyut)

Python unit tests. Run via pytest with marker filtering.

| File | Content | Hardware? |
|------|---------|-----------|
| `test_task_interface.py` | nanobind extension API tests | No |
| `test_runtime_builder.py` (mocked classes) | RuntimeBuilder discovery, error handling, build logic | No |
| `test_runtime_builder.py::TestRuntimeBuilderIntegration` | Real compilation across platform × runtime | Yes (`@pytest.mark.requires_hardware`) |

### `examples/` — Small examples (sim + onboard)

Small, fast examples that run on both simulation and real hardware. Organized as `examples/{arch}/{runtime}/{name}/`. Discovered and executed by `ci.sh`.

### `tests/st/` — Scene tests (onboard only)

Large-scale, feature-rich hardware tests. Too slow or using instructions unsupported by the simulator. Organized as `tests/st/{arch}/{runtime}/{name}/`. Discovered and executed by `ci.sh` only when the platform is not sim.

### Shared structure

Both `examples/` and `tests/st/` cases follow the same layout:

```
{name}/
  golden.py                      # generate_inputs() + compute_golden()
  kernels/
    kernel_config.py             # KERNELS, ORCHESTRATION, RUNTIME_CONFIG
    orchestration/*.cpp
    aic/*.cpp                    # optional
    aiv/*.cpp                    # optional
```

A case is discoverable when both `golden.py` and `kernels/kernel_config.py` exist.

## Marker Scheme

A single pytest marker controls hardware/non-hardware splitting:

```python
@pytest.mark.requires_hardware
class TestRuntimeBuilderIntegration:
    ...
```

Selection:

```bash
# Non-hardware tests (github-hosted)
pytest tests -m "not requires_hardware" -v

# Hardware integration tests (device runner)
pytest tests -m requires_hardware --platform a2a3 -v
```

Tests without the marker are assumed to need no hardware.

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

## `ci.sh` — Scene Test Runner

`ci.sh` handles scene test execution (examples + st). It does **not** run pytest tests — those are invoked directly by the CI workflow.

### Key features

- **Parallel execution**: `--parallel` runs sim tasks concurrently; hardware tasks use a shared device queue with `flock`-based locking.
- **Device queue**: Hardware tasks are distributed across devices specified by `-d`. Workers pop tasks from a shared queue atomically.
- **Retry**: Failed tasks are retried up to 3 times. Hardware workers quarantine a device after exhausting retries.
- **PTO-ISA pinning**: `-c <commit>` pins the PTO-ISA dependency. On first failure, ci.sh cleans the cached clone and retries with the pinned commit.
- **Watchdog**: `-t <seconds>` sets a timeout. The entire run is aborted if it exceeds the limit.
- **Summary table**: After all tasks complete, a formatted results table is printed with pass/fail status, timing, device, and attempt count.

### Usage

```bash
# Simulation (github-hosted)
./ci.sh -p a2a3sim -t 600

# Hardware with device range
./ci.sh -p a2a3 -d 4-7 --parallel -t 600

# Filter by runtime
./ci.sh -p a2a3sim -r tensormap_and_ringbuffer

# Pin PTO-ISA commit
./ci.sh -p a2a3sim -c 6622890
```

### Task discovery

`ci.sh` scans two directories:

1. `examples/` — included for both sim and onboard platforms.
2. `tests/st/` — included only for onboard platforms (non-sim).

For each directory, it walks subdirectories looking for `kernels/kernel_config.py` + `golden.py`. The arch and runtime are extracted from the path: `{root}/{arch}/{runtime}/{case_name}/`.

### Execution flow

```
1. Parse arguments (-p, -d, --parallel, -r, -c, -t)
2. Discover platforms and runtimes from src/
3. Discover tasks from examples/ and tests/st/
4. Run sim tasks (parallel or sequential)
   └── On failure + -c flag: pin PTO-ISA, retry
5. Run hardware tasks (device queue with workers)
   └── On failure + -c flag: pin PTO-ISA, retry
6. Print summary table
7. Exit 0 if all passed, 1 otherwise
```
