# CI Pipeline

## Overview

The CI pipeline maps test categories (st, ut-py, ut-cpp) × hardware tiers to GitHub Actions jobs. See [testing.md](testing.md) for full test organization and hardware classification.

Design principles:

1. **Merge by runner, not by language** — Python and C++ unit tests share setup cost and run as steps within a single job per runner tier (`ut`, `ut-a2a3`, `ut-a5`).
2. **Runner matches hardware tier** — no-hardware tests run on `ubuntu-latest`; platform-specific tests run on self-hosted runners with the matching label (`a2a3`, `a5`).
3. **`--platform` is the only filter** — pytest uses `--platform` + the `requires_hardware` marker; ctest uses label `-LE` exclusion. No `-m st`, no `-m "not requires_hardware"`.
4. **sim = no hardware** — `a2a3sim`/`a5sim` jobs run on github-hosted runners alongside unit tests.
5. **Skip irrelevant platforms for scene tests** — `detect-changes` gates `st-sim-*` and `st-onboard-*` so pure-a5 PRs skip a2a3 scene-test runs and vice versa. **UT jobs (`ut`, `ut-a2a3`, `ut-a5`) are unconditional** — unit tests cover shared contracts and the cost of a falsely-skipped regression outweighs the savings.

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
  ├── ut                     (ubuntu + macOS)        — Python + C++ UT, no hardware [always]
  ├── detect-changes         (ubuntu-latest)         — outputs a{2a3,5}_changed flags
  ├── st-sim-a2a3            (ubuntu + macOS)        — gated by a2a3_changed
  ├── st-sim-a5              (ubuntu + macOS)        — gated by a5_changed
  ├── ut-a2a3                (a2a3 self-hosted)      — Python + C++ UT, a2a3 hardware [always]
  ├── st-onboard-a2a3        (a2a3 self-hosted)      — gated by a2a3_changed
  ├── ut-a5                  (a5 self-hosted)        — Python + C++ UT, a5 hardware [always]
  └── st-onboard-a5          (a5 self-hosted)        — gated by a5_changed
```

| Job | Runner | What it runs |
| --- | ------ | ------------ |
| `ut` | `ubuntu-latest`, `macos-latest` | `pytest tests/ut` + `ctest -LE requires_hardware` |
| `st-sim-a2a3` | `ubuntu-latest`, `macos-latest` | `pytest examples tests/st --platform a2a3sim` |
| `st-sim-a5` | `ubuntu-latest`, `macos-latest` | `pytest examples tests/st --platform a5sim` |
| `ut-a2a3` | a2a3 self-hosted | `pytest tests/ut --platform a2a3` + `ctest -L "^requires_hardware(_a2a3)?$" --resource-spec-file ...` + build `tools/cann-examples/query` and run `query version` (no device) + build `tools/cann-examples/aicpu-device-query` and `tools/cann-examples/aicpu-kernel-launch` (host + cross-compiled device SO, link smoke only) |
| `st-onboard-a2a3` | a2a3 self-hosted | `pytest examples tests/st --platform a2a3 --device ...` |
| `ut-a5` | a5 self-hosted | `pytest tests/ut --platform a5` + `ctest -L "^requires_hardware(_a5)?$"` + build `tools/cann-examples/query` and run `query version` (no device) + build `tools/cann-examples/aicpu-device-query` and `tools/cann-examples/aicpu-kernel-launch` (link smoke only) |
| `st-onboard-a5` | a5 self-hosted | `pytest examples tests/st --platform a5 --device ...` |

### Nightly sanitizer sweep

A **separate** workflow, [`sanitizers.yml`](../.github/workflows/sanitizers.yml),
runs on a nightly `schedule` — kept out of `ci.yml` so the cron fires only the
sanitizer jobs, never the PR/self-hosted pipeline. Its
`sanitizer-sim` job builds the sim runtime + kernels with ASAN or TSAN
(`pip install --config-settings=cmake.define.SIMPLER_SANITIZER=...`) and runs a
**scoped** subset under the matching `LD_PRELOAD` (a2a3sim/a5sim, ubuntu-only).
`dlopen_count` tests are excluded everywhere (they assert exact dlopen accounting
that the sanitizers perturb by interposing `dlopen`). The full suite is avoided
because ASAN/TSAN slow the sim enough that oversubscription-heavy cases livelock
on a 4-vCPU runner — so the scope is parallelism-limited per sanitizer:

- **ASAN** (~1.7x): `prepared_callable` + `dynamic_register` (where present),
  `--max-parallel 2`, skipping `parallel_broadcast`.
- **TSAN** (~5-15x): livelocks the chip-fork L3 cases even when run serially, so it
  runs only the light `prepared_callable` L2 tests, `--max-parallel 1`, with
  `TSAN_OPTIONS=halt_on_error=0:exitcode=0` (report races without aborting *or*
  failing the job — TSAN's default `exitcode=66` would otherwise redden the cell on
  every race; the job gates on hang/crash, triaging the reported races into a
  suppressions file is a follow-up).

Both sanitizer jobs gate (no `continue-on-error`). Not a PR gate; see
[sanitizers.md](sanitizers.md) for the design + usage.

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

Sim jobs (`st-sim-a2a3`, `st-sim-a5`) run on `ubuntu-latest`, whose standard
GitHub-hosted runner currently has **4 vCPUs**. `--device 0-15` is still the
right choice for the **pool size** (some L3 cases need several virtual ids), but
the default `--max-parallel auto` caps the in-flight subprocess count to
`min(nproc, len(--device))` — on a 4-core runner that becomes `4`. Note
`os.cpu_count()` reports the host's logical CPUs and ignores any cgroup CPU
quota, so this is the true core count, not a container limit.

```bash
# Sim: --max-parallel auto resolves to 4 on a standard ubuntu-latest runner
pytest examples tests/st --platform a2a3sim --device 0-15

# Throttle further on a CPU-starved runner: 4 concurrent cases (each forking
# several chip subprocesses with many threads) can oversubscribe 4 cores and
# trigger the sim handshake/deinit failures in
# troubleshooting/sim-oversubscription-hang.md. --max-parallel 2 trades
# throughput for stability.
pytest examples tests/st --platform a2a3sim --device 0-15 --max-parallel 2
```

On hardware jobs the `auto` default is `len(--device)` because each subprocess
is device-bound (host CPU mostly waits on the NPU), so hardware runners do
not need `--max-parallel` manually.

### Scheduling constraints

- Sim scene tests and no-hardware unit tests run on github-hosted runners (no hardware).
- `detect-changes` computes two flags (`a2a3_changed`, `a5_changed`) from the PR diff. Each flag is `false` only when *every* changed file is in the opposite platform's tree (`src/{arch}/`, `examples/{arch}/`, `tests/{st,ut/cpp}/{arch}/`) or in the `NON_CODE` set (`docs/`, `.docs/`, `.claude/`, `.gitignore`, `.pre-commit-config.yaml`, and any `*.md` file anywhere). Anything else — shared C++ (`src/common/`), Python (`python/`, `simpler_setup/`), build files (`CMakeLists.txt`, `pyproject.toml`), shared test infra (`tests/ut/py/`, `tests/lint/`), tooling (`tools/`), or workflow files (`.github/`) — flips both flags to `true`.
- **Gated jobs (scene tests only):** `st-sim-{a2a3,a5}`, `st-onboard-{a2a3,a5}` run iff their platform's flag is `true`.
- **Unconditional jobs (all UT):** `ut`, `ut-a2a3`, `ut-a5` always run regardless of the flags — unit tests exercise shared contracts (nanobind bindings, RuntimeBuilder, ring buffers, etc.) and the risk of silently skipping a regression outweighs the CI minutes saved. The `tests/ut/cpp/{arch}/` entry in the gating regex only *attributes* an arch-specific C++ UT change to that platform (so it does not spuriously flip the other arch's scene-test flag); it does not gate the UT jobs themselves. A consequence: self-hosted runners (`a2a3`, `a5`) are always busy for at least the UT job, even on doc-only PRs that skip all scene tests.

## Hardware Classification

Three hardware tiers, applied to all test categories. See [testing.md](testing.md#hardware-classification) for the full table including per-category mechanisms (pytest markers, ctest labels, folder structure).

| Tier | CI Runner | Job examples |
| ---- | --------- | ------------ |
| No hardware | `ubuntu-latest` | `ut`, `st-sim-*` |
| Platform-specific (a2a3) | `[self-hosted, a2a3]` | `ut-a2a3`, `st-onboard-a2a3` |
| Platform-specific (a5) | `[self-hosted, a5]` | `ut-a5`, `st-onboard-a5` |

On a self-hosted runner, every step that touches an NPU — pytest and ctest
alike — must hold its devices exclusively while it runs. There are two a2a3
runner pools, branched at run time on `runner.arch`:

- **ARM64 a2a3 runners** share the host with interactive users, so the step
  runs through `task-submit --device <list> --run "..."`, whose per-device
  lock keeps a CI job from colliding with someone's local run (and vice
  versa).
- **X64 a2a3 runners** carry no `task-submit` — their cards are exclusive to
  the runner — so the step runs pytest/ctest directly with
  `--device ${DEVICE_RANGE}`.

a5 runners are ARM64-only and always use `task-submit`. Steps that only build
(cmake, `RuntimeBuilder`, the `cann-examples` smokes) take no lock on either
arch. The same device-lock rule applies to local onboard work — see
[.claude/rules/running-onboard.md](../.claude/rules/running-onboard.md).

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

## Platform notes

- **macOS libomp collision**: on macOS, the root `conftest.py` sets `KMP_DUPLICATE_LIB_OK=TRUE` before `import pytest` to work around a duplicate-libomp abort triggered by homebrew numpy and pip torch coexisting in one Python process (see [troubleshooting/macos-libomp-collision.md](troubleshooting/macos-libomp-collision.md)). Standalone `python test_*.py` bypasses conftest — rely on the env var being exported by the shell or `tools/verify_packaging.sh`.
- **sim hangs / `rc=-1` under CPU oversubscription**: on a few-vCPU runner, high `--max-parallel` (or many concurrent sim cases) oversubscribes the host CPUs, where sim's busy-spin handshake can livelock (hang → `rc=124`) or the deinit timeout can false-trip (`simpler_run failed with code -1`). Mitigate with `--max-parallel 2`; onboard is unaffected (see [troubleshooting/sim-oversubscription-hang.md](troubleshooting/sim-oversubscription-hang.md)).
- **local runs time out more slowly than CI**: compiled defaults are lenient for serving workloads, while CI sets tighter `PTO2_*_TIMEOUT_*` env values to fail fast. Use the same env values locally when debugging suspected hangs (see [troubleshooting/local-timeout-defaults.md](troubleshooting/local-timeout-defaults.md)).
- **`st-onboard-a2a3` mass 507899 is not OOM**: a whole-suite collapse of `507899`/`507018`/`register_callable -1` is an AICPU device-fault cascade (`simpler_aicpu_exec` exception), not memory exhaustion. Diagnosis recipe and the per-device preinstall-name fix are in [troubleshooting/a2a3-507899-aicpu-shared-so-fault.md](troubleshooting/a2a3-507899-aicpu-shared-so-fault.md).
