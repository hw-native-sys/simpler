# Testing

Use this page for test principles and quick commands. The focused guides cover
configuration, execution and authoring in detail.

## Detailed guides

| Guide | Contents |
| ----- | -------- |
| [Environments and hardware](testing/environments.md) | Dependency profiles, hardware markers and platform selection |
| [CLI and logging reference](testing/cli.md) | Compile warm-up, command options, log levels and CLI conventions |
| [Parallel execution](testing/parallel-execution.md) | Resource reuse, runtime isolation, device allocation and filtering |
| [Test suites and unit tests](testing/unit-tests.md) | Suite roles, C++/Python execution, build coverage and CTest NPU resources |
| [Writing scene tests](testing/writing-tests.md) | Scene template, sibling synchronization, output validation and manual cases |
| [Adding a C++ unit test](testing/adding-a-cpp-unit-test.md) | Directory contracts, target declarations, support archives and checks |
| [Scene-test device-memory fixtures](testing/child-memory.md) | Persistent test fixtures, round state and result readback |

## Design Principles

Three axioms govern hardware classification across all test categories:

1. **sim = no hardware**: `a2a3sim`, `a5sim`, and no `--platform` are all equivalent — no hardware required.
2. **a2a3 / a5 are distinct platforms**: Tests may support one, both, or neither.
3. **`requires_hardware` has two levels**:
   - `requires_hardware` (no argument) — needs *any* hardware (a2a3 or a5)
   - `requires_hardware("a2a3")` — needs *specifically* a2a3

These principles apply uniformly to ut-py (pytest markers), ut-cpp (ctest labels), and st (`@scene_test(platforms=[...])`).

### Assertion principle: no wall-clock proxies

All CI correctness tests (ut-py/ut-cpp, simulation and onboard hardware) must
establish the tested state and event ordering explicitly. Synchronize through
handshakes, barriers, acknowledgements or observable state transitions, then
assert the functional result. See the always-loaded
[CI correctness rule](../.claude/rules/discipline.md#7-ci-correctness-tests-use-explicit-events).

Do not use wall-clock magnitude, sleeps, calibrated spin counts or relative
execution speed as a proxy for concurrency, ordering or blocking. For example,
hold a predecessor at an observable gate while checking successor preparation,
then release it; do not assume preparation finishes inside an artificial delay.
Such assumptions change with machine load, clock frequency and host scheduling.
A fault test must confirm that its injected event occurred in the intended
state. Waiting longer for background interference, or relying on repeated CI
runs to eventually encounter it, does not establish fault coverage; keep such
probabilistic stress experiments in explicitly requested diagnostics.

Timeouts provide a finite failure backstop for event waits. Their expiry or
non-expiry does not prove an unrelated correctness property. When wall-time
*is* the subject under test (a timeout firing, a startup deadline being
respected, a configured budget), asserting it is correct. Budget generously for
CI variance, and check the mechanism's observable effect rather than relying
only on a tight measured bound. Explicit performance benchmarks are separate
from correctness tests. See [#1496](https://github.com/hw-native-sys/simpler/issues/1496).

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
pytest examples tests/st --platform a2a3sim       # A2/A3 simulation
pytest examples tests/st --platform a5sim         # A5 simulation
pytest examples tests/st -m "not sdma and not sdma_fault" --platform a2a3 --exclude-level 4          # hardware
pytest examples tests/st -m "not sdma and not sdma_fault" --platform a2a3 --exclude-level 4 --device 4-7  # hardware with device pool

# Compile the selected hardware batch without creating a Worker or using an NPU
python -m simpler_setup.tools.scene_test_compile examples tests/st \
    -m "not sdma and not sdma_fault" --platform a2a3 --exclude-level 4 --require-pto-isa --compile-workers 8

# SDMA demos and the fault-injection case use separate device tasks, as in CI.
# The fault case is selected by @pytest.mark.sdma_fault (issue #1425).
pytest examples tests/st -m "sdma and not sdma_fault" --platform a2a3 --device 4-5
pytest examples tests/st -m sdma_fault --platform a2a3 --device 4

# A5 runs the non-network1 corpus, including SDMA tests, on both host architectures
pytest examples tests/st --platform a5 --exclude-level 4 --device 0-7

# Single scene test (standalone)
python examples/a2a3/tensormap_and_ringbuffer/vector_example/test_vector_example.py -p a2a3sim --manual include

# Benchmark mode (100 rounds, skip golden comparison)
python examples/a2a3/tensormap_and_ringbuffer/vector_example/test_vector_example.py \
    -p a2a3 -d 0 --rounds 100 --skip-golden

# Profiling (single-round only; diagnostics are disabled with --rounds > 1)
python examples/a2a3/tensormap_and_ringbuffer/vector_example/test_vector_example.py \
    -p a2a3 --enable-chip-swimlane

# Args dump
python tests/st/a2a3/tensormap_and_ringbuffer/alternating_matmul_add/test_alternating_matmul_add.py \
    -p a2a3 -d 11 --dump-args
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

## Related guides

### Sanitizer builds (ASAN / UBSan / TSAN)

Opt-in `-fsanitize` instrumentation of host-compiled code via `--sanitizer`
(build with `SIMPLER_SANITIZER=...`, run with the matching runtime preloaded).
The design, load-bearing invariants, usage, and presets now live in their own
page — see **[sanitizers.md](sanitizers.md)**. The nightly CI job is in
[ci.md](ci.md#nightly-sanitizer-sweep); the scoping rationale (macOS / TSAN / LSan) is in
[investigations/2026-06-sanitizer-scope.md](investigations/2026-06-sanitizer-scope.md).

### CI Pipeline

See [ci.md](ci.md) for the full CI pipeline documentation, including the job matrix, runner constraints, and marker scheme.

### Previous section links

Old links into this overview remain available here; follow the corresponding
link for the full section.

<!-- markdownlint-disable MD033 -->

| Previous section | Current location |
| ---------------- | ---------------- |
| <a id="test-environments"></a>Test environments | [Read section](testing/environments.md#test-environments) |
| <a id="scene-test-cli-options"></a>Scene Test CLI Options | [Read section](testing/cli.md#scene-test-cli-options) |
| <a id="lock-free-kernel-warm-up"></a>Lock-free kernel warm-up | [Read section](testing/cli.md#lock-free-kernel-warm-up) |
| <a id="pytest"></a>pytest | [Read section](testing/cli.md#pytest) |
| <a id="standalone-test_py"></a>Standalone (test_*.py) | [Read section](testing/cli.md#standalone-test_py) |
| <a id="option-reference"></a>Option Reference | [Read section](testing/cli.md#option-reference) |
| <a id="log-levels"></a>Log levels | [Read section](testing/cli.md#log-levels) |
| <a id="integer-layout-python-aligned"></a>Integer layout (Python-aligned) | [Read section](testing/cli.md#integer-layout-python-aligned) |
| <a id="where-each-tier-ends-up"></a>Where each tier ends up | [Read section](testing/cli.md#where-each-tier-ends-up) |
| <a id="setting-it"></a>Setting it | [Read section](testing/cli.md#setting-it) |
| <a id="cli-design-principles"></a>CLI Design Principles | [Read section](testing/cli.md#cli-design-principles) |
| <a id="parallel-test-execution-and-resource-reuse"></a>Parallel Test Execution and Resource Reuse | [Read section](testing/parallel-execution.md#parallel-test-execution-and-resource-reuse) |
| <a id="quick-examples"></a>Quick examples | [Read section](testing/parallel-execution.md#quick-examples) |
| <a id="fail-fast-x-exitfirst"></a><a id="fail-fast--x----exitfirst"></a>Fail-fast (`-x` / `--exitfirst`) | [Read section](testing/parallel-execution.md#fail-fast-execution) |
| <a id="device-count-constraints"></a>Device-count constraints | [Read section](testing/parallel-execution.md#device-count-constraints) |
| <a id="-device-vs-max-parallel-two-separate-knobs"></a><a id="--device-vs---max-parallel-two-separate-knobs"></a>`--device` vs `--max-parallel` (two separate knobs) | [Read section](testing/parallel-execution.md#device-pool-size-and-maximum-parallelism) |
| <a id="mixed-l2-l3-in-one-file"></a><a id="mixed-l2--l3-in-one-file"></a>Mixed L2 + L3 in one file | [Read section](testing/parallel-execution.md#mixed-l2-and-l3-in-one-file) |
| <a id="profiling-under-parallelism"></a>Profiling under parallelism | [Read section](testing/parallel-execution.md#profiling-under-parallelism) |
| <a id="dispatcher-skip-conditions-normal-pytest-runs"></a>Dispatcher skip conditions (normal pytest runs) | [Read section](testing/parallel-execution.md#dispatcher-skip-conditions-normal-pytest-runs) |
| <a id="hardware-classification"></a>Hardware Classification | [Read section](testing/environments.md#hardware-classification) |
| <a id="helper-function"></a>Helper function | [Read section](testing/environments.md#helper-function) |
| <a id="ut-py-pytestmarkrequires_hardwareplatform"></a>ut-py: `@pytest.mark.requires_hardware[(platform)]` | [Read section](testing/environments.md#ut-py-pytestmarkrequires_hardwareplatform) |
| <a id="ut-cpp-ctest-labels"></a>ut-cpp: ctest labels | [Read section](testing/environments.md#ut-cpp-ctest-labels) |
| <a id="st-scene_testplatforms"></a>st: `@scene_test(platforms=[...])` | [Read section](testing/environments.md#st-scene_testplatforms) |
| <a id="test-types"></a>Test Types | [Read section](testing/unit-tests.md#test-types) |
| <a id="c-unit-tests-testsutcpp"></a>C++ Unit Tests (`tests/ut/cpp/`) | [Read section](testing/unit-tests.md#c-unit-tests-testsutcpp) |
| <a id="how-many-times-a-case-is-built"></a>How many times a case is built | [Read section](testing/unit-tests.md#how-many-times-a-case-is-built) |
| <a id="where-the-stand-ins-come-from"></a>Where the stand-ins come from | [Read section](testing/unit-tests.md#where-the-stand-ins-come-from) |
| <a id="python-unit-tests-testsut"></a>Python Unit Tests (`tests/ut/`) | [Read section](testing/unit-tests.md#python-unit-tests-testsut) |
| <a id="examples-examplesarch"></a>Examples (`examples/{arch}/`) | [Read section](testing/unit-tests.md#examples-examplesarch) |
| <a id="device-scene-tests-testsstarch"></a>Device Scene Tests (`tests/st/{arch}/`) | [Read section](testing/unit-tests.md#device-scene-tests-testsstarch) |
| <a id="writing-new-tests"></a>Writing New Tests | [Read section](testing/writing-tests.md#writing-new-tests) |
| <a id="new-c-unit-test"></a>New C++ Unit Test | [Read section](testing/unit-tests.md#new-c-unit-test) |
| <a id="c-hardware-tests-needing-npu-devices"></a>C++ hardware tests needing NPU devices | [Read section](testing/unit-tests.md#c-hardware-tests-needing-npu-devices) |
| <a id="new-scene-test"></a>New Scene Test | [Read section](testing/writing-tests.md#new-scene-test) |
| <a id="kernels-that-wait-for-siblings-mix-sync_start"></a>Kernels That Wait for Siblings (Mix, sync_start) | [Read section](testing/writing-tests.md#kernels-that-wait-for-siblings-mix-sync_start) |
| <a id="output-validation"></a>Output Validation | [Read section](testing/writing-tests.md#output-validation) |
| <a id="case-selection-and-manual-cases"></a>Case Selection and Manual Cases | [Read section](testing/writing-tests.md#case-selection-and-manual-cases) |
| <a id="sharing-an-example-between-examples-and-testsst"></a>Sharing an Example Between examples/ and tests/st/ | [Read section](testing/writing-tests.md#sharing-an-example-between-examples-and-testsst) |
| <a id="runtime-isolation-constraint-onboard"></a>Runtime Isolation Constraint (Onboard) | [Read section](testing/parallel-execution.md#runtime-isolation-constraint-onboard) |
| <a id="root-cause"></a>Root Cause | [Read section](testing/parallel-execution.md#root-cause) |
| <a id="impact"></a>Impact | [Read section](testing/parallel-execution.md#impact) |
| <a id="mitigation"></a>Mitigation | [Read section](testing/parallel-execution.md#mitigation) |
| <a id="device-allocation-orchestrator-xdist"></a><a id="device-allocation-orchestrator--xdist"></a>Device Allocation (Orchestrator + xdist) | [Read section](testing/parallel-execution.md#device-allocation-by-the-orchestrator-and-xdist) |
| <a id="resource-phase-device-bin-packing"></a><a id="resource-phase--device-bin-packing"></a>Resource phase — device bin-packing | [Read section](testing/parallel-execution.md#resource-phase-and-device-bin-packing) |
| <a id="l2-phase-xdist-fanout-per-device"></a><a id="l2-phase--xdist-fanout-per-device"></a>L2 phase — xdist fanout per device | [Read section](testing/parallel-execution.md#l2-phase-xdist-fanout-per-device) |
| <a id="l2-phase-standalone-fanout"></a><a id="l2-phase--standalone-fanout"></a>L2 phase — standalone fanout | [Read section](testing/parallel-execution.md#l2-phase-standalone-fanout) |
| <a id="sim-platforms"></a>Sim platforms | [Read section](testing/parallel-execution.md#sim-platforms) |
| <a id="per-case-device-filtering"></a>Per-Case Device Filtering | [Read section](testing/parallel-execution.md#per-case-device-filtering) |
| <a id="when-separate-directories-are-still-needed"></a>When separate directories are still needed | [Read section](testing/parallel-execution.md#when-separate-directories-are-still-needed) |
| <a id="explicit-l2-child-memory"></a>Explicit L2 child memory | [Read section](testing/child-memory.md#explicit-l2-child-memory) |

<!-- markdownlint-enable MD033 -->
