# Test environments and hardware classification

[Testing overview](../testing.md)

## Test environments

Tests requiring an isolated dependency set declare
`@pytest.mark.test_environment("torch_npu")` on a module, class, function or
parameter. Every inherited declaration applies: repeated identical names are
idempotent, conflicting or unknown names are errors, even on skipped or
otherwise deselected items. Unmarked tests belong to `default`.

`--test-environment` selects one registered environment; its default is
`default`. The same pytest selection applies to scene-test compile warm-up and
resource/runtime children. Platform, runtime, `-m`, `-k` and `manual` remain
independent filters. Environment classification is not dependency detection:
missing or incompatible dependencies fail a selected environment; they never
silently move tests or skip qualification. Test modules must remain safe to
collect without their optional backend; import it in fixtures or consumers.

The `torch_npu` profile requires `--junitxml`: a single pytest invocation writes
its final collection to the adjacent `.plan.json` file, executes it, then checks
its actual JUnit against that plan. The plan retains the actual preflight
identity, including CANN manifest version and host architecture. Source,
environment and canonical nodeids must match, and every selected case must pass
exactly once. The result and adjacent plan are both held under nonblocking session
locks in a common namespace until pytest cleanup. A concurrent invocation using either artifact is
rejected before JUnit registers, including collect-only runs; different result
files with distinct plans remain independent. Adjacent `.lock` files remain on disk, while their
OS locks are released when the owner exits. Empty collection,
missing/foreign/duplicate cases and any skip/error/failure are errors. This
strict mode supports direct execution only. SceneTestCase, resource dispatch
and xdist are rejected; default sweeps keep their existing execution strategy.
An explicitly supplied device pool and `--max-parallel N` can partition direct
tests across independent pytest subprocesses, with one physical device per
subprocess. The controller preserves the original selection and execution
options and merges the disjoint results into the requested JUnit file before
checking the full immutable collection plan. `--max-parallel 1` retains serial
execution. `--pdb` and `--trace` also execute serially in the controller, even
with multiple devices, so debugger input and output remain interactive.
`-x`, a nonzero `--maxfail`, and stepwise modes also use the controller's serial
loop to retain pytest's global stop semantics; `--maxfail=0` keeps parallel execution enabled.
An active pytest file log (`--log-file` or ini `log_file`) also uses the serial
controller so subprocesses cannot truncate or overwrite that file. The default
`/dev/null` handler and disabled logging plugin leave parallel execution enabled.
Framework CI allocates four physical devices and uses `--max-parallel 4` on both
platforms. Single-device callers retain their existing direct execution.
Per-shard logs and results are retained beside the parent plan under a unique
`<plan-stem>-shards-*` directory for each invocation. Failed, skipped, duplicate or missing receipts
still fail the complete run.
The controller owns pytest's last-failed cache and combines all shard outcomes;
children retain cache-based selection but do not write last-failed or nodeid caches.
With `-s` or `--capture=tee-sys`, shard output is streamed while tests run and retained in each shard log.
Shard stdout/stderr is flushed to its log as it is read, independently of completion
callbacks, and remaining output is drained during handled interruption. Child-side
buffering still follows pytest capture and Python rules; a log write error fails the run.
Readable shard XML is retained in the aggregate even when another shard crashes;
the missing shard and its assigned cases are reported as an error. Shards run
in separate process groups, which are terminated on timeout or interruption.
Collect-only and compile warm-up select environments without importing or
checking optional dependencies and do not emit execution plans.

For hardware runs, install dependencies and build the framework bridge before
requesting devices. Populate the managed PTO-ISA checkout by calling
`simpler_setup.pto_isa.ensure_pto_isa_root()` with the same installed-wheel
`PYTHONPATH` used by the device job; warming the source checkout alone does not
populate the installed assets' cache. `framework_eager/main.py --compile-only --build-dir <dir>`
prepares the bridge; exporting `SIMPLER_FRAMEWORK_BUILD_DIR=<absolute-dir>` makes
the fixture reuse it. Without that explicit setting the fixture retains its
existing session-local build. Run
`python examples/workers/l2/framework_eager/warm_kernels.py --platform <arch> --output <warmup.json>`
outside the allocation with the same Python, CANN environment and `PYTHONPATH`
as the device run. It calls the consumers' own compile entries for both runtimes,
including lifecycle x/y and HOST-control value arguments, and records each
callable's size and SHA256 plus the resolved kernel-cache and PTO-ISA paths.
The command refuses a device allocation before importing the backend.
Use its recorded paths to check that the device process consumes the same cache;
source, native ABI or compiler changes require warming it again. Cache misses
retain the compiler's normal rebuild behavior, so a passing device run alone
does not prove that compilation stayed outside the allocation. Audit compiler
execution separately when validating the prepared run.
The destroy-refusal case can reuse a precompiled shim through
`SIMPLER_FRAMEWORK_DESTROY_SHIM=<absolute-so-path>`; build it from
`tests/st/worker/framework_eager/native_capture_destroy_shim.c` using
`cc -std=c11 -O2 -fPIC -shared -Wall -Wextra -Werror <source> -o <output> -ldl`
before allocation. The eager entry point accepts `--bridge <existing-so>` to
load the prepared bridge without any relinking.
The GIL submission tests can reuse `SIMPLER_FRAMEWORK_NATIVE_PROBE=<absolute-so-path>`;
prepare it with `python tools/framework_queue_gil/probe.py build --output <dir>`
before allocation. These opt-in prepared paths leave existing callers' local
fixture fallbacks available.

After preparation, an allocated four-card run can use the existing pytest
selection with `--device "$TASK_DEVICE" --max-parallel 4 -p no:xdist`.
Pass the prepared paths through `task-submit --env` and perform the architecture
precheck inside that same allocation. Use `--max-parallel 1` as the serial
control and compare the complete selected-case receipts across both runs;
elapsed time alone does not prove that the same tests ran.

The ScopeStats consumer uses the same declared environment and shared bridge.
It checks off/on/on/off selection, per-call exports and nested scope attribution
on TMR. HBG is explicitly metadata only; neither path treats allocation
frontiers as live usage or claims performance.

The diagnostic-selection consumer uses that environment and bridge to check
per-call PMU and chip-swimlane snapshots and exports. Its off/on/on/off sequence
mutates the original selectors and prefixes after submission, checks the original
output destinations and rejects records for off calls or poison destinations.
The CPU checker contracts exercise export validation; actual collector behavior
requires the framework hardware cases. Diagnostic collection may wait, and this
consumer does not qualify other DFX collectors or capture/replay.

The framework eager tests live in `tests/st/worker/framework_eager` and run on
`--platform a2a3` or `--platform a5`. Each consumer uses that platform for
compilation, Worker construction and its result record; compile caches include
the platform. A shared host bridge does not establish device compatibility.
Each platform needs its own actual plan, JUnit and environment identity.

The framework eager corpus shares one session-scoped optional bridge fixture.
Its same-stream, cross-stream, alias/INOUT, callback-failure recovery and
per-call-value and persistent-DEVICE-state modules are discovered by their
environment declaration; adding
a module does not require an expected-nodeid list or workflow path edit. Run the
declared environment under an exclusive device allocation. Use one device with
`--max-parallel 1` as the serial control; a filtered plan proves only that filtered set.
The recovery consumer also exercises the public direct-chip close path: accepted
handles are drained before workspace ownership can be released, without
individual user waits. Native timeout/context refusals and retryable close debt
are covered separately by CPU lifecycle contracts.

The pre-admission consumer checks stale Buffer registration and a distinct
caller-created RTS context in real queue callbacks. Each refusal is followed
by healthy work on the same Worker. The wrong-context callback restores the
caller context before its healthy submit, and a third healthy request follows.
The allocation stays alive throughout; no invalid device access is issued.

The argument-capture consumer is discovered in the same environment. It
checks OFF/FULL/HYBRID/OFF snapshots and exact two-task argument payloads on
A2/A3 (both runtimes) and A5 TMR, including a runtime-owned intermediate.
A5 HBG instead exercises the existing typed unsupported refusal for FULL and
HYBRID, verifies no accepted run, tensor writes or dump files, and checks healthy
OFF calls before and after plus caller-stream reuse after Worker close. Its
refusal result explicitly reports no verified payloads; actual A5 HBG per-task
capture remains unimplemented and tracked under
[#2666](https://github.com/hw-native-sys/simpler/issues/2666); refusal tests do not
deliver that capability. A real Worker materialization CPU
contract covers native geometry; CPU recording tests do not qualify device
collection or framework ordering.

The registry and verifier live in `simpler_setup/test_environments.py`. Package
pins have one source in the profile's constraints file. Selector and preflight
share a minimum CANN version line of 9.0, without an architecture allowlist.
They compare the numeric major/minor prefix of the actual install manifest;
separated CANN release/build labels do not change admission. For example,
9.0.RC1, 9.1.T500, 9.2.0-beta.2 and 10.0 are accepted; missing, malformed or
lower version lines are rejected. Admission does not mean every combination
has been tested. Framework execution evidence covers 9.0.0 on x86_64 and
9.2.0-beta.2 on A3 ARM64 with the pinned Torch pair. See
[the CI environment contract](../ci.md) for the experiment link.
Adding a test to an existing environment requires no workflow edit. A new
environment requires an explicit registry entry and reviewed installation/runner
strategy; profiles do not install arbitrary dependencies or infer new CI runners.

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
| `LABELS "no_hardware"` | run | skip | skip |
| `LABELS "requires_hardware"` | skip | run | run |
| `LABELS "requires_hardware_a2a3"` | skip | run | skip |
| `LABELS "requires_hardware_a5"` | skip | skip | run |

Selection:

| Runner | Command |
| ------ | ------- |
| No hardware | `ctest -LE requires_hardware` |
| a2a3 | `ctest -L "^requires_hardware(_a2a3)?$"` |
| a5 | `ctest -L "^requires_hardware(_a5)?$"` |

`-LE` (exclude regex) on no-hw runner: `requires_hardware` matches all three
hardware label variants. Unlabeled tests and tests labeled `no_hardware` run.
`-L` (include regex) on device runners: only labeled tests run, unlabeled ones are excluded.

### st: `@scene_test(platforms=[...])`

`platforms` lists all supported platform names (both sim and device).

| `--platform` | Behavior |
| ------------ | -------- |
| `a2a3sim` | Run if `"a2a3sim"` in `platforms`, else skip |
| `a2a3` | Run if `"a2a3"` in `platforms`, else skip |
| *(none)* | Skip with `--platform required` |

Pass `--platform` explicitly. The runner selects cases supporting that
platform; it does not automatically parametrize over the declared simulators.

Examples:

```python
@scene_test(level=2, platforms=["a2a3sim", "a2a3", "a5sim", "a5"], ...)
class TestFoo(SceneTestCase): ...
# --platform a2a3sim  → TestFoo[a2a3sim]
# --platform a2a3     → TestFoo[a2a3]
# (none)              → skip (--platform required)

@scene_test(level=2, platforms=["a2a3"], ...)
class TestHwOnly(SceneTestCase): ...
# --platform a2a3     → TestHwOnly[a2a3]
# (none)              → skip (--platform required)
```

No separate `st` or `requires_hardware` marker — `platforms` is the sole declaration.
