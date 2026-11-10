# Parallel test execution and device allocation

[Testing overview](../testing.md)

## Parallel Test Execution and Resource Reuse

Tests are dispatched through a **test dispatcher** (`conftest.py::_dispatch_test_phases` and `scene_test.py::_dispatch_test_phases_standalone`) that pipelines work along two complementary axes:

1. **Resource reuse** — every `ChipWorker` init costs three `dlopen`s plus a device-context acquire. The dispatcher keeps one worker alive for the lifetime of a test group so every class, case, and round on that device reuses the same worker.
2. **Parallelism** — when `--device` names more than one id, independent work is spread across subprocesses so N devices do N-way work.

Pytest follows the hierarchy below. Standalone (`python test_*.py`) uses the
same Resource/L2 scheduling for `SceneTestCase` classes only; marked pytest
functions and leftover pytest items require pytest execution.

```text
Layer 1  Level axis
│
├─ Resource phase (runs first)
│   One isolated subprocess per job; scheduled by device_count, bin-packed
│   against the --device pool. Two job kinds share this phase:
│     - L3 SceneTestCase classes (one job per class)
│     - Standalone pytest functions marked with @pytest.mark.device_count
│       (one job per function). @pytest.mark.runtime is optional: a
│       runtime-agnostic function gets the same devices and the same
│       isolation, and its child is narrowed by nodeid alone.
│   CANN isolation is automatic (each job is its own process). Cross-runtime
│   jobs can overlap when their devices don't.
│
├─ L2 phase (runs after Resource drains)
│   │
│   ├─ Layer 2  Runtime — serial subprocess per runtime (CANN isolation)
│   │   └─ Layer 3  Device — parallel subprocess per device (xdist for pytest,
│   │               custom fanout for standalone). Capacity bounded by --device size
│   │       └─ Layer 4  Class — one ChipWorker per (runtime, device), reused
│   │                   across every class assigned to that device
│   │           └─ Layer 5  Case — serial within a class
│   │               └─ Layer 6  Rounds — `--rounds N` loop, reuses Worker;
│   │                           L2 child memory is declared per tensor, independent of N
│
└─ Leftover phase (runs last, in the dispatching process)
    Every selected item the first two phases do not claim — a plain
    `requires_hardware` function, say, which matches neither a device_count
    marker nor a SceneTestCase runtime. Running them here is what keeps
    "selected" and "run" the same set: an item no phase claims used to be
    reported in neither the passed nor the skipped count. A test that needs
    a process of its own declares `@pytest.mark.device_count` and becomes a
    Resource job instead.
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

### Fail-fast execution

- Default: all cases run; the dispatcher summarizes pass/fail at the end. This is the right mode for local development — you want every failure surfaced at once.
- With `-x` / `--exitfirst`: first failure cancels the pending queue, sends `SIGTERM` to running children, and **skips the L2 phase if L3 failed**. Intended for CI. The short form mirrors pytest's built-in `-x` so the flag behaves identically across pytest and standalone.

### Device-count constraints

If any L3 case declares `device_count > len(--device pool)` the dispatcher fails the whole batch up front rather than deadlocking the scheduler. Either widen `--device` or reduce the case's `device_count`. When `device_count` exceeds the *currently free* pool (but fits within the total), the case waits for an in-flight job to finish and then claims its slot.

### Device pool size and maximum parallelism

On hardware these two concepts collapse: one device = one subprocess's worth of host CPU, so `--device 0-7` naturally implies 8-way parallelism. On sim they diverge:

| `--device` controls | `--max-parallel` controls |
| ------------------- | ------------------------- |
| Size of the virtual device-id pool | Max simultaneous `subprocess.Popen` |
| Must be ≥ any L3 case's `device_count` | Should be ~nproc on sim (CPU-bound) |

This matters on CPU-constrained CI runners. Example: an L3 case needs `device_count=8` but the runner has 2 CPUs.

- `--device 0-1` — can't run the L3 case; static check fails.
- `--device 0-15` (no `--max-parallel`) — L3 case fits; when `os.cpu_count()`
  reports 2, the `auto` default limits top-level subprocesses and L2 xdist
  workers to 2.
- `--device 0-15 --max-parallel 2` — L3 case fits in the pool; at most 2 subprocesses run at a time. This is what the `auto` default computes on a 2-core runner.

`--max-parallel` counts top-level subprocesses, like `make -j`. One L3 class
is one job and runs its selected cases serially in that process; those cases
may share process state. Its Worker internally forks N chip processes for
`device_count`, and those forks do **not** count toward `--max-parallel`.

### Mixed L2 and L3 in one file

A single file can declare both L2 and L3 classes; they're grouped by
`(runtime, level)` internally. L3 classes run in the Resource phase, one
subprocess per class with serial cases, alongside resource-marked pytest
functions. L2 classes run in the L2 phase with a shared Worker per device.

### Profiling under parallelism

Each test case sets its own `CallConfig.output_prefix` (chosen by `scene_test.py::build_output_prefix` as `outputs/<ClassName>_<case>_<YYYYMMDD_HHMMSS>/`). The C++ runtime writes all diagnostic artifacts under that prefix with fixed filenames:

- `outputs/<case>_<ts>/chip_swimlane_records.json` — swimlane (`--enable-chip-swimlane`)
- `outputs/<case>_<ts>/args_dump/` — args dump (`--dump-args`)
- `outputs/<case>_<ts>/pmu.csv` — PMU counters (`--enable-pmu`)

The directory name combines the class-and-case label with a timestamp at
one-second resolution. Independent runs of the same label within the same
second can share a directory; use separate checkouts to isolate such runs.
There is no per-file timestamp, env-var scoping, or post-run flatten step.
`CallConfig::validate()` throws if any diagnostic flag is enabled but
`output_prefix` is empty; `scene_test.py::run_class_cases` fills it from the
case label.

Standalone invocations of CLIs (`python -m simpler_setup.tools.swimlane_converter`, etc.) auto-detect the latest `outputs/*/chip_swimlane_records.json` (sorted by mtime); pass `--input <path>` to override.

### Dispatcher skip conditions (normal pytest runs)

The dispatcher only takes over when there's actual work to parallelize or isolate. It falls through to plain pytest when:

- `--collect-only`
- At most one SceneTestCase runtime is present, no Resource jobs are collected
  (neither L3 classes nor resource-marked pytest functions), and `--level` was
  not explicitly set
- The internal `--resource-child` marker is set on a dispatcher subprocess

## Runtime Isolation Constraint (Onboard)

**One device can only run one runtime per process.** Switching runtimes on the same device within a single process causes AICPU kernel hangs.

### Root Cause

CANN's AICPU dispatch uses a framework SO (`libaicpu_extend_kernels.so`) with a global singleton `BackendServerHandleManager` that:

1. **`SaveSoFile`**: Writes the user AICPU .so to disk on first call, then sets `firstCreatSo_ = true` to skip all subsequent writes.
2. **`SetTileFwkKernelMap`**: `dlopen`s the .so and caches function pointers on first call, then sets `firstLoadSo_ = true` to skip all subsequent loads.

When a second runtime launches on the same device (same CANN process context), the first `simpler_aicpu_exec` launch hits the cached flags — the new AICPU .so is never written or loaded, so it calls function pointers from the first runtime's .so, which operates on incompatible data structures and hangs.

### Impact

| Scenario | Result |
| -------- | ------ |
| Same runtime, same device, sequential | Works (same .so, cached pointers valid) |
| Different runtime, same device, sequential | **Hangs** (stale .so, wrong function pointers) |
| Different runtime, different device | Works (separate CANN context per device) |
| Different runtime, different process, same device | Works (`rtDeviceReset` between processes clears context) |

### Mitigation

When the dispatcher runs, it spawns a separate subprocess per runtime for L2
work and per job for Resource work (L3 classes and resource-marked pytest
functions). These subprocesses isolate CANN runtime state. A top-level `-n N`
with N > 0 bypasses the dispatcher and this isolation; use `--max-parallel` to
size L2 parallelism. A direct run that falls through the skip conditions above
also has no dispatcher-created subprocess boundary.

## Device allocation by the orchestrator and xdist

When running `pytest --platform a2a3 --device 8-11`, the dispatcher does this:

### Resource phase and device bin-packing

For every collected resource job (L3 ``SceneTestCase`` classes and standalone pytest functions marked with ``@pytest.mark.device_count``), the scheduler (`simpler_setup/parallel_scheduler.py`) maintains a free-device set starting at `[8, 9, 10, 11]`. It pops the next queued job and, if the free set can cover its `device_count`, grabs that many ids and spawns:

```text
# L3 class job
pytest <nodeid> --resource-child --runtime <rt> --level 3 --device <alloc-range>

# Standalone resource function job
pytest <nodeid> --resource-child --runtime <rt> --device <alloc-range>
```

When a subprocess completes, its devices return to the free set and the queue is re-tried. Jobs that need more devices than currently free **wait**; jobs that need more than the whole pool **fail the batch up front**.

### L2 phase xdist fanout per device

After the Resource phase drains, one subprocess is spawned per runtime:

```text
pytest --resource-child --runtime <rt> --level 2 --device 8-11 -n 4 --dist loadfile
```

`pytest-xdist` starts 4 workers (`gw0`..`gw3`). Each worker's `pytest_configure` slices `--device 8-11` down to a single id (`gw0` → `8`, `gw1` → `9`, ...), and `st_worker` is session-scoped, so the worker initializes exactly one `ChipWorker(device=N)` and reuses it for every L2 class routed to it. `--dist loadfile` keeps all cases from one test file on the same worker, amortizing any file-level setup cost.

An explicit `-p no:xdist` or `-n 0` is authoritative for the L2 children
inherited from the top-level invocation. The dispatcher does not append xdist
options in either case, so each L2 runtime subprocess executes serially. `--pdb`
reaches the same state, and reaches it on its own gate rather than through the
worker count: xdist zeroes that count only for `-n auto` / `-n logical`, while
a bare `--pdb` leaves it unset — and since the L2 child inherits `--pdb`, an
appended `-n` would make xdist reject the child with
`--pdb is incompatible with distributing tests`. An explicit top-level `--dist`
(or its `-d` shortcut) is preserved too; the dispatcher adds only the options
the invocation left unset.

A top-level `-n N` with N > 0 is a different case: it puts xdist in
distribution mode, whose `pytest_runtestloop` claims the session before this
dispatcher's hook runs, so the phase split does not happen at all and every
collected level runs in one flat xdist fanout. Size L2 parallelism with
`--max-parallel`, not with a top-level `-n`.

Resource-phase subprocess concurrency continues to follow `--max-parallel`
because it does not use pytest-xdist. If plugin autoloading is disabled without
blocking xdist explicitly, the dispatcher warns before falling back to serial
L2 execution.

### L2 phase standalone fanout

Standalone (`python test_*.py -d 8-11`) uses the same scheduler module: classes are round-robin assigned to `len(device_ids)` chunks, one subprocess per chunk launched with a single device and explicit `--case ClassName::` selectors.

### Sim platforms

On sim (`a2a3sim`, `a5sim`), device IDs are virtual — no hardware state, no isolation constraint. All tests share a single virtual pool with auto-incrementing IDs. The same dispatcher + xdist path is used; speedup on sim comes from actual CPU parallelism, not hardware parallelism.

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
