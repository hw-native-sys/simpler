# Test CLI and logging reference

[Testing overview](../testing.md)

## Scene Test CLI Options

### Lock-free kernel warm-up

`python -m simpler_setup.tools.scene_test_compile` accepts the same pytest
paths and selection arguments as the subsequent scene-test command. It runs
pytest collection, compiles each selected `SceneTestCase` class once, and
stores the resulting `ChipCallable` under `build/cache/kernels/`. It does not
create a `Worker` or access an NPU. The default compiler budget is
`min(8, max(1, available CPUs - 2))`; pass `--compile-workers N` to override it.
The budget counts actual orchestration and incore compiler subprocesses rather
than class threads. Compilation may proceed across classes and across the
artifacts of one large callable, but all levels share the same process-local
semaphore. Consequently class-level and per-callable parallelism do not
multiply the number of active compilers within one process.
simpler's two onboard CI jobs (a2a3, a5) explicitly use eight; the sim jobs have
no warm-up step and use the same automatic token pool during cold pytest runs.

The tool walks `SceneTestCase` classes, so it does not reach a standalone case
that owns its own `Worker`. Such a case warms the same cache itself: the
DeepSeek smokes and Qwen daily cases are compiled by their respective
`examples/<arch>/<runtime>/<model>/main.py --compile-only` entry points. Those
drivers call the same `scene_test.compile_chip_callable_spec` and take the same
`--compile-workers` option.

The persistent cache has two levels. An unchanged callable loads its complete
`callable.bin` directly. When that entry misses, each incore kernel loads an
independent artifact from `build/cache/kernels/incore/`; only missing or changed
kernels are compiled before the new callable is assembled. The incore key is
independent of orchestration source, callable ABI, `func_id`, signature, and
the owning test class. It uses the same compiler-visible paths as the compiler:
paths under the checkout are relative to its root, while paths outside it stay
absolute. This keeps `__FILE__` and `__BASE_FILE__` correct while allowing a
cache restored on another CI runner to hit. Callables
across case directories still share an artifact when they reference the same
physical kernel path with the same source closure and compilation inputs.

A class that fails to compile is reported and skipped rather than aborting the
pass, so the run that follows still recompiles it and reports the error against
the case that owns it.

Pass the same paths, `-m`, `--platform`, `--runtime`, `--level`, and
`--exclude-level` selections to warm-up and execution. A normal pytest or
standalone run loads matching
artifacts from the persistent cache; changed orchestration, incore, or included
source content, compiler identity, fixed compilation flags, compilation logic,
or compilation schema produces the corresponding new cache entry automatically.
Entries unused for 14 days are pruned. When `build/` is not writable — a wheel
installed into a read-only `site-packages` — the cache is skipped with a warning
and every callable is compiled in-process, exactly as before the cache existed.

Scene tests support CLI options for benchmarking, profiling, and runtime control.
The long options have the same meanings in pytest and standalone mode; short
aliases differ where pytest or its plugins reserve a letter.

> "Profiling" is the umbrella for three parallel diagnostics sub-features: `--enable-chip-swimlane` (chip swimlane), `--dump-args` (unified argument dump), and `--enable-pmu` (PMU CSV). They are independent and can be combined.

### pytest

```bash
pytest --platform a2a3sim                                        # default: 1 round + golden
pytest --platform a2a3 --rounds 100 --skip-golden                # benchmark mode
pytest --platform a2a3 --enable-chip-swimlane                             # chip swimlane (single round)
pytest --platform a2a3 --enable-pmu                              # PMU CSV
pytest --platform a2a3sim --log-level debug                        # verbose C++ logging
```

### Standalone (test_*.py)

```bash
python test_xxx.py -p a2a3sim                                    # default: 1 round + golden
python test_xxx.py -p a2a3 -d 0 --rounds 100 --skip-golden       # benchmark mode
python test_xxx.py -p a2a3 --enable-chip-swimlane                         # chip swimlane (single round)
python test_xxx.py -p a2a3 --dump-args                         # dump unified argument artifacts
python test_xxx.py -p a2a3 --enable-pmu 4                        # PMU CSV (MEMORY)
python test_xxx.py -p a2a3sim --log-level debug                  # verbose C++ logging
python test_xxx.py -p a2a3 --case TestFoo::Case1 --prepare-only  # compile to cache, no Worker/device
```

For shared-host benchmarks, prepare selected cases before allocating a device
with `tools/benchmark_rounds.sh --prepare-only`. Submit the standalone scene
test command for each case through the host's queue tool. Add `--single-case`
to require the selector to match exactly one case, and parse timing logs only
after the queued command exits. Keep the multi-case benchmark harness outside
the device allocation.

### Option Reference

| Option | Short | Default | Description |
| ------ | ----- | ------- | ----------- |
| `--rounds N` | | 1 | Run each case N times (reuses the same Worker across rounds) |
| `--device IDS` | `-d` (standalone only) | `0` | Single id (`0`), range (`0-7`), or list (`0,2,5`). Sets the device-id pool for L3 cases and the available slots for L2 fanout. Use `--device` in pytest; pytest-xdist reserves `-d` for distribution. |
| `--max-parallel N` | | `auto` | Max in-flight subprocesses (make-style). `auto` = `min(nproc, len(--device))` on sim, `len(--device)` on hardware. Decouples device-id pool size from parallelism; use to throttle sim on a CPU-constrained runner. |
| `--runtime NAME` | | (all) | Restrict to one runtime; normal dispatch still applies. |
| `--level {2,3,4}` | | (all) | Restrict to one scene-test level. Level 4 selects network1 wrappers. |
| `--exclude-level {2,3,4}` | | (none) | Exclude tests explicitly carrying that scene-test level. Ordinary onboard lanes use `--exclude-level 4`. |
| `--case SEL` | | (all) | Case selector, repeatable: `Foo`, `ClassA::Foo`, `ClassA::` |
| `--prepare-only` | | false | Standalone only: compile selected callables into the persistent cache and exit before creating a Worker. |
| `--single-case` | | false | Standalone only: fail unless selectors match exactly one case. |
| `--manual` | | `exclude` | `exclude`/`include`/`only` for manual scene-test cases and standalone pytest tests |
| `--skip-golden` | | false | Skip golden comparison (for benchmarking) |
| `--enable-chip-swimlane [PERF_LEVEL]` | | `0` | Enable chip swimlane collection. The flag takes an integer perf_level 0–4 (bare = 4); see [docs/dfx/chip-swimlane-profiling.md](../dfx/chip-swimlane-profiling.md#31-enable-chip-swimlane) for the level table. Records land under `outputs/<case>_<ts>/`; see [output-prefix collision limits](parallel-execution.md#profiling-under-parallelism). Disabled when `--rounds > 1`. |
| `--dump-args` | | `0` | Dump tensors plus scalar args into unified runtime artifacts (bare flag = `1`; supports `0/1/2/3`) |
| `--enable-pmu [EVENT_TYPE]` | | `0` | Enable a2a3 PMU CSV collection. Bare flag selects `PIPE_UTILIZATION` (`2`); pass an event type such as `4` for `MEMORY`. |
| `--exitfirst` | `-x` | false | Stop on first failing test (fail-fast, primarily for CI) |
| `--log-level LEVEL` | | `timing` | Simpler logger threshold. Accepts `debug` / `info` / `timing` / `warn` / `error` / `null` (case-insensitive). TIMING and NUL/NULL are registered with Python logging before pytest validates the option. The "simpler" Python logger is the single source of truth; `Worker.init()` snapshots it once and pushes the threshold to HostLogger, AICPU, and the onboard CANN mapping. Changing the Python logger afterwards does not affect an existing worker. See [Log levels](#log-levels). |

Chip swimlane, args dump, PMU, dep-gen, scope stats, and swimlane-overhead analysis are disabled when `--rounds > 1` so benchmark rounds stay uninstrumented. Output tensors are reset to their initial values between rounds.

## Log levels

Simpler uses one integer threshold for host, simulation, and onboard device
logging. For implementation details
(cross-DSO shared state, self-contained logger consumers, host vs device
backends, output formats), see [logging.md](../logging.md).

### Integer layout (Python-aligned)

```text
DEBUG   = 10                                          (Python logging.DEBUG)
INFO    = 20                                          (Python logging.INFO)
TIMING  = 25  ← default threshold
WARN    = 30                                          (Python logging.WARNING)
ERROR   = 40                                          (Python logging.ERROR)
NUL     = 60                                          suppress standard levels
```

The Python `simpler` logger filters by Python's standard `level >= threshold`
rule; C++ and AICPU use the same single-axis rule. `null` suppresses the standard
levels through ERROR. Python records at level 60 or higher can still reach a
handler; it does not disable Python logging entirely.

### Where each tier ends up

| Macro | Level |
| ----- | ----- |
| `LOG_DEBUG` | DEBUG (`10`) — detailed diagnostics |
| `LOG_INFO` | INFO (`20`) — lifecycle and summaries |
| `LOG_TIMING` | TIMING (`25`) — stable performance markers such as `[STRACE]` |
| `LOG_WARN` | WARN (`30`) |
| `LOG_ERROR` | ERROR (`40`) |

At the default TIMING threshold, `[STRACE]`, warnings, and errors are visible;
ordinary INFO and DEBUG messages are silent.

### Setting it

```bash
pytest --platform a2a3sim --log-level debug
python test_xxx.py -p a2a3sim --log-level debug
```

The `simpler` Python logger (`logging.getLogger("simpler")`) is the single
source of truth; `--log-level` sets it. Importing `simpler` puts it at TIMING
unless you have already called `setLevel` — root-logger inheritance is
deliberately not used, so `import simpler` keeps timing markers visible despite
Python's default `WARNING`. `TIMING` and `NUL` are registered with
`logging.addLevelName` at import, so name-based callers and `%(levelname)s`
formatters accept them.

Two consequences for test authors:

- **The threshold is snapshotted once, at `Worker.init()`.** Calling
  `logger.setLevel(...)` afterwards does not reach a live worker — recreate the
  worker to change levels mid-run.
- **Onboard AICPU severity comes from CANN, not from this threshold.**
  `simpler_init` maps the snapshot onto `dlog_setlevel` for you, but an
  `ASCEND_GLOBAL_LOG_LEVEL` already in the environment wins. Since CANN has no
  TIMING level, the default threshold does not open CANN's INFO stream. If you
  set it yourself, set it **before `Worker.init()`** — the AICPU latches CANN's
  answer during that call and never re-reads it.

After submission with a non-empty `output_prefix`, native host records are
written to per-process `host.<pid>.log` files in a transient session directory;
without a prefix they go to stderr. See [host log session routing](../logging.md#layer-3-backend-primitives).
Sim AICPU uses the bound host logger's destination; onboard AICPU lands wherever
CANN is configured to write.

For the mechanism behind all of the above — the process-owned host-log state,
module binding order, the CANN mapping table, and how forked chip subprocesses
inherit the threshold — see
[logging.md § Configuration flow](../logging.md#configuration-flow).

## CLI Design Principles

The long options must have the same semantics in pytest and standalone
(`python test_*.py`). Short aliases must be documented per entry point:

1. **Mirror pytest.** If pytest (or a pytest plugin the project depends on) exposes a flag with a particular short form, standalone must register the same short for the same meaning. Users routinely copy commands between the two entry points; letter-level semantic drift is the fastest way to create confusing bugs.
2. **Avoid new short-form collisions with pytest's ecosystem.** Check pytest and
   its plugins before adding an alias. Existing standalone `-p` and `-d` are
   entry-point-specific exceptions: use `--platform` and `--device` when copying
   those commands into pytest.

Worked examples:

| Flag | Long form | Short | Why |
| ---- | --------- | ----- | --- |
| `--platform` | both | `-p` (standalone only) | pytest reserves `-p` for plugin loading; use `--platform` in pytest. |
| `--device` | both | `-d` (standalone only) | pytest-xdist uses `-d` for distribution; use `--device` in pytest. |
| `--exitfirst` | both | `-x` | pytest ships `-x` → standalone mirrors it so the flag behaves identically across entry points. |
| `--rounds` | both | **(none)** | pytest-xdist already uses `-n` for worker count. Standalone originally had `-n` for `--rounds`, creating a letter-level collision whenever a user switched between pytest (`-n 8` = 8 workers) and standalone (`-n 8` = 8 rounds). Removed in [#574](https://github.com/hw-native-sys/simpler/pull/574); do not reintroduce. |
| `--max-parallel` | both | **(none)** | `-j` would be the natural make-style short, but pytest reserves all lowercase single letters (`parser.addoption` rejects lowercase shorts). Standalone mirrors this to keep both CLIs identical — no short in either, always spell out `--max-parallel`. |
| `--runtime` / `--level` | both | **(none)** | Selection filters; no short keeps them distinctive. A hidden child marker controls dispatcher recursion. |
| `--skip-golden`, `--enable-chip-swimlane`, `--dump-args`, `--enable-pmu`, `--manual`, `--case`, `--log-level` | both | **(none)** | Low-frequency; long form reads better in scripts and docs. Not worth reserving letters. |

Practical guidance when adding a new CLI option:

- First decide the long name and semantics. Register it on **both** `conftest.py::pytest_addoption` and `simpler_setup/scene_test.py::run_module`.
- Check pytest's built-in shorts (`pytest --help`) and loaded plugins (`pytest-xdist`, `pytest-timeout`, etc.) for any letter you're considering.
  - If pytest uses the letter for a matching concept → register the same short in standalone.
  - If pytest uses the letter for anything else → pick a different short, or drop the short entirely.
  - pytest reserves lowercase short options for itself and plugins. If the
    project cannot register the alias in both entry points, prefer a long-only
    option rather than adding another standalone exception.
