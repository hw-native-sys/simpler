# Log System

Architecture and contracts for the host and device logging subsystem.

For the user-facing level model and CLI flags, see
[testing.md § Log levels](testing.md#log-levels). This document covers the
implementation, cross-DSO state ABI, build wiring, and output formats.

## Mental model

```text
logging.getLogger("simpler").setLevel(N)
                  │
                  ▼  Worker.init() snapshots + normalizes the threshold
       _task_interface owns SimplerHostLogState
                  │
                  ├─ its own HostLogger reads that state
                  ├─ state is inherited before hierarchical worker forks
                  └─ ChipWorker passes the state pointer to loaded host modules
                       │
                       ├─ libcpu_sim_context.so (sim, RTLD_GLOBAL for PTO hooks)
                       ├─ libhost_runtime.so (RTLD_LOCAL)
                       ├─ generated host orchestration SOs
                       └─ sim AICore SOs

Each host module:
       private host_log.cpp + unified_log_host.cpp
                  │
                  └─ simpler_host_log_bind_state(state*)

Device logging:
       dev_vlog_* compatibility interface
            ├─ sim: bound HostLogger → same state, envelope, and destination
            └─ onboard: separate CANN backend sampled during device init
```

One threshold controls `DEBUG / INFO / TIMING / WARN / ERROR`; `NUL` suppresses
all output. The values match Python logging (`10 / 20 / 25 / 30 / 40 / 60`).
CANN has no TIMING level, so onboard setup maps both TIMING and WARN to CANN
WARN. The default TIMING threshold keeps host `[STRACE]` markers without
opening CANN's INFO stream.

## File layout

```text
src/common/log/
├── include/
│   ├── common/
│   │   ├── host_log_binding.h     loader-side binder resolution helper
│   │   ├── host_log_state.h       cross-DSO state ABI + binder declaration
│   │   ├── host_span.h            host-span data ABI
│   │   ├── log_level.h            shared levels + CANN mapping
│   │   └── unified_log.h          LOG_* ABI used by host and device code
│   └── host_log.h                 private HostLogger implementation interface
├── host_log.cpp                   host envelope, filter, STRACE grammar
└── unified_log_host.cpp           unified_log_* adapters → HostLogger

src/common/platform/
├── include/aicpu/device_log.h               device backend declarations
├── shared/aicpu/unified_log_device.cpp      LOG_* ABI → dev_vlog_* adapter
├── onboard/aicpu/device_log.cpp             onboard CANN backend
└── sim/aicpu/device_log.cpp                 dev_vlog_* → bound HostLogger adapter
```

There is no standalone `libsimpler_log.so`. Host consumers compile the two
host logger translation units directly. This makes each DSO self-contained:
loading a runtime can no longer fail because a logger artifact or a global
logger symbol was missing.

Each bindable private logger starts with a `NUL` fallback threshold and stays
silent until its loader binds the process-owned state. The `_task_interface`
owner is seeded with the configured user-facing threshold (`TIMING` by default)
before it forks workers or loads runtime modules. A missed binding therefore
appears as an observably silent module instead of output filtered at the wrong
level. A standalone executable that compiles the logger and has no loader must
seed its own state before logging; `query_device_hal` does this explicitly.

## Three-layer ABI

### Layer 1 — consumer macros

`common/unified_log.h` defines the macros consumers use:

```cpp
LOG_DEBUG(fmt, ...)
LOG_INFO(fmt, ...)
LOG_TIMING(fmt, ...)
LOG_WARN(fmt, ...)
LOG_ERROR(fmt, ...)
```

Each macro injects the source location and passes `__FUNCTION__` separately.

Host spans use `SimplerHostSpan` and `unified_log_host_span`. Callers supply
what happened—name, duration, invocation identity, and attributes—while
`HostLogger::log_host_span` owns the PID, TID, timestamp envelope, escaping,
and the single `[STRACE]` format string. Before constructing attributes, hot
call sites query `unified_log_host_span_enabled`; the logger rechecks the
TIMING threshold before validating or encoding a record.

### Layer 2 — unified logging ABI

The macros expand to five `extern "C"` functions declared by
`common/unified_log.h`:

```cpp
void unified_log_error(const char *func, const char *fmt, ...);
void unified_log_warn(const char *func, const char *fmt, ...);
void unified_log_timing(const char *func, const char *fmt, ...);
void unified_log_info(const char *func, const char *fmt, ...);
void unified_log_debug(const char *func, const char *fmt, ...);
```

Host targets compile `unified_log_host.cpp`; AICPU targets compile
`unified_log_device.cpp`. Host definitions have hidden visibility and are
resolved inside the target that contains them. The device implementation
forwards to its local backend.

### Layer 3 — backend primitives

`HostLogger::vlog` is the host-side authority for level gating. It formats one
bounded record (at most the portable 512-byte `PIPE_BUF` floor) and submits it
to the process-owned queue. The queue has 4096 fixed slots and a lock-free
multi-producer claim path. Once the writer is published, producers never perform
file or stderr I/O, wait on a condition variable, acquire the drain mutex, or
grow memory. During hierarchical initialization only, before the final local
fork and writer startup, records use the synchronous destination so startup
failures are not silently discarded. A full queue or exhausted bounded claim is
a dropped record and increments `dropped_record_count`. An immediately-unlinked
named semaphore provides the writer wakeup because unnamed semaphores are not
available on every supported host OS.

Any ordinary human-readable record whose formatted envelope and body exceed
512 bytes is truncated to that fixed size and ends in `~\n`. This keeps one
record to one atomic pipe write when several forked processes share captured
stderr; callers that need a large payload must split it into separate records.

One background writer drains the queue. When `CallConfig.output_prefix` is
present it appends every C++ host-log record to the process-private
`host.<pid>.log`; otherwise it writes to stderr. The bound directory is the
destination, not a preference: a record it cannot take is counted as a drop
rather than relocated, so a run's log is never split across two places.
Severity and span depth no longer choose synchronous producer-side flush paths.
Explicit lifecycle drains wait boundedly for records already accepted by the
process.

`HostLogger::log_host_span` additionally bounds and escapes machine-readable
fields so a STRACE record fits the portable `PIPE_BUF` floor. Its early gate
keeps direct ABI and legacy STRACE callers from paying those encoding costs
when TIMING is disabled; `vlog` remains the final check if the threshold changes
between the caller's query and emission.

The AICPU `dev_vlog_*` interface remains source-compatible on both platforms.
Sim implements it as a thin `va_list` adapter into its bound `HostLogger`, so it
shares the live threshold, envelope, queue, destination, and fallback with
the other host-side modules in that process. Only real-silicon AICPU retains a
separate backend because its records go through CANN dlog rather than a host
process.

## Cross-DSO host state

Each host DSO has a private `HostLogger` object, but every copy in one process
reads the same `SimplerHostLogState`:

```c
struct SimplerHostLogState;
typedef int (*SimplerHostLogEnqueueFn)(
    void *context, struct SimplerHostLogState *state,
    const char *record, uint32_t size);

typedef struct SimplerHostLogState {
    int32_t threshold;
    int32_t log_directory_bound;
    char log_directory[1024];
    int32_t sink_owner_pid;
    int32_t sink_process_pid;
    void *sink_context;
    SimplerHostLogEnqueueFn sink_enqueue;
    uint64_t dropped_record_count;
    uint64_t pending_record_count;
    uint64_t sink_producer_state;
    uint64_t dropped_by_reason[4];
    uint64_t reported_drop_count;
} SimplerHostLogState;

int simpler_host_log_bind_state(SimplerHostLogState *state);
```

The native `_task_interface` extension owns the state for the process.
Thresholds and counters use fixed-width integers; `sink_context` and
`sink_enqueue` are process-local native pointer values shared only by modules
from the same build. `host_log.cpp` accesses mutable fields with atomic
builtins. The struct carries no version or size word: every module that binds it
is compiled from this repository in the same build, and the orchestration SOs
compiled at run time hash this header's whole include closure into their
scene-test cache key, so a stale layout cannot reach a binding. A module that
binds is rejected only for a null pointer or a threshold outside the ladder.

Only `simpler_host_log_bind_state` is exported from a host logging consumer.
`HostLogger` and every `unified_log_*` definition are hidden. This avoids
accidentally recreating the old global-symbol singleton through ELF
interposition while still giving loaders one stable binding entry point.

The first non-empty `log_directory` binding wins, so every bound DSO in the
process chooses the same output without moving a file already in use.

The owner publishes a C callback and opaque context in the same state. A private
logger in any bound DSO can therefore submit to the one process queue without
exporting a C++ object or relying on ELF interposition. The callback accepts a
record only after it owns a queue slot. Binding marks that private logger as a
consumer: it can recognize and use an existing sink, but it cannot create one.
Only the extension copy that owns the process state creates the writer, so a
transient DSO cannot leave its callback or thread behind after `dlclose()`.
`pending_record_count` tracks accepted
work for bounded drains; `dropped_record_count` tracks enqueue rejection and
final write failure. The high bit of `sink_producer_state` closes admission
before a fork boundary; its low bits keep `sink_context` alive until every DSO
caller that may have loaded it has returned. `_host_log_dropped_records()`
exposes the drop counter to diagnostics and tests, while
`_host_log_pending_records()` distinguishes accepted work still waiting for the
writer. Python and C++ flush defaults are both 1000 ms. Teardown and `os._exit()`
paths report a timeout with both counters instead of silently abandoning the
accepted backlog.

### Attributing a drop, and saying so in the log

`dropped_by_reason` splits the total four ways, because the total alone is not
actionable — `queue_full` says the queue is too small for the burst,
`claim_exhausted` says the lock-free claim budget lost to contention with room
still in the queue, `output_failed` says the destination rejected the write, and
`not_admitted` says there was no sink to submit to. Those call for four different
fixes. Every drop increments the total and exactly one reason, so the breakdown
always sums to the total; `_host_log_dropped_records_by_reason()` returns it as a
dict.

The counters live in process memory and die with the process, so a reader holding
only `host.<pid>.log` would otherwise have no way to know records are missing —
truncation leaves a record header behind, a drop leaves nothing. `prepare_to_fork()`
therefore writes the breakdown into the log itself, at ERROR, whenever the total
has grown since the last report:

```text
[HOSTLOG_DROPS] v=1 pid=1234 new=3 total=5 queue_full=5 claim_exhausted=0 output_failed=0 not_admitted=0
```

It reports a growth rather than a running total, so a process that quiesces at
several fork boundaries does not restate the same losses each time. Every path
that stops logging passes through that boundary, including `~HostLogger`.
`strace_timing.py` parses these records and warns before it prints any timing,
since a run that dropped records produces numbers derived from an incomplete log.

### Load and bind order

Python seeds the extension-owned state before C++ loads consumers:

```python
_initialize_host_log(level, defer_writer=is_hierarchical)
# Hierarchical workers perform all local forks here.
_start_host_log_writer()
self._impl.init(host_path, aicpu_path, aicore_path, dispatcher_path,
                device_id, prewarm_config, enable_sdma, sim_context_path)
# At submit, after CallConfig is available:
_set_host_log_directory(config.output_prefix)
```

`ChipWorker::init` then performs the module-specific work:

1. On sim, load `libcpu_sim_context.so` with `RTLD_GLOBAL` once per path and
   retain it in a process-wide registry. Global visibility remains necessary
   for PTO simulator hooks, not for logging. Resolve its binder on the first
   load and pass the shared state.
2. Load `libhost_runtime.so` with `RTLD_LOCAL`.
3. Resolve the runtime's binder from its handle and pass the shared state before
   resolving or calling its regular C API.
4. Call `simpler_init(...)` to configure CANN, attach the device, take ownership
   of executor binaries, and provision the async-DMA workspace.
5. When the runtime later loads a generated host orchestration SO or sim
   AICore SO, resolve that SO's binder from its own handle and pass the same
   state before invoking its entry point.

RAII guards close partially loaded DSOs when initialization fails. A successful
host-runtime handle remains owned by `ChipWorker` and is closed during
finalization. Successful sim-context handles remain in the process registry;
this preserves their process-lifetime simulator state and pthread keys across
sequential `ChipWorker` instances.

## Output formats

### Host

```text
[mono_ns=MONOTONIC_NS][T0xTID][LEVEL] func: [file.cpp:line] message
```

The prefix clock is monotonic nanoseconds, so envelope ordering and host span
timestamps share one clock.
`T0x...` is `pthread_self()`.

`strace_timing.py` keeps the Chrome trace/swimlane timestamps monotonic. See
[Host trace](dfx/host-trace.md) for the rendering contract.

### AICPU sim

```text
[mono_ns=MONOTONIC_NS][T0xTID][LEVEL] func: [file.cpp:line] message
```

Sim AICPU runs on a host CPU inside the host process and uses the same envelope
and destination as every other bound host module. The `dev_vlog_*` names remain
as the compatibility boundary `unified_log_device.cpp` consumes. Onboard AICPU
uses the CANN dlog format. Device TIMING uses CANN WARN and adds a `[TIMING]`
message tag.

## Configuration flow

| Stage | Action | Source |
| ----- | ------ | ------ |
| Python import | Register `TIMING` / `NUL`; default the `simpler` logger to TIMING | `python/simpler/_log.py` |
| `Worker.init()` | Normalize the Python level, attach it to the host logger, quiesce an old writer before local forks, then start a new writer after the final fork | `python/simpler/worker.py` |
| `ChipWorker.init()` | Re-seed inherited native state in a chip child, then enter C++ | `python/simpler/task_interface.py` |
| `_ChipWorker.init()` | Load sim context and host runtime, then bind each module's logger state | `src/common/worker/chip_worker.cpp` |
| `simpler_init` | Onboard maps the bound threshold to CANN; attach and take executor binaries | `src/common/platform/{onboard,sim}/host/c_api_shared.cpp` |
| Nested host load | Bind generated host orchestration/AICore logger state before entry | runtime maker / sim device runner |
| AICPU init | Sim binds the live host state; onboard snapshots CANN policy **and latches `InitArgs.log_level` once for the Worker's life** | platform AICPU init |

The Python level is still sampled during worker initialization. Calling
`logger.setLevel(...)` does not itself call the native setter; recreate or
reinitialize the worker to apply a new Python configuration. Within a process,
all bound host modules observe a native state update immediately instead of
requiring threshold fan-out to every DSO.

### The threshold is live on the host and fixed on the device

Two different lifetimes, and the boundary is the silicon rather than the
language:

| reader | when a `set_level` takes effect |
| ------ | ------------------------------- |
| Every host module in the process, including a `dlopen`ed one | **immediately** — each reads `state()->threshold` on every record |
| Sim AICPU | **immediately** — it runs in the host process as a bound consumer of the same state |
| **Onboard AICPU** | **never; it keeps the threshold it was given at device init** |

Onboard AICPU receives the threshold once, in `InitArgs.log_level`, and
`simpler_aicpu_init` pushes it into a device-side flag. That entry runs once per
Worker, so the value it latched is the value for that Worker's life.

**This is a design decision, not a gap.** Raising verbosity mid-run to chase a
device-side problem is not a workflow this runtime supports: recreate the Worker
with the level you want. Two reasons it is not worth supporting:

- A `Worker` is cheap to recreate, so the workaround costs nothing a debugging
  session would notice.
- Device log volume is exactly where an accidental `DEBUG` is most expensive —
  `codestyle.md` rule 7 forbids logging on AICPU hot paths precisely because
  `device_log` writes serialize on the single AICPU op and can trip the
  op-execute timeout. A threshold that can be raised into that from outside is a
  liability rather than a feature.

Delivering it would also cost more than it looks. The threshold could ride an
existing per-run payload, but making it *immediately* live needs either a device
launch per `set_level` — the opposite direction from #2092, which exists to make
`simpler_aicpu_init` launch exactly once — or a new host-writable device-resident
location polled outside any launch.

So: **any claim that "the log level is live" is scoped to host modules.** State
it that way when writing one.

### The Python logger is a client, not a second system

Seeding the native threshold also installs a handler on the `simpler` logger that
forwards each record through `unified_log_*`. Before that, Python and C++ agreed
only on a threshold: a Python record carried `[%(levelname)s] %(message)s` — no
timestamp, no thread id — so it could not be ordered against a C++ record no
matter where either was written, and it went to whatever handler happened to be
on the root logger rather than to the host logger's own output. Now one envelope,
one clock and one destination cover every record in the process.

Two properties this deliberately keeps:

- **Propagation is untouched**, so a record still reaches the root logger as
  well. That is what keeps an interactive console readable and what keeps
  `caplog.at_level(..., logger="simpler")` working in the tests that assert on
  warnings. The host log is the complete copy; the console is a view of it.
- **The handler is installed where the threshold is seeded, not at import.**
  `import simpler` must keep working when `_task_interface` is missing or stale —
  the build-stamp guard in `simpler/task_interface.py` raises on every
  source-tree move until a rebuild — so putting the extension on the import path
  of the logging surface would lose the logger exactly when it is needed to say
  why. Records logged before a worker is initialized stay on the root logger's
  handler.

A record's level is rounded toward the milder name on the way through (`35`
becomes `WARN`), the opposite of how a *threshold* is normalized: asking for a
threshold of 35 must not silently admit warnings, but a record at 35 is a warning
someone gave a custom number.

### Forked chip subprocesses

The hierarchical parent seeds native state and joins any prior writer before
`fork()`. It starts its new writer only after the final local child exists and
before remote activation creates other threads. A generic fork child starts a
writer after its fallible setup returns (setup may recursively fork a lower
subtree); a chip child starts one while initializing its `ChipWorker`. Normal
`os._exit()` paths and Worker/ChipWorker teardown perform a bounded drain. This
covers both chip-owning L3 workers and higher-level processes that emit
scheduler spans without loading a chip runtime.

### Onboard AICPU severity is CANN-owned

Onboard AICPU reads severity from CANN's dlog rather than from the host state.
`simpler_init` calls `dlog_setlevel(-1, HostLogger.cann_level(), 0)` before
device attach unless `ASCEND_GLOBAL_LOG_LEVEL` is already set; the environment
therefore wins over the Python logger.

| Simpler threshold | CANN level |
| ----------------- | ---------- |
| `debug` | DEBUG |
| `info` | INFO |
| `timing` | WARN |
| `warn` | WARN |
| `error` | ERROR |
| `null` | NUL |

The AICPU samples CANN during its init kernel and latches the result. Configure
`ASCEND_GLOBAL_LOG_LEVEL` before `Worker.init()` if direct CANN control is
needed.

## Build orchestration

There is no logger build step or logger field in `RuntimeBinaries`. Instead:

- `_task_interface`, all host runtimes, sim-context, and sim AICore targets add
  `host_log.cpp` and `unified_log_host.cpp` to their source lists.
- Sim AICPU targets add `host_log.cpp` while keeping `unified_log_device.cpp`;
  the device ABI delegates to HostLogger there.
- Host-compiled generated orchestration SOs receive the same sources through
  `KernelCompiler.get_orchestration_cache_inputs`; those sources therefore
  participate in the scene-test cache key.
- AICPU orchestration built for silicon keeps the device logging ABI and does
  not compile the host logger.
- `libcpu_sim_context.so` remains a standalone process-global simulator helper
  and is built once for sim platforms.

## Where to look

| You want to … | Look at |
| ------------- | ------- |
| Change the user-facing level model | `python/simpler/_log.py` and `docs/testing.md` |
| Change host output or STRACE grammar | `src/common/log/host_log.cpp` |
| Change the shared-state ABI | `src/common/log/include/common/host_log_state.h` |
| Change sim AICPU adaptation | `src/common/platform/sim/aicpu/device_log.cpp` |
| Change onboard CANN tagging | `src/common/platform/onboard/aicpu/device_log.cpp` |
| Add a host logging consumer | compile both host logger sources, include `src/common/log/include`, and bind state during module init |
| Add a level | `log_level.h`, `_log.py`, `simpler_setup/log_config.py`, and AICPU `set_log_level` |
