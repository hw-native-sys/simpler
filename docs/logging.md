# Log System

Architecture and contracts for the host + device logging subsystem.

For the **user-facing model** — one Python knob, V0..V9 layout, CLI flags
— see [testing.md § Log levels](testing.md#log-levels). This file
documents the implementation: layering, multi-`.so` singleton sharing, ABIs,
build wiring, and output formats.

## Mental model

```text
Python: logging.getLogger("simpler").setLevel(N)
                  │
                  ▼  Worker.init() snapshots level once, splits into (severity, info_v)
                  │
       ChipWorker.init(device_id, bins, sev, v)            ◀── Python wrapper
                  │
       1. ctypes.CDLL(libsimpler_log.so, RTLD_GLOBAL)  ◀── one HostLogger per process
       2. simpler_log_init(sev, v)   ──→ HostLogger.set_level/set_info_v
                                     (seeds HostLogger BEFORE any host_runtime /
                                      sim_context / aicore SO is dlopen'd, so
                                      any LOG_* macro firing during dlopen-time
                                      constructors already sees the right filter)
       3. ctypes.CDLL(libcpu_sim_context.so, RTLD_GLOBAL)  (sim only)
       4. _ChipWorker.init(host_lib, aicpu, aicore, device_id)   ◀── C++
            ├─ dlopen libhost_runtime.so  RTLD_LOCAL   ──→ undefined HostLogger /
            │                                              unified_log_* symbols
            │                                              resolve via (1)
            └─ simpler_init(ctx, device_id, aicpu*, aicore*)
                                     ──→ (onboard) dlog_setlevel(HostLogger.level())
                                     ──→ attach thread + transfer executor binaries

Per kernel launch:
       runner reads HostLogger.level() / .info_v() ──→ KernelArgs.log_level/info_v
       AICPU receives sev/v in KernelArgs → sets g_is_log_enable_* + g_log_info_v
```

Two axes throughout: **severity** (DEBUG/INFO/WARN/ERROR/NUL, CANN-aligned)
and **INFO sub-verbosity** (V0..V9). Python collapses both into a single
integer level; C++ stores them as separate fields.

## File layout

```text
src/common/log/                              ← libsimpler_log.so + public ABI
├── CMakeLists.txt                           libsimpler_log SHARED target
├── include/
│   ├── common/
│   │   └── unified_log.h                    public ABI (host AND device #include)
│   └── host_log.h                           HostLogger class (public — pto_runtime_c_api uses it)
├── host_log.cpp                             HostLogger impl
└── unified_log_host.cpp                     C ABI → HostLogger adapter

src/{a5,a2a3}/platform/                      ← per-arch device-side log
├── include/aicpu/device_log.h               low-level dev_vlog_* declarations
├── src/aicpu/unified_log_device.cpp         C ABI → dev_vlog_* adapter
├── onboard/aicpu/device_log.cpp             onboard backend (CANN dlog)
└── sim/aicpu/device_log.cpp                 sim backend (fprintf to stderr)
```

The `aicpu/device_log.h` and `aicpu/unified_log_device.cpp` files are
byte-identical between a5 and a2a3 today but kept per-arch on purpose:
device-side log code is owned by each platform, not by `src/common/`.

## Three-layer ABI

### Layer 1 — public macros (consumer-facing)

`common/unified_log.h` defines the only macros consumers should use:

```cpp
LOG_DEBUG(fmt, ...)      // severity DEBUG
LOG_WARN(fmt, ...)       // severity WARN
LOG_ERROR(fmt, ...)      // severity ERROR
LOG_INFO_V0(fmt, ...)    // INFO sub-tier 0 (most verbose)
LOG_INFO_V1(fmt, ...)
...
LOG_INFO_V9(fmt, ...)    // INFO sub-tier 9 (most must-see)
```

Each macro auto-injects `[__FILENAME__:__LINE__]` in front of the format
string and threads `__FUNCTION__` as a separate argument.

### Layer 2 — C ABI

The macros expand to four `extern "C"` functions declared in
`common/unified_log.h`:

```cpp
void unified_log_error(const char *func, const char *fmt, ...);
void unified_log_warn (const char *func, const char *fmt, ...);
void unified_log_debug(const char *func, const char *fmt, ...);
void unified_log_info_v(const char *func, int v, const char *fmt, ...);
```

Two implementations link the same ABI symbols:

| Symbol owner | Implementation file | Backend |
| ------------ | ------------------- | ------- |
| `libsimpler_log.so` (host) | `src/common/log/unified_log_host.cpp` | `HostLogger` → stderr |
| AICPU binary (device) | `src/{arch}/platform/src/aicpu/unified_log_device.cpp` | `dev_vlog_*` → backend |

The host `.so` is loaded with `RTLD_GLOBAL` so all consumer `.so`s
(`host_runtime`, `cpu_sim_context`, sim `aicore_kernel`, the binding) resolve
to the **same** `HostLogger` instance. The AICPU binary is independent — it
links its own copy of the ABI implementation that talks to `dev_log_*`
locally.

### Layer 3 — backend primitives

**Host** (`HostLogger` in `host_log.{h,cpp}`):

```cpp
class HostLogger {
public:
    static HostLogger &get_instance();        // process-wide singleton
    void set_level(LogLevel);                 // Python pushes via simpler_init
    void set_info_v(int);
    void vlog       (LogLevel, func, fmt, va_list);   // adapter entry point
    void vlog_info_v(int v,    func, fmt, va_list);
};
```

`vlog{,_info_v}` is the single authority for level gating — the C ABI
adapter does not pre-check.

**Device** (`dev_vlog_*` in `aicpu/device_log.h`):

```cpp
void dev_vlog_debug (const char *func, const char *fmt, va_list);
void dev_vlog_warn  (const char *func, const char *fmt, va_list);
void dev_vlog_error (const char *func, const char *fmt, va_list);
void dev_vlog_info_v(int v, const char *func, const char *fmt, va_list);
```

`unified_log_device.cpp` forwards the caller's `va_list` directly into
`dev_vlog_*` — no intermediate `vsnprintf`-to-buffer round-trip in this
layer. The sim backend is buffer-free (single `vfprintf(stderr, ...)`); the
onboard backend still buffers internally because CANN's `dlog` is variadic
only (no `va_list` variant).

## Multi-`.so` singleton

The host log code lives in **one** `.so` (`libsimpler_log.so`) at
`build/lib/libsimpler_log.so` — process-global, not per arch or variant
(the source has zero arch-specific code, so a single shared copy per host
toolchain is sufficient). Every other `.so` that calls `LOG_*` resolves the
symbols against this single instance via `RTLD_GLOBAL` load order.

### Load order — `ChipWorker.init` (Python wrapper) → `_ChipWorker.init` (C++)

```python
# python/simpler/task_interface.py — ChipWorker.init(device_id, bins, sev, v)
_preload_global(bins.simpler_log_path)              # 1: ctypes.CDLL(RTLD_GLOBAL)
log_handle.simpler_log_init(sev, v)                 # 2: seed HostLogger BEFORE
                                                    #    any consumer SO is opened
if bins.sim_context_path:
    _preload_global(bins.sim_context_path)          # 3: ctypes.CDLL(RTLD_GLOBAL), sim only
self._impl.init(host_path, aicpu_path, aicore_path, device_id)   # 4: C++ _ChipWorker.init
```

```cpp
// src/common/worker/chip_worker.cpp — _ChipWorker.init(...)
handle = dlopen(host_lib_path, RTLD_NOW | RTLD_LOCAL);  // 4a: undefined HostLogger /
                                                        //     unified_log_* resolve via (1)
simpler_init_fn(device_ctx, device_id,
                aicpu_bytes, aicpu_size,
                aicore_bytes, aicore_size);             // 4b: attach thread +
                                                        //     transfer binaries +
                                                        //     (onboard) sync dlog
```

`_preload_global` keeps a process-wide `path → ctypes.CDLL` registry so the
RTLD_GLOBAL load happens exactly once per path (mirrors the old C++
`std::once_flag`). `_task_interface.so` itself has no undefined `HostLogger`
symbols, so the preload only has to precede `_ChipWorker.init`, not module
import.

Each `.so` that needs the host symbols is built **without** compiling
`host_log.cpp` / `unified_log_host.cpp`. On macOS this requires
`-undefined dynamic_lookup`; on Linux undefined symbols in shared libraries
are allowed by default. CMake blocks live in:

- `src/{a5,a2a3}/platform/sim/host/CMakeLists.txt`
- `src/{a5,a2a3}/platform/sim/aicore/CMakeLists.txt`
- `src/common/sim_context/CMakeLists.txt`

(Onboard host `.so` builds Linux-only and needs no flag.)

### Verifying singleton sharing

`cpu_sim_context.cpp::pto_cpu_sim_acquire_device` emits a `LOG_INFO_V0`
diagnostic on first call. With `--log-level v0`:

```text
[2026-05-06 ...][T0x...][INFO_V0] pto_cpu_sim_acquire_device: cpu_sim_context.cpp:167] cpu_sim_context: acquired device 0
[2026-05-06 ...][T0x...][INFO_V0] init_runtime_impl:           runtime_maker.cpp:119] Registering 3 kernel(s) ...
```

Both lines carry the same `HostLogger`-formatted prefix (timestamp, thread
id, level tag), proving that `cpu_sim_context.so` and `host_runtime.so`
resolve to the same `HostLogger` instance. If singleton sharing were
broken, `cpu_sim_context.so` would have its own `HostLogger` defaulting to
V5 and the V0 diagnostic would be silenced entirely.

## Output formats

### Host (`HostLogger::emit`)

```text
[YYYY-MM-DD HH:MM:SS.uuuuuu][T0xTID][LEVEL] func: [file.cpp:line] message
```

Timestamp is local time with microsecond precision; `T0x...` is
`pthread_self()`. Both prefixes are added before the level/func segments so
parallel-test stderr from `pytest-xdist` is recoverable via `sort -k1`
(timestamp) and `grep T0x...` (per-thread).

### AICPU sim

```text
[INFO_V<N>] func: [file.cpp:line] message
[DEBUG]     func: [file.cpp:line] message
[WARN]      func: [file.cpp:line] message
[ERROR]     func: [file.cpp:line] message
```

No timestamp/tid — the AICPU sim path is its own `dev_vlog_*` writing
straight to stderr. Onboard AICPU goes through CANN `dlog` and inherits
its format; the V tier is encoded into the message text as `[V<N>]` since
CANN's level enum has no INFO sub-tiers.

## Configuration flow

| Stage | Action | Source |
| ----- | ------ | ------ |
| Python import | `_log.py` registers `V0..V9` / `NUL` with `logging.addLevelName`; sets `simpler` logger to V5 if untouched | `python/simpler/_log.py` |
| `Worker.init()` | reads `simpler` logger's effective level, splits via `_split_threshold(t) → (sev, info_v)` | `python/simpler/_log.py:get_current_config()` |
| `ChipWorker.init()` (Python) | `ctypes.CDLL(libsimpler_log.so, RTLD_GLOBAL)` → `simpler_log_init(sev,v)` (seeds HostLogger) → `ctypes.CDLL(libcpu_sim_context.so, RTLD_GLOBAL)` (sim) → `_ChipWorker.init` | `python/simpler/task_interface.py` |
| `simpler_log_init` | `HostLogger.set_level/set_info_v` — only writer of log filter | `src/common/log/host_log.cpp` |
| `_ChipWorker.init()` (C++) | `dlopen(host_runtime.so, RTLD_LOCAL)` → dlsym → `simpler_init` | `src/common/worker/chip_worker.cpp` |
| `simpler_init` (per platform) | (onboard) `dlog_setlevel(HostLogger.level())` — must precede device-context open so CANN snapshots the requested level for the device-side log session; then attach thread, transfer executor binaries to runner | `src/{arch}/platform/{onboard,sim}/host/pto_runtime_c_api.cpp` |
| Per kernel launch | runner reads `HostLogger.level() / info_v()` directly, writes into `KernelArgs`; AICPU `kernel.cpp` calls `set_log_level/set_log_info_v` on entry | `src/{arch}/platform/onboard/aicpu/kernel.cpp` |

The Python-side level snapshot is **one-shot** at `Worker.init()`. Calling
`logger.setLevel(...)` afterwards has no effect on a live `ChipWorker` —
recreate the worker if mid-run reconfiguration is needed.

## Build orchestration

`libsimpler_log.so` is built **once per `pip install`** (not per
arch/variant) before any consumer `.so`. `libcpu_sim_context.so` follows
the same pattern (one process-global copy, sim builds only).

| Step | File | Function |
| ---- | ---- | -------- |
| CMake project | `src/common/log/CMakeLists.txt` | `add_library(simpler_log SHARED ...)` |
| Compile invocation | `simpler_setup/runtime_compiler.py` | `RuntimeCompiler.compile_simpler_log()` |
| Build / lookup wrapper | `simpler_setup/runtime_builder.py` | `RuntimeBuilder.ensure_simpler_log()` |
| Top-level orchestration | `simpler_setup/build_runtimes.py` | builds `simpler_log` once before the per-platform loop |
| Output path | — | `build/lib/libsimpler_log.so` |
| Path resolution at runtime | `simpler_setup/runtime_builder.py` | `RuntimeBinaries.simpler_log_path` |

`ChipWorker.init(device_id, bins)` reads `bins.simpler_log_path` (and, on sim,
`bins.sim_context_path`) from `RuntimeBinaries` and `ctypes.CDLL(..., mode=
RTLD_GLOBAL)`s them before handing off to the C++ `_ChipWorker.init`.

## Where to look for what

| You want to … | Look at |
| ------------- | ------- |
| Change the user-facing single-knob model | `python/simpler/_log.py` + `docs/testing.md § Log levels` |
| Change the host output format / pattern | `src/common/log/host_log.cpp::HostLogger::emit` |
| Change the sim AICPU output format | `src/{arch}/platform/sim/aicpu/device_log.cpp::dev_vlog_*` |
| Change the onboard AICPU CANN dlog tagging | `src/{arch}/platform/onboard/aicpu/device_log.cpp::dev_vlog_*` |
| Add a new C ABI entry point (e.g. dynamic config push) | `src/common/log/include/common/unified_log.h` + `unified_log_host.cpp` + `src/{arch}/platform/src/aicpu/unified_log_device.cpp` |
| Hook a new consumer `.so` | declare `target_include_directories(target PRIVATE src/common/log/include)`; for host code also link `simpler_log` (or use undefined symbol resolution at runtime via `RTLD_GLOBAL` load) |
| Add a new severity / verbosity tier | `python/simpler/_log.py` (Python integer + `addLevelName`) + `host_log.h::LogLevel` (if a new severity) + `_split_threshold` (band mapping) + AICPU `set_log_*` setters |
