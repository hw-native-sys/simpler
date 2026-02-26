# Platform Layer

## Purpose

The platform layer (`src/platform/`) abstracts hardware-specific device operations behind shared interfaces so the same runtime code runs on real Ascend hardware (`a2a3`) or a thread-based host simulation (`a2a3sim`). All platform-specific differences are confined here; the runtime layer above sees only the shared headers.

## Directory Layout

```
src/platform/
├── include/          Shared interface headers (the contract)
│   ├── host/         Host-side APIs (C API, memory allocator, register access)
│   ├── aicpu/        AICPU-side APIs (logging, malloc, timing, registers)
│   ├── aicore/       AICore-side APIs (qualifiers, inner_kernel.h switch)
│   └── common/       Cross-component types (CoreType, KernelArgs, platform_config, barriers)
├── src/              Shared implementation (perf collector, unified logging, platform regs)
├── a2a3/             Real Ascend hardware backend
│   ├── host/         CANN runtime APIs (rtMalloc, rtLaunchKernel, rtSetDevice)
│   ├── aicpu/        CANN dlog, HAL memory, ARM64 CNTVCT_EL0 timer
│   └── aicore/       SPR register access via __asm__ instructions
└── a2a3sim/          Thread-based simulation backend
    ├── host/         std::thread, dlopen/dlsym, malloc/free
    ├── aicpu/        printf logging, malloc, std::chrono timer
    └── aicore/       Simulated registers via thread-local volatile memory
```

## Shared Interfaces

All headers live in `include/` and are implemented by **both** backends.

### Host (`include/host/`)

| Header | Role |
|--------|------|
| `pto_runtime_c_api.h` | C API for Python bindings: `init_runtime`, `launch_runtime`, `finalize_runtime`, `device_malloc`, `copy_to_device`, etc. |
| `memory_allocator.h` | RAII allocator that tracks all device allocations and frees them on `finalize()` |
| `host_regs.h` | Retrieves per-core register base addresses for AICore dispatch |
| `function_cache.h` | Caches kernel binaries already uploaded to device memory |
| `performance_collector.h` | Polls and exports swimlane profiling data |
| `platform_compile_info.h` | Reports platform name string |
| `runtime_compile_info.h` | Reports which toolchain (ccec/g++/aarch64-g++) to use per component |

### AICPU (`include/aicpu/`)

| Header | Role |
|--------|------|
| `device_log.h` | Logging macros: `DEV_INFO`, `DEV_DEBUG`, `DEV_WARN`, `DEV_ERROR` |
| `device_malloc.h` | Device-side memory allocation (`aicpu_device_malloc` / `aicpu_device_free`) |
| `device_time.h` | Timestamping via `get_sys_cnt_aicpu()` |
| `aicpu_regs.h` | Register read/write for AICPU-to-AICore task dispatch |
| `platform_regs.h` | Platform-level register access helpers |
| `performance_collector_aicpu.h` | AICPU-side performance recording |

### AICore (`include/aicore/`)

| Header | Role |
|--------|------|
| `aicore.h` | Memory qualifiers (`__aicore__`, `__gm__`) and includes platform-specific `inner_kernel.h` |
| `performance_collector_aicore.h` | AICore-side performance recording |

### Common (`include/common/`)

| Header | Role |
|--------|------|
| `platform_config.h` | Architectural constants: `PLATFORM_MAX_BLOCKDIM` (24), `PLATFORM_CORES_PER_BLOCKDIM` (3: 1 AIC + 2 AIV), `PLATFORM_MAX_AICPU_THREADS` (4), register offsets, `RegId` enum |
| `core_type.h` | `CoreType` enum: `AIC` (compute), `AIV` (vector) |
| `kernel_args.h` | `KernelArgs` struct shared across host, AICPU, and AICore |
| `memory_barrier.h` | `rmb()` / `wmb()` — ARM64 `dsb ld`/`dsb st` on hardware, compiler barriers on x86 |
| `compile_strategy.h` | `ToolchainType` enum: `CCEC`, `HOST_GXX_15`, `HOST_GXX`, `AARCH64_GXX` |
| `perf_profiling.h` | `PerfRecord` and `PerfBuffer` structures |
| `unified_log.h` | Unified logging interface for host and device |

## Key Abstractions

### DeviceRunner

Singleton that manages the device lifecycle. Same public interface, completely different internals:

| Operation | a2a3 (hardware) | a2a3sim (simulation) |
|-----------|-----------------|---------------------|
| Set device | `rtSetDevice(device_id)` | No-op |
| Allocate memory | `rtMalloc()` on HBM | `malloc()` on host heap |
| Copy to device | `rtMemcpy(HOST_TO_DEVICE)` | `memcpy()` |
| Upload kernel | Copy `.o` binary to device GM | `dlopen()` the `.so`, `dlsym("kernel_entry")` |
| Launch AICPU | `rtLaunchKernel()` on AICPU stream | `std::thread(aicpu_execute_func_)` |
| Launch AICore | `rtLaunchKernel()` on AICore stream | `std::thread(aicore_execute_func_)` per core |
| Synchronize | `rtStreamSynchronize()` | `thread.join()` |

### Register Access

All register communication uses a unified `read_reg(RegId)` / `write_reg(RegId, value)` interface with the `RegId` enum (`DATA_MAIN_BASE`, `COND`, `FAST_PATH_ENABLE`).

| | a2a3 | a2a3sim |
|-|------|---------|
| AICore read | `__asm__("MOV %0, DATA_MAIN_BASE")` | Read from `g_sim_reg_base + reg_offset(reg)` via volatile pointer |
| AICPU write | MMIO write to physical register address | Write to `reg_base_array[core_id * SIM_REG_BLOCK_SIZE + offset]` |

### Device Logging

Layered macro system: `DEV_INFO(fmt, ...)` → platform-specific `dev_log_info()`.

| | a2a3 | a2a3sim |
|-|------|---------|
| Backend | CANN `dlog_info()` | `printf()` |
| Level control | AICPU dlog level flags | `PTO_LOG_LEVEL` env variable |

### Device Memory and Timing

| | a2a3 | a2a3sim |
|-|------|---------|
| `aicpu_device_malloc` | `halMemAlloc()` on HBM | `malloc()` |
| `get_sys_cnt_aicpu()` | ARM64 `CNTVCT_EL0` register (50 MHz) | `std::chrono::high_resolution_clock` |

## Compilation

Each component uses a different toolchain depending on the platform:

| Component | a2a3 | a2a3sim | Output |
|-----------|------|---------|--------|
| Host | `g++` (host x86/aarch64) | `g++` | `.so` |
| AICPU | `aarch64-linux-gnu-g++` (cross-compile) | `g++` | `.so` |
| AICore | `ccec` (Bisheng CCE) | `g++-15` or `g++` | `.o` (hw) / `.so` (sim) |

The Python layer queries `get_incore_compiler()` and `get_orchestration_compiler()` via the C API to determine which toolchain to use, keeping the decision in C++ rather than hardcoding it in Python.

## Design Principles

1. **Behavioral equivalence** — Simulation must produce identical results for identical inputs. Same handshake protocol, same memory layout, same task ordering.

2. **Shared interface, separate implementation** — Every API is declared once in `include/`, implemented separately in `a2a3/` and `a2a3sim/`. Adding a new function requires implementing it in both backends.

3. **Three-way code split** — `include/` exports the interface consumed by the runtime layer, `src/` holds implementation shared across all platforms (e.g., performance collector, unified logging, platform registers), and each backend directory (`a2a3/`, `a2a3sim/`) holds platform-specific implementations. This separates interface, common logic, and platform-specific code into distinct locations.

4. **Compile-time platform selection** — No runtime polymorphism or vtables. The platform is selected at build time by linking against the appropriate backend's `.so`.

## Platform-Specific Behavior

### Shared Memory and Cache Coherency (a2a3)

All shared-memory struct fields use `volatile`. The barrier requirements differ by communication path on real hardware:

- **Host ↔ AICPU** — Both processors are cache-coherent. Use `rmb()` / `wmb()` (ARM64 `dsb ld` / `dsb st`) to enforce memory ordering.
- **AICPU ↔ AICore** — AICore has no hardware cache coherency. AICPU (coherent side) does not need extra barriers, but AICore must use `dcci` (data cache clean and invalidate) to see updates written by AICPU.

### Shared Memory and Cache Coherency (a2a3sim)

In simulation, all threads run on the same host CPU. Memory barriers reduce to compiler fences (`__asm__ volatile("" ::: "memory")`), which prevent the compiler from reordering accesses but do not emit hardware barrier instructions.

## See Also

- [README.md](../README.md) — API layers (C++ → C → Python) and high-level architecture diagram
- [.ai-instructions/coding/platform-porting.md](../.ai-instructions/coding/platform-porting.md) — How to add a new platform backend
