/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
/**
 * Onboard host common helpers — shared between a2a3 and a5 onboard host
 * runtime libraries (`libhost_runtime.so`).
 *
 * Migration target for code that's line-identical between arches; arch-specific
 * extensions (e.g. a2a3's `init_ffts_base_addr`) live as free functions in
 * the arch's own `device_runner.h` rather than being declared here.
 *
 * Current contents:
 *   - `DeviceArgs`: per-task AICPU device-args struct (offsets fixed by
 *     libaicpu_extend_kernels' ABI; layout identical on both archs).
 *   - `KernelArgsHelper`: host-side `KernelArgs` wrapper with device-memory
 *     management for the 3 H2D copies (`DeviceArgs`, `Runtime`, `KernelArgs`).
 *
 * Future migrations (see `.docs/ONBOARD_HOST_COMMON_REFACTOR.md`):
 *   - `DeviceRunnerBase` (lifecycle + registration + profiling init).
 *   - C-API common shims.
 */

#ifndef SIMPLER_COMMON_PLATFORM_ONBOARD_HOST_DEVICE_RUNNER_HELPERS_H
#define SIMPLER_COMMON_PLATFORM_ONBOARD_HOST_DEVICE_RUNNER_HELPERS_H

#include <cstdint>

#include "common/kernel_args.h"  // arch-specific KernelArgs layout
#include "host/memory_allocator.h"
#include "runtime.h"

/**
 * DeviceArgs structure for AICPU device arguments.
 *
 * Layout offsets are still nominally fixed by libaicpu_extend_kernels.so for
 * `aicpu_so_bin` / `aicpu_so_len` (at offsets 96 / 104), but per-task AICPU
 * launches go through `rtsLaunchCpuKernel` against the cached `rtFuncHandle`
 * on `LoadAicpuOp` — none of our code reads these fields. The fields are
 * kept (zero-initialized, never assigned) so the H2D struct layout matches
 * the historical contract on both archs; the runner-side `AicpuSoInfo`
 * allocation that used to back them was removed in PR #877.
 */
struct DeviceArgs {
    uint64_t unused[12] = {0};
    uint64_t aicpu_so_bin{0};
    uint64_t aicpu_so_len{0};
};

/**
 * Helper class for managing `KernelArgs` with device memory.
 *
 * Wraps `KernelArgs` (defined per-arch in `common/kernel_args.h`) and provides
 * host-side initialization methods for allocating device memory and copying
 * data to the device. Separates device-memory management (host-only) from the
 * structure layout (shared with kernels).
 *
 * The helper provides implicit conversion to `KernelArgs *` for seamless use
 * with runtime APIs.
 *
 * Arch-specific extensions (a2a3-only `init_ffts_base_addr`, etc.) live as
 * free functions in the arch's own `device_runner.h`.
 */
struct KernelArgsHelper {
    KernelArgs args;
    MemoryAllocator *allocator_{nullptr};
    KernelArgs *device_k_args_{nullptr};  // Device copy of KernelArgs for AICore

    /**
     * Initialize device arguments by allocating device memory and copying data.
     *
     * @param host_device_args  Host-side device arguments to copy.
     * @param allocator         Memory allocator to use.
     * @return 0 on success, error code on failure.
     */
    int init_device_args(const DeviceArgs &host_device_args, MemoryAllocator &allocator);

    /** Free device memory allocated for device arguments. */
    int finalize_device_args();

    /**
     * Initialize runtime arguments by allocating device memory and copying data.
     *
     * @param host_runtime  Host-side runtime to copy to device.
     * @param allocator     Memory allocator to use.
     * @return 0 on success, error code on failure.
     */
    int init_runtime_args(const Runtime &host_runtime, MemoryAllocator &allocator);

    /** Free device memory allocated for runtime arguments. */
    int finalize_runtime_args();

    /**
     * Allocate device memory for the host-resident `KernelArgs` and copy the
     * struct over. AICore's `KERNEL_ENTRY` expects a `KernelArgs *` (not a
     * `Runtime *`) so it can read the profiling enablement bits + ring address
     * tables and forward them into AICore platform state. Call this after
     * every `kernel_args.args.*` field is populated for the run.
     */
    int init_device_kernel_args(MemoryAllocator &allocator);

    /** Free device memory allocated for the device-resident `KernelArgs` copy. */
    int finalize_device_kernel_args();

    /**
     * Implicit conversion operators for seamless use with runtime APIs.
     *
     * These allow `KernelArgsHelper` to be used wherever `KernelArgs *` is
     * expected, enabling transparent device memory management while
     * maintaining API compatibility.
     */
    operator KernelArgs *() { return &args; }
    KernelArgs *operator&() { return &args; }
};

#endif  // SIMPLER_COMMON_PLATFORM_ONBOARD_HOST_DEVICE_RUNNER_HELPERS_H
