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
 * AICPU Dispatcher — bootstrap-only upload helper.
 *
 * Architecture
 * ============
 *
 * This dispatcher SO has one job: write the bundled runtime SO bytes to the
 * main aicpu_scheduler's process directory. CANN manages its bootstrap load;
 * per-task launches target the runtime SO directly.
 *
 * Bootstrap flow (host → libaicpu_extend_kernels → dispatcher → preinstall):
 *
 *   1. host calls `rtAicpuKernelLaunchExWithArgs` (kernel_type =
 *      `KERNEL_TYPE_AICPU_KFC`) targeting libaicpu_extend_kernels with
 *      DeviceArgs containing:
 *        - aicpu_so_bin / aicpu_so_len  → dispatcher SO bytes (libaicpu_extend_kernels reads)
 *        - inner_so_bin / inner_so_len  → runtime SO bytes    (dispatcher reads)
 *        - result_addr → bounded upload result (dispatcher publishes, host reads)
 *   2. libaicpu_extend_kernels loads the bundled dispatcher,
 *      dlsym's the three CANN-contract symbols (Static + DynInit + Dyn),
 *      invokes our `DynTileFwkBackendKernelServerInit`.
 *   3. Our Init reads inner_so_bin/inner_so_len/device_id from DeviceArgs,
 *      fingerprints the bytes (ELF Build-ID), and writes them to
 *      `<preinstall>/<hostPid>_<localDeviceId>/simpler_inner_<fp>_<device_id>.so`.
 *      The sched thread (HwHiAiUser) owns this dir, so the write succeeds. The
 *      device_id suffix keeps paired dies sharing this filesystem from writing
 *      and executing one shared file (which corrupted the image under
 *      concurrent bootstrap and faulted simpler_aicpu_exec).
 *   4. host verifies the upload result after stream completion, then computes
 *      the same fingerprint + uses the same device_id to derive
 *      the basename. CANN can resolve it in the process-owned directory.
 *   5. Per-task launches: host calls `rtsBinaryLoadFromFile` to
 *      JSON-register the preinstall file (cpuKernelMode=0), resolves
 *      `simpler_aicpu_exec` via `rtsFuncGetByName`,
 *      then dispatches each task through `rtsLaunchCpuKernel` on the cached
 *      `rtFuncHandle`. The main aicpu_scheduler owns the dlopen of the
 *      preinstall file; this dispatcher SO is no longer in the picture.
 *
 * Each runner stages its runtime SO in CANN's process-owned directory. An
 * identical image reuses its immutable file across sequential Workers.
 * CANN removes the directory at session teardown; the dispatcher does not scan
 * historical files or reuse another process's uploads.
 */

#pragma once

#include <cstdint>
#include <string>

#include "aicpu_loader/bootstrap_protocol.h"

namespace simpler_dispatcher {
std::string MakeInnerSoPath(uint64_t fp, uint64_t device_id, BootstrapResult *result = nullptr);
bool WriteBytes(const std::string &path, const char *data, uint64_t len, BootstrapResult &result);
}  // namespace simpler_dispatcher

// CANN resolves all three exports during bootstrap. Only DynInit uploads;
// the other two entries log and return success.
extern "C" {
__attribute__((visibility("default"))) int StaticTileFwkBackendKernelServer(void *args);
__attribute__((visibility("default"))) uint32_t DynTileFwkBackendKernelServerInit(void *args);
__attribute__((visibility("default"))) uint32_t DynTileFwkBackendKernelServer(void *args);
}
