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
 * AICPU Dispatcher — transient bootstrap-only upload helper.
 *
 * Architecture
 * ============
 *
 * This dispatcher SO has one job: write the bundled runtime SO bytes to the
 * main aicpu_scheduler's preinstall path. It is **never** written to disk
 * itself and **never** dispatches at per-task launch time.
 *
 * Bootstrap flow (host → libaicpu_extend_kernels → dispatcher → preinstall):
 *
 *   1. host calls `rtAicpuKernelLaunchExWithArgs` (kernel_type =
 *      `KERNEL_TYPE_AICPU_KFC`) targeting libaicpu_extend_kernels with
 *      DeviceArgs containing:
 *        - aicpu_so_bin / aicpu_so_len  → dispatcher SO bytes (libaicpu_extend_kernels reads)
 *        - inner_so_bin / inner_so_len  → runtime SO bytes    (dispatcher reads)
 *   2. libaicpu_extend_kernels writes the dispatcher bytes to its own private
 *      path (some /tmp on device, often unlinked after open), dlopens us,
 *      dlsym's the three CANN-contract symbols (Static + DynInit + Dyn),
 *      invokes our `DynTileFwkBackendKernelServerInit`.
 *   3. Our Init reads inner_so_bin/inner_so_len/device_id from DeviceArgs,
 *      fingerprints the bytes (ELF Build-ID), and writes them to
 *      `/usr/lib64/aicpu_kernels/0/aicpu_kernels_device/simpler_inner_<fp>_<device_id>.so`.
 *      The sched thread (HwHiAiUser) owns this dir, so the write succeeds. The
 *      device_id suffix keeps paired dies sharing this filesystem from writing
 *      and executing one shared file (which corrupted the image under
 *      concurrent bootstrap and faulted simpler_aicpu_exec).
 *   4. host computes the same fingerprint + uses the same device_id to derive
 *      the same preinstall filename.
 *   5. Per-task launches: host calls `rtsBinaryLoadFromFile` to
 *      JSON-register the preinstall file (cpuKernelMode=0), resolves
 *      `simpler_aicpu_exec` via `rtsFuncGetByName`,
 *      then dispatches each task through `rtsLaunchCpuKernel` on the cached
 *      `rtFuncHandle`. The main aicpu_scheduler owns the dlopen of the
 *      preinstall file; this dispatcher SO is no longer in the picture.
 *
 * Multi-runtime in one host process: each DeviceRunner bootstraps with the
 * same dispatcher bytes + its own runtime SO bytes. A process-level
 * fingerprint cache in LoadAicpuOp short-circuits repeat invocations for
 * the same runtime SO content, so libaicpu_extend_kernels' one-shot
 * `firstCreatSo_` latch fires at most once per (process, fingerprint).
 */

#ifndef COMMON_AICPU_DISPATCHER_AICPU_DISPATCHER_H_
#define COMMON_AICPU_DISPATCHER_AICPU_DISPATCHER_H_

#include <cstdint>

// C-style exports required by libaicpu_extend_kernels' SetTileFwkKernelMap
// dlsym contract. Only DynInit does real work; the other two are stubs that
// log + return failure if ever invoked (they shouldn't be — dispatcher is
// upload-only and host's per-task launches target the runtime SO directly).
extern "C" {
__attribute__((visibility("default"))) int StaticTileFwkBackendKernelServer(void *args);
__attribute__((visibility("default"))) uint32_t DynTileFwkBackendKernelServerInit(void *args);
__attribute__((visibility("default"))) uint32_t DynTileFwkBackendKernelServer(void *args);
}

#endif  // COMMON_AICPU_DISPATCHER_AICPU_DISPATCHER_H_
