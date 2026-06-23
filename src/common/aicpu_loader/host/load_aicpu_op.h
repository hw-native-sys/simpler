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
 * @file load_aicpu_op.h
 * @brief Host-side AICPU operation loader.
 *
 * Three-phase architecture:
 *
 *   1. BootstrapDispatcher (per-DeviceRunner, idempotent across instances in
 *      the same process via a content-fingerprint cache): bundles dispatcher
 *      SO bytes + runtime SO bytes into a single
 *      `rtAicpuKernelLaunchExWithArgs` (kernel_type =
 *      `KERNEL_TYPE_AICPU_KFC`) targeting libaicpu_extend_kernels. Our
 *      dispatcher then writes the runtime SO to
 *      `/usr/lib64/aicpu_kernels/0/aicpu_kernels_device/simpler_inner_<fp>_<device_id>.so`
 *      using sched-thread (HwHiAiUser) write permission. The dispatcher SO
 *      itself is never persisted to disk.
 *
 *   2. Init (per-DeviceRunner): JSON-registers the runtime SO via
 *      `rtsBinaryLoadFromFile` (cpuKernelMode=0, kernelSo points at the
 *      preinstall basename), then resolves runtime SO entry points such as
 *      `simpler_aicpu_exec` and `simpler_aicpu_prewarm_callable` to
 *      `rtFuncHandle`s via `rtsFuncGetByName`. JSON is per-process
 *      (`/tmp/simpler_inner_<fp>_<pid>.json`) so concurrent multi-chip /
 *      multi-worker tests don't race on a shared file.
 *
 *   3. LaunchBuiltInOp (per-task): `rtsLaunchCpuKernel` on the cached
 *      `rtFuncHandle`. No per-launch string marshalling, no global op
 *      registry lookups.
 *
 * See common/aicpu_loader/device/aicpu_dispatcher.h for the bootstrap protocol
 * details (extended DeviceArgs with inner_so_bin/inner_so_len,
 * fingerprint-named preinstall files).
 */

#ifndef COMMON_HOST_LOAD_AICPU_OP_H_
#define COMMON_HOST_LOAD_AICPU_OP_H_

#include <cstdint>
#include <string>
#include <unordered_map>

#include "common/kernel_args.h"
#include "runtime/runtime/rts/rts_kernel.h"
#include "runtime/rt.h"

namespace host {

/**
 * @brief AICPU operation configuration for JSON descriptor generation.
 */
struct AicpuOpConfig {
    std::string functionName;
    std::string kernelSo;
    std::string opKernelLib;
    std::string computeCost = "100";
    std::string engine = "DNN_VM_AICPU";
    std::string flagAsync = "False";
    std::string flagPartial = "False";
    std::string userDefined = "False";
    std::string opType;
};

/**
 * @brief Host-side AICPU operation loader.
 *
 * One instance per DeviceRunner; manages bootstrap (dispatcher upload) +
 * JSON registration of the runtime SO + per-task launches via the runtime
 * SO's direct rtFuncHandles.
 */
class LoadAicpuOp {
public:
    LoadAicpuOp() = default;
    ~LoadAicpuOp();

    LoadAicpuOp(const LoadAicpuOp &) = delete;
    LoadAicpuOp &operator=(const LoadAicpuOp &) = delete;
    LoadAicpuOp(LoadAicpuOp &&) = delete;
    LoadAicpuOp &operator=(LoadAicpuOp &&) = delete;

    /**
     * @brief One-shot bootstrap: upload runtime SO to preinstall via dispatcher.
     *
     * @param dispatcher_so_data  Dispatcher SO bytes (caller-owned, must outlive call)
     * @param dispatcher_so_len   Dispatcher SO size
     * @param inner_so_data       Runtime SO bytes (caller-owned, must outlive call)
     * @param inner_so_len        Runtime SO size
     * @param stream              Stream on which to enqueue the bootstrap
     * @param device_id           ACL device ordinal; suffixed onto the preinstall
     *                            basename (simpler_inner_<fp>_<device_id>.so) so
     *                            paired dies sharing the filesystem do not write
     *                            and execute the same on-disk SO.
     * @return 0 on success, error code on failure
     */
    int BootstrapDispatcher(
        const void *dispatcher_so_data, size_t dispatcher_so_len, const void *inner_so_data, size_t inner_so_len,
        rtStream_t stream, int device_id
    );

    /** @brief JSON-register the runtime SO and resolve its entry handles. */
    int Init();

    /** @brief Release binary handle + function handles + temporary JSON. */
    void Finalize();

    /**
     * @brief Launch a runtime SO entry point via rtsLaunchCpuKernel.
     *
     * @param stream       RTS stream
     * @param k_args       Kernel arguments
     * @param aicpu_num    Number of AICPU threads
     * @param func_name    Lookup key in func_handles_ (KernelNames::RunName)
     * @return 0 on success, error code on failure
     */
    int LaunchBuiltInOp(rtStream_t stream, KernelArgs *k_args, int aicpu_num, const std::string &func_name);

private:
    void *binary_handle_ = nullptr;
    std::unordered_map<std::string, rtFuncHandle> func_handles_;
    std::string json_file_path_;
    uint64_t inner_fp_ = 0;
    int device_id_ = 0;
    std::string inner_so_basename_;

    bool GenerateAicpuOpJson(const std::string &json_path, const std::string &kernel_so);
    int AicpuKernelLaunch(rtFuncHandle func_handle, rtStream_t stream, KernelArgs *k_args, int aicpu_num);
};

// Runtime SO's actual exported symbol name. Looked up via the runtime SO's
// own JSON registration (no dispatcher hop at runtime).
namespace KernelNames {
constexpr const char *RunName = "simpler_aicpu_exec";  // multi-threaded exec
constexpr const char *PrewarmName = "simpler_aicpu_prewarm_callable";
}  // namespace KernelNames

}  // namespace host

#endif  // COMMON_HOST_LOAD_AICPU_OP_H_
