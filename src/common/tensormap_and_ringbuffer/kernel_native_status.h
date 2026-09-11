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
#pragma once

#include <cstdint>

#include "task_interface/kernel_dispatch_args.h"

namespace simpler::tmr {

// CANN's aicpu_common/context/utils/status.h fixes OK=0 and INNER_ERROR=2
// for direct CPU entries. Keep the SDK dependency out of simulation builds;
// the onboard protocol probe checks these values against the installed SDK.
constexpr int kAicpuKernelSuccess = 0;
constexpr int kAicpuKernelInnerError = 2;

// Only the native boundary translates status. Detailed dispatch/runtime and
// cleanup diagnostics remain in their own domains; arbitrary enum values
// must not become CANN statuses (e.g. dispatch 6 produced replay EOS).
inline int to_aicpu_native_status(int32_t status) noexcept {
    return status == 0 ? kAicpuKernelSuccess : kAicpuKernelInnerError;
}

inline int classify_kernel_dispatch_status(int32_t status, int32_t cleanup_status) noexcept {
    if (cleanup_status != 0) return static_cast<int>(KernelDispatchStatus::CleanupFailed);
    if (status == 0) return static_cast<int>(KernelDispatchStatus::Success);
    if (status > 0 && status <= static_cast<int>(KernelDispatchStatus::InvalidBinding)) return status;
    return static_cast<int>(KernelDispatchStatus::ExecutionFailed);
}

}  // namespace simpler::tmr
