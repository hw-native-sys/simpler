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

#include <acl/acl_rt.h>

#include "host/kernel_launch_binder.h"

namespace simpler::kernel_launch {

struct KernelClearRegion {
    void *address{nullptr};
    size_t bytes{0};
};

// Prepared, context-pinned platform arguments. This is never decoded from an
// invocation packet. Owners retain handles, buffers and device execution state
// through every captured graph and external quiescence, not just this call.
struct KernelNativeInvocation {
    aclrtFuncHandle aicore{nullptr};
    aclrtFuncHandle aicpu{nullptr};
    uint32_t aicore_blocks{0};
    uint32_t aicpu_blocks{0};
    const void *aicore_args{nullptr};
    size_t aicore_args_bytes{0};
    void *aicpu_args{nullptr};
    size_t aicpu_args_bytes{0};
    aclrtPlaceHolderInfo *placeholders{nullptr};
    size_t placeholder_count{0};
    aclrtLaunchKernelCfg *aicpu_config{nullptr};
    const KernelClearRegion *clear_regions{nullptr};
    size_t clear_region_count{0};
    KernelClearRegion cancel;
};

KernelLaunchResult launch_bound_kernel_native(
    const KernelInvocationBinding &binding, const KernelNativeInvocation &native, void *caller_stream,
    const KernelLaunchGateOps &owner
);

}  // namespace simpler::kernel_launch
