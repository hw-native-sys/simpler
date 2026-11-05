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
// CPU-only stand-in for the CANN context API used by the dispatcher.
#pragma once
#include <sys/types.h>
#include <cstdint>

namespace aicpu {
struct aicpuContext_t {
    uint32_t deviceId;
    uint32_t tsId;
    pid_t hostPid;
    uint32_t vfId;
};
enum status_t { AICPU_ERROR_NONE = 0, AICPU_ERROR_FAILED = 1 };
status_t __attribute__((weak)) aicpuGetContext(aicpuContext_t *ctx);
uint32_t __attribute__((weak)) GetUniqueVfId();
}  // namespace aicpu
