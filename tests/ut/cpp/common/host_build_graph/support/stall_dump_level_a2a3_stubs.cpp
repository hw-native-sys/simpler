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

#include <cstddef>
#include <cstdint>

#include "aicpu/platform_regs.h"

// a2a3 retires a whole AICore group through one entry, against a deadline; a5 has no
// equivalent, which is why this stub is per-target rather than beside the test. The
// stall-dump tests never reach it — only the link does.
int32_t __attribute__((weak))
platform_retire_aicore_group(const AicoreExitTarget *, size_t count, uint64_t, bool *released) {
    if (released != nullptr) {
        for (size_t i = 0; i < count; i++)
            released[i] = true;
    }
    return 0;
}

uint64_t __attribute__((weak)) platform_aicore_exit_deadline() { return 0; }
