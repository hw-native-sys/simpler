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

#include <cstdint>

#include "inner_kernel.h"

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    (void)args;
    // CI's resident scheduler budget is 2 s and its op-execute budget is 3 s.
    constexpr uint64_t duration = PLATFORM_PROF_SYS_CNT_FREQ * 5 / 2;
    const uint64_t start = get_sys_cnt_aicore();
    while (get_sys_cnt_aicore() - start < duration) {}
}
