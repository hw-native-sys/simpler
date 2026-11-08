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
#include "common.h"

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    const auto *destination = data<float>(args, 0);
    auto *results = data<uint32_t>(args, 1);
    auto *control = reinterpret_cast<volatile __gm__ uint32_t *>(args[2]);
    uint32_t errors = 0;
    for (uint32_t i = 0; i < 64; ++i)
        if (destination[i] != static_cast<float>(100 + i)) ++errors;
    control[16] = 1;
    results[0] = errors;
    ccu_detail::Sync(results, 64);
    ccu_detail::Sync(const_cast<__gm__ uint32_t *>(control + 16), 64);
}
