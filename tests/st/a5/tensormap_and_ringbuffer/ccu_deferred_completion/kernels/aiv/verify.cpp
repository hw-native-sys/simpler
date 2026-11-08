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
    auto *source = data<float>(args, 0);
    auto *destination = data<float>(args, 1);
    auto *scratch = data<float>(args, 2);
    auto *result = data<int32_t>(args, 3);
    const auto *domain = reinterpret_cast<__gm__ CommContext *>(args[4]);
    const uint64_t round = args[5];
    const uint32_t count = round % 2 ? 1024 : 64;
    const uint32_t offset = 16 + (round % 8) * 16;
    const uint32_t root = (round / 2) % 2;
    ccu_detail::Sync(destination, 8192);
    ccu_detail::Sync(source, 8192);
    ccu_detail::Sync(scratch, 8192);
    ccu_detail::Sync(result, (round + 1) * 16);
    int32_t status = result[round * 4] == -2 ? 0 : 1;
    for (uint32_t i = 0; i < 2048 && status == 0; ++i) {
        const float expected = i >= offset && i < offset + count ? value(round, root, i) : -1.0f;
        if (source[i] != value(round, domain->rankId, i)) status = 2;
        else if (destination[i] != expected) status = 3;
        else if (scratch[i] != -7.0f) status = 4;
    }
    result[round * 4] = status;
    result[round * 4 + 1] = root;
    result[round * 4 + 2] = count;
    result[round * 4 + 3] = offset;
    ccu_detail::Sync(result, (round + 1) * 16);
}
