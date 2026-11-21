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
#include <pto/pto-inst.hpp>
#include "intrinsic.h"
#include "tensor.h"

static inline __aicore__ __gm__ uint64_t *words(int64_t arg) {
    const auto *tensor = reinterpret_cast<const __gm__ Tensor *>(arg);
    return reinterpret_cast<__gm__ uint64_t *>(tensor->buffer.addr) + tensor->start_offset;
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    const int32_t width = static_cast<int32_t>(args[2]);
#if defined(__DAV_CUBE__)
    constexpr int32_t lane = 0;
#else
    const int32_t lane = width == 1 ? 0 : 1 + get_sub_block_id(args);
#endif
    const int32_t member = get_block_idx(args) * width + lane;
    const int32_t source = member % static_cast<int32_t>(args[3]);
    // Each member writes a private 128-byte line; the host also validates its padding.
    const uint64_t input = ld_dev(words(args[0]) + source * 16, 0);
    const uint64_t result = input + static_cast<uint64_t>(args[4]) + member;
    st_dev(result, words(args[1]) + member * 16, 0);
    dsb(DSB_DDR);
}
