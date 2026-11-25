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
#include <pto/pto-inst.hpp>
#include "intrinsic.h"
#include "tensor.h"

namespace diamond {
inline __aicore__ __gm__ uint64_t *words(int64_t arg) {
    const auto *tensor = reinterpret_cast<const __gm__ Tensor *>(arg);
    return reinterpret_cast<__gm__ uint64_t *>(tensor->buffer.addr) + tensor->start_offset;
}

inline __aicore__ void fork(__gm__ int64_t *args) {
    auto *x = words(args[0]);
    auto *a = words(args[1]);
    st_dev(ld_dev(x, 0) + 1, a, 0);
    st_dev(ld_dev(x + 1, 0), a + 1, 0);
    dsb(DSB_DDR);
}

// The tagged checksum gives the optional longer branch real, verifiable Scalar work.
template <bool IsB>
inline __aicore__ void branch(__gm__ int64_t *args) {
    auto *a = words(args[0]);
    auto *out = words(args[1]);
    const uint64_t value = ld_dev(a, 0);
    const uint64_t result = IsB ? value + 3 : 2 * value + 1;
    const uint64_t steps = ((value & 1) == static_cast<uint64_t>(IsB)) ? ld_dev(a + 1, 0) : 0;
    uint64_t checksum = result;
    for (uint64_t i = 0; i < steps; ++i) {
        checksum ^= checksum << 13;
        checksum ^= checksum >> 7;
        checksum ^= checksum << 17;
    }
    st_dev(result, out, 0);
    st_dev(checksum, out + 1, 0);
    dsb(DSB_DDR);
}

inline __aicore__ void join(__gm__ int64_t *args) {
    auto *b = words(args[0]);
    auto *c = words(args[1]);
    auto *d = words(args[2]);
    st_dev(ld_dev(b, 0) + 2 * ld_dev(c, 0), d, 0);
    st_dev(ld_dev(b + 1, 0) ^ ld_dev(c + 1, 0), d + 1, 0);
    dsb(DSB_DDR);
}
}  // namespace diamond
