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
#include <pto/pto-inst.hpp>
#include "intrinsic.h"
#include "tensor.h"
#include "layout.h"

namespace sdma_reuse {
inline __aicore__ uint32_t elements(uint32_t index) { return index == 0 ? kLarge : 8; }
inline __aicore__ uint32_t offset(uint32_t index) {
    return index == 0 ? kGuard : kLarge + 2 * kGuard + (index - 1) * kStride + kGuard;
}
template <typename T>
inline __aicore__ __gm__ T *data(int64_t arg) {
    auto *tensor = reinterpret_cast<__gm__ Tensor *>(arg);
    return reinterpret_cast<__gm__ T *>(tensor->buffer.addr) + tensor->start_offset;
}
inline __aicore__ void flush(__gm__ void *address) {
    dcci(address, cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
    dsb((mem_dsb_t)0);
}
inline __aicore__ uint64_t read(__gm__ uint64_t *address) {
    dcci(address, cache_line_t::SINGLE_CACHE_LINE);
    dsb((mem_dsb_t)0);
    return *reinterpret_cast<volatile __gm__ uint64_t *>(address);
}
}  // namespace sdma_reuse
