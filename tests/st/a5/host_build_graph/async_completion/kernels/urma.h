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
#include "device.h"
#include <pto/comm/async/urma/urma_types.hpp>
namespace async_case {
inline __aicore__ bool urma_ready(__gm__ uint64_t *meta) {
    auto *cq = reinterpret_cast<__gm__ pto::comm::urma::UrmaCqCtx *>(observe(meta + 128));
    if (cq == nullptr) return true;
    const uint32_t target = static_cast<uint32_t>(observe(meta + 144));
    const uint32_t sequence = target - 1;
    auto *word = reinterpret_cast<__gm__ uint32_t *>(cq->bufAddr + 64 * (sequence & (cq->depth - 1)));
    const uint32_t value = ld_dev(word, 0);
    return ((value >> 2) & 1u) != ((sequence / cq->depth) & 1u);
}
}  // namespace async_case
