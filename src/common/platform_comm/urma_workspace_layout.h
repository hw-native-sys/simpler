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

#include <pto/comm/async/urma/urma_types.hpp>

#if defined(__CCE_AICORE__)
#define URMA_LAYOUT_DEVICE [aicore]
#else
#define URMA_LAYOUT_DEVICE
#endif

namespace simpler::urma {

// Simpler-owned sender headers follow the ISA workspace in the same aligned GM allocation.
// The one-past-last sender offset is the number of bytes required for the whole allocation.
URMA_LAYOUT_DEVICE inline constexpr uint64_t sender_view_offset(uint64_t workspace_size, uint32_t sender_id) {
    constexpr uint64_t mask = pto::comm::urma::kCacheLineSize - 1;
    constexpr uint64_t stride = (sizeof(pto::comm::urma::UrmaInfo) + mask) & ~mask;
    const uint64_t sender_offset = stride * sender_id;
    if (workspace_size > UINT64_MAX - mask - sender_offset) return UINT64_MAX;
    return ((workspace_size + mask) & ~mask) + sender_offset;
}

}  // namespace simpler::urma

#undef URMA_LAYOUT_DEVICE
