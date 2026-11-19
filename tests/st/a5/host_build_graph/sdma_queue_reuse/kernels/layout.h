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

namespace sdma_reuse {
constexpr uint32_t kMaxLanes = 128;
constexpr uint32_t kPostsPerLane = 33;
constexpr uint32_t kMaxPosts = kMaxLanes * kPostsPerLane;
constexpr uint32_t kWords = 16;
constexpr uint32_t kLarge = 1024 * 1024;
constexpr uint32_t kGuard = 16;
constexpr uint32_t kStride = 48;
constexpr uint32_t kAllocated = kLarge + 2 * kGuard + (kMaxPosts - 1) * kStride;
}  // namespace sdma_reuse
