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

#include <cstddef>
#include <cstdint>
#include <type_traits>

struct GlobalContext {
    // Meaningful for cooperative AIV kernels: left = 0, right = 1.
    int32_t sub_block_id;
    uint32_t async_present;
    uint64_t async_image_address;
    uint64_t async_image_bytes;
    uint64_t async_run_epoch;
    uint64_t async_group_offset;
    uint32_t async_group_index;
    uint32_t async_output_capture_refused;
};

static_assert(sizeof(GlobalContext) == 48);
static_assert(offsetof(GlobalContext, async_present) == 4 && offsetof(GlobalContext, async_group_index) == 40);
static_assert(std::is_standard_layout_v<GlobalContext> && std::is_trivially_copyable_v<GlobalContext>);

struct LocalContext {
    int32_t block_idx;
    // Logical width, including multi-wave execution, independent of physical core count.
    int32_t block_num;
};

static_assert(sizeof(LocalContext) == 8);
static_assert(std::is_standard_layout_v<LocalContext> && std::is_trivially_copyable_v<LocalContext>);

namespace simpler::percounter {

inline constexpr uint32_t PERCOUNTER_CONTEXT_SSBUF_OFFSET = 2816;
inline constexpr uint32_t PERCOUNTER_CONTEXT_SSBUF_STRIDE = 64;

struct PercounterInvocationContext {
    LocalContext local;
    GlobalContext global;
    uint32_t async_touched;  // Accessed by the kernel and scheduler on the same lane.
};
static_assert(sizeof(PercounterInvocationContext) == PERCOUNTER_CONTEXT_SSBUF_STRIDE);
static_assert(
    std::is_standard_layout_v<PercounterInvocationContext> && std::is_trivially_copyable_v<PercounterInvocationContext>
);

struct PercounterInvocationContextSlot {
    PercounterInvocationContext value;
};
static_assert(sizeof(PercounterInvocationContextSlot) == PERCOUNTER_CONTEXT_SSBUF_STRIDE);
static_assert(
    std::is_standard_layout_v<PercounterInvocationContextSlot> &&
    std::is_trivially_copyable_v<PercounterInvocationContextSlot>
);

}  // namespace simpler::percounter
