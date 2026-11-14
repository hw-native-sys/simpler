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

#include <array>
#include <cstdint>

// Immutable Host metadata for one successfully initialized communication allocation.
// The bootstrap token is shared by its ranks. Allocation IDs cannot be reused
// inside that bootstrap namespace, so generation 1 denotes its sole lifetime.
struct HostNotificationContext {
    uint64_t bootstrap_token{0};
    uint64_t allocation_id{0};
    uint64_t generation{1};
    uint64_t context_address{0};
    uint64_t window_bytes{0};
    uint32_t local_rank{0};
    uint32_t rank_count{0};
    std::array<uint32_t, 64> rank_ids{};
    std::array<uint64_t, 64> windows_in{};
    std::array<uint64_t, 64> windows_out{};
};
