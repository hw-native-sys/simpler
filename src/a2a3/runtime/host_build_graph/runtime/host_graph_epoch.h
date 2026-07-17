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

#include <atomic>
#include <cstdint>
#include <type_traits>

inline constexpr int32_t PTO2_HOST_GRAPH_EPOCH_SLOT_COUNT = 2;

struct PTO2HostGraphEpochRange {
    int32_t task_begin{0};
    int32_t task_end{0};
    int32_t dep_begin{0};
    int32_t dep_end{0};
    int32_t inline_completed{0};
    int32_t final_epoch{0};
};

struct alignas(64) PTO2HostGraphEpochSlot {
    std::atomic<uint64_t> owner_epoch;
    PTO2HostGraphEpochRange range;

    void init() {
        owner_epoch.store(0, std::memory_order_relaxed);
        range = PTO2HostGraphEpochRange{};
    }
};

struct alignas(64) PTO2HostGraphEpochControl {
    std::atomic<uint64_t> host_publish_epoch;
    std::atomic<uint64_t> device_release_epoch;
    std::atomic<uint64_t> device_exec_done_epoch;
    std::atomic<uint64_t> device_buffer_free_epoch;
    std::atomic<uint64_t> failed_epoch;
    PTO2HostGraphEpochSlot slots[PTO2_HOST_GRAPH_EPOCH_SLOT_COUNT];

    void init() {
        host_publish_epoch.store(0, std::memory_order_relaxed);
        device_release_epoch.store(0, std::memory_order_relaxed);
        device_exec_done_epoch.store(0, std::memory_order_relaxed);
        device_buffer_free_epoch.store(0, std::memory_order_relaxed);
        failed_epoch.store(0, std::memory_order_relaxed);
        for (int32_t slot = 0; slot < PTO2_HOST_GRAPH_EPOCH_SLOT_COUNT; ++slot) {
            slots[slot].init();
        }
    }
};

static_assert(std::is_standard_layout_v<PTO2HostGraphEpochRange>);
static_assert(std::is_standard_layout_v<PTO2HostGraphEpochSlot>);
static_assert(std::is_standard_layout_v<PTO2HostGraphEpochControl>);
static_assert(sizeof(PTO2HostGraphEpochSlot) % 64 == 0);
static_assert(sizeof(PTO2HostGraphEpochControl) % 64 == 0);
