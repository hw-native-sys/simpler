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
#include "aicore_completion_mailbox_types.h"
#include "host_build_graph/runtime_types.h"

// Storage declarations required by common Host arena initialization.
// WSE rejects deferred task completion and has no async completion scheduler.
inline constexpr int32_t MAX_ASYNC_WAITS = 64;
struct CompletionCondition {
    AsyncEngine engine{ASYNC_ENGINE_SDMA};
    int32_t completion_type{COMPLETION_TYPE_COUNTER};
    bool satisfied{false};
    bool retired{false};
    volatile uint32_t *counter_addr{nullptr};
    uint64_t addr{0};
    uint64_t backend_cookie{0};
    uint32_t expected_value{0};
};

struct AsyncWaitEntry {
    ChipTaskSlotState *slot_state{nullptr};
    TaskId task_token{TaskId::invalid()};
    CompletionCondition conditions[MAX_COMPLETIONS_PER_TASK];
    int32_t condition_count{0};
    int32_t waiting_completion_count{0};
    bool normal_done{false};
};

struct AsyncWaitList {
    std::atomic<int32_t> busy{0};
    AsyncWaitEntry entries[MAX_ASYNC_WAITS];
    int32_t count{0};
    std::atomic<uint64_t> mpsc_skipped_count{0};

    // Called by common Host arena initialization.
    void reset_for_reuse() {
        busy.store(0, std::memory_order_relaxed);
        count = 0;
        mpsc_skipped_count.store(0, std::memory_order_relaxed);
    }
};
