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
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "runtime_core.h"
#include "shared_memory.h"

TEST(NativeReplayReconstruction, RewiresDestroyedRuntimeAndClearsInvocationStateWithoutClearingUserHeap) {
    uint64_t windows[CHIP_MAX_RING_DEPTH];
    uint64_t heaps[CHIP_MAX_RING_DEPTH];
    int32_t dependencies[CHIP_MAX_RING_DEPTH];
    for (int ring = 0; ring < CHIP_MAX_RING_DEPTH; ++ring) {
        windows[ring] = 8;
        heaps[ring] = 4096;
        dependencies[ring] = 4;
    }
    DeviceArena arena;
    const auto layout = runtime_reserve_layout(arena, windows, heaps, dependencies);
    ASSERT_NE(arena.commit(DeviceArena::kDefaultBaseAlign), nullptr);
    DeviceArena shared;
    const auto shared_bytes = SharedMemoryHandle::calculate_size_per_ring(windows);
    const auto offset = shared.reserve(shared_bytes, CHIP_ALIGN_SIZE);
    ASSERT_NE(shared.commit(), nullptr);
    void *const shared_base = shared.region_ptr(offset);
    std::vector<uint8_t> user_heap(CHIP_MAX_RING_DEPTH * 4096, 0x5a);
    auto *runtime =
        runtime_init_data_from_layout(arena, layout, MODE_EXECUTE, shared_base, shared_bytes, user_heap.data(), heaps);
    ASSERT_NE(runtime, nullptr);

    for (int round = 0; round < 3; ++round) {
        runtime_wire_arena_pointers(arena, layout, runtime);
        ASSERT_NE(runtime->sm_handle, nullptr);
        ASSERT_TRUE(runtime->sm_handle->init_per_ring(shared_base, shared_bytes, windows, heaps));
        ASSERT_TRUE(runtime_reset_for_reuse(arena, layout, runtime));
        EXPECT_EQ(runtime->pending_scope_mode, ScopeMode::AUTO);
        EXPECT_EQ(runtime->total_cycles, 0u);
        EXPECT_EQ(runtime->sm_handle->header->orch_error_code.load(), 0);
        EXPECT_EQ(runtime->sm_handle->header->sched_error_code.load(), 0);
        for (int ring = 0; ring < CHIP_MAX_RING_DEPTH; ++ring) {
            EXPECT_EQ(runtime->scheduler.ring_sched_states[ring].last_task_alive, 0);
            EXPECT_EQ(runtime->scheduler.ring_sched_states[ring].advance_lock.load(), 0);
            EXPECT_EQ(runtime->orchestrator.rings[ring].task_allocator.window_size(), 8);
        }
        EXPECT_TRUE(std::all_of(user_heap.begin(), user_heap.end(), [](uint8_t value) {
            return value == 0x5a;
        }));

        runtime->pending_scope_mode = ScopeMode::MANUAL;
        runtime->total_cycles = 99;
        runtime->sm_handle->header->orch_error_code.store(9);
        runtime->sm_handle->header->sched_error_code.store(100);
        for (int ring = 0; ring < CHIP_MAX_RING_DEPTH; ++ring) {
            runtime->scheduler.ring_sched_states[ring].last_task_alive = 7;
            runtime->scheduler.ring_sched_states[ring].advance_lock.store(1);
        }
        runtime_destroy(runtime, arena);
        EXPECT_EQ(runtime->sm_handle, nullptr);
        EXPECT_EQ(runtime->aicore_mailbox, nullptr);
    }
}
