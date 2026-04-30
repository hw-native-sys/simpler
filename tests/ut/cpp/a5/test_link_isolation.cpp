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
/**
 * Behavior tests for runtime components built without pto_orchestrator.cpp.
 *
 * The CMake target for this file deliberately omits the orchestrator source.
 * Passing build and runtime assertions here verifies that scheduler, ring
 * buffer, shared memory, and TensorMap behavior does not require an
 * orchestrator link dependency.
 */

#include <gtest/gtest.h>

#include <atomic>
#include <cstring>
#include <vector>

#include "pto_orchestration_api.h"
#include "pto_ring_buffer.h"
#include "pto_runtime2_types.h"
#include "pto_shared_memory.h"
#include "pto_tensormap.h"
#include "scheduler/pto_scheduler.h"

namespace {

constexpr uint64_t kHeapSize = 64 * 1024;
constexpr int32_t kWindowSize = 64;

struct TestLookupResult {
    struct Entry {
        PTO2TensorMapEntry *entry;
        OverlapStatus overlap_status;
    };
    std::vector<Entry> entries;
    int count = 0;
};

void run_lookup(PTO2TensorMap &tmap, const Tensor &tensor, TestLookupResult &out) {
    tmap.lookup(tensor, [&](PTO2TensorMapEntry &entry, OverlapStatus status) -> bool {
        out.entries.push_back({&entry, status});
        out.count++;
        return true;
    });
}

Tensor make_tensor(uint64_t addr, uint32_t shape0 = 100, int32_t version = 0) {
    uint32_t shapes[RUNTIME_MAX_TENSOR_DIMS] = {shape0};
    return make_tensor_external(reinterpret_cast<void *>(addr), shapes, 1, DataType::FLOAT32, false, version);
}

struct DepPoolFixture {
    PTO2DepListEntry entries[512];
    std::atomic<int32_t> error_code{PTO2_ERROR_NONE};
    PTO2DepListPool pool{};

    void Init() {
        std::memset(entries, 0, sizeof(entries));
        error_code.store(PTO2_ERROR_NONE, std::memory_order_relaxed);
        pool.init(entries, 512, &error_code);
    }

    void AllocN(int count) {
        for (int i = 0; i < count; i++) {
            ASSERT_NE(pool.alloc(), nullptr);
        }
    }
};

}  // namespace

TEST(LinkIsolationDepPool, ReclaimBelowIntervalKeepsAllocatedEntries) {
    DepPoolFixture fixture;
    fixture.Init();
    fixture.AllocN(100);
    int32_t used_before = fixture.pool.used();

    PTO2SharedMemoryRingHeader ring_header{};
    fixture.pool.reclaim(ring_header, PTO2_DEP_POOL_CLEANUP_INTERVAL - 1);

    EXPECT_EQ(fixture.pool.used(), used_before);
    EXPECT_EQ(fixture.error_code.load(std::memory_order_acquire), PTO2_ERROR_NONE);
}

TEST(LinkIsolationDepPool, ReclaimAtIntervalUsesConsumedTaskMark) {
    DepPoolFixture fixture;
    fixture.Init();
    fixture.AllocN(100);

    std::vector<PTO2TaskSlotState> slots(kWindowSize);
    PTO2SharedMemoryRingHeader ring_header{};
    ring_header.slot_states = slots.data();
    ring_header.task_window_size = kWindowSize;
    ring_header.task_window_mask = kWindowSize - 1;

    int32_t last_alive = PTO2_DEP_POOL_CLEANUP_INTERVAL;
    int32_t mark_slot = (last_alive - 1) & ring_header.task_window_mask;
    slots[mark_slot].dep_pool_mark = 50;

    fixture.pool.reclaim(ring_header, last_alive);

    EXPECT_EQ(fixture.pool.used(), 51);
    EXPECT_EQ(fixture.error_code.load(std::memory_order_acquire), PTO2_ERROR_NONE);
}

TEST(LinkIsolationScheduler, ReleaseFaninPushesReadyTask) {
    PTO2SharedMemoryHandle *sm = PTO2SharedMemoryHandle::create(kWindowSize, kHeapSize);
    ASSERT_NE(sm, nullptr);

    PTO2SchedulerState sched{};
    ASSERT_TRUE(sched.init(sm->header));

    alignas(64) PTO2TaskSlotState slot{};
    slot.fanin_count = 1;
    slot.fanin_refcount.store(0, std::memory_order_relaxed);
    slot.task_state.store(PTO2_TASK_PENDING, std::memory_order_relaxed);
    slot.active_mask = ActiveMask(PTO2_SUBTASK_MASK_AIV0);

    EXPECT_TRUE(sched.release_fanin_and_check_ready(slot, nullptr));

    PTO2ResourceShape shape = slot.active_mask.to_shape();
    EXPECT_EQ(sched.ready_queues[static_cast<int>(shape)].pop(), &slot);

    sched.destroy();
    sm->destroy();
}

TEST(LinkIsolationScheduler, CompletedTaskCanBecomeConsumed) {
    PTO2SharedMemoryHandle *sm = PTO2SharedMemoryHandle::create(kWindowSize, kHeapSize);
    ASSERT_NE(sm, nullptr);

    PTO2SchedulerState sched{};
    ASSERT_TRUE(sched.init(sm->header));

    PTO2TaskDescriptor desc{};
    PTO2TaskSlotState &slot = sm->header->rings[0].get_slot_state_by_slot(0);
    slot.task = &desc;
    slot.ring_id = 0;
    slot.fanout_count = 1;
    slot.fanout_refcount.store(1, std::memory_order_relaxed);
    slot.task_state.store(PTO2_TASK_COMPLETED, std::memory_order_relaxed);
    sm->header->rings[0].fc.current_task_index.store(1, std::memory_order_relaxed);

    sched.check_and_handle_consumed(slot);

    EXPECT_EQ(slot.task_state.load(std::memory_order_acquire), PTO2_TASK_CONSUMED);

    sched.destroy();
    sm->destroy();
}

TEST(LinkIsolationReadyQueue, PushPopBatchWithoutOrchestrator) {
    PTO2ReadyQueue queue{};
    ASSERT_TRUE(ready_queue_init(&queue, 16));

    PTO2TaskSlotState items[4]{};
    PTO2TaskSlotState *in[4] = {&items[0], &items[1], &items[2], &items[3]};
    queue.push_batch(in, 4);

    PTO2TaskSlotState *out[4]{};
    EXPECT_EQ(queue.pop_batch(out, 4), 4);
    for (int i = 0; i < 4; i++) {
        EXPECT_EQ(out[i], &items[i]);
    }
    EXPECT_EQ(queue.pop(), nullptr);

    ready_queue_destroy(&queue);
}

TEST(LinkIsolationTensorMap, InsertLookupAndValidityWithoutOrchestrator) {
    int32_t window_sizes[PTO2_MAX_RING_DEPTH] = {kWindowSize, kWindowSize, kWindowSize, kWindowSize};
    PTO2TensorMap tmap{};
    ASSERT_TRUE(tmap.init(256, 1024, window_sizes));

    Tensor tensor = make_tensor(0x3000);
    for (int i = 0; i < kWindowSize; i++) {
        tmap.insert(tensor, PTO2TaskId::make(0, i));
    }

    tmap.sync_validity(0, kWindowSize / 2);

    TestLookupResult result;
    run_lookup(tmap, tensor, result);
    EXPECT_EQ(result.count, kWindowSize / 2);
    for (const auto &entry : result.entries) {
        EXPECT_GE(entry.entry->producer_task_id.local(), static_cast<uint32_t>(kWindowSize / 2));
        EXPECT_EQ(entry.overlap_status, OverlapStatus::COVERED);
    }

    tmap.destroy();
}
