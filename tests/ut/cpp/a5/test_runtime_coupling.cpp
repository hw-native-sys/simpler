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
 * Runtime behavior tests for the combined TMR components.
 *
 * These tests keep the assertions on observable component behavior: lifecycle,
 * scheduler state transitions, TensorMap lookup validity, and shared-memory
 * coordination. Structural coupling checks belong in design review rather than
 * in unit tests.
 */

#include <gtest/gtest.h>

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "pto_orchestration_api.h"
#include "pto_orchestrator.h"
#include "pto_ring_buffer.h"
#include "pto_runtime2_types.h"
#include "pto_shared_memory.h"
#include "pto_tensormap.h"
#include "scheduler/pto_scheduler.h"

namespace {

constexpr uint64_t kHeapSize = 64 * 1024;
constexpr int32_t kWindowSize = 64;
constexpr int32_t kDepPoolSize = 256;

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

struct TMRSystem {
    PTO2SharedMemoryHandle *sm = nullptr;
    PTO2SchedulerState sched{};
    PTO2OrchestratorState orch{};
    uint8_t *gm_heap = nullptr;
    bool sched_ok = false;
    bool orch_ok = false;

    bool Init(uint64_t heap_size = kHeapSize, int32_t window_size = kWindowSize) {
        sm = PTO2SharedMemoryHandle::create(window_size, heap_size);
        if (sm == nullptr) return false;

        gm_heap = static_cast<uint8_t *>(std::calloc(PTO2_MAX_RING_DEPTH, heap_size));
        if (gm_heap == nullptr) return false;

        sched_ok = sched.init(sm->header, kDepPoolSize);
        if (!sched_ok) return false;

        orch_ok = orch.init(sm->header, gm_heap, heap_size, kDepPoolSize);
        if (!orch_ok) return false;

        orch.set_scheduler(&sched);
        return true;
    }

    void Destroy() {
        if (orch_ok) {
            orch.destroy();
            orch_ok = false;
        }
        if (sched_ok) {
            sched.destroy();
            sched_ok = false;
        }
        if (gm_heap != nullptr) {
            std::free(gm_heap);
            gm_heap = nullptr;
        }
        if (sm != nullptr) {
            sm->destroy();
            sm = nullptr;
        }
    }
};

}  // namespace

TEST(RuntimeLifecycleBehavior, InitDestroyCanRepeat) {
    for (int cycle = 0; cycle < 3; cycle++) {
        TMRSystem sys;
        ASSERT_TRUE(sys.Init()) << "cycle=" << cycle;
        EXPECT_EQ(sys.sm->header->orch_error_code.load(std::memory_order_acquire), PTO2_ERROR_NONE);
        EXPECT_EQ(sys.sm->header->sched_error_code.load(std::memory_order_acquire), PTO2_ERROR_NONE);
        sys.Destroy();
    }
}

TEST(RuntimeLifecycleBehavior, OrchestratorScopeWithoutSchedulerLeavesNoFatalCode) {
    PTO2SharedMemoryHandle *sm = PTO2SharedMemoryHandle::create(kWindowSize, kHeapSize);
    ASSERT_NE(sm, nullptr);
    uint8_t *heap = static_cast<uint8_t *>(std::calloc(PTO2_MAX_RING_DEPTH, kHeapSize));
    ASSERT_NE(heap, nullptr);

    PTO2OrchestratorState orch{};
    ASSERT_TRUE(orch.init(sm->header, heap, kHeapSize, kDepPoolSize));

    orch.begin_scope();
    orch.end_scope();

    EXPECT_EQ(sm->header->orch_error_code.load(std::memory_order_acquire), PTO2_ERROR_NONE);
    EXPECT_FALSE(orch.fatal);

    orch.destroy();
    std::free(heap);
    sm->destroy();
}

TEST(RuntimeSchedulerBehavior, CompletedSlotWithSatisfiedFanoutBecomesConsumed) {
    TMRSystem sys;
    ASSERT_TRUE(sys.Init());

    PTO2TaskDescriptor desc{};
    PTO2TaskSlotState &slot = sys.sm->header->rings[0].get_slot_state_by_slot(0);
    slot.task = &desc;
    slot.ring_id = 0;
    slot.fanout_count = 1;
    slot.fanout_refcount.store(1, std::memory_order_relaxed);
    slot.task_state.store(PTO2_TASK_COMPLETED, std::memory_order_relaxed);
    sys.sm->header->rings[0].fc.current_task_index.store(1, std::memory_order_relaxed);

    sys.sched.check_and_handle_consumed(slot);

    EXPECT_EQ(slot.task_state.load(std::memory_order_acquire), PTO2_TASK_CONSUMED);

    sys.Destroy();
}

TEST(RuntimeSchedulerBehavior, RingPointerStopsAtFirstUnconsumedTask) {
    TMRSystem sys;
    ASSERT_TRUE(sys.Init());

    auto &ring_state = sys.sched.ring_sched_states[0];
    PTO2TaskDescriptor descs[3]{};

    PTO2TaskSlotState &slot0 = sys.sm->header->rings[0].get_slot_state_by_slot(0);
    slot0.task = &descs[0];
    slot0.task_state.store(PTO2_TASK_CONSUMED, std::memory_order_relaxed);

    PTO2TaskSlotState &slot1 = sys.sm->header->rings[0].get_slot_state_by_slot(1);
    slot1.task = &descs[1];
    slot1.task_state.store(PTO2_TASK_COMPLETED, std::memory_order_relaxed);

    PTO2TaskSlotState &slot2 = sys.sm->header->rings[0].get_slot_state_by_slot(2);
    slot2.task = &descs[2];
    slot2.task_state.store(PTO2_TASK_CONSUMED, std::memory_order_relaxed);

    sys.sm->header->rings[0].fc.current_task_index.store(3, std::memory_order_relaxed);

    ring_state.advance_ring_pointers();

    EXPECT_EQ(ring_state.last_task_alive, 1);
    EXPECT_EQ(sys.sm->header->rings[0].fc.last_task_alive.load(std::memory_order_acquire), static_cast<int32_t>(1));

    sys.Destroy();
}

TEST(RuntimeSchedulerBehavior, ReadyQueuesAcceptEveryResourceShape) {
    TMRSystem sys;
    ASSERT_TRUE(sys.Init());

    for (int shape = 0; shape < PTO2_NUM_RESOURCE_SHAPES; shape++) {
        PTO2TaskSlotState slot{};
        EXPECT_TRUE(sys.sched.ready_queues[shape].push(&slot)) << "shape=" << shape;
        EXPECT_EQ(sys.sched.ready_queues[shape].pop(), &slot) << "shape=" << shape;
    }

    sys.Destroy();
}

TEST(RuntimeTensorMapBehavior, StandaloneInsertLookupDoesNotNeedOrchestratorPointer) {
    int32_t window_sizes[PTO2_MAX_RING_DEPTH] = {16, 16, 16, 16};
    PTO2TensorMap tmap{};
    ASSERT_TRUE(tmap.init(256, 1024, window_sizes));

    Tensor tensor = make_tensor(0x1000);
    tmap.insert(tensor, PTO2TaskId::make(0, 0));

    TestLookupResult result;
    run_lookup(tmap, tensor, result);

    ASSERT_EQ(result.count, 1);
    EXPECT_EQ(result.entries[0].entry->producer_task_id, PTO2TaskId::make(0, 0));
    EXPECT_EQ(result.entries[0].overlap_status, OverlapStatus::COVERED);

    tmap.destroy();
}

TEST(RuntimeTensorMapBehavior, ValiditySkipsRetiredEntries) {
    int32_t window_sizes[PTO2_MAX_RING_DEPTH] = {256, 256, 256, 256};
    PTO2TensorMap tmap{};
    ASSERT_TRUE(tmap.init(256, 4096, window_sizes));

    Tensor tensor = make_tensor(0x2000);
    for (int i = 0; i < 100; i++) {
        tmap.insert(tensor, PTO2TaskId::make(0, i));
    }

    tmap.sync_validity(0, 80);

    TestLookupResult result;
    run_lookup(tmap, tensor, result);

    EXPECT_EQ(result.count, 20);
    for (const auto &entry : result.entries) {
        EXPECT_GE(entry.entry->producer_task_id.local(), 80u);
        EXPECT_EQ(entry.overlap_status, OverlapStatus::COVERED);
    }

    tmap.destroy();
}

TEST(RuntimeTensorMapBehavior, AllRingsCanProduceForSameTensor) {
    int32_t window_sizes[PTO2_MAX_RING_DEPTH];
    for (int i = 0; i < PTO2_MAX_RING_DEPTH; i++) {
        window_sizes[i] = kWindowSize;
    }
    PTO2TensorMap tmap{};
    ASSERT_TRUE(tmap.init(256, 1024, window_sizes));

    Tensor tensor = make_tensor(0x3000);
    for (int ring = 0; ring < PTO2_MAX_RING_DEPTH; ring++) {
        tmap.insert(tensor, PTO2TaskId::make(ring, 0));
    }

    TestLookupResult result;
    run_lookup(tmap, tensor, result);

    EXPECT_EQ(result.count, PTO2_MAX_RING_DEPTH);

    tmap.destroy();
}

TEST(RuntimeIntegrationBehavior, OrchestratorTensorMapUsesConfiguredWindow) {
    TMRSystem sys;
    ASSERT_TRUE(sys.Init());

    Tensor tensor = make_tensor(0x4000);
    sys.orch.begin_scope();
    sys.orch.tensor_map.insert(tensor, PTO2TaskId::make(0, 0));

    TestLookupResult result;
    run_lookup(sys.orch.tensor_map, tensor, result);

    sys.orch.end_scope();

    ASSERT_EQ(result.count, 1);
    EXPECT_EQ(result.entries[0].entry->producer_task_id, PTO2TaskId::make(0, 0));
    EXPECT_EQ(sys.sm->header->orch_error_code.load(std::memory_order_acquire), PTO2_ERROR_NONE);

    sys.Destroy();
}
