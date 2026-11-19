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

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "graph_execution.h"
#include "graph_host_state.h"
#include "scheduler/scheduler.h"
#include "host_build_graph/host/orchestrator.h"
#include "host_build_graph/shared_memory.h"
#include "utils/device_arena.h"
#include "host_build_graph/task_id.h"

// The device side of a body's fanin CSR row. The scheduler reads only the CSR
// pair and the slots it indexes, so a topology can be stated directly here
// rather than materialized from a Definition.
class HbgGraphWakeScanTest : public ::testing::Test {
protected:
    DeviceArena sm_arena;
    DeviceArena runtime_arena;
    SharedMemoryHandle *sm_handle = nullptr;
    SchedulerState sched{};
    SchedulerLayout sched_layout{};
    GraphExecution execution{};
    // ChipTaskStorage holds atomics, so it is neither copyable nor movable and
    // cannot back a vector; this constructs each element in place.
    std::unique_ptr<ChipTaskStorage[]> storage;
    // The execution's readiness array, allocated with the storage it indexes.
    std::unique_ptr<std::atomic<ChipTaskState>[]> states;
    std::vector<int32_t> fanin_offsets;
    std::vector<uint16_t> fanin_indices;

    void SetUp() override {
        sm_handle = SharedMemoryHandle::create_and_init_default(sm_arena);
        ASSERT_NE(sm_handle, nullptr);
        sched_layout = SchedulerState::reserve_layout(runtime_arena);
        ASSERT_NE(runtime_arena.commit(), nullptr);
        ASSERT_TRUE(sched.init_data_from_layout(sched_layout, runtime_arena, sm_handle->sm_base));
        sched.wire_arena_pointers(sched_layout, runtime_arena);
        sched.seed_queue_slots();
    }

    void TearDown() override {
        sched.destroy();
        runtime_arena.release();
        sm_arena.release();
    }

    // `rows[i]` holds task i's producers in row order. Every task starts as a
    // pending AIV leaf, the state materialization leaves it in.
    void build(const std::vector<std::vector<uint16_t>> &rows) {
        storage = std::make_unique<ChipTaskStorage[]>(rows.size());
        states = std::make_unique<std::atomic<ChipTaskState>[]>(rows.size());
        execution.task_states = states.get();
        fanin_offsets.assign(1, 0);
        for (size_t i = 0; i < rows.size(); ++i) {
            fanin_indices.insert(fanin_indices.end(), rows[i].begin(), rows[i].end());
            fanin_offsets.push_back(static_cast<int32_t>(fanin_indices.size()));
            ChipTaskSlotState &s = storage[i].slot;
            s.reset_for_reuse();
            s.sub_task_local_id = static_cast<int32_t>(i);
            s.active_mask = LaneMask::aiv0();
            s.graph_context = &execution;
            execution.reset_task_state(static_cast<int32_t>(i));
        }
        execution.task_count = static_cast<int32_t>(rows.size());
        execution.task_storage = storage.get();
        execution.fanin_offsets = fanin_offsets.data();
        execution.fanin_indices = fanin_indices.data();
    }

    ChipTaskSlotState &slot(size_t i) { return storage[i].slot; }

    ChipTaskSlotState *pop_ready() {
        ChipTaskSlotState *out = nullptr;
        const int popped = sched.ready_queues[static_cast<int32_t>(ResourceShape::AIV)].pop_batch(&out, 1);
        return popped == 1 ? out : nullptr;
    }
};

// A wide sub-task row exceeds what a byte cursor can index. The scan must
// report the row unfinished until every producer has published: a truncated
// cursor would wrap to a low index, find that one entry published, and stage a
// candidate whose remaining producers have not.
TEST_F(HbgGraphWakeScanTest, WideRowCursorDoesNotTruncate) {
    constexpr int32_t kProducers = 300;  // > 0xFF, the old cursor width
    std::vector<std::vector<uint16_t>> rows(kProducers);
    std::vector<uint16_t> consumer_row;
    consumer_row.reserve(kProducers);
    for (int32_t i = 0; i < kProducers; ++i)
        consumer_row.push_back(static_cast<uint16_t>(i));
    rows.push_back(consumer_row);
    build(rows);

    ChipTaskSlotState &consumer = slot(kProducers);
    consumer.ed_flags |= ED_FLAG_CANDIDATE;
    for (int32_t i = 0; i < kProducers; ++i)
        slot(i).ed_flags |= ED_FLAG_TRACKED;

    // Intake hangs it on the row's tail, and the cursor must name that entry
    // rather than a wrapped one.
    ASSERT_FALSE(sched.register_on_ed_publish_list(consumer));
    EXPECT_EQ(consumer.ed_publish_scan_cursor, kProducers - 1);

    // Publishing every producer but the first leaves the row unfinished.
    for (int32_t i = kProducers - 1; i > 0; --i) {
        execution.store_published(i);
        sched.seal_ed_publish_list(slot(i));
    }
    ChipTaskSlotState *detached = nullptr;
    while (sched.ed_publish_drain_queue.pop_batch(&detached, 1) == 1) {
        if (sched.advance_ed_publish_scan(*detached)) FAIL() << "staged while producer 0 is unpublished";
    }

    execution.store_published(0);
    sched.seal_ed_publish_list(slot(0));
    ASSERT_EQ(sched.ed_publish_drain_queue.pop_batch(&detached, 1), 1);
    EXPECT_TRUE(sched.advance_ed_publish_scan(*detached));
}

// Each classification resumes where the last one hung and never re-walks the
// row's completed tail. Completion is monotone, so the resume reaches the same
// verdict a full rescan would.
TEST_F(HbgGraphWakeScanTest, CursorResumesAndNeverRewalks) {
    build({{}, {}, {}, {0, 1, 2}});
    ChipTaskSlotState &consumer = slot(3);

    // Never hung: the sentinel folds the scan start to the row's tail.
    EXPECT_EQ(consumer.wake_scan_cursor, 0xFFFF);
    EXPECT_EQ(sched.graph_first_unmet_producer(execution, consumer), 2);
    EXPECT_EQ(consumer.wake_scan_cursor, 2);

    // The hung-at producer completes: the rescan resumes at the cursor and walks
    // down to the next unmet row entry.
    execution.store_completed(2);
    EXPECT_EQ(sched.graph_first_unmet_producer(execution, consumer), 1);
    EXPECT_EQ(consumer.wake_scan_cursor, 1);

    // An entry above the cursor completing does not move it — nothing re-walks
    // that tail — and clearing the rest yields the ready verdict.
    execution.store_completed(0);
    EXPECT_EQ(sched.graph_first_unmet_producer(execution, consumer), 1);
    EXPECT_EQ(consumer.wake_scan_cursor, 1);
    execution.store_completed(1);
    EXPECT_EQ(sched.graph_first_unmet_producer(execution, consumer), -1);
}

// The classifier names a row index, not a producer index; graph_producer_at is
// what maps one to the other. A row that is not in producer order separates them.
TEST_F(HbgGraphWakeScanTest, ClassifierNamesARowIndexNotAProducerIndex) {
    build({{}, {}, {}, {2, 0, 1}});
    ChipTaskSlotState &consumer = slot(3);

    EXPECT_EQ(sched.graph_first_unmet_producer(execution, consumer), 2);
    EXPECT_EQ(&sched.graph_producer_at(execution, consumer, 2), &slot(1));
    EXPECT_EQ(&sched.graph_producer_at(execution, consumer, 0), &slot(2));
}

// A waiter with exactly one producer is by construction waiting only on the
// producer draining it, so the drain routes it without a classification pass.
TEST_F(HbgGraphWakeScanTest, SingleProducerWaiterIsRoutedByTheDrain) {
    build({{}, {0}});
    ChipTaskSlotState &producer = slot(0);
    ChipTaskSlotState &consumer = slot(1);

    sched.register_graph_wake(execution, &producer, &consumer);
    ASSERT_EQ(producer.wake_list_head.load(std::memory_order_relaxed), &consumer);

    execution.store_completed(0);
    EXPECT_EQ(sched.drain_graph_wake_list(execution, producer), 1u);
    EXPECT_EQ(pop_ready(), &consumer);
    // Decided from the row's length alone, so the never-hung sentinel stands.
    EXPECT_EQ(consumer.wake_scan_cursor, 0xFFFF);
}

// A multi-producer waiter takes the classification path instead: it re-registers
// on the row's next unmet entry and reaches the queue only once that clears too.
TEST_F(HbgGraphWakeScanTest, MultiProducerWaiterReRegistersUntilItsRowClears) {
    build({{}, {}, {0, 1}});
    ChipTaskSlotState &consumer = slot(2);

    sched.register_graph_wake(execution, &slot(1), &consumer);
    execution.store_completed(1);
    EXPECT_EQ(sched.drain_graph_wake_list(execution, slot(1)), 1u);

    // Producer 0 is still pending, so the waiter moved rather than became ready.
    EXPECT_EQ(pop_ready(), nullptr);
    EXPECT_EQ(slot(0).wake_list_head.load(std::memory_order_relaxed), &consumer);
    EXPECT_EQ(consumer.wake_scan_cursor, 0);

    execution.store_completed(0);
    EXPECT_EQ(sched.drain_graph_wake_list(execution, slot(0)), 1u);
    EXPECT_EQ(pop_ready(), &consumer);
}
