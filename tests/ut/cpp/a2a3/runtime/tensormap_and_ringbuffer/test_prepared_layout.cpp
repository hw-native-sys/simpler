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
#include <cstring>
#include <limits>
#include <vector>
#include "runtime_core.h"

namespace {
ArenaSizingKey sizing_key() {
    ArenaSizingKey key{};
    for (int r = 0; r < CHIP_MAX_RING_DEPTH; ++r) {
        key.task_window_sizes[r] = 64ULL << r;
        key.heap_sizes[r] = 4096ULL * (r + 1);
        key.dep_pool_capacities[r] = 128 << r;
    }
    return key;
}

TEST(PreparedLayout, MatchesExistingLayoutForUnequalRings) {
    auto key = sizing_key();
    PreparedRunLayout plan{};
    ASSERT_TRUE(build_prepared_run_layout(key, plan));
    uint64_t heap_total = 0;
    for (int r = 0; r < CHIP_MAX_RING_DEPTH; ++r) {
        const auto expected = sm_layout::ring_segment_offsets(key.task_window_sizes, r);
        EXPECT_EQ(plan.rings[r].descriptors, expected.descriptors);
        EXPECT_EQ(plan.rings[r].payloads, expected.payloads);
        EXPECT_EQ(plan.rings[r].slot_states, expected.slot_states);
        heap_total += key.heap_sizes[r];
    }
    EXPECT_EQ(plan.sm_bytes, SharedMemoryHandle::calculate_size_per_ring(key.task_window_sizes));
    EXPECT_EQ(plan.heap_bytes, heap_total);
    EXPECT_EQ(sizeof(plan), 112U);
}

TEST(PreparedLayout, RejectsOverflowWithoutPublishingPartialPlan) {
    auto key = sizing_key();
    PreparedRunLayout plan{};
    ASSERT_TRUE(build_prepared_run_layout(key, plan));
    const auto saved = plan;
    key.heap_sizes[0] = std::numeric_limits<uint64_t>::max();
    ASSERT_FALSE(build_prepared_run_layout(key, plan));
    EXPECT_EQ(std::memcmp(&plan, &saved, sizeof(plan)), 0);
    key = sizing_key();
    key.task_window_sizes[2] = static_cast<uint64_t>(INT32_MAX) + 1;
    ASSERT_FALSE(build_prepared_run_layout(key, plan));
    EXPECT_EQ(std::memcmp(&plan, &saved, sizeof(plan)), 0);
}

class PreparedRuntime : public ::testing::Test {
protected:
    ArenaSizingKey key = sizing_key();
    RuntimeArenaLayout layout{};
    DeviceArena sm_arena;
    DeviceArena arena;
    std::vector<char> heap;
    void *sm = nullptr;
    RuntimeContext *rt = nullptr;

    void SetUp() override {
        layout = runtime_reserve_layout(arena, key.task_window_sizes, key.heap_sizes, key.dep_pool_capacities);
        ASSERT_TRUE(build_prepared_run_layout(layout.sizing, layout.prepared));
        const auto off = sm_arena.reserve(layout.prepared.sm_bytes, CHIP_ALIGN_SIZE);
        ASSERT_NE(sm_arena.commit(), nullptr);
        sm = sm_arena.region_ptr(off);
        heap.resize(layout.prepared.heap_bytes);
        ASSERT_NE(arena.commit(), nullptr);
        rt = runtime_init_data_from_layout(
            arena, layout, MODE_SIMULATE, sm, layout.prepared.sm_bytes, heap.data(), key.heap_sizes
        );
        ASSERT_NE(rt, nullptr);
        runtime_wire_arena_pointers(arena, layout, rt);
        ASSERT_TRUE(rt->sm_handle->init_per_ring(sm, layout.prepared.sm_bytes, key.task_window_sizes, key.heap_sizes));
    }

    void check_header() {
        auto *header = rt->sm_handle->header;
        for (int r = 0; r < CHIP_MAX_RING_DEPTH; ++r) {
            const auto off = sm_layout::ring_segment_offsets(key.task_window_sizes, r);
            const auto &ring = header->rings[r];
            EXPECT_EQ(
                ring.task_descriptors, reinterpret_cast<TaskDescriptor *>(static_cast<char *>(sm) + off.descriptors)
            );
            EXPECT_EQ(ring.task_payloads, reinterpret_cast<TaskPayload *>(static_cast<char *>(sm) + off.payloads));
            EXPECT_EQ(
                ring.slot_states, reinterpret_cast<ChipTaskSlotState *>(static_cast<char *>(sm) + off.slot_states)
            );
            EXPECT_EQ(ring.task_window_size, key.task_window_sizes[r]);
            EXPECT_EQ(ring.task_window_mask, key.task_window_sizes[r] - 1);
            EXPECT_EQ(ring.heap_size, key.heap_sizes[r]);
            EXPECT_EQ(ring.task_descriptors_offset, off.descriptors);
            EXPECT_EQ(ring.fc.current_task_index.load(), 0);
            EXPECT_EQ(ring.fc.last_task_alive.load(), 0);
        }
        EXPECT_EQ(header->total_size, layout.prepared.sm_bytes);
        EXPECT_EQ(header->orchestrator_done.load(), 0);
        EXPECT_EQ(header->orch_error_code.load(), SIMPLER_ERROR_NONE);
        EXPECT_EQ(header->sched_error_bitmap.load(), 0U);
        EXPECT_EQ(header->sched_error_code.load(), SIMPLER_ERROR_NONE);
        EXPECT_EQ(header->sched_error_thread.load(), -1);
        EXPECT_EQ(header->sched_stall_detail.load(), SIMPLER_STALL_DETAIL_NONE);
        EXPECT_EQ(header->sched_stall_completed.load(), 0);
        EXPECT_EQ(header->sched_stall_total.load(), 0);
        EXPECT_EQ(header->sched_stall_cnt_running.load(), 0);
        EXPECT_EQ(header->sched_stall_cnt_ready.load(), 0);
        EXPECT_EQ(header->sched_stall_cnt_waiting.load(), 0);
        EXPECT_EQ(header->sched_stall_orch_done.load(), 0);
        EXPECT_EQ(header->sched_stall_task_id.load(), TaskId::invalid());
        EXPECT_EQ(header->sched_stall_core.load(), -1);
    }

    void check_restart(bool prepared, uint32_t epoch) {
        auto &orch = rt->orchestrator;
        orch.fatal = true;
        orch.inline_completed_tasks = 42;
        orch.submit_seq = 19;
        orch.fanin_seen_current_epoch = epoch;
        orch.tensor_map.current_epoch = epoch;
        orch.tensor_map.next_entry_idx = 5;
        orch.tensor_map.free_num = 3;
        orch.scope_tasks_size = 9;
        orch.scope_stack_top = 2;
        orch.total_cluster_count = 24;
        orch.total_aiv_count = 48;
        rt->gm_heap_size = 1;
        rt->gm_heap_owned = true;
        rt->total_cycles = 999;
        for (int r = 0; r < CHIP_MAX_RING_DEPTH; ++r) {
            std::fill_n(orch.fanin_seen_epoch[r], key.task_window_sizes[r], 17U);
            std::fill_n(orch.tensor_map.task_entry_head_epochs[r], key.task_window_sizes[r], 17U);
            orch.tensor_map.last_task_alives[r] = 8;
            orch.tensor_map.last_cleanup[r] = 7;
        }
        std::fill_n(orch.tensor_map.bucket_epochs, orch.tensor_map.num_buckets, 17U);
        const bool ok =
            prepared ? runtime_reset_for_reuse_prepared(layout, rt) : runtime_reset_for_reuse(arena, layout, rt);
        ASSERT_TRUE(ok);
        EXPECT_FALSE(orch.fatal);
        EXPECT_EQ(orch.inline_completed_tasks, 0);
        EXPECT_EQ(orch.submit_seq, 0U);
        const uint32_t next = epoch == UINT32_MAX ? 1 : epoch + 1;
        EXPECT_EQ(orch.fanin_seen_current_epoch, next);
        EXPECT_EQ(orch.tensor_map.current_epoch, next);
        EXPECT_EQ(orch.tensor_map.next_entry_idx, 0);
        EXPECT_EQ(orch.tensor_map.free_num, 0);
        EXPECT_EQ(orch.scope_tasks_size, 0);
        EXPECT_EQ(orch.scope_stack_top, -1);
        EXPECT_EQ(orch.total_cluster_count, 0);
        EXPECT_EQ(orch.total_aiv_count, 0);
        EXPECT_EQ(rt->gm_heap_size, layout.prepared.heap_bytes);
        EXPECT_EQ(orch.gm_heap_size, layout.prepared.heap_bytes);
        EXPECT_FALSE(rt->gm_heap_owned);
        EXPECT_EQ(rt->total_cycles, 0U);
        EXPECT_EQ(rt->pending_scope_mode, ScopeMode::AUTO);
        for (int r = 0; r < CHIP_MAX_RING_DEPTH; ++r) {
            EXPECT_EQ(orch.rings[r].task_allocator.task_head(), 0);
            EXPECT_EQ(orch.rings[r].task_allocator.window_size(), key.task_window_sizes[r]);
            EXPECT_EQ(orch.rings[r].task_allocator.heap_capacity(), key.heap_sizes[r]);
            EXPECT_EQ(orch.tensor_map.last_task_alives[r], 0);
            EXPECT_EQ(orch.tensor_map.last_cleanup[r], 0);
            for (size_t i = 0; i < key.task_window_sizes[r]; ++i) {
                EXPECT_EQ(orch.fanin_seen_epoch[r][i], epoch == UINT32_MAX ? 0U : 17U);
                EXPECT_EQ(orch.tensor_map.task_entry_head_epochs[r][i], epoch == UINT32_MAX ? 0U : 17U);
            }
        }
    }
};

TEST_F(PreparedRuntime, RestoresHeaderFromPreviousRun) {
    check_header();
    auto *h = rt->sm_handle->header;
    h->orch_error_code.store(123);
    h->sched_error_code.store(456);
    h->sched_stall_completed.store(89);
    h->orchestrator_done.store(1);
    for (auto &ring : h->rings) {
        ring.fc.current_task_index.store(67);
        ring.fc.last_task_alive.store(41);
    }
    ASSERT_TRUE(rt->sm_handle->init_prepared(
        sm, layout.prepared.sm_bytes, key.task_window_sizes, key.heap_sizes, layout.prepared
    ));
    check_header();
}

TEST_F(PreparedRuntime, RejectsInvalidCapacityBeforeWritingHeader) {
    rt->sm_handle->header->orch_error_code.store(123);
    EXPECT_FALSE(rt->sm_handle->init_prepared(
        sm, layout.prepared.sm_bytes - 1, key.task_window_sizes, key.heap_sizes, layout.prepared
    ));
    EXPECT_EQ(rt->sm_handle->header->orch_error_code.load(), 123);
    EXPECT_FALSE(rt->sm_handle->init_prepared(
        nullptr, layout.prepared.sm_bytes, key.task_window_sizes, key.heap_sizes, layout.prepared
    ));
    EXPECT_FALSE(runtime_reset_for_reuse_prepared(layout, nullptr));
}

TEST_F(PreparedRuntime, GenericAndPreparedRestartNormalEpoch) {
    check_restart(false, 5);
    check_restart(true, 5);
}

TEST_F(PreparedRuntime, GenericAndPreparedRestartWrappedEpoch) {
    check_restart(false, UINT32_MAX);
    check_restart(true, UINT32_MAX);
}
TEST_F(PreparedRuntime, DestroyRewirePreservesAllocatorResourcesAcrossRuns) {
    for (int invocation = 0; invocation < 3; ++invocation) {
        runtime_destroy(rt, arena);
        EXPECT_EQ(rt->orchestrator.tensor_map.buckets, nullptr);
        runtime_wire_arena_pointers(arena, layout, rt);
        ASSERT_TRUE(rt->sm_handle->init_prepared(
            sm, layout.prepared.sm_bytes, key.task_window_sizes, key.heap_sizes, layout.prepared
        ));
        ASSERT_TRUE(runtime_reset_for_reuse_prepared(layout, rt));
        uint64_t heap_offset = 0;
        for (int ring = 0; ring < CHIP_MAX_RING_DEPTH; ++ring) {
            auto &allocator = rt->orchestrator.rings[ring].task_allocator;
            const auto first = allocator.alloc(CHIP_ALIGN_SIZE);
            ASSERT_FALSE(first.failed());
            EXPECT_EQ(first.task_id, 0);
            EXPECT_EQ(first.packed_base, heap.data() + heap_offset);
            EXPECT_EQ(rt->sm_handle->header->rings[ring].fc.current_task_index.load(), 1);
            const auto second = allocator.alloc(CHIP_ALIGN_SIZE);
            ASSERT_FALSE(second.failed());
            EXPECT_EQ(second.task_id, 1);
            EXPECT_EQ(second.packed_base, heap.data() + heap_offset + CHIP_ALIGN_SIZE);
            heap_offset += key.heap_sizes[ring];
        }
    }
}
}  // namespace
