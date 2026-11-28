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
#include <algorithm>
#include <cstring>
#include <functional>
#include <memory>
#include <vector>

#include "rdma/in_die_completion.h"
#include "orchestrator.h"
#include "scheduler/scheduler.h"

namespace {

using pto::comm::rdma::in_die::Queue;
using simpler::rdma::InDieCompletionCursor;
using simpler::rdma::InDiePollState;
using simpler::rdma::poll_in_die_completion;

struct MemoryReader {
    std::function<void(uint64_t)> after_read;
    uint32_t read32(uint64_t address) {
        uint32_t value;
        std::memcpy(&value, reinterpret_cast<const void *>(address), sizeof(value));
        if (after_read) after_read(address);
        return value;
    }
    uint64_t read64(uint64_t address) {
        uint64_t value;
        std::memcpy(&value, reinterpret_cast<const void *>(address), sizeof(value));
        return value;
    }
};

struct InDieFixture : testing::Test {
    struct Workspace {
        pto::comm::rdma::RdmaInfo info{};
        Queue queues[2]{};
        pto::comm::rdma::RdmaMemInfo memory[3]{};
    } workspace;
    std::array<std::array<uint32_t, 8>, 64> cqe{};
    uint32_t tail = 0;
    uint32_t head = 1;
    MemoryReader reader;
    InDieCompletionCursor cursor;

    InDieFixture() {
        auto &info = workspace.info;
        info.magic = pto::comm::rdma::kRdmaWorkspaceMagic;
        info.version = pto::comm::rdma::kRdmaWorkspaceVersion;
        info.backend = pto::comm::RdmaBackend::IN_DIE;
        info.rankCount = 2;
        info.qpNum = 1;
        info.sqPtr = reinterpret_cast<uint64_t>(workspace.queues);
        info.scqPtr = info.sqPtr;
        info.memPtr = reinterpret_cast<uint64_t>(workspace.memory);
        for (auto &queue : workspace.queues) {
            queue.cq.buffer = reinterpret_cast<uint64_t>(cqe.data());
            queue.cq.tail = reinterpret_cast<uint64_t>(&tail);
            queue.cq.depth = 64;
            queue.cq.stride = 32;
            queue.sq.head = reinterpret_cast<uint64_t>(&head);
        }
    }

    auto poll(uint32_t target = 1, uint32_t queue = 1, uint64_t epoch = 7) {
        return poll_in_die_completion(
            (uint64_t(queue) << 32) | target, reinterpret_cast<uint64_t>(&workspace), epoch, cursor, reader
        );
    }
};

TEST_F(InDieFixture, PendingOwnerDoesNotCompleteTheTransfer) {
    EXPECT_EQ(poll().state, InDiePollState::Pending);
    cqe[0][0] = 0x80;
    EXPECT_EQ(poll().state, InDiePollState::Ready);
    EXPECT_EQ(tail, 0U);
}

TEST_F(InDieFixture, InvalidReclaimedTailCannotProveSuccess) {
    tail = 2;
    head = 1;
    EXPECT_EQ(poll().state, InDiePollState::Failed);
}

TEST_F(InDieFixture, RealTaskCompletionRequiresBothFinAndCq) {
    DeviceArena shared_arena, runtime_arena;
    SharedMemoryHandle shared{};
    constexpr int window = 8;
    constexpr uint64_t heap_bytes = 4096;
    const auto size = SharedMemoryHandle::calculate_size(window);
    const auto offset = shared_arena.reserve(size, CHIP_ALIGN_SIZE);
    ASSERT_NE(shared_arena.commit(), nullptr);
    auto *base = shared_arena.region_ptr(offset);
    std::memset(base, 0, size);
    ASSERT_TRUE(shared.init(base, size, window, heap_bytes));
    auto orch = std::make_unique<OrchestratorState>();
    auto sched = std::make_unique<SchedulerState>();
    std::vector<char> heap(heap_bytes * CHIP_MAX_RING_DEPTH);
    int32_t windows[CHIP_MAX_RING_DEPTH];
    std::fill(std::begin(windows), std::end(windows), window);
    const auto orch_layout = OrchestratorState::reserve_layout(runtime_arena, windows, 128);
    const auto sched_layout = SchedulerState::reserve_layout(runtime_arena, 128);
    ASSERT_NE(runtime_arena.commit(), nullptr);
    ASSERT_TRUE(
        orch->init_data_from_layout(orch_layout, runtime_arena, shared.sm_base, heap.data(), heap_bytes, window)
    );
    ASSERT_TRUE(sched->init_data_from_layout(sched_layout, runtime_arena, shared.sm_base));
    sched->wire_arena_pointers(sched_layout, runtime_arena);
    orch->wire_arena_pointers(orch_layout, runtime_arena, sched.get());
    orch->begin_scope();
    for (const bool cq_first : {false, true}) {
        CoreTaskArgs args;
        const auto token = orch->submit_dummy_task(args).task_id();
        ASSERT_TRUE(token.is_valid());
        auto *slot = &shared.header->rings[token.ring()].get_slot_state_by_task_id(token.local_id());
        auto &waits = sched->async_wait_list;
        auto &entry = waits.entries[0];
        entry = {};
        entry.task_token = token;
        entry.slot_state = slot;
        entry.normal_done = !cq_first;
        waits.count.store(1);
        int error = SIMPLER_ERROR_NONE;
        ASSERT_TRUE(waits.append_condition_locked(
            entry, (uint64_t{1} << 32) | 1, reinterpret_cast<uint64_t>(&workspace), 0, ASYNC_ENGINE_ROCE,
            COMPLETION_TYPE_RDMA_IN_DIE_EVENT_HANDLE, error
        ));
        cqe[0][0] = cq_first ? 0x80 : 0;
        ChipTaskSlotState *released[2]{};
        int release_count = 0;
        auto check = [&] {
            return waits.poll_and_complete<false>(
                nullptr, sched.get(), released, release_count, 2
#if SIMPLER_SCHED_PROFILING
                ,
                0
#endif
            );
        };
        EXPECT_EQ(check().completed, 0);
        EXPECT_LT(slot->task_state.load(), CHIP_TASK_COMPLETED);
        EXPECT_EQ(waits.count.load(), 1);
        entry.normal_done = true;
        cqe[0][0] = 0x80;
        const auto completed = check();
        EXPECT_EQ(completed.error_code, SIMPLER_ERROR_NONE);
        EXPECT_EQ(completed.completed, 1);
        EXPECT_GE(slot->task_state.load(), CHIP_TASK_COMPLETED);
        EXPECT_EQ(waits.count.load(), 0);
        EXPECT_EQ(release_count, 1);
        EXPECT_EQ(released[0], slot);
        EXPECT_EQ(tail, 0U);
    }
    orch->destroy();
    sched->destroy();
}

TEST_F(InDieFixture, ErrorBeforeTheTargetIsNotHiddenByLastSuccessfulCqe) {
    head = 2;
    cqe[0][0] = 0x780;
    cqe[1][0] = 0x80;
    auto result = poll(2);
    EXPECT_EQ(result.state, InDiePollState::Failed);
    EXPECT_EQ(result.error, 7U);
    cqe[0][0] = 0x80;
    EXPECT_EQ(poll(2).error, 7U);
}

TEST_F(InDieFixture, ReclaimedPrefixWinsOverARecycledCqe) {
    tail = 1;
    cqe[0][0] = 0x980;
    EXPECT_EQ(poll().state, InDiePollState::Ready);
}

TEST_F(InDieFixture, TailAdvanceDiscardsTheRacingCqeSnapshot) {
    cqe[0][0] = 0x980;
    reader.after_read = [&](uint64_t address) {
        if (address == reinterpret_cast<uint64_t>(&cqe[0][0])) tail = 1;
    };
    EXPECT_EQ(poll().state, InDiePollState::Ready);
    EXPECT_EQ(cursor.error, 0U);
}

TEST_F(InDieFixture, OwnerPolarityChangesAtTheRingBoundary) {
    tail = 64;
    head = 65;
    cqe[0][0] = 0x80;
    EXPECT_EQ(poll(65).state, InDiePollState::Pending);
    cqe[0][0] = 0;
    EXPECT_EQ(poll(65).state, InDiePollState::Ready);
}

TEST_F(InDieFixture, EachPollHasABoundedScanBudget) {
    head = 40;
    for (auto &entry : cqe)
        entry[0] = 0x80;
    EXPECT_EQ(poll(40).state, InDiePollState::Pending);
    EXPECT_EQ(cursor.prefix, 32U);
    EXPECT_EQ(poll(40).state, InDiePollState::Ready);
    EXPECT_EQ(tail, 0U);
}

TEST_F(InDieFixture, QueueFailureDoesNotInvalidateEarlierProvenSuccess) {
    head = 2;
    cqe[0][0] = 0x80;
    workspace.queues[1].error = 12;
    EXPECT_EQ(poll().state, InDiePollState::Ready);
    EXPECT_EQ(poll(2).state, InDiePollState::Failed);
    EXPECT_EQ(poll().state, InDiePollState::Ready);
}

TEST_F(InDieFixture, ANewOwnerEpochCannotReuseAnOldSuccessCursor) {
    cqe[0][0] = 0x80;
    EXPECT_EQ(poll().state, InDiePollState::Ready);
    cqe[0][0] = 0;
    EXPECT_EQ(poll(1, 1, 8).state, InDiePollState::Pending);
}

TEST_F(InDieFixture, InvalidLayoutAndErrorHandlesFailClosed) {
    EXPECT_EQ(poll(0).state, InDiePollState::Failed);
    EXPECT_EQ(poll(1, UINT32_MAX).state, InDiePollState::Failed);
    EXPECT_EQ(poll(1, 2).state, InDiePollState::Failed);
    workspace.info.version = 2;
    EXPECT_EQ(poll().state, InDiePollState::Failed);
}

TEST(InDieCompletion, NullWorkspaceIsRejectedBeforeReadingMemory) {
    MemoryReader reader;
    InDieCompletionCursor cursor;
    EXPECT_EQ(poll_in_die_completion(1, 0, 0, cursor, reader).state, InDiePollState::Failed);
}

}  // namespace
