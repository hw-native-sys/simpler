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
#include <array>
#include <atomic>
#include <functional>
#include <thread>
#include <vector>

#include "scheduler/scheduler_args_dump.h"
#include "scheduler/scheduler_args_dump_tensor.h"

namespace {

SchedulerArgsDumpAck
committed_ack(uint64_t generation, ArgsDumpStage stage, uint64_t tick, SchedulerArgsDumpStatus state) {
    SchedulerArgsDumpAck ack{11, 10, static_cast<uint64_t>(stage), generation, tick, 17, 65536, 0};
    ack.status = scheduler_args_dump_ack_commit(generation, ack.stage, state);
    return ack;
}

struct InterleavedAckReader {
    std::function<void(size_t)> *before_load;
    size_t index{0};
    uint64_t load(uint64_t &word) {
        (*before_load)(index++);
        return scheduler_gm_query(word);
    }
    void barrier() const { scheduler_cache_barrier(); }
};

TEST(ResidentArgsDump, NewIdentityWithOldSealCannotGrantDeparture) {
    for (const bool same_generation : {false, true}) {
        SchedulerArgsDumpTask task{};
        task.task_id = 10;
        task.active_mask = task.before_masks = 1;
        task.publisher_before = scheduler_args_dump_publisher(0, 8);
        auto ack = committed_ack(
            same_generation ? 8 : 7, ArgsDumpStage::AFTER_COMPLETION, 10, SchedulerArgsDumpStatus::Sealed
        );
        // A body prefix cannot borrow a previous phase's committed state.
        ack.stage = static_cast<uint64_t>(ArgsDumpStage::BEFORE_DISPATCH);
        ack.generation = 8;
        bool departed = false;
        EXPECT_EQ(
            scheduler_args_dump_poll_phase(&task, ArgsDumpStage::BEFORE_DISPATCH, SubtaskSlot::AIC, ack, 11, &departed),
            SchedulerArgsDumpPoll::Waiting
        );
        EXPECT_FALSE(departed);
        EXPECT_FALSE(scheduler_args_dump_can_execute(&task));
    }
}

TEST(ResidentArgsDump, AckSnapshotRejectsEveryCrossingPublicationPrefix) {
    for (const auto next_state : {SchedulerArgsDumpStatus::InProgress, SchedulerArgsDumpStatus::Aborted}) {
        const auto old = committed_ack(
            8, ArgsDumpStage::BEFORE_DISPATCH, UINT64_C(0x100000001), SchedulerArgsDumpStatus::InProgress
        );
        auto next = old;
        next.arg_index = 18;
        next.byte_cursor = 3;
        if (next_state != SchedulerArgsDumpStatus::Aborted) ++next.tick;
        next.status = scheduler_args_dump_ack_commit(next.generation, next.stage, next_state);
        for (size_t invalidate = 0; invalidate <= 10; ++invalidate) {
            for (size_t body = invalidate; body <= 10; ++body) {
                for (size_t commit = body; commit <= 10; ++commit) {
                    auto wire = old;
                    std::function<void(size_t)> publish = [&](size_t step) {
                        if (step == invalidate) wire.status = 0;
                        if (step == body) {
                            wire.arg_index = next.arg_index;
                            wire.byte_cursor = next.byte_cursor;
                            wire.tick = next.tick;
                        }
                        if (step == commit) wire.status = next.status;
                    };
                    SchedulerArgsDumpAck observed{};
                    if (scheduler_args_dump_observe_ack(&wire, &observed, InterleavedAckReader{&publish})) {
                        const bool old_record = observed.status == old.status && observed.tick == old.tick &&
                                                observed.arg_index == old.arg_index &&
                                                observed.byte_cursor == old.byte_cursor;
                        const bool new_record = observed.status == next.status && observed.tick == next.tick &&
                                                observed.arg_index == next.arg_index &&
                                                observed.byte_cursor == next.byte_cursor;
                        EXPECT_TRUE(old_record || new_record) << invalidate << '/' << body << '/' << commit;
                    }
                }
            }
        }
    }
}

TEST(ResidentArgsDump, FinalTickProbeRejectsProgressCommittedAfterTheSecondProbe) {
    auto wire = committed_ack(8, ArgsDumpStage::BEFORE_DISPATCH, 11, SchedulerArgsDumpStatus::InProgress);
    const uint64_t commit = wire.status;
    std::function<void(size_t)> publish = [&](size_t step) {
        if (step == 1) {
            wire.status = 0;
            wire.arg_index = 18;
            wire.byte_cursor = 0;
        }
        if (step == 9) {
            wire.tick = 12;
            wire.status = commit;
        }
    };
    SchedulerArgsDumpAck observed{};
    EXPECT_FALSE(scheduler_args_dump_observe_ack(&wire, &observed, InterleavedAckReader{&publish}));
    EXPECT_EQ(observed.status, 0);
}

TEST(ResidentArgsDump, AckSnapshotRejectsMalformedCommitAndKeepsFullTickWidth) {
    auto ack = committed_ack(
        UINT32_MAX, ArgsDumpStage::AFTER_COMPLETION, UINT64_C(0x100000001), SchedulerArgsDumpStatus::Sealed
    );
    SchedulerArgsDumpAck observed{};
    ASSERT_TRUE(scheduler_args_dump_observe_ack(&ack, &observed));
    EXPECT_EQ(observed.tick, UINT64_C(0x100000001));
    EXPECT_EQ(scheduler_args_dump_ack_status(observed), SchedulerArgsDumpStatus::Sealed);
    const auto good = ack;
    for (const uint64_t invalid :
         {UINT64_C(0), good.status | (UINT64_C(1) << 20), UINT64_C(3), (UINT64_C(1) << 32) | UINT64_C(0xff)}) {
        ack = good;
        ack.status = invalid;
        EXPECT_FALSE(scheduler_args_dump_observe_ack(&ack, &observed));
        EXPECT_EQ(observed.status, 0);
    }
    ack = good;
    ack.tick = 0;
    EXPECT_FALSE(scheduler_args_dump_observe_ack(&ack, &observed));
    ack = good;
    std::function<void(size_t)> wrap_low = [&](size_t step) {
        if (step == 9) ack.tick += UINT64_C(1) << 32;
    };
    EXPECT_FALSE(scheduler_args_dump_observe_ack(&ack, &observed, InterleavedAckReader{&wrap_low}));
}

TEST(ResidentArgsDump, SameTickCommittedAbortDominatesConsumedButInvalidPublicationDoesNot) {
    SchedulerArgsDumpTask task{};
    task.task_id = 10;
    task.active_mask = 1;
    task.before_masks = UINT64_C(0x101);
    task.publisher_before = scheduler_args_dump_publisher(0, 8);
    bool departed = true;
    auto ack = committed_ack(8, ArgsDumpStage::BEFORE_DISPATCH, 11, SchedulerArgsDumpStatus::Sealed);
    ack.status = 0;
    EXPECT_EQ(
        scheduler_args_dump_poll_phase(&task, ArgsDumpStage::BEFORE_DISPATCH, SubtaskSlot::AIC, ack, 11, &departed),
        SchedulerArgsDumpPoll::Consumed
    );
    ack.status = scheduler_args_dump_ack_commit(8, ack.stage, SchedulerArgsDumpStatus::Aborted);
    EXPECT_EQ(
        scheduler_args_dump_poll_phase(&task, ArgsDumpStage::BEFORE_DISPATCH, SubtaskSlot::AIC, ack, 11, &departed),
        SchedulerArgsDumpPoll::Aborted
    );
    EXPECT_EQ(task.aborted, 1);
}

TEST(ResidentArgsDump, OutputFlushVisitsOnlyPhysicalViewLinesIncludingInnermostStride) {
    simpler::hbg::Tensor tensor;
    tensor.buffer.addr = 0x1000;
    tensor.buffer.size = 0x400;
    tensor.start_offset = 16;
    tensor.dtype = DataType::UINT32;
    tensor.ndims = 2;
    tensor.shapes[0] = 2;
    tensor.shapes[1] = 3;
    tensor.strides[0] = 64;
    tensor.strides[1] = 16;
    tensor.is_contiguous = false;
    std::vector<uint64_t> lines;
    ASSERT_TRUE(scheduler_args_dump_tensor_lines(&tensor, [&](uint64_t line) {
        lines.push_back(line);
    }));
    EXPECT_EQ(lines, (std::vector<uint64_t>{0x1040, 0x1080, 0x10c0, 0x1140, 0x1180, 0x11c0}));
    tensor.is_contiguous = true;
    lines.clear();
    ASSERT_TRUE(scheduler_args_dump_tensor_lines(&tensor, [&](uint64_t line) {
        lines.push_back(line);
    }));
    EXPECT_EQ(lines, (std::vector<uint64_t>{0x1040}));
    tensor.start_offset = UINT64_MAX;
    EXPECT_FALSE(scheduler_args_dump_tensor_lines(&tensor, [&](uint64_t line) {
        lines.push_back(line);
    }));
    tensor.start_offset = 16;
    tensor.dtype = static_cast<DataType>(UINT8_MAX);
    EXPECT_FALSE(scheduler_args_dump_tensor_lines(&tensor, [&](uint64_t line) {
        lines.push_back(line);
    }));
}

TEST(ResidentArgsDump, StorageTracksActualSelectedTasksAndWorkersAndOffAllocatesNothing) {
    SchedulerArgsDumpLayout layout{};
    ASSERT_TRUE(scheduler_args_dump_plan_layout(3, 1, 2, true, &layout));
    EXPECT_EQ(layout.task_indices_offset, 128);
    EXPECT_EQ(layout.tasks_offset, 192);
    EXPECT_EQ(layout.requests_offset, 256);
    EXPECT_EQ(layout.acknowledgements_offset, 384);
    EXPECT_EQ(layout.observed_ticks_offset, 512);
    EXPECT_EQ(layout.pending_workers_offset, 528);
    EXPECT_EQ(layout.total_size, 576);
    ASSERT_TRUE(scheduler_args_dump_plan_layout(3, 3, 2, false, &layout));
    EXPECT_EQ(layout.total_size, 0);
    ASSERT_TRUE(scheduler_args_dump_plan_layout(3, 0, 2, true, &layout));
    EXPECT_EQ(layout.total_size, 0);
    EXPECT_FALSE(scheduler_args_dump_plan_layout(3, 4, 2, true, &layout));
    EXPECT_FALSE(scheduler_args_dump_plan_layout(UINT64_MAX, 1, 2, true, &layout));
    EXPECT_FALSE(scheduler_args_dump_plan_layout(3, 1, 0, true, &layout));
}

TEST(ResidentArgsDump, OverlappingStridedViewsFitTheirPhysicalExtentAndOverflowIsRefused) {
    simpler::hbg::Tensor tensor{};
    tensor.buffer.addr = 0x1000;
    tensor.buffer.size = 12;
    tensor.dtype = DataType::UINT32;
    tensor.ndims = 2;
    tensor.shapes[0] = 2;
    tensor.shapes[1] = 2;
    tensor.strides[0] = 1;
    tensor.strides[1] = 1;
    tensor.is_contiguous = false;
    std::vector<uint64_t> lines;
    ASSERT_TRUE(scheduler_args_dump_tensor_lines(&tensor, [&](uint64_t line) {
        lines.push_back(line);
    }));
    EXPECT_EQ(lines, (std::vector<uint64_t>{0x1000}));
    tensor.strides[0] = UINT64_MAX;
    EXPECT_FALSE(scheduler_args_dump_tensor_lines(&tensor, [&](uint64_t line) {
        lines.push_back(line);
    }));
    tensor.strides[0] = 1;
    tensor.buffer.addr = UINT64_MAX - 1;
    EXPECT_FALSE(scheduler_args_dump_tensor_lines(&tensor, [&](uint64_t line) {
        lines.push_back(line);
    }));
}

TEST(ResidentArgsDump, EmptyRankedOutputTouchesNoLineEvenWithANullBuffer) {
    simpler::hbg::Tensor tensor{};
    tensor.dtype = DataType::FLOAT32;
    tensor.ndims = 2;
    tensor.shapes[0] = 5;
    tensor.shapes[1] = 0;
    tensor.strides[0] = 1;
    tensor.strides[1] = 1;
    size_t visits = 0;
    EXPECT_TRUE(scheduler_args_dump_tensor_lines(&tensor, [&](uint64_t) {
        ++visits;
    }));
    EXPECT_EQ(visits, 0);
    tensor.shapes[1] = 1;
    EXPECT_FALSE(scheduler_args_dump_tensor_lines(&tensor, [&](uint64_t) {
        ++visits;
    }));
    EXPECT_EQ(visits, 0);
    tensor.ndims = 0;
    EXPECT_FALSE(scheduler_args_dump_tensor_lines(&tensor, [&](uint64_t) {
        ++visits;
    }));
    EXPECT_EQ(visits, 0);
}

TEST(ResidentArgsDump, TaskLookupDoesNotAliasMixLaneSlotsOrIndependentTasks) {
    SchedulerArgsDumpLayout layout{};
    ASSERT_TRUE(scheduler_args_dump_plan_layout(5, 2, 3, true, &layout));
    alignas(64) std::array<uint8_t, 1024> image{};
    ASSERT_TRUE(scheduler_args_dump_init_image(image.data(), image.size(), layout, 47, DumpArgsLevel::PARTIAL));
    ASSERT_TRUE(scheduler_args_dump_select_task(image.data(), 1, UINT64_C(7)));
    ASSERT_TRUE(scheduler_args_dump_select_task(image.data(), 4, UINT64_C(2)));
    auto *mix = scheduler_args_dump_task(image.data(), 1);
    auto *independent = scheduler_args_dump_task(image.data(), 4);
    ASSERT_NE(mix, nullptr);
    ASSERT_NE(independent, nullptr);
    ASSERT_NE(mix, independent);
    EXPECT_EQ(mix->task_id, 1);
    EXPECT_EQ(mix->active_mask, 7);
    EXPECT_EQ(independent->task_id, 4);
    EXPECT_EQ(independent->active_mask, 2);
    EXPECT_EQ(scheduler_args_dump_task(image.data(), 0), nullptr);
    EXPECT_EQ(scheduler_args_dump_task(image.data(), 5), nullptr);
    EXPECT_EQ(scheduler_args_dump_task(image.data(), UINT64_C(1) << 40), nullptr);
    EXPECT_FALSE(scheduler_args_dump_select_task(image.data(), 1, UINT64_C(7)));
    EXPECT_FALSE(scheduler_args_dump_select_task(image.data(), 2, UINT64_C(1)));
    scheduler_gm_store(mix->aborted, UINT64_C(1));
    EXPECT_EQ(independent->aborted, 0);
}

TEST(ResidentArgsDump, DiagnosticTailCannotOverlapCompactSmOrOverflowTheChargedArena) {
    SchedulerArgsDumpLayout layout{};
    ASSERT_TRUE(scheduler_args_dump_plan_layout(3, 1, 2, true, &layout));
    uint64_t tail = 0;
    uint64_t arena_bytes = 0;
    ASSERT_TRUE(scheduler_args_dump_plan_arena(128, 321, layout, &tail, &arena_bytes));
    EXPECT_EQ(tail, 512);
    EXPECT_EQ(arena_bytes, 1088);
    EXPECT_GE(tail, 128 + 321);
    EXPECT_EQ(tail % 64, 0);
    EXPECT_FALSE(scheduler_args_dump_plan_arena(UINT64_MAX - 2, 4, layout, &tail, &arena_bytes));
    ASSERT_TRUE(scheduler_args_dump_plan_layout(3, 1, 2, false, &layout));
    ASSERT_TRUE(scheduler_args_dump_plan_arena(128, 321, layout, &tail, &arena_bytes));
    EXPECT_EQ(tail, 0);
    EXPECT_EQ(arena_bytes, 449);
}

TEST(ResidentArgsDump, MixCannotExecuteUntilEveryParticipantConsumesTheInputSeal) {
    SchedulerArgsDumpTask task{};
    task.task_id = UINT64_C(0xfedcba9876543210);
    task.active_mask = 7;
    constexpr uint64_t epoch = UINT64_C(0x9876543210000001);

    EXPECT_EQ(
        scheduler_args_dump_arrive(&task, ArgsDumpStage::BEFORE_DISPATCH, SubtaskSlot::AIC),
        SchedulerArgsDumpArrival::Waiting
    );
    EXPECT_FALSE(scheduler_args_dump_can_execute(&task));
    EXPECT_EQ(
        scheduler_args_dump_arrive(&task, ArgsDumpStage::BEFORE_DISPATCH, SubtaskSlot::AIV1),
        SchedulerArgsDumpArrival::Waiting
    );
    EXPECT_EQ(
        scheduler_args_dump_arrive(&task, ArgsDumpStage::BEFORE_DISPATCH, SubtaskSlot::AIV0),
        SchedulerArgsDumpArrival::Publisher
    );
    task.publisher_before = scheduler_args_dump_publisher(37, 9);

    SchedulerArgsDumpAck ack{};
    ack.run_epoch = epoch;
    ack.task_id = task.task_id;
    ack.stage = static_cast<uint64_t>(ArgsDumpStage::BEFORE_DISPATCH);
    ack.generation = 9;
    ack.tick = 1;
    ack.status = scheduler_args_dump_ack_commit(ack.generation, ack.stage, SchedulerArgsDumpStatus::InProgress);
    EXPECT_FALSE(scheduler_args_dump_depart(&task, ArgsDumpStage::BEFORE_DISPATCH, SubtaskSlot::AIC, ack, epoch));
    ack.tick = 1;
    ack.status = scheduler_args_dump_ack_commit(ack.generation, ack.stage, SchedulerArgsDumpStatus::Sealed);
    EXPECT_TRUE(scheduler_args_dump_depart(&task, ArgsDumpStage::BEFORE_DISPATCH, SubtaskSlot::AIC, ack, epoch));
    EXPECT_FALSE(scheduler_args_dump_can_execute(&task));
    EXPECT_TRUE(scheduler_args_dump_depart(&task, ArgsDumpStage::BEFORE_DISPATCH, SubtaskSlot::AIV0, ack, epoch));
    EXPECT_FALSE(scheduler_args_dump_can_execute(&task));
    EXPECT_TRUE(scheduler_args_dump_depart(&task, ArgsDumpStage::BEFORE_DISPATCH, SubtaskSlot::AIV1, ack, epoch));
    EXPECT_TRUE(scheduler_args_dump_can_execute(&task));
    EXPECT_FALSE(scheduler_args_dump_can_complete(&task));
}

TEST(ResidentArgsDump, DepartedFollowerFinishesAfterPublisherReusesItsAcknowledgementForAnotherTask) {
    SchedulerArgsDumpTask task{};
    task.task_id = UINT64_C(0x100000001);
    task.active_mask = 3;
    task.publisher_after = scheduler_args_dump_publisher(2, 19);
    ASSERT_EQ(
        scheduler_args_dump_arrive(&task, ArgsDumpStage::AFTER_COMPLETION, SubtaskSlot::AIC),
        SchedulerArgsDumpArrival::Waiting
    );
    ASSERT_EQ(
        scheduler_args_dump_arrive(&task, ArgsDumpStage::AFTER_COMPLETION, SubtaskSlot::AIV0),
        SchedulerArgsDumpArrival::Publisher
    );
    SchedulerArgsDumpAck worker_ack{};
    worker_ack.run_epoch = UINT64_C(0x100000005);
    worker_ack.task_id = task.task_id;
    worker_ack.stage = static_cast<uint64_t>(ArgsDumpStage::AFTER_COMPLETION);
    worker_ack.generation = 19;
    worker_ack.tick = 1;
    worker_ack.status =
        scheduler_args_dump_ack_commit(worker_ack.generation, worker_ack.stage, SchedulerArgsDumpStatus::Sealed);
    bool follower_departed = false;
    bool publisher_departed = false;
    EXPECT_EQ(
        scheduler_args_dump_poll_phase(
            &task, ArgsDumpStage::AFTER_COMPLETION, SubtaskSlot::AIC, worker_ack, UINT64_C(0x100000005),
            &follower_departed
        ),
        SchedulerArgsDumpPoll::Waiting
    );
    ASSERT_TRUE(follower_departed);
    EXPECT_EQ(
        scheduler_args_dump_poll_phase(
            &task, ArgsDumpStage::AFTER_COMPLETION, SubtaskSlot::AIV0, worker_ack, UINT64_C(0x100000005),
            &publisher_departed
        ),
        SchedulerArgsDumpPoll::Consumed
    );
    // The follower resumes after its publisher completes and services an
    // independent queued task using that worker's same acknowledgement line.
    worker_ack.task_id = UINT64_C(0x100000002);
    worker_ack.stage = static_cast<uint64_t>(ArgsDumpStage::BEFORE_DISPATCH);
    ++worker_ack.generation;
    EXPECT_EQ(
        scheduler_args_dump_poll_phase(
            &task, ArgsDumpStage::AFTER_COMPLETION, SubtaskSlot::AIC, worker_ack, UINT64_C(0x100000005),
            &follower_departed
        ),
        SchedulerArgsDumpPoll::Consumed
    );
    worker_ack.task_id = task.task_id;
    worker_ack.stage = static_cast<uint64_t>(ArgsDumpStage::AFTER_COMPLETION);
    worker_ack.generation = 19;
    worker_ack.tick = 1;
    worker_ack.status =
        scheduler_args_dump_ack_commit(worker_ack.generation, worker_ack.stage, SchedulerArgsDumpStatus::Aborted);
    EXPECT_EQ(
        scheduler_args_dump_poll_phase(
            &task, ArgsDumpStage::AFTER_COMPLETION, SubtaskSlot::AIC, worker_ack, UINT64_C(0x100000005),
            &follower_departed
        ),
        SchedulerArgsDumpPoll::Aborted
    );
    EXPECT_EQ(scheduler_gm_query(task.aborted), 1);
    worker_ack.task_id = UINT64_C(0x100000002);
    scheduler_gm_store(task.aborted, UINT64_C(1));
    EXPECT_EQ(
        scheduler_args_dump_poll_phase(
            &task, ArgsDumpStage::AFTER_COMPLETION, SubtaskSlot::AIC, worker_ack, UINT64_C(0x100000005),
            &follower_departed
        ),
        SchedulerArgsDumpPoll::Waiting
    );
}

TEST(ResidentArgsDump, SealMustMatchTheFullRunTaskStageAndPublisherGeneration) {
    SchedulerArgsDumpTask task{};
    task.task_id = UINT64_C(0xfedcba9876543210);
    task.active_mask = 1;
    task.publisher_before = scheduler_args_dump_publisher(37, 9);
    constexpr uint64_t epoch = UINT64_C(0x9876543210000001);
    ASSERT_EQ(
        scheduler_args_dump_arrive(&task, ArgsDumpStage::BEFORE_DISPATCH, SubtaskSlot::AIC),
        SchedulerArgsDumpArrival::Publisher
    );
    SchedulerArgsDumpAck good{};
    good.run_epoch = epoch;
    good.task_id = task.task_id;
    good.stage = static_cast<uint64_t>(ArgsDumpStage::BEFORE_DISPATCH);
    good.generation = 9;
    good.tick = 1;
    good.status = scheduler_args_dump_ack_commit(good.generation, good.stage, SchedulerArgsDumpStatus::Sealed);

    auto ack = good;
    ack.run_epoch ^= UINT64_C(1) << 40;
    EXPECT_FALSE(scheduler_args_dump_depart(&task, ArgsDumpStage::BEFORE_DISPATCH, SubtaskSlot::AIC, ack, epoch));
    ack = good;
    ack.task_id ^= UINT64_C(1) << 40;
    EXPECT_FALSE(scheduler_args_dump_depart(&task, ArgsDumpStage::BEFORE_DISPATCH, SubtaskSlot::AIC, ack, epoch));
    ack = good;
    ack.stage = static_cast<uint64_t>(ArgsDumpStage::AFTER_COMPLETION);
    EXPECT_FALSE(scheduler_args_dump_depart(&task, ArgsDumpStage::BEFORE_DISPATCH, SubtaskSlot::AIC, ack, epoch));
    ack = good;
    ++ack.generation;
    EXPECT_FALSE(scheduler_args_dump_depart(&task, ArgsDumpStage::BEFORE_DISPATCH, SubtaskSlot::AIC, ack, epoch));
    EXPECT_FALSE(scheduler_args_dump_can_execute(&task));
    EXPECT_TRUE(scheduler_args_dump_depart(&task, ArgsDumpStage::BEFORE_DISPATCH, SubtaskSlot::AIC, good, epoch));
    EXPECT_TRUE(scheduler_args_dump_can_execute(&task));
}

TEST(ResidentArgsDump, ConcurrentMixArrivalsElectOnePublisherWithoutReinitializingPeers) {
    SchedulerArgsDumpTask task{};
    task.active_mask = 7;
    std::atomic<bool> start{false};
    std::array<SchedulerArgsDumpArrival, 3> results{};
    std::array<std::thread, 3> peers;
    for (uint32_t index = 0; index < peers.size(); ++index) {
        peers[index] = std::thread([&, index] {
            while (!start.load(std::memory_order_acquire)) {}
            results[index] =
                scheduler_args_dump_arrive(&task, ArgsDumpStage::BEFORE_DISPATCH, static_cast<SubtaskSlot>(index));
        });
    }
    start.store(true, std::memory_order_release);
    for (auto &peer : peers)
        peer.join();
    EXPECT_EQ(std::count(results.begin(), results.end(), SchedulerArgsDumpArrival::Publisher), 1);
    EXPECT_EQ(std::count(results.begin(), results.end(), SchedulerArgsDumpArrival::Waiting), 2);
    EXPECT_FALSE(scheduler_args_dump_can_execute(&task));
}

TEST(ResidentArgsDump, OutputSealCannotPublishCompletionUntilAllParticipantsDepart) {
    SchedulerArgsDumpTask task{};
    task.task_id = 17;
    task.active_mask = 3;
    task.publisher_before = scheduler_args_dump_publisher(37, 9);
    ASSERT_EQ(
        scheduler_args_dump_arrive(&task, ArgsDumpStage::BEFORE_DISPATCH, SubtaskSlot::AIC),
        SchedulerArgsDumpArrival::Waiting
    );
    ASSERT_EQ(
        scheduler_args_dump_arrive(&task, ArgsDumpStage::BEFORE_DISPATCH, SubtaskSlot::AIV0),
        SchedulerArgsDumpArrival::Publisher
    );
    SchedulerArgsDumpAck ack{};
    ack.run_epoch = 42;
    ack.task_id = 17;
    ack.stage = static_cast<uint64_t>(ArgsDumpStage::BEFORE_DISPATCH);
    ack.generation = 9;
    ack.tick = 1;
    ack.status = scheduler_args_dump_ack_commit(ack.generation, ack.stage, SchedulerArgsDumpStatus::Sealed);
    ASSERT_TRUE(scheduler_args_dump_depart(&task, ArgsDumpStage::BEFORE_DISPATCH, SubtaskSlot::AIC, ack, 42));
    ASSERT_TRUE(scheduler_args_dump_depart(&task, ArgsDumpStage::BEFORE_DISPATCH, SubtaskSlot::AIV0, ack, 42));
    ASSERT_TRUE(scheduler_args_dump_can_execute(&task));
    EXPECT_EQ(
        scheduler_args_dump_arrive(&task, ArgsDumpStage::AFTER_COMPLETION, SubtaskSlot::AIV0),
        SchedulerArgsDumpArrival::Waiting
    );
    EXPECT_EQ(
        scheduler_args_dump_arrive(&task, ArgsDumpStage::AFTER_COMPLETION, SubtaskSlot::AIC),
        SchedulerArgsDumpArrival::Publisher
    );
    task.publisher_after = scheduler_args_dump_publisher(2, 63);
    EXPECT_FALSE(scheduler_args_dump_depart(&task, ArgsDumpStage::AFTER_COMPLETION, SubtaskSlot::AIC, ack, 42));
    ack.stage = static_cast<uint64_t>(ArgsDumpStage::AFTER_COMPLETION);
    ack.generation = 63;
    ack.status = scheduler_args_dump_ack_commit(ack.generation, ack.stage, SchedulerArgsDumpStatus::Sealed);
    ASSERT_TRUE(scheduler_args_dump_depart(&task, ArgsDumpStage::AFTER_COMPLETION, SubtaskSlot::AIC, ack, 42));
    EXPECT_FALSE(scheduler_args_dump_can_complete(&task));
    ASSERT_TRUE(scheduler_args_dump_depart(&task, ArgsDumpStage::AFTER_COMPLETION, SubtaskSlot::AIV0, ack, 42));
    EXPECT_TRUE(scheduler_args_dump_can_complete(&task));
    scheduler_gm_store(task.aborted, UINT64_C(1));
    EXPECT_FALSE(scheduler_args_dump_can_execute(&task));
    EXPECT_FALSE(scheduler_args_dump_can_complete(&task));
}

TEST(ResidentArgsDump, ASealDoesNotStandInForMissingParticipants) {
    SchedulerArgsDumpTask task{};
    task.task_id = 17;
    task.active_mask = 3;
    task.publisher_before = scheduler_args_dump_publisher(37, 9);
    SchedulerArgsDumpAck ack{};
    ack.run_epoch = 42;
    ack.task_id = 17;
    ack.stage = static_cast<uint64_t>(ArgsDumpStage::BEFORE_DISPATCH);
    ack.generation = 9;
    ack.tick = 1;
    ack.status = scheduler_args_dump_ack_commit(ack.generation, ack.stage, SchedulerArgsDumpStatus::Sealed);
    ASSERT_EQ(
        scheduler_args_dump_arrive(&task, ArgsDumpStage::BEFORE_DISPATCH, SubtaskSlot::AIC),
        SchedulerArgsDumpArrival::Waiting
    );
    EXPECT_FALSE(scheduler_args_dump_depart(&task, ArgsDumpStage::BEFORE_DISPATCH, SubtaskSlot::AIC, ack, 42));
    EXPECT_FALSE(scheduler_args_dump_can_execute(&task));
}

}  // namespace
