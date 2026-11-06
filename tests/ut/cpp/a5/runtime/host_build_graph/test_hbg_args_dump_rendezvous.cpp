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

#include "scheduler/scheduler_args_dump.h"

namespace {
SchedulerArgsDumpAck sealed(const SchedulerArgsDumpTask &task, ArgsDumpStage stage, uint32_t generation) {
    SchedulerArgsDumpAck ack{11, task.task_ref, static_cast<uint64_t>(stage), generation, 1, 0, 0, 0};
    ack.status = scheduler_args_dump_ack_commit(generation, ack.stage, SchedulerArgsDumpStatus::Sealed);
    return ack;
}

TEST(ResidentArgsDumpBoundary, TaskLatchesRemainValidAcrossBlockWavesAndReusedWorkerAck) {
    SchedulerArgsDumpTask task{};
    task.task_ref = 10;
    task.expected_participants = 3;
    task.active_mask = 1;
    EXPECT_EQ(scheduler_args_dump_enter(&task), SchedulerArgsDumpArrival::Publisher);
    EXPECT_EQ(scheduler_args_dump_enter(&task), SchedulerArgsDumpArrival::Waiting);
    EXPECT_FALSE(scheduler_args_dump_can_execute(&task));
    task.publisher_before = scheduler_args_dump_publisher(0, 7);
    auto ack = sealed(task, ArgsDumpStage::BEFORE_DISPATCH, 7);
    auto wrong = ack;
    wrong.run_epoch = 12;
    EXPECT_FALSE(scheduler_args_dump_seal(&task, ArgsDumpStage::BEFORE_DISPATCH, wrong, 11));
    ASSERT_TRUE(scheduler_args_dump_seal(&task, ArgsDumpStage::BEFORE_DISPATCH, ack, 11));
    // The publisher can finish and reuse its worker before later blocks enter.
    ack = {};
    EXPECT_EQ(
        scheduler_args_dump_return(
            &task,
            [] {
                return true;
            }
        ),
        SchedulerArgsDumpArrival::Waiting
    );
    EXPECT_EQ(scheduler_args_dump_enter(&task), SchedulerArgsDumpArrival::Ready);
    EXPECT_EQ(
        scheduler_args_dump_return(
            &task,
            [] {
                return true;
            }
        ),
        SchedulerArgsDumpArrival::Waiting
    );
    EXPECT_EQ(scheduler_args_dump_enter(&task), SchedulerArgsDumpArrival::Ready);
    EXPECT_FALSE(scheduler_args_dump_can_complete(&task));
    EXPECT_EQ(
        scheduler_args_dump_return(
            &task,
            [] {
                return true;
            }
        ),
        SchedulerArgsDumpArrival::Publisher
    );
    EXPECT_FALSE(scheduler_args_dump_can_complete(&task));
    task.publisher_after = scheduler_args_dump_publisher(1, 9);
    ASSERT_TRUE(scheduler_args_dump_seal(
        &task, ArgsDumpStage::AFTER_COMPLETION, sealed(task, ArgsDumpStage::AFTER_COMPLETION, 9), 11
    ));
    EXPECT_TRUE(scheduler_args_dump_can_complete(&task));
}

TEST(ResidentArgsDumpBoundary, MixFinalWriterCannotSealBeforeAllBlocksAndSubtasksReturn) {
    SchedulerArgsDumpTask task{};
    task.task_ref = 3;
    task.active_mask = 7;
    task.expected_participants = 9;
    ASSERT_EQ(scheduler_args_dump_enter(&task), SchedulerArgsDumpArrival::Publisher);
    task.publisher_before = scheduler_args_dump_publisher(0, 1);
    ASSERT_TRUE(scheduler_args_dump_seal(
        &task, ArgsDumpStage::BEFORE_DISPATCH, sealed(task, ArgsDumpStage::BEFORE_DISPATCH, 1), 11
    ));
    std::array<int, 9> writes{};
    for (size_t i = 0; i < writes.size(); ++i) {
        writes[i] = static_cast<int>(i + 1);
        const auto arrived = scheduler_args_dump_return(&task, [&] {
            EXPECT_EQ(scheduler_gm_query(task.returned_participants), i);
            EXPECT_EQ(writes[i], static_cast<int>(i + 1));
            return true;
        });
        if (i + 1 != writes.size()) {
            EXPECT_EQ(arrived, SchedulerArgsDumpArrival::Waiting);
            EXPECT_FALSE(scheduler_args_dump_can_complete(&task));
        } else {
            EXPECT_EQ(arrived, SchedulerArgsDumpArrival::Publisher);
            EXPECT_EQ(writes, (std::array<int, 9>{1, 2, 3, 4, 5, 6, 7, 8, 9}));
        }
    }
    task.publisher_after = scheduler_args_dump_publisher(2, 3);
    ASSERT_TRUE(scheduler_args_dump_seal(
        &task, ArgsDumpStage::AFTER_COMPLETION, sealed(task, ArgsDumpStage::AFTER_COMPLETION, 3), 11
    ));
    EXPECT_TRUE(scheduler_args_dump_can_complete(&task));
    EXPECT_EQ(
        scheduler_args_dump_return(
            &task,
            [] {
                return true;
            }
        ),
        SchedulerArgsDumpArrival::Invalid
    );
    EXPECT_FALSE(scheduler_args_dump_can_complete(&task));
}

TEST(ResidentArgsDumpBoundary, FailedVisibilityPublicationCannotCountAWriterOrSeal) {
    SchedulerArgsDumpTask task{};
    task.task_ref = 7;
    task.active_mask = task.expected_participants = 1;
    ASSERT_EQ(scheduler_args_dump_enter(&task), SchedulerArgsDumpArrival::Publisher);
    task.publisher_before = scheduler_args_dump_publisher(0, 1);
    ASSERT_TRUE(scheduler_args_dump_seal(
        &task, ArgsDumpStage::BEFORE_DISPATCH, sealed(task, ArgsDumpStage::BEFORE_DISPATCH, 1), 11
    ));
    EXPECT_EQ(
        scheduler_args_dump_return(
            &task,
            [] {
                return false;
            }
        ),
        SchedulerArgsDumpArrival::Invalid
    );
    EXPECT_EQ(task.returned_participants, 0u);
    EXPECT_EQ(task.aborted, 1u);
    EXPECT_FALSE(scheduler_args_dump_can_complete(&task));
}

TEST(ResidentArgsDumpBoundary, AbortedOrEmptyTaskCannotAcquireOrPublishSuccess) {
    SchedulerArgsDumpTask task{};
    task.active_mask = 1;
    EXPECT_EQ(scheduler_args_dump_enter(&task), SchedulerArgsDumpArrival::Invalid);
    task.expected_participants = 1;
    task.aborted = 1;
    EXPECT_EQ(scheduler_args_dump_enter(&task), SchedulerArgsDumpArrival::Invalid);
    EXPECT_EQ(
        scheduler_args_dump_return(
            &task,
            [] {
                return true;
            }
        ),
        SchedulerArgsDumpArrival::Invalid
    );
    EXPECT_FALSE(scheduler_args_dump_can_complete(&task));
}
enum class AckMutation { None, TickAfterBody, ClearCommitAfterBody };
struct ChangingAckReader {
    SchedulerArgsDumpAck *ack;
    AckMutation mutation;
    uint64_t load(uint64_t &word) const {
        if (&word == &ack->byte_cursor) {
            if (mutation == AckMutation::TickAfterBody) ++ack->tick;
            if (mutation == AckMutation::ClearCommitAfterBody) ack->status = 0;
        }
        return word;
    }
    void barrier() const {}
};

TEST(ResidentArgsDumpBoundary, AckObservationRejectsSameCommitWithANewTickAndIncompletePublication) {
    for (AckMutation mutation : {AckMutation::None, AckMutation::TickAfterBody, AckMutation::ClearCommitAfterBody}) {
        SCOPED_TRACE(static_cast<int>(mutation));
        SchedulerArgsDumpAck ack{11, 3, 0, 7, 1, 2, 65536, 0};
        ack.status = scheduler_args_dump_ack_commit(7, 0, SchedulerArgsDumpStatus::InProgress);
        SchedulerArgsDumpAck observed{};
        const bool accepted = scheduler_args_dump_observe_ack(&ack, &observed, ChangingAckReader{&ack, mutation});
        EXPECT_EQ(accepted, mutation == AckMutation::None);
        if (accepted) {
            EXPECT_EQ(observed.tick, 1u);
            EXPECT_EQ(observed.arg_index, 2u);
            EXPECT_EQ(observed.byte_cursor, 65536u);
            EXPECT_EQ(scheduler_args_dump_ack_status(observed), SchedulerArgsDumpStatus::InProgress);
        } else {
            EXPECT_EQ(observed.tick, 0u);
            EXPECT_EQ(observed.status, 0u);
        }
    }
}
}  // namespace
