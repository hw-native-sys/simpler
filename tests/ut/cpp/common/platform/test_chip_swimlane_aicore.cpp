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

#include "inner_kernel.h"
#undef OUT_OF_ORDER_STORE_BARRIER
#include "aicore/chip_swimlane_collector_aicore.h"

#include <gtest/gtest.h>

#include <cstdint>

TEST(ChipSwimlaneAicoreTest, CommitUsesReservedBufferGeneration) {
    ChipSwimlaneAicoreTaskBuffer first{};
    ChipSwimlaneAicoreTaskBuffer second{};
    ChipSwimlaneActiveHead head{};
    head.current_buf_ptr = reinterpret_cast<uint64_t>(&first);
    head.current_buf_seq = 0;

    ChipSwimlaneAicoreLocalState local{};
    local.cached_buf_seq = UINT32_MAX;

    ChipSwimlaneAicoreTaskRecord *reserved = chip_swimlane_aicore_reserve_task_record(&head, &local);
    ASSERT_EQ(reserved, &first.records[0]);

    head.current_buf_ptr = reinterpret_cast<uint64_t>(&second);
    head.current_buf_seq = 1;

    chip_swimlane_aicore_commit_task_record(reserved, 0x1234, 17, 100, 120, 180);

    EXPECT_EQ(first.records[0].task_token_raw, 0x1234u);
    EXPECT_EQ(first.records[0].reg_task_id, 17u);
    EXPECT_EQ(first.records[0].start_time, 120u);
    EXPECT_EQ(first.records[0].end_time, 180u);
    EXPECT_EQ(first.records[0].receive_to_start_cycles, 20u);
    EXPECT_EQ(second.records[0].task_token_raw, 0u);

    ChipSwimlaneAicoreTaskRecord *next = chip_swimlane_aicore_reserve_task_record(&head, &local);
    EXPECT_EQ(next, &second.records[0]);
}

// A record is attributable by *residency*: it belongs to whichever buffer it
// was reserved from, so a consumer reads its run from that buffer's stamp
// rather than from whatever the head currently advertises. This is what makes
// rotation across a run boundary safe, since AICore cannot read the AICPU SO's
// epoch and therefore never knows the "current" run at all.
//
// Scope note: this asserts the residency rule using the real AICore reserve /
// commit functions. It does NOT prove that AICPU stamped the right epoch —
// nothing here runs AICPU's prime/rotate, so the epochs below are fixtures.
// The stamping itself is covered on device.
TEST(ChipSwimlaneAicoreTest, RecordIsAttributableToTheBufferItWasReservedFrom) {
    ChipSwimlaneAicoreTaskBuffer first{};
    ChipSwimlaneAicoreTaskBuffer second{};
    // Stand in for what AICPU's prime / aicore_rotate write before publishing.
    first.run_epoch = 11;
    first.local_seq = 0;
    second.run_epoch = 22;
    second.local_seq = 1;

    ChipSwimlaneActiveHead head{};
    head.current_buf_ptr = reinterpret_cast<uint64_t>(&first);
    head.current_buf_seq = 0;

    ChipSwimlaneAicoreLocalState local{};
    local.cached_buf_seq = UINT32_MAX;

    ChipSwimlaneAicoreTaskRecord *reserved = chip_swimlane_aicore_reserve_task_record(&head, &local);
    ASSERT_EQ(reserved, &first.records[0]);

    // Rotation to the next run's buffer happens between reserve and commit.
    head.current_buf_ptr = reinterpret_cast<uint64_t>(&second);
    head.current_buf_seq = 1;

    chip_swimlane_aicore_commit_task_record(reserved, 0x1234, 17, 100, 120, 180);

    // Resolve the record's run the way a consumer does — from the buffer the
    // record actually landed in — and check that this is the pre-rotation one.
    const ChipSwimlaneAicoreTaskBuffer *owner = (reserved == &first.records[0]) ? &first : &second;
    EXPECT_EQ(owner, &first) << "a mid-task rotation moved the record to the successor's buffer";
    EXPECT_EQ(owner->run_epoch, 11u);
    EXPECT_EQ(first.records[0].reg_task_id, 17u);
    EXPECT_EQ(second.records[0].reg_task_id, 0u) << "the commit leaked into the successor's buffer";

    // The next reserve follows the head into the successor's buffer, and that
    // record resolves to the successor's run.
    ChipSwimlaneAicoreTaskRecord *next = chip_swimlane_aicore_reserve_task_record(&head, &local);
    ASSERT_EQ(next, &second.records[0]);
    chip_swimlane_aicore_commit_task_record(next, 0x5678, 18, 200, 220, 280);
    const ChipSwimlaneAicoreTaskBuffer *next_owner = (next == &first.records[0]) ? &first : &second;
    EXPECT_EQ(next_owner, &second);
    EXPECT_EQ(next_owner->run_epoch, 22u);
}

// `current_buf_seq` is the rotation generation AICore latches against, and
// nothing else. Making it a cross-run monotonic counter — an idea the §2.4
// matrix floated and withdrew — would break this latch, so pin the contract:
// a reload happens exactly when the head's generation differs from the cached
// one, and identity plays no part in that decision.
TEST(ChipSwimlaneAicoreTest, RotationGenerationAloneDrivesTheBufferReload) {
    ChipSwimlaneAicoreTaskBuffer first{};
    ChipSwimlaneAicoreTaskBuffer second{};
    first.run_epoch = 7;
    second.run_epoch = 7;  // same run: identity cannot be what triggers a reload

    ChipSwimlaneActiveHead head{};
    head.current_buf_ptr = reinterpret_cast<uint64_t>(&first);
    head.current_buf_seq = 4;

    ChipSwimlaneAicoreLocalState local{};
    local.cached_buf_seq = UINT32_MAX;

    // First call always reloads: the sentinel cannot match any real generation.
    ASSERT_EQ(chip_swimlane_aicore_reserve_task_record(&head, &local), &first.records[0]);
    EXPECT_EQ(local.cached_buf_seq, 4u);

    // Same generation, different pointer: no reload, so the stale buffer is
    // still the one written. This is why AICPU must bump the generation last.
    head.current_buf_ptr = reinterpret_cast<uint64_t>(&second);
    EXPECT_EQ(chip_swimlane_aicore_reserve_task_record(&head, &local), &first.records[1]);

    // Generation change is what moves AICore to the new buffer.
    head.current_buf_seq = 5;
    EXPECT_EQ(chip_swimlane_aicore_reserve_task_record(&head, &local), &second.records[0]);
    EXPECT_EQ(local.cached_buf_seq, 5u);
}
