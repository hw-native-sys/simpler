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

#include <cstdlib>

#include "common/chip_swimlane_extension.h"
#include "common/chip_swimlane_profiling.h"
#include "aicpu/chip_swimlane_collector_aicpu.h"
#include "aicpu/device_run_result_base_aicpu.h"
#include "host/chip_swimlane_collector.h"

TEST(ChipSwimlaneCollectorTest, BeginRunReleasesRuntimeExtensionSlots) {
    ChipSwimlaneCollector collector;

    collector.begin_run("first", ChipSwimlaneLevel::TASK_TIMING);
    ASSERT_TRUE(collector.set_json_extension(ChipSwimlaneExtensionSection::AicoreTasks, "[]"));
    EXPECT_FALSE(collector.set_json_extension(ChipSwimlaneExtensionSection::AicoreTasks, "[]"));

    collector.begin_run("second", ChipSwimlaneLevel::TASK_TIMING);
    EXPECT_TRUE(collector.set_json_extension(ChipSwimlaneExtensionSection::AicoreTasks, "[]"));
}

namespace {

void *swimlane_test_alloc(size_t size) { return std::calloc(1, size); }

int swimlane_test_free(void *ptr) {
    std::free(ptr);
    return 0;
}

}  // namespace

// Identity reaches the host copy, and stays with it once the device buffer is
// handed back and re-stamped by a later run. Both halves are production code:
// the buffer layout is the real one, and the copy runs through the collector's
// own `on_buffer_collected`.
//
// This does not assert that AICPU stamped the right epoch — the stamp here is a
// fixture. That part is covered on device.
TEST(ChipSwimlaneCollectorTest, CollectedRecordsCarryTheirRunAfterTheBufferIsReused) {
    ChipSwimlaneCollector collector;
    ASSERT_EQ(
        collector.initialize(
            /*num_aicore=*/1, /*aicpu_thread_num=*/1, /*device_id=*/0, ChipSwimlaneLevel::TASK_TIMING,
            swimlane_test_alloc, nullptr, swimlane_test_free
        ),
        0
    );
    collector.begin_run("identity", ChipSwimlaneLevel::TASK_TIMING);

    // One AICore buffer, stamped the way prime / aicore_rotate stamp it.
    ChipSwimlaneAicoreTaskBuffer buf{};
    buf.count = 1;
    buf.run_epoch = 91;
    buf.local_seq = 3;
    buf.records[0].start_time = 1000;  // non-zero: the start_time==0 filter keeps it
    buf.records[0].end_time = 1200;
    buf.records[0].reg_task_id = 5;
    buf.records[0].task_token_raw = 0xabc;

    ReadyBufferInfo info{};
    info.type = ProfBufferType::AICORE_TASK;
    info.index = 0;
    info.dev_buffer_ptr = &buf;
    info.host_buffer_ptr = &buf;
    info.buffer_seq = 3;
    collector.on_buffer_collected(info, /*collector_shard=*/0);

    // Re-stamp the same storage as a later run would after it was returned.
    buf.run_epoch = 92;
    buf.local_seq = 0;
    buf.records[0].reg_task_id = 99;

    const auto records = collector.collected_aicore_records_for_test();
    ASSERT_EQ(records.size(), 1u);
    ASSERT_EQ(records[0].size(), 1u);
    EXPECT_EQ(records[0][0].run_epoch, 91u) << "the host copy followed the device buffer's re-stamp";
    EXPECT_EQ(records[0][0].local_seq, 3u);
    EXPECT_EQ(records[0][0].record.reg_task_id, 5u) << "the host copy aliased the device buffer instead of copying";

    collector.finalize(nullptr, swimlane_test_free);
}

// The first buffer of the AICPU task pool is hand-popped by
// chip_swimlane_aicpu_init rather than handed out by the engine, so it has to
// stamp identity itself. A run short enough never to rotate publishes only out
// of that buffer, which makes this the path an identity test has to drive —
// a hand-filled buffer cannot see a missing stamp there.
//
// Drives the production sequence: init (primes the buffer) -> complete_task
// (writes a record) -> flush (publishes it) -> the collector's own copy.
TEST(ChipSwimlaneCollectorTest, InitStampsTheFirstBufferOfAShortRun) {
    ChipSwimlaneCollector collector;
    ASSERT_EQ(
        collector.initialize(
            /*num_aicore=*/1, /*aicpu_thread_num=*/1, /*device_id=*/0, ChipSwimlaneLevel::SCHEDULE_TIMING,
            swimlane_test_alloc, nullptr, swimlane_test_free
        ),
        0
    );
    void *shm = collector.get_chip_swimlane_setup_device_ptr();
    ASSERT_NE(shm, nullptr);

    // Poison the buffer the prime will hand out, standing in for a pooled
    // buffer a previous run already stamped. A prime that fails to stamp leaves
    // this value in place, which is the mis-attribution the fix prevents.
    auto *pool = get_perf_buffer_state(shm, 0);
    const uint64_t queued = pool->free_queue.buffer_ptrs[pool->free_queue.head % PLATFORM_PROF_SLOT_COUNT];
    ASSERT_NE(queued, 0u);
    reinterpret_cast<ChipSwimlaneAicpuTaskBuffer *>(queued)->run_epoch = 4242;

    collector.begin_run("short-run", ChipSwimlaneLevel::SCHEDULE_TIMING);

    constexpr uint64_t kEpoch = 77;
    set_platform_run_result(/*region_base=*/0, kEpoch);
    set_chip_swimlane_enabled(true);
    set_platform_chip_swimlane_base(reinterpret_cast<uint64_t>(shm));
    set_platform_chip_swimlane_aicore_rotation_table(0);
    chip_swimlane_aicpu_init(/*worker_count=*/1);

    // One record, far short of a rotation boundary.
    ASSERT_EQ(
        chip_swimlane_aicpu_complete_task(
            /*core_id=*/0, /*thread_idx=*/0, /*reg_task_id=*/1, /*dispatch_time=*/100, /*finish_time=*/200
        ),
        0
    );
    const int cores[] = {0};
    chip_swimlane_aicpu_flush(/*thread_idx=*/0, cores, /*core_num=*/1);

    // Collect what the flush published, through the production copy path.
    auto *header = get_chip_swimlane_header(shm);
    const uint32_t tail = header->queue_tails[0];
    ASSERT_GT(tail, 0u) << "flush published nothing for the short run";
    size_t task_records = 0;
    for (uint32_t i = 0; i < tail; i++) {
        const ReadyQueueEntry &entry = header->queues[0][i];
        if (entry.kind != ChipSwimlaneBufferKind::AicpuTask) continue;
        ReadyBufferInfo info{};
        info.type = ProfBufferType::AICPU_TASK;
        info.index = entry.core_index;
        info.dev_buffer_ptr = reinterpret_cast<void *>(entry.buffer_ptr);
        info.host_buffer_ptr = reinterpret_cast<void *>(entry.buffer_ptr);
        info.buffer_seq = entry.buffer_seq;
        collector.on_buffer_collected(info, /*collector_shard=*/0);
        task_records++;
    }
    ASSERT_GT(task_records, 0u) << "the AICPU task buffer never reached the ready queue";

    const auto &collected = collector.collected_perf_records_for_test();
    ASSERT_FALSE(collected.empty());
    ASSERT_EQ(collected[0].size(), 1u);
    EXPECT_EQ(collected[0][0].record.reg_task_id, 1u);
    EXPECT_EQ(collected[0][0].run_epoch, kEpoch)
        << "init's hand-popped first buffer kept a stale or unset run identity";

    collector.finalize(nullptr, swimlane_test_free);
}
