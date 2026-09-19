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
#include <vector>

#include "common/chip_swimlane_extension.h"
#include "common/chip_swimlane_profiling.h"
#include "common/memory_barrier.h"
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

// ---------------------------------------------------------------------------
// Buffer retention: a buffer is released only by a successful enqueue.
//
// AICPU is the free queue's consumer and never its producer, so it has no way
// to put a buffer back. Whenever it cannot hand one to the host — nothing to
// publish, or the ready queue full — the buffer has to stay this pool's, and
// the next run's init has to reuse it in place. Clearing `current_buf_ptr`
// instead strands the storage: the host never sees it, so nothing recycles it.
//
// Drives the production functions (init / complete_task / flush) against the
// real shared-memory layout.
// ---------------------------------------------------------------------------

namespace {

uint32_t free_queue_depth(void *shm, int core) {
    const auto *pool = get_perf_buffer_state(shm, core);
    return pool->free_queue.tail - pool->free_queue.head;
}

uint32_t aicore_free_queue_depth(void *shm, int core) {
    const auto *pool = get_aicore_buffer_state(shm, core);
    return pool->free_queue.tail - pool->free_queue.head;
}

bool free_queue_holds(const ChipSwimlaneFreeQueue &fq, uint64_t buf_ptr) {
    for (uint32_t i = fq.head; i != fq.tail; i++) {
        if (fq.buffer_ptrs[i % PLATFORM_PROF_SLOT_COUNT] == buf_ptr) return true;
    }
    return false;
}

// The same protocol for an AICore pool's free queue. Both pool kinds share the
// ChipSwimlaneFreeQueue layout and slot count, so this differs from the task-pool
// helper only in which queue it is handed.
void host_push_aicore_free_queue(ChipSwimlaneFreeQueue &fq, uint64_t buf_ptr) {
    ASSERT_LT(fq.tail - fq.head, static_cast<uint32_t>(PLATFORM_PROF_SLOT_COUNT)) << "free queue is full";
    fq.buffer_ptrs[fq.tail % PLATFORM_PROF_SLOT_COUNT] = buf_ptr;
    wmb();
    fq.tail = fq.tail + 1;
    wmb();
}

// The host's half of the SPSC protocol: write the slot, fence, then publish the
// tail. AICPU is the queue's consumer and has no push of its own, so a test
// that needs a buffer back in the pool has to play the host here.
void host_push_free_queue(ChipSwimlaneFreeQueue &fq, uint64_t buf_ptr) {
    ASSERT_LT(fq.tail - fq.head, static_cast<uint32_t>(PLATFORM_PROF_SLOT_COUNT)) << "free queue is full";
    fq.buffer_ptrs[fq.tail % PLATFORM_PROF_SLOT_COUNT] = buf_ptr;
    wmb();
    fq.tail = fq.tail + 1;
    wmb();
}

// One run's device-side sequence at the given level, with `dispatches` AICore
// dispatches and `completes` AICPU task records.
void run_once(void *shm, uint64_t epoch, ChipSwimlaneLevel level, int dispatches, int completes) {
    set_platform_run_result(/*region_base=*/0, epoch);
    set_chip_swimlane_enabled(true);
    set_platform_chip_swimlane_base(reinterpret_cast<uint64_t>(shm));
    set_platform_chip_swimlane_aicore_rotation_table(0);
    get_chip_swimlane_header(shm)->chip_swimlane_level = static_cast<uint32_t>(level);
    chip_swimlane_aicpu_init(/*worker_count=*/1);
    for (int i = 0; i < dispatches; i++) {
        chip_swimlane_aicpu_on_aicore_dispatch(/*core_id=*/0, /*thread_idx=*/0, static_cast<uint32_t>(i + 1));
    }
    for (int i = 0; i < completes; i++) {
        chip_swimlane_aicpu_complete_task(
            /*core_id=*/0, /*thread_idx=*/0, static_cast<uint32_t>(i + 1), /*dispatch_time=*/100,
            /*finish_time=*/200
        );
    }
    const int cores[] = {0};
    chip_swimlane_aicpu_flush(/*thread_idx=*/0, cores, /*core_num=*/1);
}

}  // namespace

TEST(ChipSwimlaneBufferReturnTest, IdleCoreDoesNotConsumeABufferPerRun) {
    ChipSwimlaneCollector collector;
    ASSERT_EQ(
        collector.initialize(
            /*num_aicore=*/1, /*aicpu_thread_num=*/1, /*device_id=*/0, ChipSwimlaneLevel::TASK_TIMING,
            swimlane_test_alloc, nullptr, swimlane_test_free
        ),
        0
    );
    void *shm = collector.get_chip_swimlane_setup_device_ptr();
    const uint32_t depth_at_start = free_queue_depth(shm, 0);
    const uint32_t ac_depth_at_start = aicore_free_queue_depth(shm, 0);
    ASSERT_GT(depth_at_start, 1u) << "need more than one free buffer for per-run consumption to be visible";
    ASSERT_GT(ac_depth_at_start, 1u) << "need more than one free AICore buffer for the same reason";

    // The first run legitimately draws one from each pool: they hold none yet.
    collector.begin_run("idle", ChipSwimlaneLevel::TASK_TIMING);
    run_once(shm, /*epoch=*/1, ChipSwimlaneLevel::TASK_TIMING, /*dispatches=*/0, /*completes=*/0);
    const uint32_t depth_after_first = free_queue_depth(shm, 0);
    const uint32_t ac_depth_after_first = aicore_free_queue_depth(shm, 0);
    ASSERT_EQ(depth_after_first, depth_at_start - 1);
    ASSERT_EQ(ac_depth_after_first, ac_depth_at_start - 1);

    // Every later run must reuse what the previous one retained. Drawing again
    // strands the retained buffer and shrinks the pool once per run, which is
    // what eventually starves rotation. The AICore pool is the one that starves
    // first: it is seeded with PLATFORM_AICORE_BUFFERS_PER_CORE buffers, and
    // AICore has no fallback to drop into when rotation finds none.
    for (uint64_t epoch = 2; epoch <= 4; epoch++) {
        collector.begin_run("idle", ChipSwimlaneLevel::TASK_TIMING);
        run_once(shm, epoch, ChipSwimlaneLevel::TASK_TIMING, /*dispatches=*/0, /*completes=*/0);
        EXPECT_EQ(free_queue_depth(shm, 0), depth_after_first)
            << "run " << epoch << " drew a task buffer instead of reusing the retained one";
        EXPECT_NE(get_perf_buffer_state(shm, 0)->head.current_buf_ptr, 0u)
            << "run " << epoch << " released a task buffer it never handed to the host";
        EXPECT_EQ(aicore_free_queue_depth(shm, 0), ac_depth_after_first)
            << "run " << epoch << " drew an AICore buffer instead of reusing the retained one";
        EXPECT_NE(get_aicore_buffer_state(shm, 0)->head.current_buf_ptr, 0u)
            << "run " << epoch << " released an AICore buffer it never handed to the host";
    }

    collector.finalize(nullptr, swimlane_test_free);
}

TEST(ChipSwimlaneBufferReturnTest, PartialAicoreTailStampsItsRealCountAtTaskTiming) {
    ChipSwimlaneCollector collector;
    ASSERT_EQ(
        collector.initialize(
            /*num_aicore=*/1, /*aicpu_thread_num=*/1, /*device_id=*/0, ChipSwimlaneLevel::TASK_TIMING,
            swimlane_test_alloc, nullptr, swimlane_test_free
        ),
        0
    );
    void *shm = collector.get_chip_swimlane_setup_device_ptr();
    collector.begin_run("partial", ChipSwimlaneLevel::TASK_TIMING);

    // Three AICore dispatches, far short of a rotation boundary. TASK_TIMING is
    // the level whose flush used to stamp full capacity instead.
    run_once(shm, /*epoch=*/9, ChipSwimlaneLevel::TASK_TIMING, /*dispatches=*/3, /*completes=*/0);

    auto *header = get_chip_swimlane_header(shm);
    bool saw_aicore = false;
    for (uint32_t i = 0; i < header->queue_tails[0]; i++) {
        const ReadyQueueEntry &entry = header->queues[0][i];
        if (entry.kind != ChipSwimlaneBufferKind::AicoreTask) continue;
        saw_aicore = true;
        const auto *buf = reinterpret_cast<const ChipSwimlaneAicoreTaskBuffer *>(entry.buffer_ptr);
        EXPECT_EQ(buf->count, 3u) << "the tail buffer was stamped with capacity rather than its record count";

        // A successful hand-over releases the pointer. Retention covers the
        // failed hand-over only; keeping this one would leave the buffer owned
        // by the device and the host at once.
        const auto *ac_state = get_aicore_buffer_state(shm, 0);
        EXPECT_EQ(ac_state->head.current_buf_ptr, 0u) << "the published AICore buffer is still the pool's active one";
        EXPECT_FALSE(free_queue_holds(ac_state->free_queue, entry.buffer_ptr))
            << "AICPU pushed the free queue it only consumes";
    }
    EXPECT_TRUE(saw_aicore) << "the AICore tail buffer never reached the ready queue";

    collector.finalize(nullptr, swimlane_test_free);
}

// A failed enqueue charges dropped by the buffer's real record count and keeps
// the buffer, so the next run's init can reuse it.
//
// The production cause is a full ready queue, but reaching it means sitting in
// the backpressure gate for PLATFORM_DFX_BACKPRESSURE_TIMEOUT_CYCLES — 30
// seconds of spinning, which a unit test has no business doing. The gate also
// rejects an out-of-range thread index outright, which fails the same
// `enqueue_ready` for the same caller and runs the same failure branch. So the
// branch is covered here; the gate's own timeout behaviour is not, and belongs
// to the engine's tests.
TEST(ChipSwimlaneBufferReturnTest, AFailedEnqueueChargesDroppedOnceAndKeepsTheBuffer) {
    ChipSwimlaneCollector collector;
    ASSERT_EQ(
        collector.initialize(
            /*num_aicore=*/1, /*aicpu_thread_num=*/1, /*device_id=*/0, ChipSwimlaneLevel::SCHEDULE_TIMING,
            swimlane_test_alloc, nullptr, swimlane_test_free
        ),
        0
    );
    void *shm = collector.get_chip_swimlane_setup_device_ptr();
    collector.begin_run("full", ChipSwimlaneLevel::SCHEDULE_TIMING);

    set_platform_run_result(/*region_base=*/0, /*run_epoch=*/5);
    set_chip_swimlane_enabled(true);
    set_platform_chip_swimlane_base(reinterpret_cast<uint64_t>(shm));
    set_platform_chip_swimlane_aicore_rotation_table(0);
    auto *header = get_chip_swimlane_header(shm);
    header->chip_swimlane_level = static_cast<uint32_t>(ChipSwimlaneLevel::SCHEDULE_TIMING);
    chip_swimlane_aicpu_init(/*worker_count=*/1);

    auto *pool = get_perf_buffer_state(shm, 0);
    const uint64_t held = pool->head.current_buf_ptr;
    ASSERT_NE(held, 0u);
    const uint32_t depth_after_init = free_queue_depth(shm, 0);

    ASSERT_EQ(
        chip_swimlane_aicpu_complete_task(
            /*core_id=*/0, /*thread_idx=*/0, /*reg_task_id=*/1, /*dispatch_time=*/10, /*finish_time=*/20
        ),
        0
    );
    ASSERT_EQ(reinterpret_cast<const ChipSwimlaneAicpuTaskBuffer *>(held)->count, 1u);

    // Make the enqueue fail without entering the backpressure gate.
    const uint32_t dropped_before = pool->head.dropped_record_count;
    const int cores[] = {0};
    chip_swimlane_aicpu_flush(/*thread_idx=*/PLATFORM_MAX_AICPU_THREADS, cores, /*core_num=*/1);

    EXPECT_EQ(pool->head.dropped_record_count, dropped_before + 1)
        << "the failed enqueue charged dropped by something other than the buffer's record count";
    EXPECT_EQ(pool->head.current_buf_ptr, held)
        << "a buffer the host never received was released, so nothing can return it";
    EXPECT_EQ(free_queue_depth(shm, 0), depth_after_init) << "the failed flush drew from the free queue";

    // The retained buffer is reusable: the next run's init takes it in place
    // and re-stamps it rather than drawing a replacement.
    collector.begin_run("full-next", ChipSwimlaneLevel::SCHEDULE_TIMING);
    set_platform_run_result(/*region_base=*/0, /*run_epoch=*/6);
    header->chip_swimlane_level = static_cast<uint32_t>(ChipSwimlaneLevel::SCHEDULE_TIMING);
    chip_swimlane_aicpu_init(/*worker_count=*/1);
    EXPECT_EQ(pool->head.current_buf_ptr, held) << "init drew a replacement instead of reusing the retained buffer";
    const auto *reused = reinterpret_cast<const ChipSwimlaneAicpuTaskBuffer *>(held);
    EXPECT_EQ(reused->count, 0u);
    EXPECT_EQ(reused->run_epoch, 6u) << "a reused buffer kept the previous run's identity";

    collector.finalize(nullptr, swimlane_test_free);
}

// The AICore half of the same rule. It is a separate branch in the same flush,
// with its own accounting: the task pool zeroes `count` when it charges dropped,
// while the AICore path charges the mark it stamped and leaves `count` for the
// next init to reset. Both must keep the buffer.
//
// Three dispatches and no completions, so the task pool has nothing to publish
// and the AICore enqueue is the only one attempted. The enqueue is failed by an
// out-of-range thread index for the reason given on the task-pool test above.
TEST(ChipSwimlaneBufferReturnTest, AFailedAicoreEnqueueChargesDroppedOnceAndKeepsTheBuffer) {
    ChipSwimlaneCollector collector;
    ASSERT_EQ(
        collector.initialize(
            /*num_aicore=*/1, /*aicpu_thread_num=*/1, /*device_id=*/0, ChipSwimlaneLevel::TASK_TIMING,
            swimlane_test_alloc, nullptr, swimlane_test_free
        ),
        0
    );
    void *shm = collector.get_chip_swimlane_setup_device_ptr();
    collector.begin_run("aicore-full", ChipSwimlaneLevel::TASK_TIMING);

    set_platform_run_result(/*region_base=*/0, /*run_epoch=*/11);
    set_chip_swimlane_enabled(true);
    set_platform_chip_swimlane_base(reinterpret_cast<uint64_t>(shm));
    set_platform_chip_swimlane_aicore_rotation_table(0);
    auto *header = get_chip_swimlane_header(shm);
    header->chip_swimlane_level = static_cast<uint32_t>(ChipSwimlaneLevel::TASK_TIMING);
    chip_swimlane_aicpu_init(/*worker_count=*/1);

    auto *ac_state = get_aicore_buffer_state(shm, 0);
    const uint64_t held = ac_state->head.current_buf_ptr;
    ASSERT_NE(held, 0u);
    const uint32_t ac_depth_after_init = aicore_free_queue_depth(shm, 0);

    constexpr int kDispatches = 3;
    for (int i = 0; i < kDispatches; i++) {
        chip_swimlane_aicpu_on_aicore_dispatch(/*core_id=*/0, /*thread_idx=*/0, static_cast<uint32_t>(i + 1));
    }

    const int cores[] = {0};
    chip_swimlane_aicpu_flush(/*thread_idx=*/PLATFORM_MAX_AICPU_THREADS, cores, /*core_num=*/1);

    EXPECT_EQ(ac_state->head.dropped_record_count, static_cast<uint32_t>(kDispatches))
        << "the failed AICore enqueue charged dropped by capacity rather than the buffer's record count";
    EXPECT_EQ(ac_state->head.current_buf_ptr, held)
        << "an AICore buffer the host never received was released, so nothing can return it";
    EXPECT_EQ(aicore_free_queue_depth(shm, 0), ac_depth_after_init) << "the failed flush drew from the AICore pool";
    EXPECT_FALSE(free_queue_holds(ac_state->free_queue, held)) << "AICPU pushed the free queue it only consumes";

    // Reusable in place, re-stamped and re-zeroed by the next run's init.
    collector.begin_run("aicore-full-next", ChipSwimlaneLevel::TASK_TIMING);
    set_platform_run_result(/*region_base=*/0, /*run_epoch=*/12);
    header->chip_swimlane_level = static_cast<uint32_t>(ChipSwimlaneLevel::TASK_TIMING);
    chip_swimlane_aicpu_init(/*worker_count=*/1);
    EXPECT_EQ(ac_state->head.current_buf_ptr, held)
        << "init drew a replacement instead of reusing the retained AICore buffer";
    const auto *reused = reinterpret_cast<const ChipSwimlaneAicoreTaskBuffer *>(held);
    EXPECT_EQ(reused->count, 0u) << "a reused AICore buffer kept the failed run's mark";
    EXPECT_EQ(reused->run_epoch, 12u) << "a reused AICore buffer kept the previous run's identity";
    EXPECT_EQ(ac_state->head.current_buf_seq, 0u);

    collector.finalize(nullptr, swimlane_test_free);
}

// The other side of retention: a buffer the host *did* receive must not also
// stay the device's. Retention is scoped to a failed hand-over, so a successful
// enqueue still releases the pointer, and the pool must not hand that buffer out
// again until the host has put it back.
//
// Drives the whole cycle per run — init -> complete_task -> flush -> the
// collector's own copy -> the host's push back into the free queue — which also
// pins the count and identity of each run's single record. PLATFORM_PROF_SLOT_COUNT
// runs later the queue comes back around to the first buffer, so the last run
// exercises reuse of storage that went through the host rather than storage that
// was retained.
TEST(ChipSwimlaneBufferReturnTest, APublishedBufferIsReturnedByTheHostAndNotRetainedByTheDevice) {
    ChipSwimlaneCollector collector;
    ASSERT_EQ(
        collector.initialize(
            /*num_aicore=*/1, /*aicpu_thread_num=*/1, /*device_id=*/0, ChipSwimlaneLevel::SCHEDULE_TIMING,
            swimlane_test_alloc, nullptr, swimlane_test_free
        ),
        0
    );
    void *shm = collector.get_chip_swimlane_setup_device_ptr();
    auto *pool = get_perf_buffer_state(shm, 0);
    auto *header = get_chip_swimlane_header(shm);

    const uint32_t depth_at_start = free_queue_depth(shm, 0);
    ASSERT_EQ(depth_at_start, static_cast<uint32_t>(PLATFORM_PROF_SLOT_COUNT))
        << "this test's FIFO reasoning assumes a fully seeded free queue";

    std::vector<uint64_t> published;
    uint32_t consumed_tail = 0;

    for (uint64_t epoch = 1; epoch <= static_cast<uint64_t>(PLATFORM_PROF_SLOT_COUNT) + 1; epoch++) {
        collector.begin_run("publish-cycle", ChipSwimlaneLevel::SCHEDULE_TIMING);
        run_once(shm, epoch, ChipSwimlaneLevel::SCHEDULE_TIMING, /*dispatches=*/0, /*completes=*/1);

        // A successful hand-over releases the pointer; anything else is a buffer
        // owned by the device and the host at once.
        ASSERT_EQ(pool->head.current_buf_ptr, 0u) << "run " << epoch << " retained a buffer it published";

        // Exactly one new ready entry, carrying this run's one record.
        const uint32_t tail = header->queue_tails[0];
        ASSERT_EQ(tail, consumed_tail + 1) << "run " << epoch << " published something other than one buffer";
        const ReadyQueueEntry &entry = header->queues[0][consumed_tail];
        ASSERT_EQ(entry.kind, ChipSwimlaneBufferKind::AicpuTask);
        const uint64_t handed_over = entry.buffer_ptr;
        consumed_tail = tail;

        EXPECT_FALSE(free_queue_holds(pool->free_queue, handed_over))
            << "run " << epoch << " both published a buffer and left it in the free queue";

        ReadyBufferInfo info{};
        info.type = ProfBufferType::AICPU_TASK;
        info.index = entry.core_index;
        info.dev_buffer_ptr = reinterpret_cast<void *>(handed_over);
        info.host_buffer_ptr = reinterpret_cast<void *>(handed_over);
        info.buffer_seq = entry.buffer_seq;
        collector.on_buffer_collected(info, /*collector_shard=*/0);

        const auto &collected = collector.collected_perf_records_for_test();
        ASSERT_FALSE(collected.empty());
        ASSERT_EQ(collected[0].size(), 1u) << "run " << epoch << " collected other than its own single record";
        EXPECT_EQ(collected[0][0].run_epoch, epoch);
        EXPECT_EQ(collected[0][0].record.reg_task_id, 1u);

        if (epoch == static_cast<uint64_t>(PLATFORM_PROF_SLOT_COUNT) + 1) {
            // FIFO brings the first run's storage back to the queue head exactly
            // now. It is the same buffer, freshly stamped — reuse after a return
            // is correct, and only reuse *without* a return is not.
            EXPECT_EQ(handed_over, published.front())
                << "the returned buffer never came back around; the FIFO assumption above is wrong";
            break;
        }
        published.push_back(handed_over);
        host_push_free_queue(pool->free_queue, handed_over);
    }

    collector.finalize(nullptr, swimlane_test_free);
}

// The AICore pool's copy of the same cycle. It is not redundant with the task
// pool's: this pool's count comes from the dispatch derivation rather than from
// a per-record increment, its buffer is written by a program that cannot read
// the AICPU SO, and its records survive reuse — `init` zeroes `count` but not
// the payload, so a reused buffer still holds its previous occupant's records
// and only the count stamp keeps the host from reading them.
//
// The record payloads are a fixture: AICore's writer is `__aicore__` code
// compiled by ccec and tested in test_chip_swimlane_aicore. Everything that
// decides ownership and accounting here is production — the dispatch counter,
// the flush's mark and release, the collector's copy, and `init`'s re-acquire.
TEST(ChipSwimlaneBufferReturnTest, APublishedAicoreBufferIsReturnedByTheHostAndReacquired) {
    ChipSwimlaneCollector collector;
    ASSERT_EQ(
        collector.initialize(
            /*num_aicore=*/1, /*aicpu_thread_num=*/1, /*device_id=*/0, ChipSwimlaneLevel::TASK_TIMING,
            swimlane_test_alloc, nullptr, swimlane_test_free
        ),
        0
    );
    void *shm = collector.get_chip_swimlane_setup_device_ptr();
    auto *ac_state = get_aicore_buffer_state(shm, 0);
    auto *header = get_chip_swimlane_header(shm);

    ASSERT_EQ(aicore_free_queue_depth(shm, 0), static_cast<uint32_t>(PLATFORM_PROF_SLOT_COUNT))
        << "this test's FIFO reasoning assumes a fully seeded AICore free queue";

    constexpr uint64_t kLastEpoch = static_cast<uint64_t>(PLATFORM_PROF_SLOT_COUNT) + 1;
    std::vector<uint64_t> published;
    uint32_t consumed_tail = 0;

    for (uint64_t epoch = 1; epoch <= kLastEpoch; epoch++) {
        // The wrap-around run writes fewer records than the buffer's first
        // occupant did, so the stale tail is there to be mis-read.
        const int records = (epoch == kLastEpoch) ? 1 : 3;

        collector.begin_run("aicore-publish-cycle", ChipSwimlaneLevel::TASK_TIMING);
        set_platform_run_result(/*region_base=*/0, epoch);
        set_chip_swimlane_enabled(true);
        set_platform_chip_swimlane_base(reinterpret_cast<uint64_t>(shm));
        set_platform_chip_swimlane_aicore_rotation_table(0);
        header->chip_swimlane_level = static_cast<uint32_t>(ChipSwimlaneLevel::TASK_TIMING);
        chip_swimlane_aicpu_init(/*worker_count=*/1);

        const uint64_t held = ac_state->head.current_buf_ptr;
        ASSERT_NE(held, 0u) << "run " << epoch << " acquired no AICore buffer";
        auto *buf = reinterpret_cast<ChipSwimlaneAicoreTaskBuffer *>(held);
        ASSERT_EQ(buf->count, 0u) << "run " << epoch << " acquired a buffer whose count was not reset";

        for (int i = 0; i < records; i++) {
            chip_swimlane_aicpu_on_aicore_dispatch(/*core_id=*/0, /*thread_idx=*/0, static_cast<uint32_t>(i + 1));
            buf->records[i].start_time = 1000 + static_cast<uint64_t>(i);
            buf->records[i].end_time = 2000 + static_cast<uint64_t>(i);
            buf->records[i].reg_task_id = static_cast<uint32_t>(epoch * 100 + static_cast<uint64_t>(i));
        }

        const int cores[] = {0};
        chip_swimlane_aicpu_flush(/*thread_idx=*/0, cores, /*core_num=*/1);

        ASSERT_EQ(ac_state->head.current_buf_ptr, 0u) << "run " << epoch << " retained an AICore buffer it published";

        const uint32_t tail = header->queue_tails[0];
        ASSERT_EQ(tail, consumed_tail + 1) << "run " << epoch << " published other than one buffer";
        const ReadyQueueEntry &entry = header->queues[0][consumed_tail];
        ASSERT_EQ(entry.kind, ChipSwimlaneBufferKind::AicoreTask);
        ASSERT_EQ(entry.buffer_ptr, held);
        consumed_tail = tail;

        EXPECT_EQ(buf->count, static_cast<uint32_t>(records))
            << "run " << epoch << " stamped a mark other than the dispatches it made";
        EXPECT_FALSE(free_queue_holds(ac_state->free_queue, held))
            << "run " << epoch << " both published an AICore buffer and left it in the free queue";

        ReadyBufferInfo info{};
        info.type = ProfBufferType::AICORE_TASK;
        info.index = entry.core_index;
        info.dev_buffer_ptr = reinterpret_cast<void *>(held);
        info.host_buffer_ptr = reinterpret_cast<void *>(held);
        info.buffer_seq = entry.buffer_seq;
        collector.on_buffer_collected(info, /*collector_shard=*/0);

        const auto &collected = collector.collected_aicore_records_for_test();
        ASSERT_FALSE(collected.empty());
        ASSERT_EQ(collected[0].size(), static_cast<size_t>(records))
            << "run " << epoch << " collected other than its own records";
        for (int i = 0; i < records; i++) {
            EXPECT_EQ(collected[0][static_cast<size_t>(i)].run_epoch, epoch);
            EXPECT_EQ(
                collected[0][static_cast<size_t>(i)].record.reg_task_id,
                static_cast<uint32_t>(epoch * 100 + static_cast<uint64_t>(i))
            );
        }

        if (epoch == kLastEpoch) {
            EXPECT_EQ(held, published.front())
                << "the returned AICore buffer never came back around; the FIFO assumption above is wrong";
            break;
        }
        published.push_back(held);
        host_push_free_queue(ac_state->free_queue, held);
    }

    collector.finalize(nullptr, swimlane_test_free);
}

// ---------------------------------------------------------------------------
// AICore record accounting: published + live + dropped == total, with the
// current buffer's count tracked per dispatch rather than derived from
// `total_record_count - current_buf_seq * BUFFER_SIZE`.
// ---------------------------------------------------------------------------

class ChipSwimlaneAccountingTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_EQ(
            collector_.initialize(
                /*num_aicore=*/1, /*aicpu_thread_num=*/1, /*device_id=*/0, ChipSwimlaneLevel::TASK_TIMING,
                swimlane_test_alloc, nullptr, swimlane_test_free
            ),
            0
        );
        shm_ = collector_.get_chip_swimlane_setup_device_ptr();
        ASSERT_NE(shm_, nullptr);
        set_chip_swimlane_enabled(true);
        set_platform_chip_swimlane_base(reinterpret_cast<uint64_t>(shm_));
        set_platform_chip_swimlane_aicore_rotation_table(0);
    }

    void TearDown() override {
        set_platform_run_result(0, 0);
        set_chip_swimlane_enabled(false);
        collector_.finalize(nullptr, swimlane_test_free);
    }

    // One run's device sequence. Deliberately does not call begin_run(): the
    // caller decides whether the host's per-run clear happens, because whether it
    // does is the thing under test.
    void device_run(uint64_t epoch, int dispatches) {
        set_platform_run_result(/*region_base=*/0, epoch);
        get_chip_swimlane_header(shm_)->chip_swimlane_level = static_cast<uint32_t>(ChipSwimlaneLevel::TASK_TIMING);
        chip_swimlane_aicpu_init(/*worker_count=*/1);
        for (int i = 0; i < dispatches; i++) {
            chip_swimlane_aicpu_on_aicore_dispatch(/*core_id=*/0, /*thread_idx=*/0, static_cast<uint32_t>(i + 1));
        }
        const int cores[] = {0};
        chip_swimlane_aicpu_flush(/*thread_idx=*/0, cores, /*core_num=*/1);
    }

    // The `count` stamped on every AICore buffer published so far, in order.
    std::vector<uint32_t> published_marks() const {
        std::vector<uint32_t> marks;
        const auto *header = get_chip_swimlane_header(shm_);
        for (uint32_t i = 0; i < header->queue_tails[0]; i++) {
            const ReadyQueueEntry &entry = header->queues[0][i];
            if (entry.kind != ChipSwimlaneBufferKind::AicoreTask) continue;
            const uint32_t mark = reinterpret_cast<const ChipSwimlaneAicoreTaskBuffer *>(entry.buffer_ptr)->count;
            marks.push_back(mark);
        }
        return marks;
    }

    static bool accounting_balances(const ChipSwimlaneAicoreTaskPool *ac_state) {
        const uint32_t published = ac_state->head.published_record_count;
        const uint32_t live = ac_state->head.live_record_count;
        const uint32_t dropped = ac_state->head.dropped_record_count;
        const uint32_t total = ac_state->head.total_record_count;
        return published + live + dropped == total;
    }

    ChipSwimlaneCollector collector_;
    void *shm_ = nullptr;
};

// The tail count survives a window spanning two runs — which is the whole point
// of tracking it instead of deriving it.
//
// The old form was `total_record_count - current_buf_seq * BUFFER_SIZE`. Both
// operands were reset together every run: the host cleared `total` in
// `publish_run_config`, and `init` reset `seq` to 0. Take the per-run clear away —
// which is exactly what continuous collection does — and the two no longer agree:
// `total` carries the earlier run's dispatches while `seq` restarts at 0, so the
// derivation reports the whole window as live in the current buffer.
//
// Here run 1 dispatches 3 and publishes them; run 2 starts without the host's
// clear and dispatches 2. The derivation would have marked run 2's tail 5 — more
// records than were ever written into that buffer — and charged the difference to
// whichever side of the accounting read it next. The tracked counter says 2.
TEST_F(ChipSwimlaneAccountingTest, TheTailCountSurvivesAWindowSpanningTwoRuns) {
    // Run 1, with the host's per-run clear, as today.
    collector_.begin_run("run-one", ChipSwimlaneLevel::TASK_TIMING);
    device_run(/*epoch=*/1, /*dispatches=*/3);
    ASSERT_EQ(published_marks().size(), 1u);
    EXPECT_EQ(published_marks()[0], 3u) << "run 1 published something other than its 3 dispatches";

    auto *ac_state = get_aicore_buffer_state(shm_, 0);
    EXPECT_EQ(ac_state->head.total_record_count, 3u);
    EXPECT_EQ(ac_state->head.published_record_count, 3u);
    EXPECT_EQ(ac_state->head.live_record_count, 0u);
    EXPECT_EQ(ac_state->head.dropped_record_count, 0u);

    // Run 2 without begin_run(): `total_record_count` keeps the first run's 3
    // while `init` resets `current_buf_seq` to 0. This is the state the old
    // derivation could not survive.
    device_run(/*epoch=*/2, /*dispatches=*/2);
    ASSERT_EQ(published_marks().size(), 2u);
    EXPECT_EQ(
        published_marks()[1], 2u
    ) << "the tail was marked from a cross-run derivation rather than this run's own count";

    EXPECT_EQ(ac_state->head.total_record_count, 5u) << "the window's attempt tally should span both runs";
    EXPECT_EQ(ac_state->head.published_record_count, 5u);
    EXPECT_EQ(ac_state->head.live_record_count, 0u);
    EXPECT_EQ(ac_state->head.dropped_record_count, 0u);
    EXPECT_TRUE(accounting_balances(ac_state)) << "published + live + dropped != total";
}

// An idle run leaves the identity trivially satisfied and publishes nothing, so
// a retained buffer never contributes a phantom count.
TEST_F(ChipSwimlaneAccountingTest, AnIdleRunLeavesNothingLive) {
    collector_.begin_run("idle", ChipSwimlaneLevel::TASK_TIMING);
    device_run(/*epoch=*/7, /*dispatches=*/0);

    auto *ac_state = get_aicore_buffer_state(shm_, 0);
    EXPECT_TRUE(published_marks().empty()) << "an idle run published a buffer";
    EXPECT_EQ(ac_state->head.live_record_count, 0u);
    EXPECT_EQ(ac_state->head.published_record_count, 0u);
    EXPECT_EQ(ac_state->head.total_record_count, 0u);
    EXPECT_TRUE(accounting_balances(ac_state));
}

// Dispatches made with no active buffer belong to `dropped`, not to whichever
// buffer the pool hands over next.
//
// The path, with capacity B: `init` finds the free queue empty and leaves
// `current_buf_ptr` at 0, so AICore resolves a null buffer and its reserve refuses
// every record. B dispatches later the host returns a buffer and the B+1'th
// dispatch rotates into it — a rotation whose outgoing pointer is 0, so it stashes
// nothing and settles nothing. Had those B dispatches been counted as live, the
// recovered buffer would be marked with B+1 records while holding 1.
//
// Checking only `published + live + dropped == total` cannot see this: the sum
// balances either way. So this asserts the buffer's own mark and `dropped`
// separately, which is where the two outcomes actually differ.
TEST_F(ChipSwimlaneAccountingTest, DispatchesWithNoBufferAreDroppedNotCarriedIntoTheNextOne) {
    auto *ac_state = get_aicore_buffer_state(shm_, 0);

    // Keep one real buffer aside, then present an empty free queue to init.
    const uint64_t spare = ac_state->free_queue.buffer_ptrs[ac_state->free_queue.head % PLATFORM_PROF_SLOT_COUNT];
    ASSERT_NE(spare, 0u);
    ac_state->free_queue.head = ac_state->free_queue.tail;
    wmb();

    collector_.begin_run("starved", ChipSwimlaneLevel::TASK_TIMING);
    set_platform_run_result(/*region_base=*/0, /*run_epoch=*/61);
    get_chip_swimlane_header(shm_)->chip_swimlane_level = static_cast<uint32_t>(ChipSwimlaneLevel::TASK_TIMING);
    chip_swimlane_aicpu_init(/*worker_count=*/1);
    ASSERT_EQ(ac_state->head.current_buf_ptr, 0u) << "init found a buffer; the starvation setup did not take";

    // A full batch of dispatches with nowhere to land.
    constexpr uint32_t kCapacity = static_cast<uint32_t>(PLATFORM_AICORE_BUFFER_SIZE);
    for (uint32_t i = 0; i < kCapacity; i++) {
        chip_swimlane_aicpu_on_aicore_dispatch(/*core_id=*/0, /*thread_idx=*/0, i + 1);
    }
    EXPECT_EQ(ac_state->head.live_record_count, 0u)
        << "dispatches with no active buffer were credited to a buffer that does not exist";
    EXPECT_EQ(ac_state->head.dropped_record_count, kCapacity);

    // The host returns a buffer; the next dispatch is the rotation boundary, so it
    // acquires that buffer and is the only record it will hold.
    host_push_aicore_free_queue(ac_state->free_queue, spare);
    chip_swimlane_aicpu_on_aicore_dispatch(/*core_id=*/0, /*thread_idx=*/0, kCapacity + 1);
    ASSERT_NE(ac_state->head.current_buf_ptr, 0u) << "the returned buffer was never picked up";
    EXPECT_EQ(ac_state->head.live_record_count, 1u);

    const int cores[] = {0};
    chip_swimlane_aicpu_flush(/*thread_idx=*/0, cores, /*core_num=*/1);

    const std::vector<uint32_t> marks = published_marks();
    ASSERT_EQ(marks.size(), 1u) << "expected exactly the recovered buffer to be published";
    EXPECT_EQ(marks[0], 1u) << "the recovered buffer was marked with records it never received";
    EXPECT_EQ(ac_state->head.published_record_count, 1u);
    EXPECT_EQ(ac_state->head.dropped_record_count, kCapacity);
    EXPECT_EQ(ac_state->head.total_record_count, kCapacity + 1);
    EXPECT_TRUE(accounting_balances(ac_state));
}

// A rotated-out buffer reaches the host only on the ACK that gates it, and the
// accounting identity is open for exactly that long.
//
// The rotation does not hand the just-filled buffer over: tensormap_and_ringbuffer's
// AICore executor writes FIN before the swimlane record, so the FIN that gated the
// boundary dispatch does not prove the old buffer's tail record has drained. The
// buffer is stashed and released only when AICore ACKs the first task of the NEW
// buffer, whose token the rotation stored as the gate.
//
// Nothing here reads the stash itself — it is file-local to the AICPU translation
// unit — but every effect of it is observable from the shared memory the host reads:
// the buffer is in neither queue, `published_record_count` has not moved, and the
// count is missing from the identity. So this pins the window by its consequences
// rather than by its storage.
TEST_F(ChipSwimlaneAccountingTest, ARotatedBufferReachesTheHostOnlyOnItsGatingAck) {
    auto *ac_state = get_aicore_buffer_state(shm_, 0);

    collector_.begin_run("ack-gate", ChipSwimlaneLevel::TASK_TIMING);
    set_platform_run_result(/*region_base=*/0, /*run_epoch=*/71);
    get_chip_swimlane_header(shm_)->chip_swimlane_level = static_cast<uint32_t>(ChipSwimlaneLevel::TASK_TIMING);
    chip_swimlane_aicpu_init(/*worker_count=*/1);

    const uint64_t rotated = ac_state->head.current_buf_ptr;
    ASSERT_NE(rotated, 0u) << "init found no buffer";
    ASSERT_GT(aicore_free_queue_depth(shm_, 0), 0u) << "no replacement buffer available for the rotation";

    // Fill the active buffer to capacity. The rotation fires on the NEXT dispatch,
    // not this batch's last one.
    constexpr uint32_t kCapacity = static_cast<uint32_t>(PLATFORM_AICORE_BUFFER_SIZE);
    for (uint32_t i = 0; i < kCapacity; i++) {
        chip_swimlane_aicpu_on_aicore_dispatch(/*core_id=*/0, /*thread_idx=*/0, i + 1);
    }
    ASSERT_EQ(ac_state->head.current_buf_ptr, rotated) << "rotated before the capacity boundary";
    ASSERT_EQ(ac_state->head.live_record_count, kCapacity);

    // The boundary dispatch. Its token becomes the ACK gate.
    constexpr uint32_t kGate = kCapacity + 1;
    chip_swimlane_aicpu_on_aicore_dispatch(/*core_id=*/0, /*thread_idx=*/0, kGate);
    ASSERT_NE(ac_state->head.current_buf_ptr, rotated) << "the rotation did not take";

    // Mid-hand-off. The rotated buffer belongs to neither side: the host cannot
    // see it, and AICPU has not put it back — it has no push on the free queue.
    EXPECT_TRUE(published_marks().empty()) << "the rotated buffer was published before its gating ACK";
    EXPECT_FALSE(free_queue_holds(ac_state->free_queue, rotated)) << "AICPU pushed a buffer onto the free queue";
    EXPECT_EQ(ac_state->head.published_record_count, 0u);
    EXPECT_EQ(ac_state->head.live_record_count, 1u) << "the boundary dispatch belongs to the replacement buffer";
    EXPECT_EQ(ac_state->head.dropped_record_count, 0u);
    EXPECT_EQ(ac_state->head.total_record_count, kGate);

    // This is the checkpoint-vs-running-invariant gap, measured rather than
    // asserted away: the stashed buffer's records are in neither `published` nor
    // `live`, so the identity is short by exactly that buffer's count.
    EXPECT_FALSE(accounting_balances(ac_state)) << "the identity closed while a buffer was still mid-hand-off";
    EXPECT_EQ(
        ac_state->head.total_record_count - ac_state->head.published_record_count - ac_state->head.live_record_count -
            ac_state->head.dropped_record_count,
        kCapacity
    ) << "the shortfall is not the stashed buffer's count";

    // An ACK for any other task leaves it stashed. The gate is a specific token,
    // not "the next completion to arrive".
    chip_swimlane_aicpu_on_aicore_ack(/*core_id=*/0, /*thread_idx=*/0, kGate - 1);
    EXPECT_TRUE(published_marks().empty()) << "a non-gating ACK released the stashed buffer";
    EXPECT_EQ(ac_state->head.published_record_count, 0u);

    // The gating ACK releases it, marked with the count captured at rotation.
    chip_swimlane_aicpu_on_aicore_ack(/*core_id=*/0, /*thread_idx=*/0, kGate);
    const std::vector<uint32_t> after_ack = published_marks();
    ASSERT_EQ(after_ack.size(), 1u) << "the gating ACK did not release the stashed buffer";
    EXPECT_EQ(after_ack[0], kCapacity) << "the released buffer was marked with a count it does not hold";
    EXPECT_EQ(ac_state->head.published_record_count, kCapacity);
    EXPECT_TRUE(accounting_balances(ac_state)) << "the identity did not close once the hand-off completed";

    // The gate is consumed, so a repeat of the same ACK cannot publish it twice.
    chip_swimlane_aicpu_on_aicore_ack(/*core_id=*/0, /*thread_idx=*/0, kGate);
    EXPECT_EQ(published_marks().size(), 1u) << "a repeated gating ACK published the buffer a second time";
    EXPECT_EQ(ac_state->head.published_record_count, kCapacity);

    // Run end hands over the replacement, holding just the boundary dispatch.
    const int cores[] = {0};
    chip_swimlane_aicpu_flush(/*thread_idx=*/0, cores, /*core_num=*/1);

    const std::vector<uint32_t> marks = published_marks();
    ASSERT_EQ(marks.size(), 2u) << "the replacement buffer was not published at flush";
    EXPECT_EQ(marks[1], 1u);
    EXPECT_EQ(ac_state->head.published_record_count, kGate);
    EXPECT_EQ(ac_state->head.live_record_count, 0u);
    EXPECT_TRUE(accounting_balances(ac_state));
}
