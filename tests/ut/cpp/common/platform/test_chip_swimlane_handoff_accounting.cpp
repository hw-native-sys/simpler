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
 * Successful handoff accounting: what the device committed, what the host
 * received, and what the host kept, as three separate quantities.
 *
 * Every case drives the production path — real AICPU dispatch, rotation and
 * flush through all four producer classes, the collector's own
 * `on_buffer_collected`, then the production terminal read and
 * classification.
 *
 * Host and device share process memory here, so nothing below is evidence
 * about device cache visibility.
 */

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <thread>

#include "aicpu/chip_swimlane_collector_aicpu.h"
#include "aicpu/device_run_result_base_aicpu.h"
#include "common/chip_swimlane_profiling.h"
#include "common/memory_barrier.h"
#include "common/scheduler_profiling.h"
#include "host/chip_swimlane_collector.h"

using Handoff = ChipSwimlaneCollector::HandoffVerdict;
using Coverage = ChipSwimlaneCollector::HandoffCoverage;

namespace {

void *ha_alloc(size_t size) { return std::calloc(1, size); }

int ha_free(void *ptr) {
    std::free(ptr);
    return 0;
}

int init_collector(
    ChipSwimlaneCollector &collector, int num_aicore, ChipSwimlaneLevel level = ChipSwimlaneLevel::TASK_TIMING
) {
    return collector.initialize(num_aicore, /*aicpu_thread_num=*/1, /*device_id=*/0, level, ha_alloc, nullptr, ha_free);
}

void arm_run(
    ChipSwimlaneCollector &collector, int num_aicore, uint64_t epoch, const char *prefix,
    ChipSwimlaneLevel level = ChipSwimlaneLevel::TASK_TIMING
) {
    void *shm = collector.get_chip_swimlane_setup_device_ptr();
    ASSERT_NE(shm, nullptr);
    collector.begin_run(prefix, level);
    set_platform_run_result(/*region_base=*/0, epoch);
    set_chip_swimlane_enabled(true);
    set_platform_chip_swimlane_base(reinterpret_cast<uint64_t>(shm));
    set_platform_chip_swimlane_aicore_rotation_table(0);
    set_platform_chip_swimlane_run_terminal_bank(
        reinterpret_cast<uint64_t>(collector.arm_run_terminal_bank(/*bank_index=*/0, epoch))
    );
    chip_swimlane_aicpu_init(num_aicore);
}

// Dispatch `records` AICore tasks on `core_id`, writing each slot so the host
// accepts them. Crossing PLATFORM_AICORE_BUFFER_SIZE rotates the buffer.
void dispatch_aicore(ChipSwimlaneCollector &collector, int core_id, int records) {
    void *shm = collector.get_chip_swimlane_setup_device_ptr();
    auto *ac_state = get_aicore_buffer_state(shm, core_id);
    for (int i = 0; i < records; i++) {
        chip_swimlane_aicpu_on_aicore_dispatch(core_id, /*thread_idx=*/0, static_cast<uint32_t>(i + 1));
        auto *buf = reinterpret_cast<ChipSwimlaneAicoreTaskBuffer *>(ac_state->head.current_buf_ptr);
        if (buf == nullptr) continue;
        const uint32_t slot = ac_state->head.live_record_count > 0 ? ac_state->head.live_record_count - 1 : 0;
        if (slot >= static_cast<uint32_t>(PLATFORM_AICORE_BUFFER_SIZE)) continue;
        buf->records[slot].start_time = 1000 + static_cast<uint64_t>(i);
        buf->records[slot].end_time = 2000 + static_cast<uint64_t>(i);
        buf->records[slot].reg_task_id = static_cast<uint32_t>(i + 1);
    }
}

// Commit `records` AICPU task records on `core_id`. Crossing
// PLATFORM_PROF_BUFFER_SIZE rotates through the shared device engine.
void complete_aicpu_tasks(int core_id, int records) {
    for (int i = 0; i < records; i++) {
        chip_swimlane_aicpu_complete_task(
            core_id, /*thread_idx=*/0, static_cast<uint32_t>(i + 1), /*dispatch_time=*/10 + i,
            /*finish_time=*/20 + i
        );
    }
}

void record_sched_phases(int thread_idx, int records) {
    for (int i = 0; i < records; i++) {
        chip_swimlane_aicpu_record_sched_phase(
            thread_idx, SchedPhaseKind::Dispatch, /*start_time=*/100 + i, /*end_time=*/200 + i,
            /*loop_iter=*/static_cast<uint32_t>(i), /*tasks_processed=*/1
        );
    }
}

void flush_cores(const int *cores, int core_num) { chip_swimlane_aicpu_flush(/*thread_idx=*/0, cores, core_num); }

void flush_core(int core_id) {
    const int cores[] = {core_id};
    flush_cores(cores, /*core_num=*/1);
}

// Hand every published buffer to the collector exactly as the mgmt thread
// would, and return how many entries were delivered.
uint32_t collect_published(ChipSwimlaneCollector &collector, uint32_t from_tail) {
    void *shm = collector.get_chip_swimlane_setup_device_ptr();
    auto *header = get_chip_swimlane_header(shm);
    uint32_t delivered = 0;
    for (uint32_t i = from_tail; i < header->queue_tails[0]; i++) {
        const ReadyQueueEntry &entry = header->queues[0][i];
        ReadyBufferInfo info{};
        info.type = static_cast<ProfBufferType>(static_cast<uint32_t>(entry.kind));
        info.index = entry.core_index;
        info.dev_buffer_ptr = reinterpret_cast<void *>(entry.buffer_ptr);
        info.host_buffer_ptr = reinterpret_cast<void *>(entry.buffer_ptr);
        info.buffer_seq = entry.buffer_seq;
        collector.on_buffer_collected(info, /*collector_shard=*/0);
        delivered++;
    }
    return delivered;
}

const ChipSwimlaneRunTerminal *terminal_at(ChipSwimlaneCollector &collector, int producer_index) {
    void *shm = collector.get_chip_swimlane_setup_device_ptr();
    return get_run_terminal(get_run_terminal_bank(shm, /*bank_index=*/0), producer_index);
}

const ChipSwimlaneRunTerminal *aicore_terminal(ChipSwimlaneCollector &collector, int core_id) {
    return terminal_at(collector, PLATFORM_RUN_TERMINAL_AICORE_TASK_BASE + core_id);
}

const ChipSwimlaneRunTerminal *aicpu_terminal(ChipSwimlaneCollector &collector, int core_id) {
    return terminal_at(collector, PLATFORM_RUN_TERMINAL_AICPU_TASK_BASE + core_id);
}

// The host's recycle step, as the buffer-pool manager performs it: park a
// buffer the host is done with in the pool's free queue.
void recycle_buffer(ChipSwimlaneFreeQueue *free_queue, uint64_t buffer_ptr) {
    const uint32_t tail = free_queue->tail;
    free_queue->buffer_ptrs[tail % PLATFORM_PROF_SLOT_COUNT] = buffer_ptr;
    wmb();
    free_queue->tail = tail + 1;
}

// Report and read back one run, exactly as the host does after completion.
void settle_run(ChipSwimlaneCollector &collector, uint64_t epoch) {
    collector.reconcile_counters();
    collector.report_run_terminal_snapshot(/*bank_index=*/0, epoch);
}

}  // namespace

// A rotation and the tail flush are two commits, and the published record
// count is their two saved counts.
TEST(ChipSwimlaneHandoffAccountingTest, RotationAndTailFlushAreTwoCommits) {
    ChipSwimlaneCollector collector;
    ASSERT_EQ(init_collector(collector, /*num_aicore=*/1), 0);
    constexpr uint64_t kEpoch = 8100;
    arm_run(collector, /*num_aicore=*/1, kEpoch, "rotate");

    dispatch_aicore(collector, /*core_id=*/0, PLATFORM_AICORE_BUFFER_SIZE + 3);
    flush_core(/*core_id=*/0);

    const ChipSwimlaneRunTerminal *t = aicore_terminal(collector, 0);
    EXPECT_EQ(t->run_epoch, kEpoch);
    EXPECT_EQ(t->published_buffers, 2u) << "one rotation plus the tail flush";
    EXPECT_EQ(t->published_records, static_cast<uint32_t>(PLATFORM_AICORE_BUFFER_SIZE) + 3u);
    EXPECT_EQ(t->live_at_close, 0u) << "the flush settles live before the terminal is written";

    collect_published(collector, 0);
    settle_run(collector, kEpoch);
    const auto r = collector.handoff_report_for_test().aicore_task;
    EXPECT_EQ(r.verdict, Handoff::Match);
    EXPECT_EQ(r.coverage, Coverage::Complete);
    EXPECT_EQ(r.received_buffers, r.published_buffers);
    EXPECT_EQ(r.received_records, r.published_records);
    EXPECT_TRUE(r.silent_loss_known);
    EXPECT_EQ(r.silent_loss, 0u);

    collector.finalize(nullptr, ha_free);
}

// The AICPU task pool rotates through the shared device engine, so its
// successful commit is the engine's optional hook rather than a site in the
// swimlane module. Rotation and tail flush are two commits there too.
TEST(ChipSwimlaneHandoffAccountingTest, EngineRotationCountsTheCommittedBuffer) {
    ChipSwimlaneCollector collector;
    ASSERT_EQ(init_collector(collector, /*num_aicore=*/1), 0);
    constexpr uint64_t kEpoch = 8150;
    arm_run(collector, /*num_aicore=*/1, kEpoch, "engine");

    complete_aicpu_tasks(/*core_id=*/0, PLATFORM_PROF_BUFFER_SIZE + 4);
    flush_core(/*core_id=*/0);

    const ChipSwimlaneRunTerminal *t = aicpu_terminal(collector, 0);
    EXPECT_EQ(t->published_buffers, 2u) << "the engine's rotation plus the tail flush";
    EXPECT_EQ(t->published_records, static_cast<uint32_t>(PLATFORM_PROF_BUFFER_SIZE) + 4u);
    EXPECT_EQ(t->total, static_cast<uint32_t>(PLATFORM_PROF_BUFFER_SIZE) + 4u);
    EXPECT_EQ(t->dropped, 0u);

    collect_published(collector, 0);
    settle_run(collector, kEpoch);
    const auto r = collector.handoff_report_for_test().aicpu_task;
    EXPECT_EQ(r.verdict, Handoff::Match);
    EXPECT_EQ(r.coverage, Coverage::Complete);
    EXPECT_EQ(r.received_buffers, 2u);
    EXPECT_EQ(r.received_records, r.published_records);
    EXPECT_EQ(r.silent_loss, 0u);

    collector.finalize(nullptr, ha_free);
}

// The sched-phase pool reaches the same two commit sites from its own entry
// points: capacity rotation through the engine, and the run-end phase flush.
TEST(ChipSwimlaneHandoffAccountingTest, SchedPhaseRotationAndFlushAreTwoCommits) {
    ChipSwimlaneCollector collector;
    ASSERT_EQ(init_collector(collector, /*num_aicore=*/1, ChipSwimlaneLevel::SCHED_PHASES), 0);
    constexpr uint64_t kEpoch = 8160;
    arm_run(collector, /*num_aicore=*/1, kEpoch, "sched", ChipSwimlaneLevel::SCHED_PHASES);
    chip_swimlane_aicpu_init_phase(/*worker_count=*/1, /*num_sched_phase_threads=*/1, /*num_orch_phase_threads=*/0);

    record_sched_phases(/*thread_idx=*/0, PLATFORM_PHASE_RECORDS_PER_THREAD + 5);
    chip_swimlane_aicpu_flush_sched_phase_buffer(/*thread_idx=*/0);

    const ChipSwimlaneRunTerminal *t = terminal_at(collector, PLATFORM_RUN_TERMINAL_SCHED_PHASE_BASE + 0);
    EXPECT_EQ(t->run_epoch, kEpoch);
    EXPECT_EQ(t->published_buffers, 2u);
    EXPECT_EQ(t->published_records, static_cast<uint32_t>(PLATFORM_PHASE_RECORDS_PER_THREAD) + 5u);

    collect_published(collector, 0);
    settle_run(collector, kEpoch);
    const auto r = collector.handoff_report_for_test().sched_phase;
    EXPECT_EQ(r.received_buffers, 2u);
    EXPECT_EQ(r.received_records, r.published_records);
    // No host-independent phase denominator exists, so even a class that lines
    // up exactly is not reported as a closed set.
    EXPECT_EQ(r.coverage, Coverage::Unknown);
    EXPECT_EQ(r.verdict, Handoff::Unknown);
    EXPECT_FALSE(r.silent_loss_known);

    collector.finalize(nullptr, ha_free);
}

// The orchestrator's pool files under a pool ordinal that differs from its
// thread index, so its commit is counted on its own class.
TEST(ChipSwimlaneHandoffAccountingTest, OrchPhaseFlushCountsItsCommit) {
    ChipSwimlaneCollector collector;
    ASSERT_EQ(init_collector(collector, /*num_aicore=*/1, ChipSwimlaneLevel::ORCH_PHASES), 0);
    constexpr uint64_t kEpoch = 8170;
    arm_run(collector, /*num_aicore=*/1, kEpoch, "orch", ChipSwimlaneLevel::ORCH_PHASES);
    chip_swimlane_aicpu_init_phase(/*worker_count=*/1, /*num_sched_phase_threads=*/1, /*num_orch_phase_threads=*/1);
    chip_swimlane_aicpu_set_orch_thread_idx(/*thread_idx=*/0);

    for (int i = 0; i < 6; i++) {
        chip_swimlane_aicpu_record_orch_phase(
            /*start_time=*/300 + i, /*end_time=*/400 + i, /*task_id=*/static_cast<uint64_t>(i), /*submit_idx=*/i
        );
    }
    chip_swimlane_aicpu_flush_orch_phase_buffer(/*thread_idx=*/0);

    const ChipSwimlaneRunTerminal *t = terminal_at(collector, PLATFORM_RUN_TERMINAL_ORCH_PHASE_BASE + 0);
    EXPECT_EQ(t->published_buffers, 1u);
    EXPECT_EQ(t->published_records, 6u);

    collect_published(collector, 0);
    settle_run(collector, kEpoch);
    const auto r = collector.handoff_report_for_test().orch_phase;
    EXPECT_EQ(r.received_buffers, 1u);
    EXPECT_EQ(r.received_records, 6u);
    EXPECT_EQ(r.coverage, Coverage::Unknown);

    collector.finalize(nullptr, ha_free);
}

// A record the producer could not place is charged as dropped and never
// appears as a handoff. This is the record-placement refusal only: it says
// nothing about a failed `enqueue_ready`, whose publication and retry
// semantics are a different path and are not covered here.
TEST(ChipSwimlaneHandoffAccountingTest, UnplaceableRecordIsDroppedNotPublished) {
    ChipSwimlaneCollector collector;
    ASSERT_EQ(init_collector(collector, /*num_aicore=*/1), 0);
    constexpr uint64_t kEpoch = 8200;
    arm_run(collector, /*num_aicore=*/1, kEpoch, "no_room");

    void *shm = collector.get_chip_swimlane_setup_device_ptr();
    auto *state = get_perf_buffer_state(shm, 0);
    auto *buf = reinterpret_cast<ChipSwimlaneAicpuTaskBuffer *>(state->head.current_buf_ptr);
    ASSERT_NE(buf, nullptr);
    buf->count = static_cast<uint32_t>(PLATFORM_PROF_BUFFER_SIZE);

    chip_swimlane_aicpu_complete_task(
        /*core_id=*/0, /*thread_idx=*/0, /*reg_task_id=*/1, /*dispatch_time=*/10, /*finish_time=*/20
    );

    // Asserted on the head the terminal copies from, before any flush, so the
    // refusal is isolated from the tail publication that would follow it.
    EXPECT_EQ(state->head.total_record_count, 1u);
    EXPECT_EQ(state->head.dropped_record_count, 1u);
    EXPECT_EQ(state->head.published_buffer_count, 0u) << "nothing was handed over";
    EXPECT_EQ(state->head.published_record_count, 0u);

    collector.finalize(nullptr, ha_free);
}

// The accounting uses the count read before publication, and the buffer is
// taken away inside the commit that published it.
//
// The free queue is emptied first, so the rotation blocks in the engine's pop
// gate after its enqueue: the host thread below then owns the published
// buffer, overwrites its count, and only then recycles it — which is what
// releases the producer. The producer therefore cannot return until the
// overwrite has landed, so any read of that buffer from the pop onwards would
// be wrong. The window between the ready-queue tail advance and the very next
// statement is narrower than any test outside the producer's own instruction
// stream can hold open, and this does not close it.
TEST(ChipSwimlaneHandoffAccountingTest, PublishedCountSurvivesRecycleInsideTheCommit) {
    ChipSwimlaneCollector collector;
    ASSERT_EQ(init_collector(collector, /*num_aicore=*/1), 0);
    constexpr uint64_t kEpoch = 8300;
    arm_run(collector, /*num_aicore=*/1, kEpoch, "recycle");

    void *shm = collector.get_chip_swimlane_setup_device_ptr();
    auto *header = get_chip_swimlane_header(shm);
    auto *state = get_perf_buffer_state(shm, 0);
    ChipSwimlaneFreeQueue *free_queue = &state->free_queue;

    // Keep one buffer aside, then leave the pool with nothing to rotate into.
    ASSERT_NE(free_queue->head, free_queue->tail);
    const uint64_t spare = free_queue->buffer_ptrs[free_queue->head % PLATFORM_PROF_SLOT_COUNT];
    free_queue->head = free_queue->tail;
    wmb();

    std::atomic<bool> took_ownership{false};
    std::thread host([&] {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        uint64_t published = 0;
        while (std::chrono::steady_clock::now() < deadline) {
            if (header->queue_tails[0] != 0) {
                rmb();
                published = header->queues[0][0].buffer_ptr;
                break;
            }
        }
        if (published == 0) {
            // Never leave the producer sitting on its 30-second backpressure
            // budget; the assertion below is what fails the case.
            recycle_buffer(free_queue, spare);
            return;
        }
        auto *buf = reinterpret_cast<ChipSwimlaneAicpuTaskBuffer *>(published);
        buf->count = 7;
        buf->run_epoch = 0;
        took_ownership.store(true, std::memory_order_release);
        recycle_buffer(free_queue, published);
    });

    complete_aicpu_tasks(/*core_id=*/0, PLATFORM_PROF_BUFFER_SIZE);
    host.join();

    ASSERT_TRUE(took_ownership.load(std::memory_order_acquire))
        << "the host thread never took the buffer, so nothing was raced";
    EXPECT_EQ(state->head.published_buffer_count, 1u);
    EXPECT_EQ(state->head.published_record_count, static_cast<uint32_t>(PLATFORM_PROF_BUFFER_SIZE))
        << "counted from the value saved before publication";

    collector.finalize(nullptr, ha_free);
}

// Counters that would wrap saturate instead — both of them, driven across the
// boundary by the production increments — and a saturated class yields no
// numeric verdict. In particular it must not read as an empty, complete run.
TEST(ChipSwimlaneHandoffAccountingTest, SimultaneousTotalAndDropWrapSaturates) {
    ChipSwimlaneCollector collector;
    ASSERT_EQ(init_collector(collector, /*num_aicore=*/1), 0);
    constexpr uint64_t kEpoch = 8400;
    arm_run(collector, /*num_aicore=*/1, kEpoch, "saturate");

    void *shm = collector.get_chip_swimlane_setup_device_ptr();
    auto *state = get_perf_buffer_state(shm, 0);
    auto *buf = reinterpret_cast<ChipSwimlaneAicpuTaskBuffer *>(state->head.current_buf_ptr);
    ASSERT_NE(buf, nullptr);

    // Start just below the boundary and reach it through the real commit path:
    // a full buffer makes every one of these attempts both a commit attempt
    // and a drop, so total and dropped cross together.
    state->head.total_record_count = UINT32_MAX - 2;
    state->head.dropped_record_count = UINT32_MAX - 2;
    buf->count = static_cast<uint32_t>(PLATFORM_PROF_BUFFER_SIZE);
    for (int i = 0; i < 6; i++) {
        EXPECT_EQ(
            chip_swimlane_aicpu_complete_task(
                /*core_id=*/0, /*thread_idx=*/0, /*reg_task_id=*/static_cast<uint32_t>(i + 1), /*dispatch_time=*/10,
                /*finish_time=*/20
            ),
            -1
        );
    }
    EXPECT_EQ(state->head.total_record_count, UINT32_MAX) << "saturated, not wrapped to a small value";
    EXPECT_EQ(state->head.dropped_record_count, UINT32_MAX) << "the drop side saturates on the same path";

    buf->count = 0;  // the tail flush must not republish the refusal buffer
    flush_core(/*core_id=*/0);

    const ChipSwimlaneRunTerminal *t = aicpu_terminal(collector, 0);
    EXPECT_EQ(t->total, UINT32_MAX);
    EXPECT_EQ(t->dropped, UINT32_MAX);

    collect_published(collector, 0);
    settle_run(collector, kEpoch);
    const auto r = collector.handoff_report_for_test().aicpu_task;
    EXPECT_EQ(r.verdict, Handoff::Saturated);
    EXPECT_FALSE(r.silent_loss_known) << "no exact loss figure may be derived from a bound";

    collector.finalize(nullptr, ha_free);
}

// An unsettled producer is reported as such, never as loss: its own accounting
// never closed, so the three device figures are not a closed set.
TEST(ChipSwimlaneHandoffAccountingTest, UnsettledLiveAtCloseSuppressesTheVerdict) {
    ChipSwimlaneCollector collector;
    ASSERT_EQ(init_collector(collector, /*num_aicore=*/1), 0);
    constexpr uint64_t kEpoch = 8500;
    arm_run(collector, /*num_aicore=*/1, kEpoch, "unsettled");

    dispatch_aicore(collector, /*core_id=*/0, 3);
    flush_core(/*core_id=*/0);

    // Stand in for a producer reaped before its settlement step ran.
    void *shm = collector.get_chip_swimlane_setup_device_ptr();
    get_aicore_buffer_state(shm, 0)->head.live_record_count = 7;
    chip_swimlane_aicpu_flush_sched_phase_buffer(/*thread_idx=*/0);

    collect_published(collector, 0);
    collector.reconcile_counters();
    // Re-close the AICore terminal with the unsettled live value in place.
    const ChipSwimlaneRunTerminal *t = aicore_terminal(collector, 0);
    const_cast<ChipSwimlaneRunTerminal *>(t)->live_at_close = 7;
    collector.report_run_terminal_snapshot(/*bank_index=*/0, kEpoch);

    const auto r = collector.handoff_report_for_test().aicore_task;
    EXPECT_EQ(r.verdict, Handoff::Unsettled);
    EXPECT_FALSE(r.silent_loss_known);
    EXPECT_EQ(r.live_at_close, 7u);

    collector.finalize(nullptr, ha_free);
}

// A buffer the host cannot attribute is still counted, at the layer that can
// hold it: a kind outside the four classes has no class receipt at all, and an
// unattributable index or epoch never discharges this run's handoff.
TEST(ChipSwimlaneHandoffAccountingTest, UnattributableBuffersAreCountedButNeverReceived) {
    ChipSwimlaneCollector collector;
    ASSERT_EQ(init_collector(collector, /*num_aicore=*/1), 0);
    constexpr uint64_t kEpoch = 8600;
    arm_run(collector, /*num_aicore=*/1, kEpoch, "quarantine");

    dispatch_aicore(collector, /*core_id=*/0, 2);
    flush_core(/*core_id=*/0);
    collect_published(collector, 0);

    void *shm = collector.get_chip_swimlane_setup_device_ptr();
    auto *header = get_chip_swimlane_header(shm);
    auto *real = reinterpret_cast<ChipSwimlaneAicoreTaskBuffer *>(header->queues[0][0].buffer_ptr);

    // Out-of-range core index: must not reach a per-producer array.
    ReadyBufferInfo bad_index{};
    bad_index.type = ProfBufferType::AICORE_TASK;
    bad_index.index = 4096;
    bad_index.dev_buffer_ptr = real;
    bad_index.host_buffer_ptr = real;
    collector.on_buffer_collected(bad_index, /*collector_shard=*/0);

    // Another run's stamp: must not count toward this run's receipt.
    ChipSwimlaneAicoreTaskBuffer foreign{};
    foreign.run_epoch = kEpoch + 99;
    foreign.count = 3;
    ReadyBufferInfo foreign_info{};
    foreign_info.type = ProfBufferType::AICORE_TASK;
    foreign_info.index = 0;
    foreign_info.dev_buffer_ptr = &foreign;
    foreign_info.host_buffer_ptr = &foreign;
    collector.on_buffer_collected(foreign_info, /*collector_shard=*/0);

    // A kind no class owns: counted before the kind picks a class, because
    // afterwards there is nowhere to put it.
    ReadyBufferInfo unroutable = foreign_info;
    unroutable.type = static_cast<ProfBufferType>(9);
    collector.on_buffer_collected(unroutable, /*collector_shard=*/0);

    settle_run(collector, kEpoch);
    const auto report = collector.handoff_report_for_test();
    const auto r = report.aicore_task;
    EXPECT_EQ(report.presented_buffers, 4u) << "all four were handed to the collector";
    EXPECT_EQ(report.unroutable_buffers, 1u);
    EXPECT_EQ(report.transport_retired_buffers, 0u) << "the drain path retired nothing in this test";
    EXPECT_EQ(r.observed_buffers, 3u) << "only the three with a class of their own";
    EXPECT_EQ(r.invalid_index_buffers, 1u);
    EXPECT_EQ(r.foreign_epoch_buffers, 1u);
    EXPECT_EQ(r.received_buffers, 1u) << "only the run's own buffer discharges its handoff";
    EXPECT_EQ(r.verdict, Handoff::Match);

    collector.finalize(nullptr, ha_free);
}

// A malformed count still counts the handoff — it happened — but its record
// figure is not trusted, and the class says so instead of reporting a number.
// Record trust is reported alongside the verdict, not folded into it.
TEST(ChipSwimlaneHandoffAccountingTest, MalformedCountKeepsTheBufferAndDistrustsTheRecords) {
    ChipSwimlaneCollector collector;
    ASSERT_EQ(init_collector(collector, /*num_aicore=*/1), 0);
    constexpr uint64_t kEpoch = 8700;
    arm_run(collector, /*num_aicore=*/1, kEpoch, "malformed");

    dispatch_aicore(collector, /*core_id=*/0, 2);
    flush_core(/*core_id=*/0);

    void *shm = collector.get_chip_swimlane_setup_device_ptr();
    auto *header = get_chip_swimlane_header(shm);
    auto *published = reinterpret_cast<ChipSwimlaneAicoreTaskBuffer *>(header->queues[0][0].buffer_ptr);
    published->count = static_cast<uint32_t>(PLATFORM_AICORE_BUFFER_SIZE) + 11u;

    collect_published(collector, 0);
    settle_run(collector, kEpoch);

    const auto r = collector.handoff_report_for_test().aicore_task;
    EXPECT_EQ(r.received_buffers, 1u) << "the handoff still happened";
    EXPECT_EQ(r.malformed_count_buffers, 1u);
    EXPECT_EQ(r.received_records, 0u) << "an out-of-range count is not used as a record figure";
    EXPECT_FALSE(r.records_trusted);
    EXPECT_EQ(r.coverage, Coverage::Complete) << "coverage is unaffected by what a buffer's count said";
    EXPECT_EQ(r.verdict, Handoff::RecordsUntrusted);

    collector.finalize(nullptr, ha_free);
}

// Equal buffer totals say nothing about the records inside them. One buffer
// carrying five records, received as one buffer carrying two, is not a match.
TEST(ChipSwimlaneHandoffAccountingTest, EqualBufferTotalsWithUnequalRecordsIsNotMatch) {
    ChipSwimlaneCollector collector;
    ASSERT_EQ(init_collector(collector, /*num_aicore=*/1), 0);
    constexpr uint64_t kEpoch = 8750;
    arm_run(collector, /*num_aicore=*/1, kEpoch, "mismatch");

    dispatch_aicore(collector, /*core_id=*/0, 5);
    flush_core(/*core_id=*/0);

    // In range, so the count is trusted — and wrong. Only a record comparison
    // can see this.
    void *shm = collector.get_chip_swimlane_setup_device_ptr();
    auto *header = get_chip_swimlane_header(shm);
    reinterpret_cast<ChipSwimlaneAicoreTaskBuffer *>(header->queues[0][0].buffer_ptr)->count = 2;

    collect_published(collector, 0);
    settle_run(collector, kEpoch);

    const auto r = collector.handoff_report_for_test().aicore_task;
    EXPECT_EQ(r.published_buffers, 1u);
    EXPECT_EQ(r.received_buffers, 1u);
    EXPECT_TRUE(r.records_trusted);
    EXPECT_EQ(r.published_records, 5u);
    EXPECT_EQ(r.received_records, 2u);
    EXPECT_EQ(r.verdict, Handoff::RecordMismatch);

    collector.finalize(nullptr, ha_free);
}

// The host layer counts a zero-record handoff as a handoff. The four device
// commit sites all guard against publishing an empty buffer today
// (`chip_swimlane_collector_aicpu.cpp:829`, `:886`, `:1214`, and the engine
// only rotates at capacity), so this pins the host half of the rule.
TEST(ChipSwimlaneHandoffAccountingTest, ZeroRecordReceiptStillCountsAsABuffer) {
    ChipSwimlaneCollector collector;
    ASSERT_EQ(init_collector(collector, /*num_aicore=*/1), 0);
    constexpr uint64_t kEpoch = 8800;
    arm_run(collector, /*num_aicore=*/1, kEpoch, "empty");

    ChipSwimlaneAicoreTaskBuffer empty{};
    empty.run_epoch = kEpoch;
    empty.count = 0;
    ReadyBufferInfo info{};
    info.type = ProfBufferType::AICORE_TASK;
    info.index = 0;
    info.dev_buffer_ptr = &empty;
    info.host_buffer_ptr = &empty;
    collector.on_buffer_collected(info, /*collector_shard=*/0);

    settle_run(collector, kEpoch);
    const auto r = collector.handoff_report_for_test().aicore_task;
    EXPECT_EQ(r.observed_buffers, 1u);
    EXPECT_EQ(r.received_buffers, 1u);
    EXPECT_EQ(r.received_records, 0u);

    collector.finalize(nullptr, ha_free);
}

// A producer that closed with zeros covered its index; the run is complete and
// the comparison holds. This is the case an absent producer must not be
// confused with — see the two below.
TEST(ChipSwimlaneHandoffAccountingTest, ZeroValuedTerminalIsCompleteNotAbsent) {
    ChipSwimlaneCollector collector;
    ASSERT_EQ(init_collector(collector, /*num_aicore=*/1), 0);
    constexpr uint64_t kEpoch = 8850;
    arm_run(collector, /*num_aicore=*/1, kEpoch, "idle");

    flush_core(/*core_id=*/0);  // nothing recorded; both pools still close
    collect_published(collector, 0);
    settle_run(collector, kEpoch);

    const auto r = collector.handoff_report_for_test().aicore_task;
    EXPECT_EQ(r.coverage, Coverage::Complete);
    EXPECT_EQ(r.reported_producers, 1);
    EXPECT_EQ(r.missing_producers, 0);
    EXPECT_EQ(r.published_buffers, 0u);
    EXPECT_EQ(r.received_buffers, 0u);
    EXPECT_EQ(r.verdict, Handoff::Match);
    EXPECT_TRUE(r.silent_loss_known);

    collector.finalize(nullptr, ha_free);
}

// An expected producer that published no terminal entry leaves a partial sum.
// Summing it as if it were the whole class is exactly how a missing producer
// reads as a complete, matching run, so no exact verdict is offered.
TEST(ChipSwimlaneHandoffAccountingTest, MissingTaskTerminalIsIncompleteNotComplete) {
    ChipSwimlaneCollector collector;
    ASSERT_EQ(init_collector(collector, /*num_aicore=*/2), 0);
    constexpr uint64_t kEpoch = 8870;
    arm_run(collector, /*num_aicore=*/2, kEpoch, "partial");

    dispatch_aicore(collector, /*core_id=*/0, 3);
    dispatch_aicore(collector, /*core_id=*/1, 4);
    // Only core 0 is flushed, so core 1's producers never close their entries.
    flush_core(/*core_id=*/0);
    collect_published(collector, 0);
    settle_run(collector, kEpoch);

    const auto report = collector.handoff_report_for_test();
    for (const auto *r : {&report.aicore_task, &report.aicpu_task}) {
        EXPECT_EQ(r->coverage, Coverage::Incomplete);
        EXPECT_EQ(r->expected_producers, 2);
        EXPECT_EQ(r->reported_producers, 1);
        EXPECT_EQ(r->missing_producers, 1);
        EXPECT_EQ(r->verdict, Handoff::Incomplete);
        EXPECT_FALSE(r->silent_loss_known) << "a partial sum is not a loss figure";
    }

    collector.finalize(nullptr, ha_free);
}

// A class with no expected set and no entries is unknown, not "no producer of
// this class exists". Nothing available to the host distinguishes a disabled
// phase pool from one whose producers never closed.
TEST(ChipSwimlaneHandoffAccountingTest, AbsentPhaseClassIsUnknownNotNotApplicable) {
    ChipSwimlaneCollector collector;
    ASSERT_EQ(init_collector(collector, /*num_aicore=*/1), 0);
    constexpr uint64_t kEpoch = 8900;
    arm_run(collector, /*num_aicore=*/1, kEpoch, "absent");

    dispatch_aicore(collector, /*core_id=*/0, 2);
    flush_core(/*core_id=*/0);
    collect_published(collector, 0);
    settle_run(collector, kEpoch);

    const auto report = collector.handoff_report_for_test();
    EXPECT_EQ(report.aicore_task.verdict, Handoff::Match) << "this class did report";
    EXPECT_EQ(report.orch_phase.coverage, Coverage::Unknown);
    EXPECT_EQ(report.orch_phase.verdict, Handoff::Unknown) << "absence without a denominator is not n/a";
    EXPECT_EQ(report.orch_phase.reported_producers, 0);

    collector.finalize(nullptr, ha_free);
}

namespace {

uint32_t sched_phase_free_depth(void *shm, int thread_idx) {
    const auto *pool = get_sched_phase_buffer_state(shm, thread_idx);
    return pool->free_queue.tail - pool->free_queue.head;
}

uint32_t orch_phase_free_depth(void *shm) {
    const auto *pool = get_orch_phase_buffer_state(shm, 0);
    return pool->free_queue.tail - pool->free_queue.head;
}

void record_orch_phases(int records) {
    for (int i = 0; i < records; i++) {
        chip_swimlane_aicpu_record_orch_phase(
            /*start_time=*/300 + i, /*end_time=*/400 + i, /*task_id=*/static_cast<uint64_t>(i), /*submit_idx=*/i
        );
    }
}

}  // namespace

// A run that publishes nothing keeps its primed sched-phase buffer on the head,
// and the next run takes that same allocation over. Popping a replacement would
// strand it: the host never received it, and AICPU is the free queue's consumer
// and never its producer.
TEST(ChipSwimlaneHandoffAccountingTest, AnEmptySchedPhaseFlushGivesItsBufferToTheNextRun) {
    ChipSwimlaneCollector collector;
    ASSERT_EQ(init_collector(collector, /*num_aicore=*/1, ChipSwimlaneLevel::SCHED_PHASES), 0);
    constexpr uint64_t kFirst = 8940;
    constexpr uint64_t kSecond = 8950;
    arm_run(collector, /*num_aicore=*/1, kFirst, "phase-empty", ChipSwimlaneLevel::SCHED_PHASES);
    chip_swimlane_aicpu_init_phase(/*worker_count=*/1, /*num_sched_phase_threads=*/1, /*num_orch_phase_threads=*/0);

    void *shm = collector.get_chip_swimlane_setup_device_ptr();
    auto *state = get_sched_phase_buffer_state(shm, 0);
    const uint64_t primed = state->head.current_buf_ptr;
    ASSERT_NE(primed, 0u);
    const uint32_t depth_after_prime = sched_phase_free_depth(shm, 0);

    chip_swimlane_aicpu_flush_sched_phase_buffer(/*thread_idx=*/0);

    EXPECT_EQ(state->head.current_buf_ptr, primed) << "an empty final flush released a buffer the host never took";
    EXPECT_EQ(state->head.published_buffer_count, 0u);
    const ChipSwimlaneRunTerminal *first = terminal_at(collector, PLATFORM_RUN_TERMINAL_SCHED_PHASE_BASE + 0);
    EXPECT_EQ(first->run_epoch, kFirst) << "an enabled-but-idle pool still closes its terminal";
    EXPECT_EQ(first->total, 0u);
    EXPECT_EQ(first->published_buffers, 0u);

    arm_run(collector, /*num_aicore=*/1, kSecond, "phase-reuse", ChipSwimlaneLevel::SCHED_PHASES);
    chip_swimlane_aicpu_init_phase(/*worker_count=*/1, /*num_sched_phase_threads=*/1, /*num_orch_phase_threads=*/0);

    EXPECT_EQ(state->head.current_buf_ptr, primed) << "init drew a replacement instead of reusing the retained buffer";
    EXPECT_EQ(sched_phase_free_depth(shm, 0), depth_after_prime) << "the adopted buffer cost a free-queue entry";
    const auto *reused = reinterpret_cast<const ChipSwimlaneAicpuSchedPhaseBuffer *>(primed);
    EXPECT_EQ(reused->count, 0u);
    EXPECT_EQ(reused->run_epoch, kSecond) << "a reused buffer kept the previous run's identity";
    EXPECT_EQ(reused->local_seq, 0u);
    EXPECT_EQ(state->head.current_buf_seq, 0u);

    // The adopted buffer is this run's writable head, and what it carries
    // reaches the host under this run's identity.
    record_sched_phases(/*thread_idx=*/0, 3);
    chip_swimlane_aicpu_flush_sched_phase_buffer(/*thread_idx=*/0);

    const ChipSwimlaneRunTerminal *second = terminal_at(collector, PLATFORM_RUN_TERMINAL_SCHED_PHASE_BASE + 0);
    EXPECT_EQ(second->run_epoch, kSecond);
    EXPECT_EQ(second->published_buffers, 1u);
    EXPECT_EQ(second->published_records, 3u);
    EXPECT_EQ(second->dropped, 0u);

    EXPECT_EQ(collect_published(collector, 0), 1u);
    settle_run(collector, kSecond);
    const auto r = collector.handoff_report_for_test().sched_phase;
    EXPECT_EQ(r.received_buffers, 1u);
    EXPECT_EQ(r.received_records, 3u);

    collector.finalize(nullptr, ha_free);
}

// A failed final handoff charges the buffer's records to `dropped`, resets its
// count and keeps it: the host never took it, so nothing else can return it.
// The next run adopts that buffer and republishes from it under its own epoch.
//
// The production cause is a full ready queue, and reaching it means sitting in
// the backpressure gate for PLATFORM_DFX_BACKPRESSURE_TIMEOUT_CYCLES — 30
// seconds of spinning, which a unit test has no business doing. The gate also
// rejects an out-of-range thread index outright, which fails the same
// `enqueue_ready` from the same caller and runs the same branch; the orch flush
// reads its pool at ordinal 0, so that index reaches nothing but the enqueue.
// The gate's own timeout behaviour is not covered here.
TEST(ChipSwimlaneHandoffAccountingTest, AFailedOrchPhaseHandoffKeepsItsBufferForTheNextRun) {
    ChipSwimlaneCollector collector;
    ASSERT_EQ(init_collector(collector, /*num_aicore=*/1, ChipSwimlaneLevel::ORCH_PHASES), 0);
    constexpr uint64_t kFirst = 8960;
    constexpr uint64_t kSecond = 8970;
    arm_run(collector, /*num_aicore=*/1, kFirst, "orch-refused", ChipSwimlaneLevel::ORCH_PHASES);
    chip_swimlane_aicpu_init_phase(/*worker_count=*/1, /*num_sched_phase_threads=*/1, /*num_orch_phase_threads=*/1);
    chip_swimlane_aicpu_set_orch_thread_idx(/*thread_idx=*/0);

    void *shm = collector.get_chip_swimlane_setup_device_ptr();
    auto *state = get_orch_phase_buffer_state(shm, 0);
    const uint64_t held = state->head.current_buf_ptr;
    ASSERT_NE(held, 0u);
    const uint32_t depth_after_prime = orch_phase_free_depth(shm);
    record_orch_phases(4);
    ASSERT_EQ(reinterpret_cast<const ChipSwimlaneAicpuOrchPhaseBuffer *>(held)->count, 4u);

    chip_swimlane_aicpu_flush_orch_phase_buffer(/*thread_idx=*/PLATFORM_MAX_AICPU_THREADS);

    EXPECT_EQ(state->head.current_buf_ptr, held) << "a buffer the host never received was released";
    EXPECT_EQ(reinterpret_cast<const ChipSwimlaneAicpuOrchPhaseBuffer *>(held)->count, 0u)
        << "a retained buffer must hold no records once they are charged";
    EXPECT_EQ(state->head.dropped_record_count, 4u);
    EXPECT_EQ(state->head.published_buffer_count, 0u);
    EXPECT_EQ(orch_phase_free_depth(shm), depth_after_prime) << "the failed flush drew from the free queue";
    const ChipSwimlaneRunTerminal *first = terminal_at(collector, PLATFORM_RUN_TERMINAL_ORCH_PHASE_BASE + 0);
    EXPECT_EQ(first->run_epoch, kFirst);
    EXPECT_EQ(first->total, 4u);
    EXPECT_EQ(first->dropped, 4u) << "the refusal is charged, not silent";

    arm_run(collector, /*num_aicore=*/1, kSecond, "orch-reuse", ChipSwimlaneLevel::ORCH_PHASES);
    chip_swimlane_aicpu_init_phase(/*worker_count=*/1, /*num_sched_phase_threads=*/1, /*num_orch_phase_threads=*/1);
    chip_swimlane_aicpu_set_orch_thread_idx(/*thread_idx=*/0);

    EXPECT_EQ(state->head.current_buf_ptr, held) << "init drew a replacement instead of reusing the retained buffer";
    EXPECT_EQ(orch_phase_free_depth(shm), depth_after_prime);
    const auto *reused = reinterpret_cast<const ChipSwimlaneAicpuOrchPhaseBuffer *>(held);
    EXPECT_EQ(reused->run_epoch, kSecond) << "the second run would publish under the first run's identity";
    EXPECT_EQ(reused->local_seq, 0u);

    record_orch_phases(2);
    chip_swimlane_aicpu_flush_orch_phase_buffer(/*thread_idx=*/0);

    const ChipSwimlaneRunTerminal *second = terminal_at(collector, PLATFORM_RUN_TERMINAL_ORCH_PHASE_BASE + 0);
    EXPECT_EQ(second->run_epoch, kSecond);
    EXPECT_EQ(second->published_buffers, 1u);
    EXPECT_EQ(second->published_records, 2u);
    EXPECT_EQ(second->dropped, 0u) << "the predecessor's charge is not this run's";

    EXPECT_EQ(collect_published(collector, 0), 1u);
    settle_run(collector, kSecond);
    const auto r = collector.handoff_report_for_test().orch_phase;
    EXPECT_EQ(r.received_buffers, 1u);
    EXPECT_EQ(r.received_records, 2u);

    collector.finalize(nullptr, ha_free);
}

// A pool that is primed and then neither written nor flushed still owns its
// buffer at the next run's init.
//
// This is the orchestrator's pool on a run below ORCH_PHASES: the
// tensormap_and_ringbuffer cold path passes `orch_phase_threads = 1` for every
// run at SCHED_PHASES or above, while both the orch emit and the orch flush
// require ORCH_PHASES. The pool's buffers come from a run that did ask for
// ORCH_PHASES — the host stocks the device orch pool at that level and nothing
// returns those buffers between runs — which is why the level the collector is
// initialized at differs from the level the runs below arm.
//
// The sched pool is driven through its whole successful lifecycle alongside, so
// the two dispositions are asserted against each other: a pool whose buffer the
// host took draws a replacement, a pool that still owns its buffer does not.
TEST(ChipSwimlaneHandoffAccountingTest, AnUnflushedOrchPhasePoolStaysOwnedAcrossRuns) {
    ChipSwimlaneCollector collector;
    ASSERT_EQ(init_collector(collector, /*num_aicore=*/1, ChipSwimlaneLevel::ORCH_PHASES), 0);
    constexpr uint64_t kFirst = 8980;
    constexpr uint64_t kSecond = 8990;
    arm_run(collector, /*num_aicore=*/1, kFirst, "orch-idle", ChipSwimlaneLevel::SCHED_PHASES);
    chip_swimlane_aicpu_init_phase(/*worker_count=*/1, /*num_sched_phase_threads=*/1, /*num_orch_phase_threads=*/1);

    void *shm = collector.get_chip_swimlane_setup_device_ptr();
    auto *orch = get_orch_phase_buffer_state(shm, 0);
    auto *sched = get_sched_phase_buffer_state(shm, 0);
    const uint64_t orch_primed = orch->head.current_buf_ptr;
    ASSERT_NE(orch_primed, 0u);
    const uint32_t orch_depth = orch_phase_free_depth(shm);
    const uint32_t sched_depth = sched_phase_free_depth(shm, 0);

    // The sched side's run, end to end. The orch pool is never emitted into and
    // never flushed, exactly as the level gates leave it.
    record_sched_phases(/*thread_idx=*/0, 2);
    chip_swimlane_aicpu_flush_sched_phase_buffer(/*thread_idx=*/0);
    ASSERT_EQ(sched->head.current_buf_ptr, 0u) << "a published buffer belongs to the host";
    EXPECT_EQ(orch->head.current_buf_ptr, orch_primed);
    EXPECT_EQ(orch->head.total_record_count, 0u);

    arm_run(collector, /*num_aicore=*/1, kSecond, "orch-idle-next", ChipSwimlaneLevel::SCHED_PHASES);
    chip_swimlane_aicpu_init_phase(/*worker_count=*/1, /*num_sched_phase_threads=*/1, /*num_orch_phase_threads=*/1);

    EXPECT_EQ(orch->head.current_buf_ptr, orch_primed) << "an unflushed pool lost the buffer it still owned";
    EXPECT_EQ(orch_phase_free_depth(shm), orch_depth) << "the adopted buffer cost a free-queue entry";
    const auto *reused = reinterpret_cast<const ChipSwimlaneAicpuOrchPhaseBuffer *>(orch_primed);
    EXPECT_EQ(reused->count, 0u);
    EXPECT_EQ(reused->run_epoch, kSecond);
    EXPECT_EQ(reused->local_seq, 0u);

    EXPECT_NE(sched->head.current_buf_ptr, 0u);
    EXPECT_NE(sched->head.current_buf_ptr, orch_primed);
    EXPECT_EQ(sched_phase_free_depth(shm, 0), sched_depth - 1)
        << "a pool whose buffer the host took must draw a replacement";

    collector.finalize(nullptr, ha_free);
}
