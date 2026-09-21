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
 * dep_gen run identity and buffer ownership, driven end to end through
 * production code: the real host collector allocates and seeds the pool, the
 * real AICPU producer acquires, records and flushes, and the collector's own
 * `on_buffer_collected` copies what was published.
 *
 * Two properties, one chain. Identity: a run's records must be attributable to
 * that run after the buffer has gone back to the pool and been re-stamped, so
 * the epoch has to be copied out with the records rather than read back from
 * the device buffer. Ownership: a buffer is released only by a successful
 * enqueue, because AICPU consumes the free queue and never produces into it, so
 * reuse by the next run's init is its only way to hand one back.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <vector>

#include "aicpu/dep_gen_collector_aicpu.h"
#include "aicpu/device_run_result_base_aicpu.h"
#include "common/dep_gen.h"
#include "common/memory_barrier.h"
#include "host/dep_gen_collector.h"

namespace {

// The collector stores this byte and never reads it back: DepFlags is declared
// in each runtime's own types.h, which is why the entry point takes a plain
// uint8_t. Naming the enumerators here would tie this case to one runtime's
// spelling of a value the code under test does not interpret.
constexpr uint8_t kCreatorEdgeDepKinds = 0x3;  // wait | retain

void *dep_gen_test_alloc(size_t size) { return std::calloc(1, size); }

int dep_gen_test_free(void *ptr) {
    std::free(ptr);
    return 0;
}

uint32_t free_queue_depth(void *shm) {
    const DepGenBufferState *state = get_dep_gen_buffer_state(shm, 0);
    return state->free_queue.tail - state->free_queue.head;
}

bool free_queue_holds(const DepGenFreeQueue &fq, uint64_t buf_ptr) {
    for (uint32_t i = fq.head; i != fq.tail; i++) {
        if (fq.buffer_ptrs[i % PLATFORM_DEP_GEN_SLOT_COUNT] == buf_ptr) return true;
    }
    return false;
}

// The host's half of the SPSC protocol: write the slot, fence, publish the tail.
// AICPU is the queue's consumer and has no push of its own, so a test that needs
// a buffer back in the pool has to play the host here.
void host_push_free_queue(DepGenFreeQueue &fq, uint64_t buf_ptr) {
    ASSERT_LT(fq.tail - fq.head, static_cast<uint32_t>(PLATFORM_DEP_GEN_SLOT_COUNT)) << "free queue is full";
    fq.buffer_ptrs[fq.tail % PLATFORM_DEP_GEN_SLOT_COUNT] = buf_ptr;
    wmb();
    fq.tail = fq.tail + 1;
    wmb();
}

class DepGenRunIdentityTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_EQ(
            collector_.init(/*num_threads=*/1, dep_gen_test_alloc, nullptr, dep_gen_test_free, /*device_id=*/0), 0
        );
        shm_ = collector_.get_dep_gen_shm_device_ptr();
        ASSERT_NE(shm_, nullptr);
        state_ = get_dep_gen_buffer_state(shm_, 0);
        header_ = get_dep_gen_header(shm_);

        set_dep_gen_enabled(true);
        set_platform_dep_gen_base(reinterpret_cast<uint64_t>(shm_));
        dep_gen_aicpu_set_orch_thread_idx(0);
    }

    void TearDown() override {
        dep_gen_aicpu_finalize();
        set_dep_gen_enabled(false);
        set_platform_dep_gen_base(0);
        set_platform_run_result(0, 0);
        collector_.finalize(nullptr, dep_gen_test_free);
    }

    // One run's device-side sequence: the host opens the window, the device
    // acquires a buffer, records `submits` tasks and flushes.
    void run_once(uint64_t epoch, int submits, int orch_idx_at_flush = 0) {
        collector_.begin_run();
        run_once_without_begin_run(epoch, submits, orch_idx_at_flush);
    }

    // The same device sequence with the host's per-run clear left out, which is
    // what a collection window spanning several runs will look like.
    void run_once_without_begin_run(uint64_t epoch, int submits, int orch_idx_at_flush = 0) {
        set_platform_run_result(/*region_base=*/0, epoch);
        dep_gen_aicpu_set_orch_thread_idx(0);
        dep_gen_aicpu_init();
        for (int i = 0; i < submits; i++) {
            record_submit(0x1000 + static_cast<uint64_t>(i));
        }
        dep_gen_aicpu_set_orch_thread_idx(orch_idx_at_flush);
        dep_gen_aicpu_flush();
        dep_gen_aicpu_set_orch_thread_idx(0);
    }

    void record_submit(uint64_t task_id_raw) {
        const int32_t kernel_ids[3] = {-1, -1, -1};
        dep_gen_aicpu_record_submit(
            task_id_raw, /*in_manual_scope=*/false, /*early_dispatch=*/false, /*tensor_count=*/0,
            /*tensor_ptrs=*/nullptr, /*arg_types=*/nullptr, /*explicit_dep_count=*/0, /*explicit_deps_raw=*/nullptr,
            /*explicit_dep_kinds_raw=*/nullptr, kCreatorEdgeDepKinds, /*block_num=*/1, kernel_ids
        );
    }

    // Hand every ready entry the device published since `consumed_tail_` to the
    // collector's own copy path, and return how many buffers that was.
    int collect_published() {
        const uint32_t tail = header_->queue_tails[0];
        int collected = 0;
        for (uint32_t i = consumed_tail_; i < tail; i++) {
            const DepGenReadyQueueEntry &entry = header_->queues[0][i];
            DepGenReadyBufferInfo info{};
            info.instance_index = entry.instance_index;
            info.thread_index = 0;
            info.dev_buffer_ptr = reinterpret_cast<void *>(entry.buffer_ptr);
            info.host_buffer_ptr = reinterpret_cast<void *>(entry.buffer_ptr);
            info.buffer_seq = entry.buffer_seq;
            collector_.on_buffer_collected(info);
            collected++;
        }
        consumed_tail_ = tail;
        return collected;
    }

    uint64_t published_buffer_at(uint32_t index) const { return header_->queues[0][index].buffer_ptr; }

    DepGenCollector collector_;
    void *shm_ = nullptr;
    DepGenBufferState *state_ = nullptr;
    DepGenDataHeader *header_ = nullptr;
    uint32_t consumed_tail_ = 0;
};

// A run's records reach the host carrying that run's epoch, and the grouping
// keeps them separate from a later run's. The stamp is production: the engine's
// pop hook writes it, and nothing in the test touches `run_epoch`.
TEST_F(DepGenRunIdentityTest, EachRunsRecordsCarryItsOwnEpoch) {
    run_once(/*epoch=*/11, /*submits=*/2);
    ASSERT_EQ(collect_published(), 1) << "the first run published no buffer";

    run_once(/*epoch=*/12, /*submits=*/1);
    ASSERT_EQ(collect_published(), 1) << "the second run published no buffer";

    // begin_run() still clears per run, so the collector holds only the latest.
    // What this pins is that the run it holds is stamped with its own epoch and
    // not the previous one's — the buffer may well be the same storage.
    const auto &runs = collector_.runs();
    ASSERT_EQ(runs.size(), 1u);
    EXPECT_EQ(runs.begin()->first, 12u) << "the second run's records carried the first run's identity";
    EXPECT_EQ(runs.begin()->second.size(), 1u);

    uint64_t epoch = 0;
    const std::vector<DepGenRecord> *records = collector_.window_records(&epoch);
    ASSERT_NE(records, nullptr);
    EXPECT_EQ(epoch, 12u);
    EXPECT_EQ(records->size(), 1u);
}

// Identity survives the buffer going back to the pool and being re-stamped:
// the host copy must not be reading it back off the device buffer.
//
// The window deliberately spans several runs (no `begin_run()` after the first)
// because that is the only state in which the question has an answer — with the
// per-run clear in place the host holds one run at a time. The pool is FIFO with
// PLATFORM_DEP_GEN_SLOT_COUNT buffers, so returning each buffer as it is
// collected brings the first run's storage back around on the run after the last
// slot.
TEST_F(DepGenRunIdentityTest, TheHostCopyKeepsItsEpochAfterTheBufferIsRestamped) {
    constexpr uint64_t kFirstEpoch = 21;
    run_once(kFirstEpoch, /*submits=*/1);
    ASSERT_EQ(collect_published(), 1);
    const uint64_t first = published_buffer_at(0);
    ASSERT_NE(first, 0u);
    ASSERT_EQ(reinterpret_cast<const DepGenBuffer *>(first)->run_epoch, kFirstEpoch);
    host_push_free_queue(state_->free_queue, first);

    // Cycle the pool. The last of these reacquires `first`.
    const uint64_t last_epoch = kFirstEpoch + static_cast<uint64_t>(PLATFORM_DEP_GEN_SLOT_COUNT);
    for (uint64_t epoch = kFirstEpoch + 1; epoch <= last_epoch; epoch++) {
        run_once_without_begin_run(epoch, /*submits=*/1);
        ASSERT_EQ(collect_published(), 1) << "run " << epoch << " published nothing";
        host_push_free_queue(state_->free_queue, published_buffer_at(consumed_tail_ - 1));
    }

    ASSERT_EQ(published_buffer_at(consumed_tail_ - 1), first)
        << "the returned buffer never came back around; the FIFO assumption above is wrong";
    EXPECT_EQ(reinterpret_cast<const DepGenBuffer *>(first)->run_epoch, last_epoch)
        << "the reused buffer was not re-stamped, so this test cannot show the host copy is independent";

    // The device storage now says `last_epoch`. The host copy taken back when it
    // said 21 must still say 21, and must still hold that run's one record.
    const auto &runs = collector_.runs();
    ASSERT_EQ(runs.count(kFirstEpoch), 1u) << "the first run's records lost their identity";
    EXPECT_EQ(runs.at(kFirstEpoch).size(), 1u);
    EXPECT_EQ(runs.size(), static_cast<size_t>(PLATFORM_DEP_GEN_SLOT_COUNT) + 1)
        << "runs were merged or dropped across the window";
}

// An idle run must not cost the pool a buffer. dep_gen's flush already returns
// early when the buffer is empty, leaving the pointer set; the loss was at
// init, which popped a replacement over it.
TEST_F(DepGenRunIdentityTest, IdleRunsDoNotConsumeABufferEach) {
    const uint32_t depth_at_start = free_queue_depth(shm_);
    ASSERT_GT(depth_at_start, 1u) << "need more than one free buffer for per-run consumption to be visible";

    // The first run legitimately draws one: the instance holds none yet.
    run_once(/*epoch=*/31, /*submits=*/0);
    const uint32_t depth_after_first = free_queue_depth(shm_);
    ASSERT_EQ(depth_after_first, depth_at_start - 1);
    ASSERT_NE(state_->current_buf_ptr, 0u);
    const uint64_t held = state_->current_buf_ptr;

    for (uint64_t epoch = 32; epoch <= 34; epoch++) {
        run_once(epoch, /*submits=*/0);
        EXPECT_EQ(free_queue_depth(shm_), depth_after_first)
            << "run " << epoch << " drew a buffer instead of reusing the retained one";
        EXPECT_EQ(state_->current_buf_ptr, held) << "run " << epoch << " swapped the buffer it already held";
        EXPECT_EQ(reinterpret_cast<const DepGenBuffer *>(held)->run_epoch, epoch)
            << "run " << epoch << " reused the buffer without re-stamping it";
    }

    EXPECT_EQ(header_->queue_tails[0], 0u) << "an idle run published a buffer";
}

// A failed enqueue charges dropped once, zeroes the count, and keeps the
// buffer for the next run's init.
//
// The production cause is a full ready queue, which means sitting in the
// backpressure gate; the gate also rejects an out-of-range orchestrator thread
// index outright, which fails the same enqueue for the same caller and runs the
// same branch. So the branch is covered; the gate's own timeout is not, and
// belongs to the engine's tests.
TEST_F(DepGenRunIdentityTest, AFailedFlushChargesDroppedOnceAndKeepsTheBuffer) {
    collector_.begin_run();
    set_platform_run_result(/*region_base=*/0, /*run_epoch=*/41);
    dep_gen_aicpu_set_orch_thread_idx(0);
    dep_gen_aicpu_init();

    const uint64_t held = state_->current_buf_ptr;
    ASSERT_NE(held, 0u);
    const uint32_t depth_after_init = free_queue_depth(shm_);

    record_submit(0x900);
    record_submit(0x901);
    ASSERT_EQ(reinterpret_cast<const DepGenBuffer *>(held)->count, 2u);

    dep_gen_aicpu_set_orch_thread_idx(PLATFORM_MAX_AICPU_THREADS);
    dep_gen_aicpu_flush();
    dep_gen_aicpu_set_orch_thread_idx(0);

    EXPECT_EQ(state_->dropped_record_count, 2u)
        << "the failed enqueue charged dropped by something other than the buffer's record count";
    EXPECT_EQ(state_->current_buf_ptr, held)
        << "a buffer the host never received was released, so nothing can return it";
    EXPECT_EQ(free_queue_depth(shm_), depth_after_init) << "the failed flush drew from the free queue";
    EXPECT_FALSE(free_queue_holds(state_->free_queue, held)) << "AICPU pushed the free queue it only consumes";
    EXPECT_EQ(reinterpret_cast<const DepGenBuffer *>(held)->count, 0u)
        << "the retained buffer still claims records that were charged to dropped";

    // Reusable in place: the next run takes it and re-stamps it.
    collector_.begin_run();
    set_platform_run_result(/*region_base=*/0, /*run_epoch=*/42);
    dep_gen_aicpu_init();
    EXPECT_EQ(state_->current_buf_ptr, held) << "init drew a replacement instead of reusing the retained buffer";
    const auto *reused = reinterpret_cast<const DepGenBuffer *>(held);
    EXPECT_EQ(reused->count, 0u);
    EXPECT_EQ(reused->run_epoch, 42u) << "a reused buffer kept the previous run's identity";
    EXPECT_EQ(state_->current_buf_seq, 0u);
}

// A published buffer is the host's, not the pool's: retention covers the failed
// hand-over only, so a successful flush still releases the pointer, and the pool
// must not hand that buffer out again before the host returns it.
TEST_F(DepGenRunIdentityTest, APublishedBufferIsNotAlsoRetainedByTheDevice) {
    run_once(/*epoch=*/51, /*submits=*/1);

    EXPECT_EQ(state_->current_buf_ptr, 0u) << "the published buffer is still the instance's active one";
    ASSERT_EQ(collect_published(), 1);
    const uint64_t published = published_buffer_at(0);
    EXPECT_FALSE(free_queue_holds(state_->free_queue, published))
        << "the buffer was both published and left in the free queue";

    // Nothing is retained, so the next run draws from the pool — and it must not
    // draw the one the host has not given back.
    collector_.begin_run();
    set_platform_run_result(/*region_base=*/0, /*run_epoch=*/52);
    dep_gen_aicpu_init();
    const uint64_t acquired = state_->current_buf_ptr;
    EXPECT_NE(acquired, 0u) << "init drew nothing after the previous run released its buffer";
    EXPECT_NE(acquired, published) << "the pool handed out a buffer the host had not returned";
    EXPECT_EQ(reinterpret_cast<const DepGenBuffer *>(acquired)->run_epoch, 52u);
}

// A run that submitted nothing has an empty graph, not a missing one. Its
// deps.json must still be written, or "nothing submitted" becomes
// indistinguishable from "collection failed" — and the replay writer accepts
// `num_records == 0` precisely so that file can exist.
TEST_F(DepGenRunIdentityTest, AZeroRecordRunStillYieldsAnEmptyGraph) {
    // A real run that acquires a buffer, records nothing, and flushes: the flush
    // publishes nothing, so no buffer ever reaches the host.
    run_once(/*epoch=*/71, /*submits=*/0);
    ASSERT_EQ(collect_published(), 0) << "an idle run published a buffer";
    ASSERT_TRUE(collector_.runs().empty());

    uint64_t epoch = 12345;
    const std::vector<DepGenRecord> *records = collector_.window_records(&epoch);
    ASSERT_NE(records, nullptr) << "a zero-record run was refused a graph, so no deps.json would be written";
    EXPECT_TRUE(records->empty());
    EXPECT_EQ(epoch, 0u) << "no buffer was stamped, so there is no run identity to report";
}

// The accessor refuses only what it cannot answer. One deps.json describes one
// graph, so a window holding several must not silently emit one of them as if it
// were the whole graph — but a window holding none is answerable, and is covered
// by the case above.
TEST_F(DepGenRunIdentityTest, WindowRecordsRefusesOnlyWhenSeveralRunsShareAWindow) {
    run_once(/*epoch=*/61, /*submits=*/1);
    ASSERT_EQ(collect_published(), 1);
    uint64_t epoch = 0;
    ASSERT_NE(collector_.window_records(&epoch), nullptr);
    EXPECT_EQ(epoch, 61u);

    // A second run collected without an intervening begin_run() is what
    // continuous collection will look like. Until session output decides how a
    // path names its run, that must not resolve to one graph.
    run_once_without_begin_run(/*epoch=*/62, /*submits=*/1);
    ASSERT_EQ(collect_published(), 1);
    EXPECT_EQ(collector_.runs().size(), 2u);
    EXPECT_EQ(collector_.window_records(), nullptr) << "two runs in one window resolved to a single graph";
}

}  // namespace
