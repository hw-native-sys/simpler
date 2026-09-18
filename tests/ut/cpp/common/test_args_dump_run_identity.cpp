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
 * args_dump run identity and buffer ownership, end to end through production
 * code: the real host collector allocates and seeds the pool, the real AICPU
 * producer acquires, records and flushes, and the collector's own
 * `on_buffer_collected` / `export_dump_files` copy and write out what was
 * published.
 *
 * Identity: an arg's run must be recoverable after the buffer has gone back to
 * the pool and been re-stamped, so the epoch is copied out with the records
 * rather than read back from the device buffer. Ownership: a buffer is released
 * only by a successful enqueue, because AICPU consumes the free queue and never
 * produces into it, so reuse by the next run's init is its only way back.
 *
 * Payloads are deliberately out of scope here: every record is a SCALAR with
 * `capture_payload = 0`, so nothing touches the arena. The arena needs no
 * identity of its own — `arena_write_offset` is monotonic across runs, so a
 * record's `payload_offset` is already unique and its run is the record's.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <unistd.h>
#include <vector>

#include "aicpu/args_dump_aicpu.h"
#include "aicpu/device_run_result_base_aicpu.h"
#include "common/args_dump.h"
#include "common/memory_barrier.h"
#include "host/args_dump_collector.h"

namespace {

void *args_dump_test_alloc(size_t size) { return std::calloc(1, size); }

int args_dump_test_free(void *ptr) {
    std::free(ptr);
    return 0;
}

uint32_t free_queue_depth(void *shm, int thread_idx) {
    const DumpBufferState *state = get_dump_buffer_state(shm, thread_idx);
    return state->free_queue.tail - state->free_queue.head;
}

bool free_queue_holds(const DumpFreeQueue &fq, uint64_t buf_ptr) {
    for (uint32_t i = fq.head; i != fq.tail; i++) {
        if (fq.buffer_ptrs[i % PLATFORM_DUMP_SLOT_COUNT] == buf_ptr) return true;
    }
    return false;
}

// The host's half of the SPSC protocol: write the slot, fence, publish the tail.
// AICPU is the queue's consumer and has no push of its own, so a test that needs
// a buffer back in the pool has to play the host here. This models the return,
// not the asynchronous transfer that accompanies it on real hardware.
void host_push_free_queue(DumpFreeQueue &fq, uint64_t buf_ptr) {
    ASSERT_LT(fq.tail - fq.head, static_cast<uint32_t>(PLATFORM_DUMP_SLOT_COUNT)) << "free queue is full";
    fq.buffer_ptrs[fq.tail % PLATFORM_DUMP_SLOT_COUNT] = buf_ptr;
    wmb();
    fq.tail = fq.tail + 1;
    wmb();
}

class ArgsDumpRunIdentityTest : public ::testing::Test {
protected:
    static constexpr int kThreadIdx = 0;

    void SetUp() override {
        dir_ = std::filesystem::temp_directory_path() /
               ("args_dump_identity_" + std::to_string(::getpid()) + "_" + std::to_string(++instance_));
        std::filesystem::remove_all(dir_);
        ASSERT_TRUE(std::filesystem::create_directories(dir_));

        collector_.begin_run(dir_.string(), DumpArgsLevel::FULL);
        ASSERT_EQ(
            collector_.initialize(
                /*num_dump_threads=*/1, /*device_id=*/0, DumpArgsLevel::FULL, args_dump_test_alloc, nullptr,
                args_dump_test_free
            ),
            0
        );
        shm_ = collector_.get_dump_shm_device_ptr();
        ASSERT_NE(shm_, nullptr);
        state_ = get_dump_buffer_state(shm_, kThreadIdx);
        header_ = get_dump_header(shm_);

        set_dump_args_enabled(true);
        set_platform_dump_base(reinterpret_cast<uint64_t>(shm_));
    }

    void TearDown() override {
        set_dump_args_enabled(false);
        set_platform_dump_base(0);
        set_platform_run_result(0, 0);
        collector_.finalize(nullptr, args_dump_test_free);
        std::filesystem::remove_all(dir_);
    }

    // One run's device-side sequence: the host opens the window, the device
    // acquires a buffer, records `args` scalars and flushes.
    void run_once(uint64_t epoch, int args) {
        collector_.begin_run(dir_.string(), DumpArgsLevel::FULL);
        run_once_without_begin_run(epoch, args);
    }

    // The same sequence with the host's per-run clear left out, which is what a
    // collection window spanning several runs will look like.
    void run_once_without_begin_run(uint64_t epoch, int args) {
        set_platform_run_result(/*region_base=*/0, epoch);
        dump_args_init(/*num_dump_threads=*/1);
        for (int i = 0; i < args; i++) {
            record_scalar(static_cast<uint64_t>(0x500 + i), static_cast<uint32_t>(i));
        }
        dump_args_flush(kThreadIdx);
    }

    void record_scalar(uint64_t task_id, uint32_t arg_index) {
        ArgsDumpInfo info{};
        info.task_id = task_id;
        info.role = ArgsDumpRole::INPUT;
        info.stage = ArgsDumpStage::BEFORE_DISPATCH;
        info.arg_index = arg_index;
        info.kind = static_cast<uint8_t>(ArgsDumpKind::SCALAR);
        info.scalar_value = 0x1234;
        info.capture_payload = 0;
        info.func_count = 1;
        info.func_ids[0] = 7;
        info.ndims = 0;
        ASSERT_EQ(dump_arg_record(kThreadIdx, info), 0);
    }

    // Hand every ready entry published since `consumed_tail_` to the collector's
    // own copy path, and return how many buffers that was.
    int collect_published() {
        const uint32_t tail = header_->queue_tails[kThreadIdx];
        int collected = 0;
        for (uint32_t i = consumed_tail_; i < tail; i++) {
            const DumpReadyQueueEntry &entry = header_->queues[kThreadIdx][i];
            DumpReadyBufferInfo info{};
            info.thread_index = entry.thread_index;
            info.dev_buffer_ptr = reinterpret_cast<void *>(entry.buffer_ptr);
            info.host_buffer_ptr = reinterpret_cast<void *>(entry.buffer_ptr);
            info.buffer_seq = entry.buffer_seq;
            collector_.on_buffer_collected(info, /*collector_shard=*/0);
            collected++;
        }
        consumed_tail_ = tail;
        return collected;
    }

    uint64_t published_buffer_at(uint32_t index) const { return header_->queues[kThreadIdx][index].buffer_ptr; }

    std::string read_manifest() {
        const std::filesystem::path path = dir_ / "args_dump" / "args_dump.json";
        std::ifstream in(path);
        if (!in.is_open()) return {};
        return std::string{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    }

    ArgsDumpCollector collector_;
    void *shm_ = nullptr;
    DumpBufferState *state_ = nullptr;
    DumpDataHeader *header_ = nullptr;
    uint32_t consumed_tail_ = 0;
    std::filesystem::path dir_;
    static int instance_;
};

int ArgsDumpRunIdentityTest::instance_ = 0;

// A run's args reach the host carrying that run's epoch. The stamp is
// production — the engine's pop hook and init's reuse path write it, and nothing
// in the test touches `run_epoch`.
TEST_F(ArgsDumpRunIdentityTest, EachRunsArgsCarryItsOwnEpoch) {
    run_once(/*epoch=*/11, /*args=*/2);
    ASSERT_EQ(collect_published(), 1) << "the first run published no buffer";

    run_once(/*epoch=*/12, /*args=*/1);
    ASSERT_EQ(collect_published(), 1) << "the second run published no buffer";

    // begin_run() still clears per run, so only the latest survives. What this
    // pins is that it is stamped with its own epoch, not the previous one's —
    // the buffer may well be the same storage.
    ASSERT_EQ(collector_.export_dump_files(), 0);
    const std::string manifest = read_manifest();
    ASSERT_FALSE(manifest.empty());
    EXPECT_NE(manifest.find("\"run_epoch\": 12"), std::string::npos)
        << "the second run's args did not carry its own identity: " << manifest;
    EXPECT_EQ(manifest.find("\"run_epoch\": 11"), std::string::npos)
        << "the cleared run's identity survived into this run's manifest: " << manifest;
}

// Identity survives the buffer going back to the pool and being re-stamped: the
// host copy must not be reading it back off the device buffer.
//
// The window deliberately spans several runs (no `begin_run()` after the first),
// because with the per-run clear in place the host holds one run at a time and
// the question has no answer. The pool is FIFO with PLATFORM_DUMP_SLOT_COUNT
// buffers, so returning each buffer as it is collected brings the first run's
// storage back around on the run after the last slot.
TEST_F(ArgsDumpRunIdentityTest, TheHostCopyKeepsItsEpochAfterTheBufferIsRestamped) {
    constexpr uint64_t kFirstEpoch = 21;
    run_once(kFirstEpoch, /*args=*/1);
    ASSERT_EQ(collect_published(), 1);
    const uint64_t first = published_buffer_at(0);
    ASSERT_NE(first, 0u);
    ASSERT_EQ(reinterpret_cast<const DumpMetaBuffer *>(first)->run_epoch, kFirstEpoch);
    host_push_free_queue(state_->free_queue, first);

    const uint64_t last_epoch = kFirstEpoch + static_cast<uint64_t>(PLATFORM_DUMP_SLOT_COUNT);
    for (uint64_t epoch = kFirstEpoch + 1; epoch <= last_epoch; epoch++) {
        run_once_without_begin_run(epoch, /*args=*/1);
        ASSERT_EQ(collect_published(), 1) << "run " << epoch << " published nothing";
        host_push_free_queue(state_->free_queue, published_buffer_at(consumed_tail_ - 1));
    }

    ASSERT_EQ(published_buffer_at(consumed_tail_ - 1), first)
        << "the returned buffer never came back around; the FIFO assumption above is wrong";
    ASSERT_EQ(reinterpret_cast<const DumpMetaBuffer *>(first)->run_epoch, last_epoch)
        << "the reused buffer was not re-stamped, so this test cannot show the host copy is independent";

    // The device storage now says `last_epoch`. The copy taken back when it said
    // 21 must still say 21, and every run in the window must be present exactly
    // once.
    ASSERT_EQ(collector_.export_dump_files(), 0);
    const std::string manifest = read_manifest();
    ASSERT_FALSE(manifest.empty());
    for (uint64_t epoch = kFirstEpoch; epoch <= last_epoch; epoch++) {
        EXPECT_NE(manifest.find("\"run_epoch\": " + std::to_string(epoch)), std::string::npos)
            << "run " << epoch << " is missing from the manifest: " << manifest;
    }
}

// An idle run must not cost the pool a buffer.
TEST_F(ArgsDumpRunIdentityTest, IdleRunsDoNotConsumeABufferEach) {
    const uint32_t depth_at_start = free_queue_depth(shm_, kThreadIdx);
    ASSERT_GT(depth_at_start, 1u) << "need more than one free buffer for per-run consumption to be visible";

    // The first run legitimately draws one: the thread holds none yet.
    run_once(/*epoch=*/31, /*args=*/0);
    const uint32_t depth_after_first = free_queue_depth(shm_, kThreadIdx);
    ASSERT_EQ(depth_after_first, depth_at_start - 1);
    const uint64_t held = state_->current_buf_ptr;
    ASSERT_NE(held, 0u);

    for (uint64_t epoch = 32; epoch <= 34; epoch++) {
        run_once(epoch, /*args=*/0);
        EXPECT_EQ(free_queue_depth(shm_, kThreadIdx), depth_after_first)
            << "run " << epoch << " drew a buffer instead of reusing the retained one";
        EXPECT_EQ(state_->current_buf_ptr, held) << "run " << epoch << " swapped the buffer it already held";
        EXPECT_EQ(reinterpret_cast<const DumpMetaBuffer *>(held)->run_epoch, epoch)
            << "run " << epoch << " reused the buffer without re-stamping it";
    }

    EXPECT_EQ(header_->queue_tails[kThreadIdx], 0u) << "an idle run published a buffer";
}

// A failed enqueue charges dropped once, zeroes the count and keeps the buffer.
//
// **Not covered here, deliberately.** args_dump's flush validates its own thread
// index before building the engine context, so the out-of-range index that
// reaches `enqueue_ready`'s guard in the swimlane and dep_gen suites cannot be
// injected: `dump_args_flush` returns before flushing. The only other way in is a
// genuinely full ready queue, which means sitting in the backpressure gate for
// PLATFORM_DFX_BACKPRESSURE_TIMEOUT_CYCLES — 30 seconds of real wall clock, since
// the host stub's `get_sys_cnt_aicpu` scales a steady_clock reading. A unit test
// has no business doing either.
//
// What makes the gap narrow rather than open: the shared engine already applies
// this exact rule on the hot path. `DeviceProfilerEngine::switch_buffer` charges
// dropped, zeroes the count and returns **without** clearing `current_ptr` on
// enqueue failure, clearing it only on success. `dump_args_flush`'s own inline
// enqueue was the deviation, and it now matches. So the rule is exercised for
// this collector by every mid-run rotation that fails; only the run-end flush
// branch rests on review.

// A published buffer is the host's, not the pool's: retention covers the failed
// hand-over only, so a successful flush still releases the pointer and the pool
// must not hand that buffer out again before the host returns it.
TEST_F(ArgsDumpRunIdentityTest, APublishedBufferIsNotAlsoRetainedByTheDevice) {
    run_once(/*epoch=*/51, /*args=*/1);

    EXPECT_EQ(state_->current_buf_ptr, 0u) << "the published buffer is still the thread's active one";
    ASSERT_EQ(collect_published(), 1);
    const uint64_t published = published_buffer_at(0);
    EXPECT_FALSE(free_queue_holds(state_->free_queue, published))
        << "the buffer was both published and left in the free queue";

    // Nothing is retained, so the next run draws from the pool — and it must not
    // draw the one the host has not given back.
    collector_.begin_run(dir_.string(), DumpArgsLevel::FULL);
    set_platform_run_result(/*region_base=*/0, /*run_epoch=*/52);
    dump_args_init(/*num_dump_threads=*/1);
    const uint64_t acquired = state_->current_buf_ptr;
    EXPECT_NE(acquired, 0u) << "init drew nothing after the previous run released its buffer";
    EXPECT_NE(acquired, published) << "the pool handed out a buffer the host had not returned";
    EXPECT_EQ(reinterpret_cast<const DumpMetaBuffer *>(acquired)->run_epoch, 52u);

    // Close the window the way a run does. `on_buffer_collected` starts the
    // payload writer thread unconditionally, and `export_dump_files()` is what
    // stops it; leaving it to `finalize()`'s export-skipping fallback instead is
    // a teardown path nothing else in the tree exercises, and not what this test
    // is about. Reaching `finalize()` through it hung this binary in CI on both
    // runners while passing here, which is logged separately.
    ASSERT_EQ(collector_.export_dump_files(), 0);
}

}  // namespace
