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
 * PMU run identity and buffer ownership.
 *
 * Production code drives everything that decides ownership and attribution: the
 * real host collector allocates and seeds the pool, the real `pmu_aicpu_init`
 * acquires or reuses a buffer and stamps it, the real `pmu_aicpu_flush_buffers`
 * publishes or retains it, and the real collector copies the rows out to CSV.
 *
 * **Record contents are a fixture, and that is a real boundary rather than a
 * shortcut.** PMU records are not written by a host-callable function on either
 * architecture: a2a3's `pmu_aicpu_record_task` reads counters through MMIO from a
 * per-core register window, and a5's arrive in an AICore-written staging ring that
 * `pmu_aicpu_complete_record` validates. Neither belongs in a host unit test, and
 * neither is what this change touches. So the test writes `count` and the record
 * fields, and asserts on the identity and ownership around them — none of which it
 * writes.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "aicpu/device_run_result_base_aicpu.h"
#include "aicpu/platform_regs.h"
#include "aicpu/pmu_collector_aicpu.h"
#include "common/memory_barrier.h"
#include "common/pmu_profiling.h"
#include "host/pmu_collector.h"

// `set_platform_pmu_reg_addrs` / `get_platform_pmu_reg_addrs` live in the
// platform register layer, whose translation unit reaches further into the
// runtime than this test needs. They are defined here instead, holding the
// address the test sets: zero, which `pmu_aicpu_init` reads as "no PMU on this
// core" and which makes it skip the MMIO programming entirely. Nothing else in
// the buffer-pool path touches a register.
extern "C" {
namespace {
uint64_t g_pmu_test_reg_addrs = 0;
}

void set_platform_pmu_reg_addrs(uint64_t pmu_regs) { g_pmu_test_reg_addrs = pmu_regs; }
uint64_t get_platform_pmu_reg_addrs() { return g_pmu_test_reg_addrs; }

// a5's PMU collector also resolves the platform register base. Zero, for the same
// reason: this test drives the buffer pool, not the counters. Unreferenced on
// a2a3, whose collector reads only the per-core PMU window.
uint64_t get_platform_regs() { return 0; }
}

// The register accessors themselves. They are defined here rather than linked
// because the two architectures put them in different translation units — a2a3 in
// the shared platform register layer, a5 in its sim variant hooks — and both of
// those pull in more of the runtime, or collide with the shared test stubs, than
// this test needs. `get_reg_ptr` still comes from the shared stubs.
//
// Reaching any of these would mean the test wandered into MMIO, so they say so
// instead of returning a plausible value.
uint64_t read_reg(uint64_t, RegId) {
    ADD_FAILURE() << "the test reached MMIO; PMU registers are out of scope here";
    return 0;
}

uint32_t reg_load_acquire(const volatile uint32_t *) {
    ADD_FAILURE() << "the test reached MMIO; PMU registers are out of scope here";
    return 0;
}

void reg_store_release(volatile uint32_t *, uint32_t) {
    ADD_FAILURE() << "the test reached MMIO; PMU registers are out of scope here";
}

// a5 declares `write_reg` out-of-line with a 64-bit value, where a2a3 has it
// inline over `reg_store_release` above. The arch's own headers carry no
// preprocessor discriminator — the counter counts are `constexpr int`, invisible
// to `#if` — so the build sets this one.
#if defined(SIMPLER_TEST_PMU_ARCH_A5)
void write_reg(uint64_t, RegId, uint64_t) {
    ADD_FAILURE() << "the test reached MMIO; PMU registers are out of scope here";
}
#endif

namespace {

void *pmu_test_alloc(size_t size) { return std::calloc(1, size); }

int pmu_test_free(void *ptr) {
    std::free(ptr);
    return 0;
}

uint32_t free_queue_depth(void *shm, int core) {
    const PmuBufferState *state = get_pmu_buffer_state(shm, core);
    return state->free_queue.tail - state->free_queue.head;
}

bool free_queue_holds(const PmuFreeQueue &fq, uint64_t buf_ptr) {
    for (uint32_t i = fq.head; i != fq.tail; i++) {
        if (fq.buffer_ptrs[i % PLATFORM_PMU_SLOT_COUNT] == buf_ptr) return true;
    }
    return false;
}

// The host's half of the SPSC protocol: write the slot, fence, publish the tail.
// AICPU is the queue's consumer and has no push of its own, so a test that needs
// a buffer back in the pool has to play the host here. This models the return,
// not the asynchronous transfer that accompanies it on real hardware.
void host_push_free_queue(PmuFreeQueue &fq, uint64_t buf_ptr) {
    ASSERT_LT(fq.tail - fq.head, static_cast<uint32_t>(PLATFORM_PMU_SLOT_COUNT)) << "free queue is full";
    fq.buffer_ptrs[fq.tail % PLATFORM_PMU_SLOT_COUNT] = buf_ptr;
    wmb();
    fq.tail = fq.tail + 1;
    wmb();
}

class PmuRunIdentityTest : public ::testing::Test {
protected:
    static constexpr int kCore = 0;
    static constexpr int kThread = 0;

    void SetUp() override {
        dir_ = std::filesystem::temp_directory_path() /
               ("pmu_identity_" + std::to_string(::getpid()) + "_" + std::to_string(++instance_));
        std::filesystem::remove_all(dir_);
        ASSERT_TRUE(std::filesystem::create_directories(dir_));
        csv_ = dir_ / "pmu.csv";

        // begin_run precedes init on the first run: the event type it binds is
        // what reaches the device header and names the CSV columns.
        collector_.begin_run(csv_.string(), PmuEventType::PIPE_UTILIZATION);
        ASSERT_EQ(
            collector_.init(
                /*num_cores=*/1, /*num_threads=*/1, pmu_test_alloc, nullptr, pmu_test_free, /*device_id=*/0
            ),
            0
        );
        shm_ = collector_.get_pmu_shm_device_ptr();
        ASSERT_NE(shm_, nullptr);
        state_ = get_pmu_buffer_state(shm_, kCore);
        header_ = get_pmu_header(shm_);

        set_pmu_enabled(true);
        set_platform_pmu_base(reinterpret_cast<uint64_t>(shm_));
        // No PMU register window: pmu_aicpu_init treats a zero reg address as
        // "no PMU on this core" and skips the MMIO programming, which is exactly
        // the part this test has no business driving. The buffer-pool half runs
        // regardless.
        set_platform_pmu_reg_addrs(0);
    }

    void TearDown() override {
        set_pmu_enabled(false);
        set_platform_pmu_base(0);
        set_platform_run_result(0, 0);
        collector_.finalize(nullptr, pmu_test_free);
        std::filesystem::remove_all(dir_);
    }

    // One run's device-side sequence: the host opens the window, the device
    // acquires a buffer, `records` records land in it, and it flushes.
    void run_once(uint64_t epoch, int records) {
        collector_.begin_run(csv_.string(), PmuEventType::PIPE_UTILIZATION);
        run_once_without_begin_run(epoch, records);
    }

    void run_once_without_begin_run(uint64_t epoch, int records) {
        set_platform_run_result(/*region_base=*/0, epoch);
        const uint32_t core_ids[] = {0};
        pmu_aicpu_init(core_ids, /*num_cores=*/1);
        if (records > 0) fill_records(state_->current_buf_ptr, records);
        const int cores[] = {kCore};
        pmu_aicpu_flush_buffers(kThread, cores, /*core_num=*/1);
    }

    // The fixture half — see the file comment for why the real writers are out
    // of scope. `run_epoch` is deliberately untouched.
    void fill_records(uint64_t buf_ptr, int n) {
        ASSERT_NE(buf_ptr, 0u);
        auto *buf = reinterpret_cast<PmuBuffer *>(buf_ptr);
        for (int i = 0; i < n; i++) {
            buf->records[i].task_id = 0x700 + static_cast<uint64_t>(i);
            buf->records[i].func_id = static_cast<uint32_t>(i);
            buf->records[i].pmu_total_cycles = 1000 + static_cast<uint64_t>(i);
        }
        buf->count = static_cast<uint32_t>(n);
        state_->total_record_count += static_cast<uint32_t>(n);
        wmb();
    }

    int collect_published() {
        const uint32_t tail = header_->queue_tails[kThread];
        int collected = 0;
        for (uint32_t i = consumed_tail_; i < tail; i++) {
            const PmuReadyQueueEntry &entry = header_->queues[kThread][i];
            PmuReadyBufferInfo info{};
            info.core_index = entry.core_index;
            info.thread_index = kThread;
            info.dev_buffer_ptr = reinterpret_cast<void *>(entry.buffer_ptr);
            info.host_buffer_ptr = reinterpret_cast<void *>(entry.buffer_ptr);
            info.buffer_seq = entry.buffer_seq;
            collector_.on_buffer_collected(info, /*collector_shard=*/0);
            collected++;
        }
        consumed_tail_ = tail;
        return collected;
    }

    uint64_t published_buffer_at(uint32_t index) const { return header_->queues[kThread][index].buffer_ptr; }

    std::string read_csv() {
        std::ifstream in(csv_);
        if (!in.is_open()) return {};
        return std::string{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    }

    // Every data row's run_epoch column, in file order.
    std::vector<std::string> csv_run_epochs(const std::string &content) {
        std::vector<std::string> out;
        std::istringstream lines(content);
        std::string header;
        if (!std::getline(lines, header)) return out;
        std::vector<std::string> cols;
        std::istringstream head_cols(header);
        for (std::string col; std::getline(head_cols, col, ',');)
            cols.push_back(col);
        size_t idx = cols.size();
        for (size_t i = 0; i < cols.size(); i++) {
            if (cols[i] == "run_epoch") idx = i;
        }
        if (idx == cols.size()) return out;
        for (std::string row; std::getline(lines, row);) {
            if (row.empty()) continue;
            std::vector<std::string> fields;
            std::istringstream row_cols(row);
            for (std::string f; std::getline(row_cols, f, ',');)
                fields.push_back(f);
            if (idx < fields.size()) out.push_back(fields[idx]);
        }
        return out;
    }

    PmuCollector collector_;
    void *shm_ = nullptr;
    PmuBufferState *state_ = nullptr;
    PmuDataHeader *header_ = nullptr;
    uint32_t consumed_tail_ = 0;
    std::filesystem::path dir_;
    std::filesystem::path csv_;
    static int instance_;
};

int PmuRunIdentityTest::instance_ = 0;

// A run's rows reach the CSV carrying that run's epoch. The stamp is production:
// init writes it at both acquisition points and nothing here touches it.
TEST_F(PmuRunIdentityTest, EachRunsRowsCarryItsOwnEpoch) {
    run_once(/*epoch=*/11, /*records=*/2);
    ASSERT_EQ(collect_published(), 1) << "the run published no buffer";
    collector_.reconcile_counters();

    const std::vector<std::string> epochs = csv_run_epochs(read_csv());
    ASSERT_EQ(epochs.size(), 2u) << "expected one row per record in:\n" << read_csv();
    for (const std::string &e : epochs) {
        EXPECT_EQ(e, "11") << "a row carried an identity other than its run's";
    }
}

// Identity is consumed with the records, not read back later: the pool reuses the
// storage and a later run re-stamps it in place.
TEST_F(PmuRunIdentityTest, RowsKeepTheirEpochAfterTheBufferIsRestamped) {
    const uint32_t seeded = free_queue_depth(shm_, kCore);
    ASSERT_GT(seeded, 1u) << "need more than one seeded buffer to cycle the pool";

    run_once(/*epoch=*/21, /*records=*/1);
    ASSERT_EQ(collect_published(), 1);
    const uint64_t first = published_buffer_at(0);
    ASSERT_EQ(reinterpret_cast<const PmuBuffer *>(first)->run_epoch, 21u);
    host_push_free_queue(state_->free_queue, first);

    // The pool is FIFO, so returning each buffer as it is collected brings the
    // first run's storage back around after as many runs as the pool was seeded
    // with. Derived from the measured depth rather than a constant, so this stays
    // correct if the pool is resized.
    const uint64_t last_epoch = 21 + static_cast<uint64_t>(seeded);
    for (uint64_t epoch = 22; epoch <= last_epoch; epoch++) {
        run_once_without_begin_run(epoch, /*records=*/1);
        ASSERT_EQ(collect_published(), 1) << "run " << epoch << " published nothing";
        host_push_free_queue(state_->free_queue, published_buffer_at(consumed_tail_ - 1));
    }

    ASSERT_EQ(published_buffer_at(consumed_tail_ - 1), first)
        << "the returned buffer never came back around; the FIFO assumption above is wrong";
    ASSERT_EQ(reinterpret_cast<const PmuBuffer *>(first)->run_epoch, last_epoch)
        << "the reused buffer was not re-stamped, so this test cannot show the rows are independent";

    collector_.reconcile_counters();
    const std::vector<std::string> epochs = csv_run_epochs(read_csv());
    ASSERT_FALSE(epochs.empty());
    EXPECT_EQ(epochs.front(), "21") << "the first run's row followed the device buffer's re-stamp";
    EXPECT_EQ(epochs.back(), std::to_string(last_epoch));
}

// An idle run must not cost the pool a buffer.
TEST_F(PmuRunIdentityTest, IdleRunsDoNotConsumeABufferEach) {
    const uint32_t depth_at_start = free_queue_depth(shm_, kCore);
    ASSERT_GT(depth_at_start, 1u) << "need more than one free buffer for per-run consumption to be visible";

    run_once(/*epoch=*/31, /*records=*/0);
    const uint32_t depth_after_first = free_queue_depth(shm_, kCore);
    ASSERT_EQ(depth_after_first, depth_at_start - 1);
    const uint64_t held = state_->current_buf_ptr;
    ASSERT_NE(held, 0u);

    for (uint64_t epoch = 32; epoch <= 34; epoch++) {
        run_once(epoch, /*records=*/0);
        EXPECT_EQ(free_queue_depth(shm_, kCore), depth_after_first)
            << "run " << epoch << " drew a buffer instead of reusing the retained one";
        EXPECT_EQ(state_->current_buf_ptr, held) << "run " << epoch << " swapped the buffer it already held";
        EXPECT_EQ(reinterpret_cast<const PmuBuffer *>(held)->run_epoch, epoch)
            << "run " << epoch << " reused the buffer without re-stamping it";
    }

    EXPECT_EQ(header_->queue_tails[kThread], 0u) << "an idle run published a buffer";
}

// A failed enqueue charges dropped once, zeroes the count and keeps the buffer.
//
// The production cause is a full ready queue, which means sitting in the
// backpressure gate; the gate also rejects an out-of-range thread index outright,
// which fails the same enqueue for the same caller and runs the same branch. PMU's
// flush passes its thread argument straight through, so the branch is reachable
// here; the gate's own timeout is not, and belongs to the engine's tests.
TEST_F(PmuRunIdentityTest, AFailedFlushChargesDroppedOnceAndKeepsTheBuffer) {
    collector_.begin_run(csv_.string(), PmuEventType::PIPE_UTILIZATION);
    set_platform_run_result(/*region_base=*/0, /*run_epoch=*/41);
    const uint32_t core_ids[] = {0};
    pmu_aicpu_init(core_ids, /*num_cores=*/1);

    const uint64_t held = state_->current_buf_ptr;
    ASSERT_NE(held, 0u);
    const uint32_t depth_after_init = free_queue_depth(shm_, kCore);
    fill_records(held, 2);
    const uint32_t dropped_before = state_->dropped_record_count;

    const int cores[] = {kCore};
    pmu_aicpu_flush_buffers(PLATFORM_MAX_AICPU_THREADS, cores, /*core_num=*/1);

    EXPECT_EQ(state_->dropped_record_count, dropped_before + 2)
        << "the failed enqueue charged dropped by something other than the buffer's record count";
    EXPECT_EQ(state_->current_buf_ptr, held)
        << "a buffer the host never received was released, so nothing can return it";
    EXPECT_EQ(free_queue_depth(shm_, kCore), depth_after_init) << "the failed flush drew from the free queue";
    EXPECT_FALSE(free_queue_holds(state_->free_queue, held)) << "AICPU pushed the free queue it only consumes";
    EXPECT_EQ(reinterpret_cast<const PmuBuffer *>(held)->count, 0u)
        << "the retained buffer still claims records that were charged to dropped";

    // Reusable in place: the next run takes it and re-stamps it.
    collector_.begin_run(csv_.string(), PmuEventType::PIPE_UTILIZATION);
    set_platform_run_result(/*region_base=*/0, /*run_epoch=*/42);
    pmu_aicpu_init(core_ids, /*num_cores=*/1);
    EXPECT_EQ(state_->current_buf_ptr, held) << "init drew a replacement instead of reusing the retained buffer";
    const auto *reused = reinterpret_cast<const PmuBuffer *>(held);
    EXPECT_EQ(reused->count, 0u);
    EXPECT_EQ(reused->run_epoch, 42u) << "a reused buffer kept the previous run's identity";
    EXPECT_EQ(state_->current_buf_seq, 0u);
}

// A published buffer is the host's, not the pool's: retention covers the failed
// hand-over only.
TEST_F(PmuRunIdentityTest, APublishedBufferIsNotAlsoRetainedByTheDevice) {
    run_once(/*epoch=*/51, /*records=*/1);

    EXPECT_EQ(state_->current_buf_ptr, 0u) << "the published buffer is still the core's active one";
    ASSERT_EQ(collect_published(), 1);
    const uint64_t published = published_buffer_at(0);
    EXPECT_FALSE(free_queue_holds(state_->free_queue, published))
        << "the buffer was both published and left in the free queue";

    collector_.begin_run(csv_.string(), PmuEventType::PIPE_UTILIZATION);
    set_platform_run_result(/*region_base=*/0, /*run_epoch=*/52);
    const uint32_t core_ids[] = {0};
    pmu_aicpu_init(core_ids, /*num_cores=*/1);
    const uint64_t acquired = state_->current_buf_ptr;
    EXPECT_NE(acquired, 0u) << "init drew nothing after the previous run released its buffer";
    EXPECT_NE(acquired, published) << "the pool handed out a buffer the host had not returned";
    EXPECT_EQ(reinterpret_cast<const PmuBuffer *>(acquired)->run_epoch, 52u);
}

}  // namespace
