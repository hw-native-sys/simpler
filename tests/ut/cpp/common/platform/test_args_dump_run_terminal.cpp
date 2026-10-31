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
 * What one ArgsDump producer thread reports for its own run.
 *
 * The producer is the real AICPU module; what this supplies is the shared
 * region and the free queue. Two kinds of loss are accounted differently and
 * must not be conflated: a record refused before anything was written for it is
 * one this producer never wrote, while a record written into a buffer that is
 * then lost is already among the ones it wrote. Adding every drop to the total
 * would count the second kind twice and make a reported loss fail conservation.
 *
 * What this cannot cover, stated rather than simulated: **the commit-failure
 * path itself**. A full ready queue is only reported as full after the device
 * backpressure budget elapses — `PLATFORM_DFX_BACKPRESSURE_TIMEOUT_CYCLES`, 30
 * seconds against the host clock this build reads — so provoking it here would
 * spend that budget per case. The accounting it produces is covered by
 * constructed counts in `test_diagnostic_run_credit.cpp`
 * (`AWrittenThenLostRecordIsCountedOnce`); the refusal side, which needs no
 * wait, is driven through the real producer below. Also uncovered: the arena
 * acknowledgment clocks, which belong to the host and have the collector's own
 * cases, and device cache visibility — every store here is host memory.
 */

#include "aicpu/args_dump_aicpu.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "aicpu/device_run_result_base_aicpu.h"
#include "common/args_dump.h"
#include "common/diagnostic_run_terminal.h"
#include "host/diagnostic_run_credit.h"
#include "support/test_task_id.h"

namespace credit = simpler::dfx::credit;
namespace runs = simpler::dfx::runs;

namespace {

constexpr int kLane = 0;
constexpr uint64_t kEpoch = 8801;
constexpr size_t kArenaBytes = 64 * 1024;

/**
 * One producer thread's shared region, with both queues under test control.
 *
 * `expected` is one: a single dump thread is brought up, so the run's
 * participant set is the first entry of a bank dimensioned by the platform
 * maximum.
 */
class ArgsDumpRunTerminalTest : public ::testing::Test {
protected:
    void SetUp() override {
        shm_bytes_ = calc_dump_data_size(/*num_dump_threads=*/1);
        shm_.reset(static_cast<uint8_t *>(std::aligned_alloc(64, (shm_bytes_ + 63) / 64 * 64)));
        ASSERT_NE(shm_, nullptr);
        std::memset(shm_.get(), 0, shm_bytes_);

        // Value-initialized rather than memset: one runtime's TaskId is a
        // non-trivial type, so the buffer is not a blob to clear.
        meta_ = std::make_unique<DumpMetaBuffer>();
        arena_.assign(kArenaBytes, 0);

        header_ = get_dump_header(shm_.get());
        header_->magic = ARGS_DUMP_MAGIC;

        header_->num_dump_threads = 1;

        state_ = get_dump_buffer_state(shm_.get(), kLane);
        state_->free_queue.buffer_ptrs[0] = reinterpret_cast<uint64_t>(meta_.get());
        state_->free_queue.tail = 1;
        state_->arena_base = reinterpret_cast<uint64_t>(arena_.data());
        state_->arena_size = kArenaBytes;
        state_->arena_write_offset = 0;

        bank_.assign(static_cast<size_t>(PLATFORM_MAX_AICPU_THREADS) * PLATFORM_DIAGNOSTIC_TERMINAL_BANKS, {});
        for (auto &entry : bank_)
            diagnostic_terminal_reset(&entry);

        set_dump_args_enabled(true);
        set_platform_dump_base(reinterpret_cast<uint64_t>(shm_.get()));
        set_platform_dump_args_run_terminal(reinterpret_cast<uint64_t>(bank_.data()));
        set_platform_run_result(/*region_base=*/0, kEpoch);
        dump_args_init(/*num_dump_threads=*/1, DumpArgsLevel::FULL);
    }

    void TearDown() override {
        set_platform_run_result(0, 0);
        set_platform_dump_args_run_terminal(0);
        set_platform_dump_base(0);
        set_dump_args_enabled(false);
    }

    /** Record one small contiguous tensor; returns the module's own result. */
    int record_tensor(int32_t task_id, uint32_t arg_index) {
        ArgsDumpInfo info{};
        info.task_id = simpler::ut::test_task_id(task_id);
        info.role = ArgsDumpRole::INPUT;
        info.stage = ArgsDumpStage::BEFORE_DISPATCH;
        info.arg_index = arg_index;
        info.kind = static_cast<uint8_t>(ArgsDumpKind::TENSOR);
        info.dtype = static_cast<uint8_t>(DataType::UINT64);
        info.ndims = 1;
        info.shapes[0] = static_cast<int64_t>(payload_.size());
        info.strides[0] = 1;
        info.start_offset = 0;
        info.buffer_addr = reinterpret_cast<uint64_t>(payload_.data());
        info.capture_payload = 1;
        info.func_count = 1;
        info.func_ids[0] = 1;
        return dump_arg_record(kLane, info);
    }

    const DiagnosticRunTerminal &terminal(int producer = kLane) const { return bank_[static_cast<size_t>(producer)]; }

    credit::RunReconciliation reconcile(uint64_t received, uint64_t epoch = kEpoch) const {
        return credit::reconcile_run(bank_.data(), PLATFORM_MAX_AICPU_THREADS, /*expected=*/1, epoch, received);
    }

    struct Free {
        void operator()(uint8_t *p) const { std::free(p); }
    };

    size_t shm_bytes_{0};
    std::unique_ptr<uint8_t, Free> shm_;
    std::unique_ptr<DumpMetaBuffer> meta_;
    std::vector<uint8_t> arena_;
    std::vector<uint64_t> payload_{1, 2, 3, 4};
    std::vector<DiagnosticRunTerminal> bank_;
    DumpDataHeader *header_{nullptr};
    DumpBufferState *state_{nullptr};
};

}  // namespace

TEST_F(ArgsDumpRunTerminalTest, ADeliveredRunConservesAndCarriesItsPayloadBytes) {
    ASSERT_EQ(record_tensor(0x901, 0), 0);
    ASSERT_EQ(record_tensor(0x902, 1), 0);
    ASSERT_EQ(meta_->count, 2u);

    dump_args_flush(kLane);
    dump_args_close_run(kLane);

    EXPECT_EQ(terminal().run_epoch, kEpoch);
    EXPECT_EQ(terminal().total, 2u);
    EXPECT_EQ(terminal().published_records, 2u);
    EXPECT_EQ(terminal().dropped, 0u);
    EXPECT_EQ(terminal().live_at_close, 0u);
    EXPECT_EQ(terminal().published_payloads, 2u) << "a payload the producer handed over was not counted";
    EXPECT_GT(terminal().published_payload_bytes, 0u) << "a payload the producer handed over was not accounted";

    const credit::RunReconciliation out = reconcile(/*received=*/2);
    EXPECT_EQ(out.verdict, runs::Verdict::Published);
    EXPECT_FALSE(out.counts_unknown);
}

TEST_F(ArgsDumpRunTerminalTest, ARecordRefusedBeforeTheWriteIsAlsoThisRunsTotal) {
    ASSERT_EQ(record_tensor(0x904, 0), 0);
    // Consume an empty replacement slot instead of waiting 30 seconds for a
    // free buffer that this fixture has no host thread to return.
    state_->free_queue.buffer_ptrs[state_->free_queue.tail % PLATFORM_DUMP_SLOT_COUNT] = 0;
    ++state_->free_queue.tail;
    dump_args_flush(kLane);  // hands the only buffer over, leaving none free

    // With no free buffer and nothing held, the next record is refused before
    // anything is written for it. It is this producer's responsibility all the
    // same, so it belongs in the total as well as in the drops.
    EXPECT_EQ(state_->current_buf_ptr, 0u);
    EXPECT_NE(record_tensor(0x905, 1), 0);

    dump_args_close_run(kLane);

    EXPECT_EQ(terminal().total, 2u) << "a refused record left the total";
    EXPECT_EQ(terminal().dropped, 1u);
    EXPECT_EQ(state_->run_dropped_record_count, 1u);
    EXPECT_EQ(terminal().published_records, 1u);

    const credit::RunReconciliation out = reconcile(/*received=*/1);
    EXPECT_EQ(out.verdict, runs::Verdict::PartialSafe);
    EXPECT_FALSE(out.counts_unknown);
}

TEST_F(ArgsDumpRunTerminalTest, AHeldBufferAtCloseIsReportedLive) {
    ASSERT_EQ(record_tensor(0x906, 0), 0);
    ASSERT_EQ(meta_->count, 1u);

    // No flush: the record is still in the producer's buffer.
    dump_args_close_run(kLane);

    EXPECT_EQ(terminal().total, 1u);
    EXPECT_EQ(terminal().published_records, 0u);
    EXPECT_EQ(terminal().live_at_close, 1u);

    const credit::RunReconciliation out = reconcile(/*received=*/0);
    EXPECT_EQ(out.verdict, runs::Verdict::PartialSafe);
    EXPECT_EQ(out.unsettled, 1u);
}

TEST_F(ArgsDumpRunTerminalTest, TheSuccessorsInitStartsItsOwnAccounting) {
    ASSERT_EQ(record_tensor(0x907, 0), 0);
    dump_args_flush(kLane);
    dump_args_close_run(kLane);
    ASSERT_EQ(terminal().total, 1u);

    // A successor's device init takes the entry over and starts its counters,
    // with no host reset between the two runs. It takes it over by claiming
    // it: the entry carries the successor's identity and no usable arithmetic
    // until that run's own close, so the predecessor's snapshot is neither
    // readable nor attributable to anyone.
    set_platform_run_result(0, kEpoch + 1);
    state_->free_queue.buffer_ptrs[1] = reinterpret_cast<uint64_t>(meta_.get());
    state_->free_queue.tail = 2;
    dump_args_init(/*num_dump_threads=*/1, DumpArgsLevel::FULL);
    EXPECT_EQ(terminal().run_epoch, kEpoch + 1) << "the successor's init did not claim the entry it publishes into";
    EXPECT_NE(terminal().counts_unknown, 0u) << "a claimed entry offered arithmetic before its run closed";
    EXPECT_EQ(terminal().total, 0u) << "the successor's init left its predecessor's count in the entry";
    EXPECT_EQ(reconcile(/*received=*/0, kEpoch + 1).verdict, runs::Verdict::CutUnknown)
        << "a run whose only participant has not closed was settled";

    ASSERT_EQ(record_tensor(0x908, 0), 0);
    dump_args_flush(kLane);
    dump_args_close_run(kLane);

    EXPECT_EQ(terminal().run_epoch, kEpoch + 1);
    EXPECT_EQ(terminal().total, 1u) << "the successor inherited its predecessor's count";
    EXPECT_EQ(reconcile(/*received=*/1, kEpoch + 1).verdict, runs::Verdict::Published);
}

TEST_F(ArgsDumpRunTerminalTest, ARunWithoutAnIdentityWritesNoEntryAtEitherEnd) {
    // Epoch zero is the entry's "no snapshot" state, so neither end of a run
    // may write it: a claim would attribute an entry to nobody and a close
    // would publish figures nobody can match. The fixture's own init already
    // claimed this lane for `kEpoch`, so what this watches is that an
    // identity-less run leaves that claim exactly as it found it — neither
    // overwritten with its own records nor retired.
    ASSERT_EQ(terminal().run_epoch, kEpoch) << "the fixture's init claimed no entry to leave alone";
    const uint32_t claimed_unknown = terminal().counts_unknown;

    set_platform_run_result(0, 0);
    dump_args_init(/*num_dump_threads=*/1, DumpArgsLevel::FULL);
    EXPECT_EQ(terminal().run_epoch, kEpoch) << "an init with no identity claimed a bank";

    ASSERT_EQ(record_tensor(0x909, 0), 0);
    dump_args_flush(kLane);
    dump_args_close_run(kLane);

    EXPECT_EQ(terminal().run_epoch, kEpoch) << "a close with no identity published into a bank";
    EXPECT_EQ(terminal().counts_unknown, claimed_unknown) << "a close with no identity settled someone else's entry";
    EXPECT_EQ(terminal().total, 0u) << "a run with no identity wrote its counters into another run's entry";
}

// ---------------------------------------------------------------------------
// A run that brings up fewer lanes than the platform could hold
// ---------------------------------------------------------------------------

namespace {

/**
 * Two dump lanes in a bank dimensioned for `PLATFORM_MAX_AICPU_THREADS`.
 *
 * This is every real run's shape and the single-lane fixture above cannot
 * express it: a collector is dimensioned by the producers the platform could
 * have, while a run brings up the lanes its scheduler actually started — on
 * both runtimes that is one fewer than the configured AICPU thread count,
 * because one thread orchestrates. The cases here go through the production
 * `dump_args_init` / `dump_args_close_run` entries rather than writing entries
 * by hand, because what they are about is which lanes those entries touch.
 */
class ArgsDumpFewerLanesTest : public ::testing::Test {
protected:
    static constexpr int kLanes = 2;

    void SetUp() override {
        ASSERT_LT(kLanes, PLATFORM_MAX_AICPU_THREADS) << "this case is about a run narrower than the platform maximum";
        shm_bytes_ = calc_dump_data_size(kLanes);
        shm_.reset(static_cast<uint8_t *>(std::aligned_alloc(64, (shm_bytes_ + 63) / 64 * 64)));
        ASSERT_NE(shm_, nullptr);
        std::memset(shm_.get(), 0, shm_bytes_);

        header_ = get_dump_header(shm_.get());
        header_->magic = ARGS_DUMP_MAGIC;

        header_->num_dump_threads = kLanes;

        for (int lane = 0; lane < kLanes; lane++) {
            meta_[lane] = std::make_unique<DumpMetaBuffer>();
            arena_[lane].assign(kArenaBytes, 0);
            DumpBufferState *state = get_dump_buffer_state(shm_.get(), lane);
            state->free_queue.buffer_ptrs[0] = reinterpret_cast<uint64_t>(meta_[lane].get());
            state->free_queue.tail = 1;
            state->arena_base = reinterpret_cast<uint64_t>(arena_[lane].data());
            state->arena_size = kArenaBytes;
            state->arena_write_offset = 0;
        }

        bank_.assign(static_cast<size_t>(PLATFORM_MAX_AICPU_THREADS) * PLATFORM_DIAGNOSTIC_TERMINAL_BANKS, {});
        for (auto &entry : bank_)
            diagnostic_terminal_reset(&entry);

        set_dump_args_enabled(true);
        set_platform_dump_base(reinterpret_cast<uint64_t>(shm_.get()));
        set_platform_dump_args_run_terminal(reinterpret_cast<uint64_t>(bank_.data()));
        set_platform_run_result(/*region_base=*/0, kEpoch);
        dump_args_init(kLanes, DumpArgsLevel::FULL);
    }

    void TearDown() override {
        set_platform_run_result(0, 0);
        set_platform_dump_args_run_terminal(0);
        set_platform_dump_base(0);
        set_dump_args_enabled(false);
    }

    int record_on(int lane, int32_t task_id) {
        ArgsDumpInfo info{};
        info.task_id = simpler::ut::test_task_id(task_id);
        info.role = ArgsDumpRole::INPUT;
        info.stage = ArgsDumpStage::BEFORE_DISPATCH;
        info.arg_index = 0;
        info.kind = static_cast<uint8_t>(ArgsDumpKind::TENSOR);
        info.dtype = static_cast<uint8_t>(DataType::UINT64);
        info.ndims = 1;
        info.shapes[0] = static_cast<int64_t>(payload_.size());
        info.strides[0] = 1;
        info.start_offset = 0;
        info.buffer_addr = reinterpret_cast<uint64_t>(payload_.data());
        info.capture_payload = 1;
        info.func_count = 1;
        info.func_ids[0] = 1;
        return dump_arg_record(lane, info);
    }

    const DiagnosticRunTerminal &terminal(int producer) const { return bank_[static_cast<size_t>(producer)]; }

    /**
     * Settled the way the collector settles it: the whole dimensioned array,
     * with the platform maximum as the expected count — which is exactly the
     * number this run does not have.
     */
    credit::RunReconciliation reconcile(uint64_t received) const {
        return credit::reconcile_run(
            bank_.data(), PLATFORM_MAX_AICPU_THREADS, /*expected=*/PLATFORM_MAX_AICPU_THREADS, kEpoch, received
        );
    }

    struct Free {
        void operator()(uint8_t *p) const { std::free(p); }
    };

    size_t shm_bytes_{0};
    std::unique_ptr<uint8_t, Free> shm_;
    std::unique_ptr<DumpMetaBuffer> meta_[kLanes];
    std::vector<uint8_t> arena_[kLanes];
    std::vector<uint64_t> payload_{1, 2, 3, 4};
    std::vector<DiagnosticRunTerminal> bank_;
    DumpDataHeader *header_{nullptr};
};

}  // namespace

TEST_F(ArgsDumpFewerLanesTest, OnlyTheLanesThisRunStartedAreItsParticipants) {
    // The regression this file exists for. `dump_args_init` claims the lanes it
    // brings up and no others: a claim on a lane the run never starts is never
    // settled, and an unsettled participant makes the whole run's accounting
    // unknown — so claiming the dimensioned array would fail every run on every
    // platform whose maximum exceeds its scheduler's lane count, which is all
    // of them.
    for (int lane = kLanes; lane < PLATFORM_MAX_AICPU_THREADS; lane++) {
        EXPECT_EQ(terminal(lane).run_epoch, 0u) << "lane " << lane << " was claimed for a run that never brought it up";
    }
    for (int lane = 0; lane < kLanes; lane++) {
        EXPECT_EQ(terminal(lane).run_epoch, kEpoch) << "lane " << lane << " was started and never claimed";
    }

    for (int lane = 0; lane < kLanes; lane++) {
        ASSERT_EQ(record_on(lane, 0x910 + lane), 0);
        dump_args_flush(lane);
        dump_args_close_run(lane);
    }

    const credit::RunReconciliation out = reconcile(/*received=*/kLanes);
    EXPECT_EQ(out.verdict, runs::Verdict::Published) << "a run narrower than the platform maximum was not settled";
    EXPECT_FALSE(out.counts_unknown);
    EXPECT_EQ(out.reported_producers, kLanes);
    EXPECT_EQ(out.produced, static_cast<uint64_t>(kLanes));
}

TEST_F(ArgsDumpFewerLanesTest, AStartedLaneThatNeverClosedStillRejectsTheRun) {
    // The other half, and the one the fix must not cost: a lane this run *did*
    // bring up and that never reached its close has published no arithmetic
    // anyone may use, and its own claim says so. Narrowing the claim to the
    // started lanes must not narrow it to the closed ones.
    for (int lane = 0; lane < kLanes; lane++) {
        ASSERT_EQ(record_on(lane, 0x920 + lane), 0);
        dump_args_flush(lane);
    }
    dump_args_close_run(0);

    EXPECT_EQ(terminal(1).run_epoch, kEpoch) << "a started lane did not claim this run";
    EXPECT_NE(terminal(1).counts_unknown, 0u) << "a lane that never closed offered usable arithmetic";

    const credit::RunReconciliation out = reconcile(/*received=*/kLanes);
    EXPECT_EQ(out.verdict, runs::Verdict::CutUnknown) << "a run with an unsettled participant was published";
    EXPECT_TRUE(out.counts_unknown);
    EXPECT_EQ(out.reported_producers, kLanes) << "the unsettled lane was not counted as a participant";
}
