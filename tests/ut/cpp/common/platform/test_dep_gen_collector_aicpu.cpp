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
 * AICPU-side dep_gen writer tests over a host-allocated stand-in for the
 * dep_gen shared-memory region.
 *
 * Focus: the orchestrator thread index contract. A record captured before
 * dep_gen_aicpu_set_orch_thread_idx() can never be published — the ready queue
 * it would go to is selected by that index — so it must be accounted as
 * dropped instead of accumulating in a buffer nobody will read.
 */

#include "aicpu/dep_gen_collector_aicpu.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "aicpu/device_run_result_base_aicpu.h"
#include "common/diagnostic_run_terminal.h"
#include "host/diagnostic_run_credit.h"
#include "support/test_task_id.h"

namespace {

// The collector stores this byte and never reads it back: DepFlags is declared
// in each runtime's own types.h, which is why the entry point takes a plain
// uint8_t. Naming the enumerators here would tie this case to one runtime's
// spelling of a value the code under test does not interpret.
constexpr uint8_t kCreatorEdgeDepKinds = 0x3;  // wait | retain
constexpr uint8_t kWaitEdgeDepKind = 0x1;
constexpr uint8_t kRetainEdgeDepKind = 0x2;

constexpr int kInstanceCount = 1;
constexpr size_t kShmAlign = alignof(DepGenDataHeader);

void *alloc_aligned_zeroed(size_t size) {
    const size_t rounded = (size + kShmAlign - 1) / kShmAlign * kShmAlign;
    void *p = aligned_alloc(kShmAlign, rounded);
    if (p != nullptr) {
        memset(p, 0, rounded);
    }
    return p;
}

// The shm header/state region and one DepGenBuffer, both cache-line aligned as
// the device-side allocations are, torn down with the fixture. DepGenBuffer is
// ~4.8 MB, so it is heap- rather than stack-allocated.
class DepGenCollectorAicpuTest : public ::testing::Test {
protected:
    void SetUp() override {
        shm_ = alloc_aligned_zeroed(calc_dep_gen_shm_size(kInstanceCount));
        ASSERT_NE(shm_, nullptr);
        buffer_ = static_cast<DepGenBuffer *>(alloc_aligned_zeroed(sizeof(DepGenBuffer)));
        ASSERT_NE(buffer_, nullptr);

        header_ = get_dep_gen_header(shm_);
        header_->num_instances = kInstanceCount;
        state_ = get_dep_gen_buffer_state(shm_, 0);
        state_->free_queue.buffer_ptrs[0] = reinterpret_cast<uint64_t>(buffer_);
        state_->free_queue.tail = 1;

        set_platform_dep_gen_base(reinterpret_cast<uint64_t>(shm_));
        set_dep_gen_enabled(true);
        dep_gen_aicpu_init();
        ASSERT_EQ(state_->current_buf_ptr, reinterpret_cast<uint64_t>(buffer_));
    }

    void TearDown() override {
        dep_gen_aicpu_finalize();
        set_dep_gen_enabled(false);
        set_platform_dep_gen_base(0);
        free(buffer_);
        free(shm_);
    }

    void record_one_submit(TaskId task_id) { record_submit(task_id, 0, nullptr, nullptr); }

    void record_submit(
        TaskId task_id, int explicit_dep_count, const TaskId *explicit_deps, const uint8_t *explicit_dep_kinds_raw,
        uint8_t default_explicit_dep_kind = kCreatorEdgeDepKinds
    ) {
        const int32_t kernel_ids[3] = {-1, -1, -1};
        dep_gen_aicpu_record_submit(
            task_id, /*in_manual_scope=*/false, /*early_dispatch=*/false, /*tensor_count=*/0,
            /*tensor_ptrs=*/nullptr, /*arg_types=*/nullptr, explicit_dep_count, explicit_deps, explicit_dep_kinds_raw,
            default_explicit_dep_kind, /*block_num=*/1, kernel_ids
        );
    }

    void record_submit_with_additional(
        TaskId task_id, int explicit_dep_count, const TaskId *explicit_deps, const uint8_t *explicit_dep_kinds_raw,
        int additional_dep_count, const TaskId *additional_deps, uint8_t additional_dep_kind = kWaitEdgeDepKind
    ) {
        const int32_t kernel_ids[3] = {-1, -1, -1};
        dep_gen_aicpu_record_submit(
            task_id, false, false, 0, nullptr, nullptr, explicit_dep_count, explicit_deps, explicit_dep_kinds_raw,
            kCreatorEdgeDepKinds, 1, kernel_ids, additional_dep_count, additional_deps, additional_dep_kind
        );
    }

    void *shm_ = nullptr;
    DepGenBuffer *buffer_ = nullptr;
    DepGenDataHeader *header_ = nullptr;
    DepGenBufferState *state_ = nullptr;
};

TEST_F(DepGenCollectorAicpuTest, SubmitBeforeOrchThreadIdxIsDropped) {
    record_one_submit(simpler::ut::test_task_id(0x1234));

    EXPECT_EQ(buffer_->count, 0u);
    EXPECT_EQ(state_->total_record_count, 1u);
    EXPECT_EQ(state_->dropped_record_count, 1u);
    EXPECT_EQ(state_->total_overflow_record_count, 0u);
    // Host reconciliation invariant: collected + dropped == total + overflow.
    EXPECT_EQ(
        buffer_->count + state_->dropped_record_count, state_->total_record_count + state_->total_overflow_record_count
    );
    EXPECT_EQ(header_->queue_tails[0], 0u);
}

TEST_F(DepGenCollectorAicpuTest, SubmitAfterOrchThreadIdxIsRecordedAndFlushed) {
    dep_gen_aicpu_set_orch_thread_idx(0);
    record_one_submit(simpler::ut::test_task_id(0x1234));

    ASSERT_EQ(buffer_->count, 1u);
    EXPECT_EQ(buffer_->records[0].task_id, simpler::ut::test_task_id(0x1234));
    EXPECT_EQ(state_->total_record_count, 1u);
    EXPECT_EQ(state_->dropped_record_count, 0u);

    dep_gen_aicpu_flush();

    EXPECT_EQ(header_->queue_tails[0], 1u);
    EXPECT_EQ(header_->queues[0][0].buffer_ptr, reinterpret_cast<uint64_t>(buffer_));
    EXPECT_EQ(state_->current_buf_ptr, 0u);
}

TEST_F(DepGenCollectorAicpuTest, NullKindsUseDefaultForEveryDependency) {
    dep_gen_aicpu_set_orch_thread_idx(0);
    constexpr int kDepCount = DEP_GEN_MAX_EXPLICIT_DEPS + 1;
    std::vector<TaskId> deps(kDepCount);
    for (int i = 0; i < kDepCount; ++i) {
        deps[i] = simpler::ut::test_task_id(0x100 + i);
    }

    record_submit(
        simpler::ut::test_task_id(0x1234), kDepCount, deps.data(), nullptr,
        /*default_explicit_dep_kind=*/3
    );

    ASSERT_EQ(buffer_->count, 2u);
    const DepGenRecord &record = buffer_->records[0];
    ASSERT_EQ(record.explicit_dep_count, DEP_GEN_MAX_EXPLICIT_DEPS);
    for (int i = 0; i < DEP_GEN_MAX_EXPLICIT_DEPS; ++i) {
        EXPECT_EQ(record.explicit_deps[i], deps[i]);
        EXPECT_EQ(record.explicit_dep_kinds[i], 3u);
    }

    const auto *overflow = reinterpret_cast<const DepGenOverflowRecord *>(&buffer_->records[1]);
    ASSERT_EQ(overflow->dep_count, 1u);
    EXPECT_EQ(overflow->deps[0], deps.back());
    EXPECT_EQ(overflow->kinds[0], 3u);
}

TEST_F(DepGenCollectorAicpuTest, MixedKindsSurviveTwoOverflowRecords) {
    dep_gen_aicpu_set_orch_thread_idx(0);
    constexpr int kDepCount = DEP_GEN_MAX_EXPLICIT_DEPS + DEP_GEN_OVERFLOW_DEPS_PER_RECORD + 1;
    std::vector<TaskId> deps(kDepCount);
    std::vector<uint8_t> kinds(kDepCount);
    for (int i = 0; i < kDepCount; ++i) {
        deps[i] = simpler::ut::test_task_id(0x1000 + i);
        kinds[i] = (i % 2 == 0) ? 1u : 3u;
    }

    record_submit(simpler::ut::test_task_id(0x5678), kDepCount, deps.data(), kinds.data());

    ASSERT_EQ(buffer_->count, 3u);
    EXPECT_EQ(state_->total_record_count, 1u);
    EXPECT_EQ(state_->total_overflow_record_count, 2u);

    const DepGenRecord &base = buffer_->records[0];
    ASSERT_EQ(base.explicit_dep_count, DEP_GEN_MAX_EXPLICIT_DEPS);
    EXPECT_NE(base.flags & DEP_GEN_FLAG_HAS_OVERFLOW, 0u);
    for (int i = 0; i < DEP_GEN_MAX_EXPLICIT_DEPS; ++i) {
        EXPECT_EQ(base.explicit_deps[i], deps[i]);
        EXPECT_EQ(base.explicit_dep_kinds[i], kinds[i]);
    }

    const auto *first = reinterpret_cast<const DepGenOverflowRecord *>(&buffer_->records[1]);
    ASSERT_EQ(first->dep_count, DEP_GEN_OVERFLOW_DEPS_PER_RECORD);
    EXPECT_NE(first->flags & DEP_GEN_FLAG_OVERFLOW, 0u);
    EXPECT_EQ(first->flags & DEP_GEN_FLAG_LAST_OVERFLOW, 0u);
    for (int i = 0; i < DEP_GEN_OVERFLOW_DEPS_PER_RECORD; ++i) {
        const int dep_idx = DEP_GEN_MAX_EXPLICIT_DEPS + i;
        EXPECT_EQ(first->deps[i], deps[dep_idx]);
        EXPECT_EQ(first->kinds[i], kinds[dep_idx]);
    }

    const auto *last = reinterpret_cast<const DepGenOverflowRecord *>(&buffer_->records[2]);
    ASSERT_EQ(last->dep_count, 1u);
    EXPECT_NE(last->flags & DEP_GEN_FLAG_OVERFLOW, 0u);
    EXPECT_NE(last->flags & DEP_GEN_FLAG_LAST_OVERFLOW, 0u);
    EXPECT_EQ(last->deps[0], deps.back());
    EXPECT_EQ(last->kinds[0], kinds.back());
}

TEST_F(DepGenCollectorAicpuTest, EmptyAdditionalSpanMatchesLegacyRecord) {
    dep_gen_aicpu_set_orch_thread_idx(0);
    const TaskId task = simpler::ut::test_task_id(0x3000);
    const std::array<TaskId, 2> deps = {simpler::ut::test_task_id(1), simpler::ut::test_task_id(2)};
    const std::array<uint8_t, 2> kinds = {kWaitEdgeDepKind, kRetainEdgeDepKind};
    record_submit(task, static_cast<int>(deps.size()), deps.data(), kinds.data());
    record_submit_with_additional(task, static_cast<int>(deps.size()), deps.data(), kinds.data(), 0, nullptr);

    ASSERT_EQ(buffer_->count, 2u);
    EXPECT_EQ(memcmp(&buffer_->records[0], &buffer_->records[1], sizeof(DepGenRecord)), 0);
    EXPECT_EQ(state_->total_record_count, 2u);
    EXPECT_EQ(state_->total_overflow_record_count, 0u);
    EXPECT_EQ(state_->dropped_record_count, 0u);
}

TEST_F(DepGenCollectorAicpuTest, AdditionalOnlySpanUsesItsFixedKind) {
    dep_gen_aicpu_set_orch_thread_idx(0);
    const std::array<TaskId, 2> deps = {simpler::ut::test_task_id(3), simpler::ut::test_task_id(4)};
    record_submit_with_additional(
        simpler::ut::test_task_id(0x3001), 0, nullptr, nullptr, static_cast<int>(deps.size()), deps.data()
    );

    ASSERT_EQ(buffer_->count, 1u);
    const auto &record = buffer_->records[0];
    ASSERT_EQ(record.explicit_dep_count, deps.size());
    for (size_t i = 0; i < deps.size(); ++i) {
        EXPECT_EQ(record.explicit_deps[i], deps[i]);
        EXPECT_EQ(record.explicit_dep_kinds[i], kWaitEdgeDepKind);
    }
}

TEST_F(DepGenCollectorAicpuTest, DependencySpansCrossBaseAndOverflowBoundaries) {
    dep_gen_aicpu_set_orch_thread_idx(0);
    struct Counts {
        int explicit_count;
        int additional_count;
    };
    constexpr int base_capacity = DEP_GEN_MAX_EXPLICIT_DEPS;
    constexpr int overflow_capacity = DEP_GEN_OVERFLOW_DEPS_PER_RECORD;
    const std::array<Counts, 8> cases = {{
        {0, base_capacity + 1},
        {1, base_capacity},
        {base_capacity - 1, 2},
        {base_capacity, 1},
        {base_capacity + 1, overflow_capacity},
        {base_capacity + overflow_capacity - 1, 2},
        {base_capacity + overflow_capacity, 1},
        {base_capacity + overflow_capacity + 1, 2},
    }};
    uint32_t expected_slots = 0;
    for (size_t c = 0; c < cases.size(); ++c) {
        SCOPED_TRACE(c);
        const auto counts = cases[c];
        std::vector<TaskId> explicit_deps(counts.explicit_count);
        std::vector<uint8_t> kinds(counts.explicit_count);
        std::vector<TaskId> additional_deps(counts.additional_count);
        for (int i = 0; i < counts.explicit_count; ++i) {
            explicit_deps[i] = simpler::ut::test_task_id(0x1000 + i);
            kinds[i] = i % 2 == 0 ? kWaitEdgeDepKind : kRetainEdgeDepKind;
        }
        for (int i = 0; i < counts.additional_count; ++i)
            additional_deps[i] = simpler::ut::test_task_id(0x2000 + i);
        const uint8_t *raw_kinds = c % 2 == 0 ? kinds.data() : nullptr;
        const TaskId task = simpler::ut::test_task_id(0x4000 + c);
        const int total = counts.explicit_count + counts.additional_count;
        const int needed = dep_gen_records_needed_for(total);
        const uint32_t start = buffer_->count;
        record_submit_with_additional(
            task, counts.explicit_count, explicit_deps.data(), raw_kinds, counts.additional_count,
            additional_deps.data()
        );

        expected_slots += static_cast<uint32_t>(needed);
        ASSERT_EQ(buffer_->count, expected_slots);
        const auto &base = buffer_->records[start];
        EXPECT_EQ(base.task_id, task);
        EXPECT_EQ((base.flags & DEP_GEN_FLAG_HAS_OVERFLOW) != 0, needed > 1);
        ASSERT_EQ(base.explicit_dep_count, base_capacity);
        int read = 0;
        auto expect_range = [&](const TaskId *deps, const uint8_t *captured_kinds, int count) {
            for (int i = 0; i < count; ++i, ++read) {
                const bool is_explicit = read < counts.explicit_count;
                const TaskId expected =
                    is_explicit ? explicit_deps[read] : additional_deps[read - counts.explicit_count];
                const uint8_t expected_kind =
                    is_explicit ? (raw_kinds != nullptr ? kinds[read] : kCreatorEdgeDepKinds) : kWaitEdgeDepKind;
                EXPECT_EQ(deps[i], expected) << read;
                EXPECT_EQ(captured_kinds[i], expected_kind) << read;
            }
        };
        expect_range(base.explicit_deps, base.explicit_dep_kinds, base.explicit_dep_count);
        for (int link = 1; link < needed; ++link) {
            const auto *overflow =
                reinterpret_cast<const DepGenOverflowRecord *>(&buffer_->records[start + static_cast<uint32_t>(link)]);
            ASSERT_EQ(overflow->dep_count, std::min(total - read, overflow_capacity));
            EXPECT_EQ(overflow->task_id, task);
            EXPECT_NE(overflow->flags & DEP_GEN_FLAG_OVERFLOW, 0u);
            EXPECT_EQ((overflow->flags & DEP_GEN_FLAG_LAST_OVERFLOW) != 0, link == needed - 1);
            expect_range(overflow->deps, overflow->kinds, overflow->dep_count);
        }
        EXPECT_EQ(read, total);
    }
    EXPECT_EQ(state_->total_record_count, cases.size());
    EXPECT_EQ(state_->total_overflow_record_count, expected_slots - cases.size());
    EXPECT_EQ(state_->dropped_record_count, 0u);
}

TEST_F(DepGenCollectorAicpuTest, RepeatedProducerPreservesRetainAndAdditionalWaitFacts) {
    dep_gen_aicpu_set_orch_thread_idx(0);
    const TaskId producer = simpler::ut::test_task_id(0x123);
    record_submit_with_additional(simpler::ut::test_task_id(0x3002), 1, &producer, &kRetainEdgeDepKind, 1, &producer);

    ASSERT_EQ(buffer_->count, 1u);
    const auto &record = buffer_->records[0];
    ASSERT_EQ(record.explicit_dep_count, 2u);
    EXPECT_EQ(record.explicit_deps[0], producer);
    EXPECT_EQ(record.explicit_dep_kinds[0], kRetainEdgeDepKind);
    EXPECT_EQ(record.explicit_deps[1], producer);
    EXPECT_EQ(record.explicit_dep_kinds[1], kWaitEdgeDepKind);
}

TEST_F(DepGenCollectorAicpuTest, CombinedCountOverflowDropsBeforeInspectingDependencyRanges) {
    dep_gen_aicpu_set_orch_thread_idx(0);
    // Both pointers name real objects. The unrepresentable combined count is
    // rejected before either declared range is inspected.
    const TaskId producer = simpler::ut::test_task_id(0x124);
    record_submit_with_additional(
        simpler::ut::test_task_id(0x3003), std::numeric_limits<int>::max(), &producer, nullptr, 1, &producer
    );

    EXPECT_EQ(buffer_->count, 0u);
    EXPECT_EQ(state_->total_record_count, 1u);
    EXPECT_EQ(state_->dropped_record_count, 1u);
    EXPECT_EQ(state_->total_overflow_record_count, 0u);
    record_one_submit(simpler::ut::test_task_id(0x3004));
    EXPECT_EQ(buffer_->count, 1u);
    EXPECT_EQ(state_->total_record_count, 2u);
    EXPECT_EQ(state_->dropped_record_count, 1u);
}

TEST_F(DepGenCollectorAicpuTest, InvalidAdditionalSpanDropsWithoutPartialCapture) {
    dep_gen_aicpu_set_orch_thread_idx(0);
    record_submit_with_additional(simpler::ut::test_task_id(0x3005), 0, nullptr, nullptr, -1, nullptr);
    record_submit_with_additional(simpler::ut::test_task_id(0x3006), 0, nullptr, nullptr, 1, nullptr);

    EXPECT_EQ(buffer_->count, 0u);
    EXPECT_EQ(state_->total_record_count, 2u);
    EXPECT_EQ(state_->dropped_record_count, 2u);
    EXPECT_EQ(state_->total_overflow_record_count, 0u);
}

TEST_F(DepGenCollectorAicpuTest, OversizedCombinedChainTruncatesAtBufferBudgetAcrossSpans) {
    dep_gen_aicpu_set_orch_thread_idx(0);
    const int max_deps =
        DEP_GEN_MAX_EXPLICIT_DEPS + (PLATFORM_DEP_GEN_RECORDS_PER_BUFFER - 1) * DEP_GEN_OVERFLOW_DEPS_PER_RECORD;
    const TaskId producer = simpler::ut::test_task_id(0x125);
    std::vector<TaskId> explicit_deps(static_cast<size_t>(max_deps - 1), producer);
    const std::array<TaskId, 2> additional = {simpler::ut::test_task_id(0x126), simpler::ut::test_task_id(0x127)};
    record_submit_with_additional(
        simpler::ut::test_task_id(0x3007), static_cast<int>(explicit_deps.size()), explicit_deps.data(), nullptr,
        static_cast<int>(additional.size()), additional.data()
    );

    ASSERT_EQ(buffer_->count, static_cast<uint32_t>(PLATFORM_DEP_GEN_RECORDS_PER_BUFFER));
    const auto *last =
        reinterpret_cast<const DepGenOverflowRecord *>(&buffer_->records[PLATFORM_DEP_GEN_RECORDS_PER_BUFFER - 1]);
    ASSERT_EQ(last->dep_count, DEP_GEN_OVERFLOW_DEPS_PER_RECORD);
    EXPECT_NE(last->flags & DEP_GEN_FLAG_LAST_OVERFLOW, 0u);
    EXPECT_EQ(last->deps[DEP_GEN_OVERFLOW_DEPS_PER_RECORD - 2], producer);
    EXPECT_EQ(last->kinds[DEP_GEN_OVERFLOW_DEPS_PER_RECORD - 2], kCreatorEdgeDepKinds);
    EXPECT_EQ(last->deps[DEP_GEN_OVERFLOW_DEPS_PER_RECORD - 1], additional[0]);
    EXPECT_EQ(last->kinds[DEP_GEN_OVERFLOW_DEPS_PER_RECORD - 1], kWaitEdgeDepKind);
    EXPECT_EQ(state_->total_record_count, 1u);
    EXPECT_EQ(state_->total_overflow_record_count, static_cast<uint32_t>(PLATFORM_DEP_GEN_RECORDS_PER_BUFFER - 1));
    EXPECT_EQ(state_->dropped_record_count, 0u);
}

// ---------------------------------------------------------------------------
// Per-run terminal accounting
// ---------------------------------------------------------------------------

/**
 * One terminal bank, owned by the test the way the host collector owns it.
 *
 * The producer publishes into it and the host-side reconciliation reads it, so
 * the unit a producer counts in is checked against the unit the reader
 * subtracts in — which is the whole hazard an overflow chain introduces.
 */
class DepGenTerminalTest : public DepGenCollectorAicpuTest {
protected:
    void SetUp() override {
        DepGenCollectorAicpuTest::SetUp();
        bank_.assign(static_cast<size_t>(DEP_GEN_TERMINAL_PRODUCERS) * PLATFORM_DIAGNOSTIC_TERMINAL_BANKS, {});
        for (auto &entry : bank_)
            diagnostic_terminal_reset(&entry);
        set_platform_dep_gen_run_terminal(reinterpret_cast<uint64_t>(bank_.data()));
        set_platform_run_result(/*region_base=*/0, kEpoch);
        dep_gen_aicpu_set_orch_thread_idx(0);
    }

    void TearDown() override {
        set_platform_run_result(0, 0);
        set_platform_dep_gen_run_terminal(0);
        DepGenCollectorAicpuTest::TearDown();
    }

    /** Reconcile this run exactly as the collector will. */
    simpler::dfx::credit::RunReconciliation reconcile(uint64_t received) const {
        return simpler::dfx::credit::reconcile_run(
            bank_.data(), DEP_GEN_TERMINAL_PRODUCERS, DEP_GEN_TERMINAL_PRODUCERS, kEpoch, received
        );
    }

    static constexpr uint64_t kEpoch = 7701;
    std::vector<DiagnosticRunTerminal> bank_;
};

TEST_F(DepGenTerminalTest, ASubmitThatSpillsIntoAChainConservesInSlots) {
    constexpr int kDepCount = DEP_GEN_MAX_EXPLICIT_DEPS + 1;
    std::vector<TaskId> deps(kDepCount);
    for (int i = 0; i < kDepCount; ++i)
        deps[i] = simpler::ut::test_task_id(0x200 + i);

    record_submit(simpler::ut::test_task_id(0x4321), kDepCount, deps.data(), nullptr);
    // One submit, two buffer slots: the base record plus its overflow.
    ASSERT_EQ(buffer_->count, 2u);
    ASSERT_EQ(state_->total_record_count, 1u);
    ASSERT_EQ(state_->total_overflow_record_count, 1u);

    dep_gen_aicpu_flush();
    dep_gen_aicpu_close_run();

    // The terminal is in slots, not submits: counting submits would report one
    // produced against two published and condemn a healthy run.
    EXPECT_EQ(bank_[0].run_epoch, kEpoch);
    EXPECT_EQ(bank_[0].total, 2u);
    EXPECT_EQ(bank_[0].published_records, 2u);
    EXPECT_EQ(bank_[0].dropped, 0u);
    EXPECT_EQ(bank_[0].live_at_close, 0u);

    const auto out = reconcile(/*received=*/2);
    EXPECT_EQ(out.verdict, simpler::dfx::runs::Verdict::Published);
    EXPECT_FALSE(out.counts_unknown);
}

TEST_F(DepGenTerminalTest, AnUnflushedBufferIsReportedLiveAndStillConserves) {
    record_one_submit(simpler::ut::test_task_id(0x11));
    record_one_submit(simpler::ut::test_task_id(0x12));
    ASSERT_EQ(buffer_->count, 2u);

    // No flush: the records are in the producer's held buffer at close.
    dep_gen_aicpu_close_run();

    EXPECT_EQ(bank_[0].total, 2u);
    EXPECT_EQ(bank_[0].published_records, 0u);
    EXPECT_EQ(bank_[0].live_at_close, 2u);

    const auto out = reconcile(/*received=*/0);
    EXPECT_EQ(out.verdict, simpler::dfx::runs::Verdict::PartialSafe);
    EXPECT_FALSE(out.counts_unknown) << "a held buffer is a known loss, not an unknown one";
    EXPECT_EQ(out.unsettled, 2u);
}

TEST_F(DepGenTerminalTest, ARunWithoutAnIdentityPublishesNothing) {
    set_platform_run_result(0, 0);
    record_one_submit(simpler::ut::test_task_id(0x13));
    dep_gen_aicpu_flush();
    dep_gen_aicpu_close_run();

    EXPECT_EQ(bank_[0].run_epoch, 0u) << "a run with no identity claimed a bank";
}

TEST_F(DepGenTerminalTest, TheRunsCountersStartAtItsOwnDeviceInit) {
    record_one_submit(simpler::ut::test_task_id(0x14));
    dep_gen_aicpu_flush();
    dep_gen_aicpu_close_run();
    ASSERT_EQ(bank_[0].total, 1u);

    // The host returns the published buffer to the free queue, as its drain
    // shard does; AICPU never produces into that queue itself.
    state_->free_queue.buffer_ptrs[1] = reinterpret_cast<uint64_t>(buffer_);
    state_->free_queue.tail = 2;

    // The successor's device init is what zeroes them, with no host reset in
    // between — which is the whole point of moving the reset to the producer.
    set_platform_run_result(0, kEpoch + 1);
    dep_gen_aicpu_init();
    ASSERT_EQ(state_->current_buf_ptr, reinterpret_cast<uint64_t>(buffer_));
    EXPECT_EQ(state_->total_record_count, 0u);
    EXPECT_EQ(state_->published_record_count, 0u);
    // The entry is the successor's from its init: the init claims it under the
    // successor's identity, so the predecessor's snapshot is gone and the
    // figures standing in it are nobody's yet. Both halves matter — an entry
    // still naming the predecessor would let its counts be read as this run's,
    // and one naming the successor with usable arithmetic would publish a run
    // that has not closed.
    EXPECT_EQ(bank_[0].run_epoch, kEpoch + 1) << "the successor's init did not claim the entry it publishes into";
    EXPECT_NE(bank_[0].counts_unknown, 0u) << "a claimed entry offered arithmetic before its run closed";
    EXPECT_EQ(bank_[0].total, 0u) << "the successor's init left its predecessor's count in the entry";
    EXPECT_EQ(bank_[0].published_records, 0u);
    // And an interrupted successor is still rejected: settled from here, the
    // run reports the unknown its own claim carries rather than the
    // predecessor's balanced figures.
    const simpler::dfx::credit::RunReconciliation interrupted = simpler::dfx::credit::reconcile_run(
        bank_.data(), DEP_GEN_TERMINAL_PRODUCERS, /*expected=*/1, kEpoch + 1, /*received=*/0
    );
    EXPECT_EQ(interrupted.verdict, simpler::dfx::runs::Verdict::CutUnknown)
        << "a successor that never closed was settled";
    EXPECT_TRUE(interrupted.counts_unknown);

    record_one_submit(simpler::ut::test_task_id(0x15));
    record_one_submit(simpler::ut::test_task_id(0x16));
    dep_gen_aicpu_flush();
    dep_gen_aicpu_close_run();

    EXPECT_EQ(bank_[0].run_epoch, kEpoch + 1);
    EXPECT_EQ(bank_[0].total, 2u) << "the successor inherited its predecessor's count";
    EXPECT_EQ(bank_[0].published_records, 2u);
}

}  // namespace
