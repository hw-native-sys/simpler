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

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

// The collector stores this byte and never reads it back: DepFlags is declared
// in each runtime's own types.h, which is why the entry point takes a plain
// uint8_t. Naming the enumerators here would tie this case to one runtime's
// spelling of a value the code under test does not interpret.
constexpr uint8_t kCreatorEdgeDepKinds = 0x3;  // wait | retain

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

    void record_one_submit(uint64_t task_id_raw) { record_submit(task_id_raw, 0, nullptr, nullptr); }

    void record_submit(
        uint64_t task_id_raw, int explicit_dep_count, const uint64_t *explicit_deps_raw,
        const uint8_t *explicit_dep_kinds_raw, uint8_t default_explicit_dep_kind = kCreatorEdgeDepKinds
    ) {
        const int32_t kernel_ids[3] = {-1, -1, -1};
        dep_gen_aicpu_record_submit(
            task_id_raw, /*in_manual_scope=*/false, /*early_dispatch=*/false, /*tensor_count=*/0,
            /*tensor_ptrs=*/nullptr, /*arg_types=*/nullptr, explicit_dep_count, explicit_deps_raw,
            explicit_dep_kinds_raw, default_explicit_dep_kind, /*block_num=*/1, kernel_ids
        );
    }

    void *shm_ = nullptr;
    DepGenBuffer *buffer_ = nullptr;
    DepGenDataHeader *header_ = nullptr;
    DepGenBufferState *state_ = nullptr;
};

TEST_F(DepGenCollectorAicpuTest, SubmitBeforeOrchThreadIdxIsDropped) {
    record_one_submit(0x1234);

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
    record_one_submit(0x1234);

    ASSERT_EQ(buffer_->count, 1u);
    EXPECT_EQ(buffer_->records[0].task_id, 0x1234u);
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
    std::vector<uint64_t> deps(kDepCount);
    for (int i = 0; i < kDepCount; ++i) {
        deps[i] = 0x100u + static_cast<uint64_t>(i);
    }

    record_submit(0x1234, kDepCount, deps.data(), nullptr, /*default_explicit_dep_kind=*/3);

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
    std::vector<uint64_t> deps(kDepCount);
    std::vector<uint8_t> kinds(kDepCount);
    for (int i = 0; i < kDepCount; ++i) {
        deps[i] = 0x1000u + static_cast<uint64_t>(i);
        kinds[i] = (i % 2 == 0) ? 1u : 3u;
    }

    record_submit(0x5678, kDepCount, deps.data(), kinds.data());

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

}  // namespace
