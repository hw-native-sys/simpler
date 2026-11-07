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
#include <cstring>
#include "backend/async_backend.h"

namespace {
struct alignas(8) BackendData {
    AsyncBackendLayout layout{};
    AsyncBackendTask tasks[2]{};
    AsyncCounterRange counters[2]{};
    AsyncSdmaTransfer transfers[2]{};
    BackendData() {
        layout.total_bytes = sizeof(*this);
        layout.task_count = 2;
        layout.counter_count = 2;
        layout.transfer_count = 2;
        layout.tasks_offset = offsetof(BackendData, tasks);
        layout.counters_offset = offsetof(BackendData, counters);
        layout.transfers_offset = offsetof(BackendData, transfers);
        tasks[0] = {1, 0, 2, 0, 2};
        tasks[1] = {3, 2, 0, 2, 0};
        counters[0] = {0x1004, 3, 16};
        counters[1] = {0x2000, 1, 4};
        transfers[0] = {0x10000, 4096, 0x20000, 512, 0x30000, 1024, 2};
        transfers[1] = {0x10000, 4096, 0x40000, 256, 0x50000, 256, 2};
    }
    AsyncBackendView view() const {
        AsyncBackendView out;
        EXPECT_TRUE(out.mount(reinterpret_cast<const uint8_t *>(this), sizeof(*this)));
        return out;
    }
};
SchedulerAsyncGroupIdentity identity(int64_t task = 1) {
    SchedulerAsyncGroupIdentity result{};
    result.run_epoch = 19;
    result.task_ref = task;
    result.capacity = 64;
    result.condition_types = CORE_ASYNC_CONDITION_TYPES;
    return result;
}
DeferredCompletionEntry counter(uint64_t address, uint32_t expected = 1) {
    return {address, 0, expected, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER, 0};
}
DeferredCompletionEntry event(uint64_t address, uint64_t expected = UINT64_C(0x100000001)) {
    return {address, expected, 0, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_SDMA_EVENT_RECORD, 0};
}

TEST(HbgAsyncBackend, DefaultRegistrationNeedsNoDeclarationImage) {
    AsyncBackendView empty;
    ASSERT_TRUE(empty.mount(nullptr, 0));
    AsyncCompletionBackend backend(empty, HBG_ASYNC_CONTRACT_UNQUALIFIED);
    EXPECT_TRUE(backend.validate_registration(identity(), 0, counter(0x1000)));
    EXPECT_TRUE(backend.validate_registration(identity(), 63, event(0x2000)));
    EXPECT_FALSE(backend.validate_registration(identity(), 64, counter(0x1000)));
    EXPECT_FALSE(backend.validate_registration(identity(), 0, counter(0)));
    EXPECT_FALSE(backend.validate_registration(identity(), 0, event(0x1004)));
    EXPECT_FALSE(backend.validate_registration(identity(), 0, counter(0x1000, UINT32_MAX)));
    AsyncCompletionBackend core_only(empty, HBG_ASYNC_CONTRACT_CORE_PIPELINES_ONLY);
    EXPECT_TRUE(core_only.validate_registration(identity(), 0, counter(0x1000)));
    EXPECT_FALSE(core_only.validate_registration(identity(), 0, event(0x2000)));
}

TEST(HbgAsyncBackend, DeclaredCountersKeepStrideHolesAndEmptySetsDistinctFromDefaults) {
    BackendData data;
    AsyncCompletionBackend backend(data.view(), 0);
    for (uint64_t address : {0x1004, 0x1014, 0x1024, 0x2000})
        EXPECT_TRUE(backend.validate_registration(identity(), 0, counter(address)));
    for (uint64_t address : {0x1000, 0x1008, 0x1034, 0x2004})
        EXPECT_FALSE(backend.validate_registration(identity(), 0, counter(address)));
    EXPECT_FALSE(backend.validate_registration(identity(3), 0, counter(0x1004)));
    EXPECT_TRUE(backend.validate_registration(identity(2), 0, counter(0x7000)));
    data.counters[0].count = UINT64_MAX;
    AsyncCompletionBackend corrupt(data.view(), 0);
    EXPECT_FALSE(corrupt.validate_registration(identity(), 0, counter(0x1004)));
}

TEST(HbgAsyncBackend, TransferSelectionHonorsWorkspaceAndBothExactViews) {
    BackendData data;
    auto view = data.view();
    AsyncSdmaTransfer selected;
    bool declared = false;
    ASSERT_TRUE(view.select_transfer(1, 0x10000, 0x20020, 0x30040, 128, &selected, &declared));
    EXPECT_TRUE(declared);
    EXPECT_EQ(selected.queue_count, 2u);
    EXPECT_FALSE(view.select_transfer(1, 0x11000, 0x20000, 0x30000, 64, &selected, &declared));
    EXPECT_FALSE(view.select_transfer(1, 0x10000, 0x201e0, 0x30000, 64, &selected, &declared));
    EXPECT_FALSE(view.select_transfer(1, 0x10000, 0x20000, 0x303f0, 64, &selected, &declared));
    EXPECT_FALSE(view.select_transfer(3, 0x10000, 0x20000, 0x30000, 64, &selected, &declared));
    EXPECT_TRUE(view.select_transfer(2, 0x10000, 0x20000, 0x30000, 64, &selected, &declared));
    EXPECT_FALSE(declared);
    data.transfers[1] = data.transfers[0];
    EXPECT_FALSE(data.view().select_transfer(1, 0x10000, 0x20000, 0x30000, 64, &selected, &declared));
}

TEST(HbgAsyncBackend, DeclaredSdmaRequiresRecordCacheLineInsideWorkspace) {
    BackendData data;
    AsyncCompletionBackend backend(data.view(), 0);
    EXPECT_TRUE(backend.validate_registration(identity(), 0, event(0x10000)));
    EXPECT_TRUE(backend.validate_registration(identity(), 0, event(0x10ff8)));
    EXPECT_FALSE(backend.validate_registration(identity(), 0, event(0x11000)));
    EXPECT_FALSE(backend.validate_registration(identity(3), 0, event(0x10000)));
}

TEST(HbgAsyncBackend, MountRejectsVersionOverflowTruncationAndOverlappingSections) {
    for (uint32_t fault = 0; fault < 6; ++fault) {
        BackendData data;
        uint64_t bytes = sizeof(data);
        if (fault == 0) ++data.layout.version;
        if (fault == 1) --bytes;
        if (fault == 2) data.layout.counter_count = UINT64_MAX;
        if (fault == 3) data.layout.counters_offset = data.layout.tasks_offset;
        if (fault == 4) data.layout.transfers_offset = data.layout.counters_offset;
        if (fault == 5) ++data.layout.tasks_offset;
        AsyncBackendView view;
        EXPECT_FALSE(view.mount(reinterpret_cast<const uint8_t *>(&data), bytes));
        EXPECT_EQ(view.base, nullptr);
    }
}

struct Reader {
    uint32_t *reads;
    int32_t signed_value;
    uint64_t post;
    int32_t counter(uint64_t) {
        ++*reads;
        return signed_value;
    }
    uint64_t sdma_post_done(uint64_t) {
        ++*reads;
        return post;
    }
};
struct Barrier {
    uint32_t *count;
    void finish() { ++*count; }
};
TEST(HbgAsyncBackend, CheckUsesOneObservationAndFullWidthThresholdWithoutResourceTables) {
    uint32_t reads = 0, barriers = 0;
    AsyncCompletionBackend<Reader, Barrier> pending({}, 0, {&reads, -1, UINT64_C(0x100000000)}, {&barriers});
    EXPECT_EQ(pending.observe_once(identity(), 0, counter(0x1000, 0)).state, SchedulerAsyncConditionState::PENDING);
    EXPECT_EQ(pending.observe_once(identity(), 0, event(0x2000)).state, SchedulerAsyncConditionState::PENDING);
    EXPECT_EQ(reads, 2u);
    EXPECT_EQ(barriers, 0u);
    AsyncCompletionBackend<Reader, Barrier> ready({}, 0, {&reads, 1, UINT64_C(0x100000001)}, {&barriers});
    EXPECT_EQ(ready.observe_once(identity(), 0, event(0x2000)).state, SchedulerAsyncConditionState::READY);
    EXPECT_EQ(reads, 3u);
    EXPECT_EQ(barriers, 1u);
    EXPECT_EQ(ready.observe_once(identity(), 0, counter(0)).state, SchedulerAsyncConditionState::FAILED);
    EXPECT_EQ(reads, 3u);
}
}  // namespace
