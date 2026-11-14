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

#include <array>
#include <cstring>
#include <vector>

#include "host_build_graph/host/async_completion_plan.h"
#include "host_build_graph/host/prepared_execution_plan.h"

static_assert(sizeof(LocalContext) == 8);

namespace {

class HbgPreparedBindings : public ::testing::Test {
protected:
    static constexpr uint64_t kDeviceBase = UINT64_C(0x100000000);
    static constexpr uint64_t kEpoch = 37;
    AicoreSchedulerLayout layout{};
    AsyncCompletionHostPlan async{};
    std::vector<uint8_t> storage;
    void *base{nullptr};

    void SetUp() override {
        ASSERT_TRUE(scheduler_plan_layout(2, 2, 1, &layout, false, 7));
        AsyncTaskPlanInput first{0, 3, LaneMask::aic(), {1, 0, 0}};
        AsyncTaskPlanInput second{1, 2, LaneMask(true, false, true), {0, 0, 2}};
        std::string error;
        ASSERT_TRUE(plan_async_completion(layout.total_size, kEpoch, 2, {first, second}, &async, &error)) << error;
        layout.async_layout_offset = async.layout_offset;
        layout.total_size = async.total_bytes;
        storage.resize(layout.total_size + SCHEDULER_STATE_BASE_ALIGNMENT);
        base = reinterpret_cast<void *>(
            (reinterpret_cast<uintptr_t>(storage.data()) + SCHEDULER_STATE_BASE_ALIGNMENT - 1) &
            ~(uintptr_t{SCHEDULER_STATE_BASE_ALIGNMENT} - 1)
        );
        ASSERT_TRUE(scheduler_init_data_from_layout(base, layout, kDeviceBase, async.initialized_bytes));
        ASSERT_TRUE(initialize_async_completion(base, layout.total_size, async));
        auto *metadata = scheduler_state_at<SchedulerTaskMetadata>(base, layout.task_metadata_offset);
        metadata[0].flags = SCHEDULER_TASK_EXECUTABLE;
        metadata[0].active_mask = LaneMask::aic();
        metadata[0].logical_block_num = 3;
        metadata[1].flags = SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_MIX;
        metadata[1].active_mask = LaneMask(true, false, true);
        metadata[1].logical_block_num = 2;
        ASSERT_TRUE(scheduler_init_prepared_execution_records(base, layout, kEpoch));
    }

    bool prepare(int64_t task) {
        const uint64_t functions[]{0x1000, 0x2000, 0x3000};
        return scheduler_prepare_host_calls(base, layout, kDeviceBase, task, functions, 0, nullptr, 0, 0);
    }
};

TEST_F(HbgPreparedBindings, OnboardCallsResolveCodeWithoutDereferencingDeviceObjects) {
    const uint64_t functions[]{
        scheduler_resolve_callable_entry(kDeviceBase, SchedulerCallableAddressSpace::DEVICE), 0, 0
    };
    ASSERT_TRUE(scheduler_prepare_host_calls(base, layout, kDeviceBase, 0, functions, 0, nullptr, 0, 0));
    const auto *calls = scheduler_state_at<SchedulerPreparedCall>(base, layout.prepared.calls_offset);
    for (uint32_t block = 0; block < 3; ++block)
        EXPECT_EQ(calls[block].function_bin_addr, UINT64_C(0x1000000c0));
}

TEST_F(HbgPreparedBindings, SimCallsUseTheResolvedSymbolRatherThanTheObjectAddress) {
    const std::array<uint8_t, 4> binary{};
    auto bytes = make_callable<CORE_MAX_TENSOR_ARGS>(nullptr, 0, binary.data(), binary.size());
    auto *callable = reinterpret_cast<CoreCallable *>(bytes.data());
    constexpr uint64_t entry = UINT64_C(0x98760000);
    callable->set_resolved_addr(entry);
    const uint64_t functions[]{
        scheduler_resolve_callable_entry(reinterpret_cast<uint64_t>(callable), SchedulerCallableAddressSpace::HOST), 0,
        0
    };
    ASSERT_TRUE(scheduler_prepare_host_calls(base, layout, kDeviceBase, 0, functions, 0, nullptr, 0, 0));
    const auto *calls = scheduler_state_at<SchedulerPreparedCall>(base, layout.prepared.calls_offset);
    for (uint32_t block = 0; block < 3; ++block)
        EXPECT_EQ(calls[block].function_bin_addr, entry);
    callable->set_resolved_addr(0);
    EXPECT_EQ(
        scheduler_resolve_callable_entry(reinterpret_cast<uint64_t>(callable), SchedulerCallableAddressSpace::HOST), 0u
    );
}

TEST_F(HbgPreparedBindings, MissingOrOverflowingObjectsCannotPublishCalls) {
    for (auto address_space : {SchedulerCallableAddressSpace::DEVICE, SchedulerCallableAddressSpace::HOST}) {
        const uint64_t functions[]{scheduler_resolve_callable_entry(0, address_space), 0, 0};
        EXPECT_FALSE(scheduler_prepare_host_calls(base, layout, kDeviceBase, 0, functions, 0, nullptr, 0, 0));
    }
    const uint64_t aligned_max = UINT64_MAX & ~(uint64_t{alignof(CoreCallable)} - 1);
    EXPECT_EQ(scheduler_resolve_callable_entry(aligned_max, SchedulerCallableAddressSpace::DEVICE), 0u);
}

TEST_F(HbgPreparedBindings, SparseAsyncLanesKeepExactExecutionIdentityWithoutRegisteringConditions) {
    std::vector<uint8_t> async_before(async.initialized_bytes - async.begin_offset);
    std::memcpy(async_before.data(), static_cast<uint8_t *>(base) + async.begin_offset, async_before.size());
    ASSERT_TRUE(prepare(0));
    ASSERT_TRUE(prepare(1));
    EXPECT_EQ(
        std::memcmp(async_before.data(), static_cast<uint8_t *>(base) + async.begin_offset, async_before.size()), 0
    );
    const auto *bindings = scheduler_state_at<SchedulerExecutionBinding>(base, layout.prepared.bindings_offset);
    const std::array<uint32_t, 7> groups{0, 1, 2, UINT32_MAX, 3, UINT32_MAX, 4};
    const std::array<int32_t, 7> blocks{0, 1, 2, 0, 0, 1, 1};
    for (uint32_t index = 0; index < groups.size(); ++index) {
        SCOPED_TRACE(index);
        const auto &binding = bindings[index];
        EXPECT_EQ(binding.local_context.block_idx, blocks[index]);
        EXPECT_EQ(binding.local_context.block_num, index < 3 ? 3 : 2);
        EXPECT_EQ(binding.global_context.async_present, 0u);
        EXPECT_EQ(binding.global_context.async_group_index, groups[index]);
        if (groups[index] == UINT32_MAX) {
            EXPECT_EQ(binding.global_context.async_image_address, 0u);
            continue;
        }
        EXPECT_EQ(binding.global_context.async_image_address, kDeviceBase);
        EXPECT_EQ(binding.global_context.async_image_bytes, layout.total_size);
        EXPECT_EQ(binding.global_context.async_run_epoch, kEpoch);
        EXPECT_EQ(
            binding.global_context.async_group_offset,
            async.layout.groups_offset + groups[index] * sizeof(pto2::resident::AsyncGroupData)
        );
    }
}

TEST_F(HbgPreparedBindings, MismatchedAsyncEpochIsRejectedBeforePublishingCalls) {
    scheduler_state_at<SchedulerAsyncLayout>(base, layout.async_layout_offset)->run_epoch += 1;
    EXPECT_FALSE(prepare(0));
    EXPECT_EQ(scheduler_state_at<SchedulerPreparedControl>(base, layout.prepared.controls_offset)[0].state, 0u);
}

}  // namespace
