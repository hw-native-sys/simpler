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

#include "hbg_scheduler_test_support.h"

namespace {

void check_payload_range(void *base, uint64_t device_base, uint64_t offset, uint64_t count) {
    for (uint64_t index = 0; index < count; ++index) {
        SCOPED_TRACE(index);
        const uint64_t payload_offset = offset + index * sizeof(DispatchPayload);
        const auto *payload = scheduler_state_at<DispatchPayload>(base, payload_offset);
        ASSERT_EQ(
            payload->args[PAYLOAD_LOCAL_CONTEXT_INDEX],
            device_base + payload_offset + offsetof(DispatchPayload, local_context)
        );
        ASSERT_EQ(
            payload->args[PAYLOAD_GLOBAL_CONTEXT_INDEX],
            device_base + payload_offset + offsetof(DispatchPayload, global_context)
        );
        EXPECT_EQ(payload->local_context.block_idx, 0);
        EXPECT_EQ(payload->local_context.block_num, 1);
        EXPECT_EQ(payload->local_context.async_ctx.task_token, TaskId::invalid());
        EXPECT_EQ(payload->local_context.async_ctx.completion_count, nullptr);
        EXPECT_EQ(payload->local_context.async_ctx.completion_error_code, nullptr);
        EXPECT_EQ(payload->local_context.async_ctx.completion_entries, nullptr);
        EXPECT_EQ(payload->local_context.async_ctx.completion_capacity, 0u);
        EXPECT_EQ(payload->global_context.sub_block_id, 0);
        EXPECT_EQ(payload->function_bin_addr, 0u);
        EXPECT_EQ(payload->src_payload, 0u);
        for (int32_t arg = 0; arg < PAYLOAD_LOCAL_CONTEXT_INDEX; ++arg)
            EXPECT_EQ(payload->args[arg], 0u);
    }
}

TEST(HbgPayloadContext, InitializesEveryOrdinarySlot) {
    AicoreSchedulerLayout layout{};
    ASSERT_TRUE(scheduler_plan_layout(1, 1, 0, &layout));
    scheduler_test::SchedulerStateBuffer storage(layout);
    check_payload_range(
        storage.base(), reinterpret_cast<uint64_t>(storage.base()), layout.dispatch_payloads_offset,
        SCHEDULER_WORKER_CAPACITY * SCHEDULER_PENDING_SLOT_COUNT
    );
}

TEST(HbgPayloadContext, InitializesEveryIndependentMixSlot) {
    AicoreSchedulerLayout layout{};
    ASSERT_TRUE(scheduler_plan_layout(1, 1, 1, &layout));
    scheduler_test::SchedulerStateBuffer storage(layout);
    check_payload_range(
        storage.base(), reinterpret_cast<uint64_t>(storage.base()), layout.mix_payloads_offset,
        SCHEDULER_MIX_SLOT_CAPACITY
    );
}

TEST(HbgPayloadContext, BindsDeviceAddressesInsteadOfHostMirrorAddresses) {
    AicoreSchedulerLayout layout{};
    ASSERT_TRUE(scheduler_plan_layout(1, 1, 1, &layout));
    scheduler_test::SchedulerStateBuffer storage(layout);
    constexpr uint64_t device_base = UINT64_C(0x400000000);
    ASSERT_NE(reinterpret_cast<uint64_t>(storage.base()), device_base);
    ASSERT_TRUE(scheduler_init_data_from_layout(storage.base(), layout, device_base));
    check_payload_range(
        storage.base(), device_base, layout.dispatch_payloads_offset,
        SCHEDULER_WORKER_CAPACITY * SCHEDULER_PENDING_SLOT_COUNT
    );
    check_payload_range(storage.base(), device_base, layout.mix_payloads_offset, SCHEDULER_MIX_SLOT_CAPACITY);
}

TEST(HbgPayloadContext, RebindClearsDirtyPayloadsAndOldDeviceAddresses) {
    AicoreSchedulerLayout layout{};
    ASSERT_TRUE(scheduler_plan_layout(2, 2, 2, &layout));
    scheduler_test::SchedulerStateBuffer storage(layout);
    ASSERT_TRUE(scheduler_init_data_from_layout(storage.base(), layout, UINT64_C(0x400000000)));
    std::memset(storage.base(), 0xa5, static_cast<size_t>(layout.total_size));
    constexpr uint64_t device_base = UINT64_C(0x800000000);
    ASSERT_TRUE(scheduler_init_data_from_layout(storage.base(), layout, device_base));
    check_payload_range(
        storage.base(), device_base, layout.dispatch_payloads_offset,
        SCHEDULER_WORKER_CAPACITY * SCHEDULER_PENDING_SLOT_COUNT
    );
    check_payload_range(storage.base(), device_base, layout.mix_payloads_offset, SCHEDULER_MIX_SLOT_CAPACITY);
}

TEST(HbgPayloadContext, RejectsInvalidDestinationBeforeChangingTheMirror) {
    AicoreSchedulerLayout layout{};
    ASSERT_TRUE(scheduler_plan_layout(1, 1, 1, &layout));
    scheduler_test::SchedulerStateBuffer storage(layout);
    auto *payload = scheduler_state_at<DispatchPayload>(storage.base(), layout.mix_payloads_offset);
    payload->function_bin_addr = 123;
    for (uint64_t device_base : {UINT64_C(0), UINT64_C(1), UINT64_C(64), UINT64_MAX - UINT64_C(127)}) {
        SCOPED_TRACE(device_base);
        EXPECT_FALSE(scheduler_init_data_from_layout(storage.base(), layout, device_base));
        EXPECT_EQ(payload->function_bin_addr, 123u);
    }
    EXPECT_FALSE(scheduler_init_data_from_layout(nullptr, layout, UINT64_C(0x400000000)));
    EXPECT_FALSE(
        scheduler_init_data_from_layout(static_cast<uint8_t *>(storage.base()) + 1, layout, UINT64_C(0x400000000))
    );
    EXPECT_EQ(payload->function_bin_addr, 123u);
}

}  // namespace
