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
#include <utility>
#include <vector>

// Include the implementation to exercise the private executor, without a
// production test hook or an alternative teardown implementation.
// NOLINTNEXTLINE(bugprone-suspicious-include)
#include "aicpu/aicpu_executor.cpp"

namespace {
std::vector<std::pair<const void *, size_t>> invalidations;
}
namespace aicpu_cache_maintenance {
void invalidate_range_impl(const void *address, size_t bytes) { invalidations.emplace_back(address, bytes); }
void flush_range_impl(const void *, size_t) {}
}  // namespace aicpu_cache_maintenance

TEST(EntryCacheDeinit, FailedSnapshotFallsBackToFullThroughExecutor) {
    Runtime runtime;
    runtime.dev.entry_args_source_ = static_cast<uint32_t>(EntryArgsSource::LaunchEnvelope);
    runtime.dev.entry_tensor_count_ = 4;
    PlatformEntryArgs view{
        &runtime, LAUNCH_ENVELOPE_HEADER_BYTES, 4, 0, static_cast<uint32_t>(EntryArgsSource::LaunchEnvelope)
    };
    view.cache_maintenance = static_cast<uint32_t>(EntryCacheMaintenance::Live);
    platform_aicpu_affinity_set_thread_idx(0);
    set_platform_entry_args(0, view);
    AicpuExecutor executor;
    for (auto verdict : {DeviceRunVerdict::Ok, DeviceRunVerdict::Error, DeviceRunVerdict::None, DeviceRunVerdict::Ok}) {
        const bool success = verdict == DeviceRunVerdict::Ok;
        executor.terminal_publisher_.take(
            RunTerminalSelection{
                verdict, success ? 0 : -7, success ? DeviceRunCodeSource::None : DeviceRunCodeSource::ThreadRc
            },
            nullptr, 0
        );
        invalidations.clear();
        executor.deinit(&runtime);
        if (success) {
            ASSERT_EQ(invalidations.size(), 2u);
            EXPECT_EQ(invalidations[0].first, &runtime);
            EXPECT_EQ(invalidations[0].second, simpler::tmr::entry_tensor_offset + 4 * sizeof(Tensor));
            EXPECT_EQ(
                invalidations[1].first,
                reinterpret_cast<const char *>(&runtime) + offsetof(DeviceRuntimeLaunchDesc, workers)
            );
        } else {
            ASSERT_EQ(invalidations.size(), 1u);
            EXPECT_EQ(invalidations[0].first, &runtime);
            EXPECT_EQ(invalidations[0].second, sizeof(runtime.dev));
        }
        EXPECT_EQ(executor.terminal_publisher_.snapshot_is_ok(), success);
    }
    set_platform_entry_args(0, {});
    platform_aicpu_affinity_set_thread_idx(-1);
}

class SchedulerContextTestPeer {
public:
    static void dirty(SchedulerContext &context, ChipTaskSlotState &running, ChipTaskSlotState &pending) {
        for (auto &state : context.core_exec_states_) {
            state.reg_addr = 0x1234;
            state.running_slot_state = &running;
            state.pending_slot_state = &pending;
            state.running_reg_task_id = 17;
            state.pending_reg_task_id = 19;
        }
    }
    static void expect_entry_reset(const SchedulerContext &context) {
        for (const auto &state : context.core_exec_states_) {
            EXPECT_EQ(state.reg_addr, 0u);
            EXPECT_EQ(state.running_slot_state, nullptr);
            EXPECT_EQ(state.pending_slot_state, nullptr);
            EXPECT_EQ(state.running_reg_task_id, 0);
            EXPECT_EQ(state.pending_reg_task_id, 0);
        }
    }
};

TEST(EntryCacheDeinit, SchedulerEntryResetsPreviousRunStateBeforeValidation) {
    SchedulerContext context;
    ChipTaskSlotState running{};
    ChipTaskSlotState pending{};
    Runtime runtime;
    runtime.dev.gm_sm_ptr_ = nullptr;
    runtime.dev.callable_table_addr_ = 0;
    runtime.dev.callable_table_len_ = 0;
    for (int workers : {3, 0, RUNTIME_MAX_WORKER, 4, 3}) {
        SCOPED_TRACE(workers);
        SchedulerContextTestPeer::dirty(context, running, pending);
        context.deinit();
        runtime.dev.worker_count = workers;
        EXPECT_EQ(context.pre_handshake_init(&runtime, 2, 1, 0), workers > 0 && workers % 3 == 0 ? 0 : -1);
        SchedulerContextTestPeer::expect_entry_reset(context);
    }
}
