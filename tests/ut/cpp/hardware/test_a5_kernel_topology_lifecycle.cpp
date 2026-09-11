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

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <set>
#include <sstream>

#include <acl/acl.h>
#include "device_runner_base.h"

#define private public
#include "device_runner.h"
#undef private

namespace {

constexpr int kInjectedError = 1701;

struct FakeTopologyTransport {
    int alloc_error{0};
    int copy_fail_on{0};
    int launch_error{0};
    int sync_error{0};
    int free_error{0};
    int set_device_error{0};
    int alloc_calls{0};
    int copy_calls{0};
    int launch_calls{0};
    int sync_calls{0};
    int free_calls{0};
    int reset_calls{0};
    void *pending_result{nullptr};
    void *last_stream{nullptr};
    std::set<void *> live;
};

FakeTopologyTransport fake;

class TopologyRunner : public DeviceRunner {
public:
    void attach_for_test() {
        ASSERT_EQ(execution_mode_latch().latch(SIMPLER_MODE_KERNEL), 0);
        device_id_ = 0;
        KernelContextOps ops{};
        ops.get_current_device = [](void *, int *device) {
            *device = 0;
            return 0;
        };
        ops.create_hidden_stream = [](void *, void **stream) {
            *stream = &fake;
            return 0;
        };
        ops.destroy_hidden_stream = [](void *, void *) {
            return 0;
        };
        ops.create_event = [](void *, uint32_t, void **event) {
            *event = &fake;
            return 0;
        };
        ops.destroy_event = [](void *, void *) {
            return 0;
        };
        ASSERT_EQ(kernel_execution_state().initialize(0, ops), 0);
    }
};

class KernelTopologyLifecycle : public ::testing::Test {
protected:
    void SetUp() override {
        fake = {};
        runner.attach_for_test();
    }

    void TearDown() override {
        fake.sync_error = 0;
        fake.free_error = 0;
        fake.set_device_error = 0;
        EXPECT_EQ(runner.finalize(), 0);
        EXPECT_TRUE(fake.live.empty());
        EXPECT_EQ(fake.reset_calls, 0);
    }

    int query() { return runner.query_aicpu_device_occupancy(output, &stream_token); }

    void expect_retained(bool in_flight) {
        EXPECT_NE(runner.kernel_topology_result_, nullptr);
        EXPECT_EQ(runner.kernel_topology_stream_ != nullptr, in_flight);
        EXPECT_EQ(runner.committed_device_memory(), sizeof(AicpuTopologyQueryResult));
        EXPECT_EQ(fake.live.size(), 1U);
    }

    TopologyRunner runner;
    pto::a5::AicpuDeviceOccupancy output;
    int stream_token{0};
};

}  // namespace

// These definitions interpose the linked onboard library's transport only.
// DeviceRunner's query, allocator bookkeeping and finalize run unchanged.
extern "C" rtError_t rtMalloc(void **out, uint64_t bytes, rtMemType_t, uint16_t) {
    ++fake.alloc_calls;
    if (fake.alloc_error != 0) return fake.alloc_error;
    *out = std::malloc(bytes);
    if (*out == nullptr) return kInjectedError;
    fake.live.insert(*out);
    return 0;
}

extern "C" rtError_t rtFree(void *ptr) {
    ++fake.free_calls;
    if (fake.free_error != 0) return fake.free_error;
    EXPECT_EQ(fake.live.erase(ptr), 1U);
    std::free(ptr);
    return 0;
}

extern "C" rtError_t rtMemcpy(void *dst, uint64_t dst_bytes, const void *src, uint64_t bytes, rtMemcpyKind_t) {
    ++fake.copy_calls;
    if (fake.copy_calls == fake.copy_fail_on) return kInjectedError;
    if (bytes > dst_bytes) return kInjectedError;
    std::memcpy(dst, src, bytes);
    return 0;
}

extern "C" rtError_t rtSetDevice(int32_t) { return fake.set_device_error; }

extern "C" rtError_t rtDeviceReset(int32_t) {
    ++fake.reset_calls;
    return kInjectedError;
}

extern "C" aclError aclrtResetDevice(int32_t) {
    ++fake.reset_calls;
    return kInjectedError;
}

extern "C" aclError aclFinalize() {
    ++fake.reset_calls;
    return kInjectedError;
}

extern "C" aclError aclrtSynchronizeStreamWithTimeout(aclrtStream stream, int32_t) {
    ++fake.sync_calls;
    EXPECT_EQ(stream, fake.last_stream);
    if (fake.sync_error != 0) return fake.sync_error;
    if (fake.pending_result != nullptr) {
        AicpuTopologyQueryResult result{};
        result.occupy = 0x3c;
        result.occupy_rc = 0;
        std::memcpy(fake.pending_result, &result, sizeof(result));
        fake.pending_result = nullptr;
    }
    return 0;
}

int DeviceRunnerBase::launch_aicpu_payload(
    rtStream_t stream, void *args, size_t args_size, const char *kernel_name, int aicpu_num
) {
    ++fake.launch_calls;
    EXPECT_STREQ(kernel_name, "simpler_aicpu_query_topology");
    EXPECT_EQ(args_size, sizeof(AicpuTopologyQueryArgs));
    EXPECT_EQ(aicpu_num, 1);
    AicpuTopologyQueryArgs copied{};
    std::memcpy(&copied, args, sizeof(copied));
    fake.pending_result = reinterpret_cast<void *>(copied.result_addr);
    fake.last_stream = stream;
    return fake.launch_error;
}

namespace {

TEST_F(KernelTopologyLifecycle, SuccessReleasesScratchAndReusesOccupancyCache) {
    ASSERT_EQ(query(), 0);
    EXPECT_EQ(output.occupy, 0x3cU);
    EXPECT_EQ(runner.committed_device_memory(), 0U);
    EXPECT_EQ(runner.kernel_topology_result_, nullptr);
    ASSERT_EQ(query(), 0);
    EXPECT_EQ(fake.alloc_calls, 1);
    EXPECT_EQ(fake.launch_calls, 1);
    EXPECT_EQ(fake.sync_calls, 1);
    EXPECT_EQ(fake.free_calls, 1);
}

TEST_F(KernelTopologyLifecycle, AllocationFailureDoesNotPublishOrLaunch) {
    fake.alloc_error = kInjectedError;
    EXPECT_NE(query(), 0);
    EXPECT_EQ(fake.launch_calls, 0);
    EXPECT_EQ(fake.free_calls, 0);
    EXPECT_FALSE(runner.aicpu_device_occupancy_cached_);
    EXPECT_EQ(runner.committed_device_memory(), 0U);
}

TEST_F(KernelTopologyLifecycle, CopyFailureReleasesScratchWithoutLaunch) {
    fake.copy_fail_on = 1;
    EXPECT_EQ(query(), kInjectedError);
    EXPECT_EQ(fake.launch_calls, 0);
    EXPECT_EQ(fake.sync_calls, 0);
    EXPECT_EQ(fake.free_calls, 1);
    EXPECT_EQ(runner.committed_device_memory(), 0U);
}

TEST_F(KernelTopologyLifecycle, CopyRollbackFailureRetainsScratchUntilCloseRetry) {
    fake.copy_fail_on = 1;
    fake.free_error = kInjectedError;
    EXPECT_EQ(query(), kInjectedError);
    expect_retained(false);
    EXPECT_EQ(runner.kernel_execution_state().phase(), KernelContextPhase::Poisoned);
    EXPECT_EQ(runner.finalize(), kInjectedError);
    expect_retained(false);
    EXPECT_EQ(fake.sync_calls, 0);
    fake.free_error = 0;
    EXPECT_EQ(runner.finalize(), 0);
    EXPECT_EQ(runner.committed_device_memory(), 0U);
}

TEST_F(KernelTopologyLifecycle, LaunchFailureRetainsPossiblyInFlightScratch) {
    fake.launch_error = kInjectedError;
    EXPECT_EQ(query(), kInjectedError);
    expect_retained(true);
    EXPECT_EQ(fake.free_calls, 0);
    EXPECT_EQ(query(), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(fake.launch_calls, 1);
    EXPECT_EQ(runner.finalize(), 0);
    EXPECT_EQ(fake.sync_calls, 1);
    EXPECT_EQ(fake.free_calls, 1);
}

TEST_F(KernelTopologyLifecycle, SyncAndCloseFailuresKeepBufferUntilConfirmedCompletion) {
    fake.sync_error = kInjectedError;
    EXPECT_EQ(query(), kInjectedError);
    expect_retained(true);
    EXPECT_EQ(runner.finalize(), kInjectedError);
    expect_retained(true);
    EXPECT_EQ(fake.free_calls, 0);
    fake.sync_error = 0;
    fake.free_error = kInjectedError;
    EXPECT_EQ(runner.finalize(), kInjectedError);
    expect_retained(false);
    const int completed_sync_calls = fake.sync_calls;
    fake.free_error = 0;
    EXPECT_EQ(runner.finalize(), 0);
    EXPECT_EQ(fake.sync_calls, completed_sync_calls);
    EXPECT_EQ(runner.committed_device_memory(), 0U);
}

TEST_F(KernelTopologyLifecycle, CloseDeviceSelectionFailurePreservesInFlightOwnership) {
    fake.sync_error = kInjectedError;
    EXPECT_EQ(query(), kInjectedError);
    fake.set_device_error = kInjectedError;
    EXPECT_EQ(runner.finalize(), kInjectedError);
    expect_retained(true);
    EXPECT_EQ(fake.sync_calls, 1);
    EXPECT_EQ(fake.free_calls, 0);
}

TEST_F(KernelTopologyLifecycle, ResultCopyFailureDoesNotPublishCache) {
    fake.copy_fail_on = 2;
    EXPECT_EQ(query(), kInjectedError);
    EXPECT_FALSE(runner.aicpu_device_occupancy_cached_);
    EXPECT_EQ(fake.free_calls, 1);
    EXPECT_EQ(runner.committed_device_memory(), 0U);
}

TEST_F(KernelTopologyLifecycle, SuccessfulQueryCleanupFailureDoesNotPublishCache) {
    fake.free_error = kInjectedError;
    EXPECT_EQ(query(), kInjectedError);
    EXPECT_FALSE(runner.aicpu_device_occupancy_cached_);
    EXPECT_EQ(output.occupy, 0U);
    expect_retained(false);
}

}  // namespace
