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

#include "kernel_binder_test_support.h"
#include "kernel_launch_native.h"
#include "aicpu_loader/host/kernel_graph_launch.h"
#include <acl/error_codes/rt_error_codes.h>

#include <array>
#include <cstring>

namespace {
using namespace kernel_binder_test;
struct NativeFake {
    Fixture fixture;
    std::array<uint64_t, 10> packet{};
    std::array<uint32_t, 4> handshake{};
    KernelInvocationPlaceholder descriptor{64, 72};
    aclrtPlaceHolderInfo placeholder{64, 72};
    KernelClearRegion clear{handshake.data(), sizeof(handshake)};
    KernelNativeInvocation native;
    std::vector<std::vector<uint8_t>> copies;
    std::vector<int> memset_values;
    std::vector<int> host_args_errors;
    int recovery_syncs{0};
    int recovery_status{0};
    void initialize() {
        fixture.initialize();
        fixture.binding.packet = reinterpret_cast<const uint8_t *>(packet.data());
        fixture.binding.packet_bytes = sizeof(packet);
        fixture.binding.placeholders = &descriptor;
        fixture.binding.placeholder_count = 1;
        native.aicore = ptr(200);
        native.aicpu = ptr(201);
        native.aicore_blocks = 24;
        native.aicpu_blocks = 6;
        native.aicore_args = handshake.data();
        native.aicore_args_bytes = sizeof(handshake);
        native.aicpu_args = packet.data();
        native.aicpu_args_bytes = sizeof(packet);
        native.placeholders = &placeholder;
        native.placeholder_count = 1;
        native.clear_regions = &clear;
        native.clear_region_count = 1;
        native.cancel = {handshake.data(), sizeof(handshake)};
    }
    KernelLaunchResult launch(void *caller = ptr(100)) {
        return launch_bound_kernel_native(fixture.binding, native, caller, fixture.fake.gate());
    }
};
NativeFake *active = nullptr;
}  // namespace

extern "C" aclError aclrtQueryEventStatus(aclrtEvent event, aclrtEventRecordedStatus *status) {
    EXPECT_EQ(event, ptr(7));
    auto &f = active->fixture.fake;
    ++f.queries;
    *status = f.complete ? ACL_EVENT_RECORDED_STATUS_COMPLETE : ACL_EVENT_RECORDED_STATUS_NOT_READY;
    return f.query_error;
}
extern "C" aclError aclrtStreamWaitEvent(aclrtStream stream, aclrtEvent event) {
    // Start and AicoreDone are consumed by aicpu, AicoreStart by aicore, and
    // only AicpuDone reaches the caller: caller and aicore never share an edge.
    EXPECT_EQ(stream, event == ptr(4) ? ptr(2) : event == ptr(6) ? ptr(100) : ptr(1));
    const auto ops = active->fixture.fake.ops();
    return ops.wait_event(ops.context, stream, event);
}
extern "C" aclError aclrtRecordEvent(aclrtEvent event, aclrtStream stream) {
    EXPECT_EQ(stream, event == ptr(5) ? ptr(2) : (event == ptr(4) || event == ptr(6)) ? ptr(1) : ptr(100));
    const auto ops = active->fixture.fake.ops();
    return ops.record_event(ops.context, event, stream);
}
extern "C" aclError aclrtMemsetAsync(void *address, size_t maximum, int32_t value, size_t count, aclrtStream stream) {
    EXPECT_EQ(address, active->handshake.data());
    EXPECT_EQ(maximum, sizeof(active->handshake));
    EXPECT_EQ(count, maximum);
    // The handshake clear belongs to the AICPU branch; only the cancel that
    // compensation issues runs on the caller's stream.
    EXPECT_EQ(stream, value == 0xff ? ptr(100) : ptr(1));
    active->memset_values.push_back(value);
    return active->fixture.fake.append(value == 0xff ? Cancel : Step::Clear);
}
extern "C" aclError
aclrtLaunchKernel(aclrtFuncHandle function, uint32_t blocks, const void *args, size_t bytes, aclrtStream stream) {
    EXPECT_EQ(function, ptr(200));
    EXPECT_EQ(blocks, 24u);
    EXPECT_EQ(args, active->handshake.data());
    EXPECT_EQ(bytes, sizeof(active->handshake));
    EXPECT_EQ(stream, ptr(2));
    return active->fixture.fake.append(Step::AicoreLaunch);
}
extern "C" aclError aclrtLaunchKernelWithHostArgs(
    aclrtFuncHandle function, uint32_t blocks, aclrtStream stream, aclrtLaunchKernelCfg *, void *args, size_t bytes,
    aclrtPlaceHolderInfo *placeholders, size_t count
) {
    EXPECT_EQ(function, ptr(201));
    EXPECT_EQ(blocks, 6u);
    EXPECT_EQ(stream, ptr(1));
    EXPECT_EQ(count, 1u);
    auto *first = static_cast<uint8_t *>(args);
    uint64_t placeholder_value = 1;
    std::memcpy(&placeholder_value, first + placeholders[0].addrOffset, sizeof(placeholder_value));
    EXPECT_EQ(placeholder_value, 0u);
    active->copies.emplace_back(first, first + bytes);
    auto &copy = active->copies.back();
    const uint64_t address = reinterpret_cast<uintptr_t>(copy.data()) + placeholders[0].dataOffset;
    std::memcpy(copy.data() + placeholders[0].addrOffset, &address, sizeof(address));
    std::memcpy(first + placeholders[0].addrOffset, &address, sizeof(address));
    if (!active->host_args_errors.empty()) {
        const int error = active->host_args_errors.front();
        active->host_args_errors.erase(active->host_args_errors.begin());
        return error;
    }
    return active->fixture.fake.append(Step::AicpuLaunch);
}

extern "C" aclError aclrtSynchronizeEvent(aclrtEvent event) {
    // This event precedes the current AICore launch; its done event is ptr(5).
    EXPECT_EQ(event, ptr(4));
    ++active->recovery_syncs;
    return active->recovery_status;
}

TEST(HbgKernelHostArgs, MemoryPressureRetriesOnceAfterThePrelaunchEvent) {
    for (const auto errors :
         {std::vector<int>{ACL_ERROR_RT_MEMORY_ALLOCATION, 0},
          std::vector<int>{ACL_ERROR_RT_MEMORY_ALLOCATION, ACL_ERROR_RT_MEMORY_ALLOCATION}, std::vector<int>{12345},
          std::vector<int>{0}}) {
        NativeFake f;
        active = &f;
        f.initialize();
        f.host_args_errors = errors;
        hbg::GraphHostArgs args;
        args.storage.assign(10, 0);
        args.bytes = 80;
        args.address_offset = 64;
        args.data_offset = 72;
        EXPECT_EQ(hbg::launch_graph_host_args(args, ptr(201), 6, ptr(1), nullptr, ptr(4)), errors.back());
        EXPECT_EQ(f.copies.size(), errors.size());
        EXPECT_EQ(f.recovery_syncs, errors.size() == 2 ? 1 : 0);
    }
}

TEST(HbgKernelHostArgs, FailedRecoverySynchronizationDoesNotRetry) {
    NativeFake f;
    active = &f;
    f.initialize();
    f.host_args_errors = {ACL_ERROR_RT_MEMORY_ALLOCATION};
    f.recovery_status = 12346;
    hbg::GraphHostArgs args;
    args.storage.assign(10, 0);
    args.bytes = 80;
    args.address_offset = 64;
    args.data_offset = 72;
    EXPECT_EQ(hbg::launch_graph_host_args(args, ptr(201), 6, ptr(1), nullptr, ptr(4)), 12346);
    EXPECT_EQ(f.copies.size(), 1u);
    EXPECT_EQ(f.recovery_syncs, 1);
}

TEST(KernelNativeBinder, RoutesThreeStreamsAndCopiesIndependentHostArgs) {
    NativeFake f;
    active = &f;
    f.initialize();
    f.packet[9] = 17;
    ASSERT_EQ(f.launch().status, 0);
    EXPECT_EQ(f.fixture.fake.trace, success);
    EXPECT_EQ(f.packet[8], 0u);
    f.packet[9] = 29;
    ASSERT_EQ(f.launch().status, 0);
    ASSERT_EQ(f.copies.size(), 2u);
    uint64_t payload = 0;
    std::memcpy(&payload, f.copies[0].data() + 72, 8);
    EXPECT_EQ(payload, 17u);
    std::memcpy(&payload, f.copies[1].data() + 72, 8);
    EXPECT_EQ(payload, 29u);
    EXPECT_EQ(f.memset_values, (std::vector<int>{0, 0}));
    EXPECT_EQ(f.fixture.fake.queries, 0);
    f.fixture.fake.complete = false;
    EXPECT_NE(f.launch(ptr(101)).status, 0);
    EXPECT_EQ(f.fixture.fake.queries, 1);
}

TEST(KernelNativeBinder, AicpuFailureCancelsWithSingleAllOnesFillAndRestoresPlaceholder) {
    NativeFake f;
    active = &f;
    f.initialize();
    f.fixture.fake.fail_at = 8;
    const auto result = f.launch();
    EXPECT_EQ(result.status, -1708);
    EXPECT_EQ(result.cleanup_status, 0);
    EXPECT_TRUE(result.tail_recorded);
    EXPECT_EQ(f.packet[8], 0u);
    EXPECT_EQ(f.memset_values, (std::vector<int>{0, 0xff}));
    EXPECT_TRUE(f.fixture.fake.poisoned);
}

TEST(KernelNativeBinder, InvalidPreparedArgumentsRejectBeforeAnyEnqueue) {
    for (int fault = 0; fault < 10; ++fault) {
        SCOPED_TRACE(fault);
        NativeFake f;
        active = &f;
        f.initialize();
        switch (fault) {
        case 0:
            f.native.aicore = nullptr;
            break;
        case 1:
            f.native.aicpu_args_bytes -= 8;
            break;
        case 2:
            f.native.cancel.bytes -= 1;
            break;
        case 3:
            f.native.cancel.address = ptr(4096);
            break;
        case 4:
            f.placeholder.addrOffset = 0;
            break;
        case 5:
            f.placeholder.dataOffset = sizeof(f.packet);
            break;
        case 6:
            f.packet[8] = 1;
            break;
        case 7:
            f.native.placeholder_count = 0;
            break;
        case 8:
            f.fixture.fake.validation_error = -99;
            break;
        case 9:
            f.native.clear_regions = nullptr;
            break;
        }
        EXPECT_NE(f.launch().status, 0);
        EXPECT_TRUE(f.fixture.fake.trace.empty());
        EXPECT_TRUE(f.copies.empty());
        EXPECT_FALSE(f.fixture.fake.poisoned);
        EXPECT_EQ(f.fixture.fake.acquisitions, f.fixture.fake.finishes);
    }
}
