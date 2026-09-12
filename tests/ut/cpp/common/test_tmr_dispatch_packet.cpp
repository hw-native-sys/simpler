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

#include "host/tmr_dispatch_packet.h"

namespace {
using namespace simpler::tmr;

const PreparedInvocationView kCallable{3, 1, 1, 17};
const TmrExecutionBindingView kBinding{0x10000, 8};

TmrEncodingCandidate candidate() {
    ChipStorageTaskArgs args{};
    const uint32_t shape[] = {2, 4};
    args.add_tensor(
        make_tensor_external(reinterpret_cast<void *>(0x20000), shape, 2, DataType::FLOAT32, AddressSpace::DEVICE)
    );
    args.add_scalar(19);
    TmrEncodingCandidate result;
    TmrEncodingCache cache;
    EXPECT_EQ(encode_tmr_invocation(args, kCallable, kBinding, cache, &result), InvocationStatus::Ok);
    return result;
}

TEST(TmrDispatchPacket, WrapsOwnedInvocationForRealDispatchWithoutEnablingExecution) {
    auto invocation = candidate();
    const auto inner = invocation.packet();
    const std::vector<uint8_t> original(inner.data, inner.data + inner.size);
    KernelCallableDeviceResidency resident{17, 0x100000, 128, 3, 0};
    std::vector<uint8_t> packet;
    ASSERT_EQ(
        make_tmr_dispatch_packet(invocation, kCallable, kBinding, reinterpret_cast<uint64_t>(&resident), &packet),
        InvocationStatus::Ok
    );
    constexpr size_t prefix = offsetof(SimplerKernelDispatchArgs, invocation);
    ASSERT_EQ(packet.size(), prefix + inner.size);
    SimplerKernelDispatchArgs header{};
    std::memcpy(&header, packet.data(), sizeof(header));
    EXPECT_EQ(header.packet_bytes, packet.size());
    EXPECT_EQ(header.invocation.payload_bytes, packet.size() - sizeof(header));
    EXPECT_EQ(std::memcmp(packet.data() + prefix, original.data(), original.size()), 0);
    EXPECT_EQ(simpler_aicpu_kernel_exec(packet.data()), static_cast<int>(KernelDispatchStatus::UnsupportedPayload));
    // Replay reads the current descriptor rather than trusting the captured generation.
    ++resident.generation;
    EXPECT_EQ(simpler_aicpu_kernel_exec(packet.data()), static_cast<int>(KernelDispatchStatus::Stale));
    // The old header-first format is not a dispatch envelope.
    auto bare = original;
    EXPECT_EQ(simpler_aicpu_kernel_exec(bare.data()), static_cast<int>(KernelDispatchStatus::InvalidArgs));
    EXPECT_EQ(std::memcmp(inner.data, original.data(), original.size()), 0);
}

TEST(TmrDispatchPacket, RejectionPreservesOutputAndEncodingCache) {
    auto invocation = candidate();
    std::vector<uint8_t> packet{1, 2, 3};
    const auto original = packet;
    for (uint64_t address : {uint64_t{0}, uint64_t{1}, UINT64_MAX}) {
        EXPECT_EQ(
            make_tmr_dispatch_packet(invocation, kCallable, kBinding, address, &packet),
            InvocationStatus::InvalidBinding
        );
        EXPECT_EQ(packet, original);
    }
    auto stale = kCallable;
    ++stale.slot_generation;
    EXPECT_EQ(make_tmr_dispatch_packet(invocation, stale, kBinding, 0x10000, &packet), InvocationStatus::StaleCallable);
    auto wrong_binding = kBinding;
    ++wrong_binding.context_generation;
    EXPECT_EQ(
        make_tmr_dispatch_packet(invocation, kCallable, wrong_binding, 0x10000, &packet),
        InvocationStatus::InvalidBinding
    );
    EXPECT_EQ(packet, original);
    EXPECT_EQ(
        make_tmr_dispatch_packet(invocation, kCallable, kBinding, 0x10000, nullptr), InvocationStatus::InvalidArgument
    );
}

}  // namespace
