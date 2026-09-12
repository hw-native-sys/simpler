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
#include <limits>

#include "kernel_dispatch_args.h"
#include "aicpu/kernel_invocation_consumer.h"

namespace {
int consumed;
int invalidations;
const void *invalidated_address;
KernelCallableDeviceResidency published;
bool publish_on_invalidate;
int consumer_result;
const void *seen_payload;
uint64_t seen_generation;
struct Packet {
    SimplerKernelDispatchArgs args;
    uint64_t payload;
};
class KernelDispatch : public testing::Test {
protected:
    KernelCallableDeviceResidency resident{17, 0x100000, 128, 3, 0};
    Packet packet{};
    void SetUp() override {
        consumed = invalidations = 0;
        invalidated_address = seen_payload = nullptr;
        publish_on_invalidate = false;
        consumer_result = 0;
        seen_generation = 0;
        packet.args.packet_bytes = sizeof(packet);
        packet.args.residency_address = reinterpret_cast<uint64_t>(&resident);
        packet.args.invocation.mode = SIMPLER_MODE_KERNEL;
        packet.args.invocation.callable_id = 3;
        packet.args.invocation.generation = 17;
        packet.args.invocation.payload_bytes = sizeof(packet.payload);
        packet.payload = 42;
    }
    int run() { return simpler_aicpu_kernel_exec(&packet); }
};
}  // namespace

// Link substitutes for the production consumer and platform cache primitive.
// The exported entry itself is compiled from the production .cpp unchanged.
namespace aicpu_cache_maintenance {
void invalidate_range_impl(const void *address, size_t size) {
    EXPECT_EQ(size, sizeof(KernelCallableDeviceResidency));
    ++invalidations;
    invalidated_address = address;
    if (publish_on_invalidate) std::memcpy(const_cast<void *>(address), &published, sizeof(published));
}
}  // namespace aicpu_cache_maintenance
int consume_kernel_invocation(
    const SimplerKernelInvocationHeader &invocation, const KernelCallableDeviceResidency &resident, const void *payload,
    size_t bytes
) {
    ++consumed;
    EXPECT_GE(invalidations, consumed);
    EXPECT_EQ(bytes, sizeof(uint64_t));
    EXPECT_EQ(*static_cast<const uint64_t *>(payload), 42);
    EXPECT_EQ(invocation.generation, resident.generation);
    seen_payload = payload;
    seen_generation = invocation.generation;
    return consumer_result;
}

TEST_F(KernelDispatch, ValidPacketReachesConsumerAndPropagatesItsResult) {
    consumer_result = -83;
    EXPECT_EQ(run(), -83);
    EXPECT_EQ(consumed, 1);
    EXPECT_EQ(seen_payload, &packet.payload);
    EXPECT_EQ(seen_generation, 17);
    EXPECT_EQ(invalidated_address, &resident);
}
TEST_F(KernelDispatch, RejectsStaleBeforePayloadOrCodeAccess) {
    packet.args.invocation.generation = 16;
    resident.device_address = 1;
    EXPECT_EQ(run(), static_cast<int>(KernelDispatchStatus::Stale));
    EXPECT_EQ(consumed, 0);
    EXPECT_EQ(invalidations, 1);
}
TEST_F(KernelDispatch, ReplayReadsUpdatedSlotBeforeComparison) {
    EXPECT_EQ(run(), 0);
    published = resident;
    published.generation = 18;
    publish_on_invalidate = true;
    EXPECT_EQ(run(), static_cast<int>(KernelDispatchStatus::Stale));
    EXPECT_EQ(packet.args.invocation.generation, 17);
    EXPECT_EQ(invalidations, 2);
    EXPECT_EQ(consumed, 1);
    packet.args.invocation.generation = 18;
    EXPECT_EQ(run(), 0);
    EXPECT_EQ(consumed, 2);
}
TEST_F(KernelDispatch, WrongSlotAndEmptySlotNeverConsume) {
    resident.callable_id = 4;
    EXPECT_EQ(run(), static_cast<int>(KernelDispatchStatus::NotResident));
    resident.callable_id = 3;
    resident.generation = 0;
    EXPECT_EQ(run(), static_cast<int>(KernelDispatchStatus::NotResident));
    resident.generation = 17;
    resident.device_address = 0;
    EXPECT_EQ(run(), static_cast<int>(KernelDispatchStatus::NotResident));
    EXPECT_EQ(consumed, 0);
}
TEST_F(KernelDispatch, InvalidIdentityNeverReadsDescriptor) {
    packet.args.residency_address = 8;
    for (int id : {-1, 64, 1000}) {
        packet.args.invocation.callable_id = id;
        EXPECT_EQ(run(), static_cast<int>(KernelDispatchStatus::InvalidArgs));
    }
    packet.args.invocation.callable_id = 3;
    packet.args.invocation.generation = 0;
    EXPECT_EQ(run(), static_cast<int>(KernelDispatchStatus::InvalidArgs));
    packet.args.invocation.generation = 17;
    packet.args.invocation.mode = SIMPLER_MODE_PROGRAM;
    EXPECT_EQ(run(), static_cast<int>(KernelDispatchStatus::InvalidArgs));
    EXPECT_EQ(invalidations, 0);
    EXPECT_EQ(consumed, 0);
}
TEST_F(KernelDispatch, MalformedEnvelopeNeverReadsDescriptor) {
    EXPECT_EQ(simpler_aicpu_kernel_exec(nullptr), static_cast<int>(KernelDispatchStatus::InvalidArgs));
    EXPECT_EQ(
        simpler_aicpu_kernel_exec(reinterpret_cast<char *>(&packet) + 1),
        static_cast<int>(KernelDispatchStatus::InvalidArgs)
    );
    packet.args.packet_bytes = sizeof(SimplerKernelDispatchArgs) - 1;
    EXPECT_EQ(run(), static_cast<int>(KernelDispatchStatus::InvalidArgs));
    packet.args.packet_bytes = sizeof(packet);
    packet.args.invocation.payload_bytes++;
    EXPECT_EQ(run(), static_cast<int>(KernelDispatchStatus::InvalidArgs));
    packet.args.invocation.payload_bytes--;
    packet.args.invocation.host_copy_tensor_count = 1;
    EXPECT_EQ(run(), static_cast<int>(KernelDispatchStatus::InvalidArgs));
    packet.args.invocation.host_copy_tensor_count = 0;
    packet.args.residency_address = 0;
    EXPECT_EQ(run(), static_cast<int>(KernelDispatchStatus::InvalidArgs));
    packet.args.residency_address = std::numeric_limits<uint64_t>::max() - 7;
    EXPECT_EQ(run(), static_cast<int>(KernelDispatchStatus::InvalidArgs));
    EXPECT_EQ(invalidations, 0);
    EXPECT_EQ(consumed, 0);
}

TEST_F(KernelDispatch, CountsAreRejectedBeforeReadingResidency) {
    packet.args.residency_address = 8;
    packet.args.invocation.tensor_count = -1;
    EXPECT_EQ(run(), static_cast<int>(KernelDispatchStatus::InvalidArgs));
    packet.args.invocation.tensor_count = 257;
    EXPECT_EQ(run(), static_cast<int>(KernelDispatchStatus::InvalidArgs));
    packet.args.invocation.tensor_count = 0;
    packet.args.invocation.scalar_count = -1;
    EXPECT_EQ(run(), static_cast<int>(KernelDispatchStatus::InvalidArgs));
    packet.args.invocation.scalar_count = 129;
    EXPECT_EQ(run(), static_cast<int>(KernelDispatchStatus::InvalidArgs));
    packet.args.invocation.scalar_count = 1;
    packet.args.invocation.tensor_count = 256;
    EXPECT_EQ(run(), static_cast<int>(KernelDispatchStatus::InvalidArgs));
    EXPECT_EQ(invalidations, 0);
    EXPECT_EQ(consumed, 0);
}
TEST_F(KernelDispatch, RepeatedValidInvocationDoesNotChangeSnapshot) {
    EXPECT_EQ(run(), 0);
    EXPECT_EQ(run(), 0);
    EXPECT_EQ(packet.args.invocation.generation, 17);
    EXPECT_EQ(consumed, 2);
    EXPECT_EQ(invalidations, 2);
}
