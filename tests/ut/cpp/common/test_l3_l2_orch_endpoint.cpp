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

#include <array>
#include <cstdint>
#include <cstring>

#include <gtest/gtest.h>

#include "aicpu/l3_l2_orch_endpoint.h"
#include "common/l3_l2_orch_comm.h"

namespace {

struct alignas(64) SignalSlot {
    volatile uint64_t value;
    uint8_t padding[L3L2_ORCH_COMM_SIGNAL_BYTES - sizeof(uint64_t)];
};

struct RegionStorage {
    std::array<uint8_t, 128> payload{};
    SignalSlot l3_to_l2{};
    SignalSlot l2_to_l3{};
};

L3L2OrchRegionDesc make_desc(RegionStorage *storage) {
    return L3L2OrchRegionDesc{
        l3_l2_orch_comm_magic_version(),
        17,
        reinterpret_cast<uint64_t>(storage->payload.data()),
        storage->payload.size(),
        reinterpret_cast<uint64_t>(&storage->l3_to_l2),
        reinterpret_cast<uint64_t>(&storage->l2_to_l3),
    };
}

TEST(L3L2OrchEndpointTest, DecodesDescriptorScalarsAndReturnsPayloadViewWithoutCopying) {
    RegionStorage storage{};
    L3L2OrchRegionDesc desc = make_desc(&storage);
    std::array<uint64_t, L3L2_ORCH_REGION_DESC_SCALAR_COUNT> scalars{};
    ASSERT_TRUE(l3_l2_orch_comm_encode_desc(desc, scalars.data(), scalars.size()));

    L3L2OrchEndpoint endpoint(scalars.data(), scalars.size());

    ASSERT_EQ(endpoint.error().kind, L3L2EndpointErrorKind::NONE) << endpoint.error().message;
    L3L2OrchPayloadView view{};
    ASSERT_TRUE(endpoint.payload_read(8, 16, &view)) << endpoint.error().message;
    EXPECT_EQ(view.gm_addr, desc.payload_base + 8);
    EXPECT_EQ(view.nbytes, 16u);
}

TEST(L3L2OrchEndpointTest, PayloadWriteCopiesSmallMetadataIntoPayloadRange) {
    RegionStorage storage{};
    L3L2OrchEndpoint endpoint(make_desc(&storage));
    const uint32_t marker = 0xA5B6C7D8u;

    ASSERT_TRUE(endpoint.payload_write(12, &marker, sizeof(marker))) << endpoint.error().message;

    uint32_t observed = 0;
    std::memcpy(&observed, storage.payload.data() + 12, sizeof(observed));
    EXPECT_EQ(observed, marker);
}

TEST(L3L2OrchEndpointTest, PayloadReadViewSeesChangingHeaderAcrossRounds) {
    RegionStorage storage{};
    L3L2OrchEndpoint endpoint(make_desc(&storage));
    L3L2OrchPayloadView view{};
    ASSERT_TRUE(endpoint.payload_read(0, sizeof(uint32_t), &view)) << endpoint.error().message;
    ASSERT_NE(view.gm_addr, 0u);
    auto *header = reinterpret_cast<uint32_t *>(static_cast<uintptr_t>(view.gm_addr));

    storage.payload[0] = 0x11;
    EXPECT_EQ(*header & 0xFFu, 0x11u);

    storage.payload[0] = 0x22;
    EXPECT_EQ(*header & 0xFFu, 0x22u);
}

TEST(L3L2OrchEndpointTest, PayloadBoundsErrorCarriesStructuredMetadata) {
    RegionStorage storage{};
    L3L2OrchEndpoint endpoint(make_desc(&storage));

    L3L2OrchPayloadView view{0xCAFE, 0xBEEF};

    EXPECT_FALSE(endpoint.payload_read(120, 16, &view));
    EXPECT_EQ(view.gm_addr, 0u);
    EXPECT_EQ(view.nbytes, 0u);
    EXPECT_EQ(endpoint.error().kind, L3L2EndpointErrorKind::OUT_OF_BOUNDS);
    EXPECT_STREQ(endpoint.error().op, "payload_read");
    EXPECT_EQ(endpoint.error().region_id, 17u);
    EXPECT_EQ(endpoint.error().seq, 0u);
    EXPECT_NE(endpoint.error().message, nullptr);
}

TEST(L3L2OrchEndpointTest, NotifyPublishesOnL2ToL3AndWaitObservesL3ToL2) {
    RegionStorage storage{};
    L3L2OrchEndpoint endpoint(make_desc(&storage));
    storage.l3_to_l2.value = 3;

    EXPECT_TRUE(endpoint.wait(3, 1'000'000)) << endpoint.error().message;
    EXPECT_TRUE(endpoint.notify(5)) << endpoint.error().message;
    EXPECT_EQ(storage.l2_to_l3.value, 5u);
}

TEST(L3L2OrchEndpointTest, WaitFutureSequenceIsProtocolError) {
    RegionStorage storage{};
    L3L2OrchEndpoint endpoint(make_desc(&storage));
    storage.l3_to_l2.value = 9;

    EXPECT_FALSE(endpoint.wait(8, 1'000'000));

    EXPECT_EQ(endpoint.error().kind, L3L2EndpointErrorKind::SIGNAL_PROTOCOL);
    EXPECT_STREQ(endpoint.error().op, "wait");
    EXPECT_EQ(endpoint.error().region_id, 17u);
    EXPECT_EQ(endpoint.error().seq, 8u);
    EXPECT_EQ(endpoint.error().observed_signal, 9u);
}

TEST(L3L2OrchEndpointTest, WaitTimeoutCarriesStructuredMetadata) {
    RegionStorage storage{};
    L3L2OrchEndpoint endpoint(make_desc(&storage));

    EXPECT_FALSE(endpoint.wait(1, 1));

    EXPECT_EQ(endpoint.error().kind, L3L2EndpointErrorKind::SIGNAL_TIMEOUT);
    EXPECT_STREQ(endpoint.error().op, "wait");
    EXPECT_EQ(endpoint.error().region_id, 17u);
    EXPECT_EQ(endpoint.error().seq, 1u);
    EXPECT_EQ(endpoint.error().observed_signal, 0u);
}

TEST(L3L2OrchEndpointTest, NotifyRejectsNonMonotonicSequence) {
    RegionStorage storage{};
    L3L2OrchEndpoint endpoint(make_desc(&storage));
    ASSERT_TRUE(endpoint.notify(4)) << endpoint.error().message;

    EXPECT_FALSE(endpoint.notify(4));

    EXPECT_EQ(endpoint.error().kind, L3L2EndpointErrorKind::SIGNAL_PROTOCOL);
    EXPECT_STREQ(endpoint.error().op, "notify");
    EXPECT_EQ(endpoint.error().region_id, 17u);
    EXPECT_EQ(endpoint.error().seq, 4u);
}

TEST(L3L2OrchEndpointTest, RejectsBadDescriptorScalars) {
    std::array<uint64_t, L3L2_ORCH_REGION_DESC_SCALAR_COUNT> scalars{};

    L3L2OrchEndpoint endpoint(scalars.data(), scalars.size());

    EXPECT_EQ(endpoint.error().kind, L3L2EndpointErrorKind::BAD_DESCRIPTOR);
    EXPECT_STREQ(endpoint.error().op, "init");
    EXPECT_NE(endpoint.error().message, nullptr);
}

}  // namespace
