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

struct RegionStorage {
    alignas(64) std::array<uint8_t, 128> payload{};
    alignas(64) std::array<int32_t, 32> counters{};
};

L3L2OrchRegionDesc make_desc(RegionStorage *storage) {
    return L3L2OrchRegionDesc{
        l3_l2_orch_comm_magic_version(),
        17,
        reinterpret_cast<uint64_t>(storage->payload.data()),
        storage->payload.size(),
        reinterpret_cast<uint64_t>(storage->counters.data()),
        storage->counters.size() * sizeof(int32_t),
    };
}

TEST(L3L2OrchEndpointTest, DecodesDescriptorScalarsAndCounterRange) {
    RegionStorage storage{};
    L3L2OrchRegionDesc desc = make_desc(&storage);
    std::array<uint64_t, L3L2_ORCH_REGION_DESC_SCALAR_COUNT> scalars{};
    ASSERT_TRUE(l3_l2_orch_comm_encode_desc(desc, scalars.data(), scalars.size()));

    L3L2OrchEndpoint endpoint(scalars.data(), scalars.size());

    ASSERT_EQ(endpoint.error().kind, L3L2EndpointErrorKind::NONE) << endpoint.error().message;
    EXPECT_EQ(endpoint.descriptor().counter_base, desc.counter_base);
    EXPECT_EQ(endpoint.descriptor().counter_bytes, desc.counter_bytes);

    uint64_t counter_addr = 0;
    ASSERT_TRUE(endpoint.counter_addr(8, &counter_addr)) << endpoint.error().message;
    EXPECT_EQ(counter_addr, desc.counter_base + 8);
}

TEST(L3L2OrchEndpointTest, ConvertsCounterTicksToNanoseconds) {
    EXPECT_EQ(l3_l2_orch_endpoint_ticks_to_ns(50'000'000, 50'000'000), 1'000'000'000);
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
    EXPECT_EQ(endpoint.error().counter_addr, 0u);
    EXPECT_NE(endpoint.error().message, nullptr);
}

TEST(L3L2OrchEndpointTest, CounterAddrRejectsBadOffsets) {
    RegionStorage storage{};
    L3L2OrchEndpoint endpoint(make_desc(&storage));
    uint64_t counter_addr = 0xCAFE;

    EXPECT_FALSE(endpoint.counter_addr(2, &counter_addr));

    EXPECT_EQ(counter_addr, 0u);
    EXPECT_EQ(endpoint.error().kind, L3L2EndpointErrorKind::OUT_OF_BOUNDS);
    EXPECT_STREQ(endpoint.error().op, "counter_addr");
    EXPECT_EQ(endpoint.error().region_id, 17u);
    EXPECT_EQ(endpoint.error().counter_addr, make_desc(&storage).counter_base + 2);
}

TEST(L3L2OrchEndpointTest, SignalNotifySetAndAddUpdateCounters) {
    RegionStorage storage{};
    L3L2OrchEndpoint endpoint(make_desc(&storage));
    uint64_t counter_addr = 0;
    ASSERT_TRUE(endpoint.counter_addr(0, &counter_addr)) << endpoint.error().message;

    EXPECT_TRUE(endpoint.signal_notify(counter_addr, 5, L3L2OrchNotifyOp::Set)) << endpoint.error().message;
    EXPECT_EQ(storage.counters[0], 5);

    EXPECT_TRUE(endpoint.signal_notify(counter_addr, -2, L3L2OrchNotifyOp::Add)) << endpoint.error().message;
    EXPECT_EQ(storage.counters[0], 3);
}

TEST(L3L2OrchEndpointTest, SignalTestCoversAllComparisonsAndMismatchIsNotError) {
    RegionStorage storage{};
    L3L2OrchEndpoint endpoint(make_desc(&storage));
    uint64_t counter_addr = 0;
    ASSERT_TRUE(endpoint.counter_addr(4, &counter_addr)) << endpoint.error().message;
    storage.counters[1] = 7;

    L3L2OrchSignalTestResult result{};
    EXPECT_TRUE(endpoint.signal_test(counter_addr, 7, L3L2OrchWaitCmp::EQ, &result)) << endpoint.error().message;
    EXPECT_TRUE(result.matched);
    EXPECT_EQ(result.observed, 7);

    EXPECT_TRUE(endpoint.signal_test(counter_addr, 8, L3L2OrchWaitCmp::NE, &result)) << endpoint.error().message;
    EXPECT_TRUE(result.matched);
    EXPECT_TRUE(endpoint.signal_test(counter_addr, 6, L3L2OrchWaitCmp::GT, &result)) << endpoint.error().message;
    EXPECT_TRUE(result.matched);
    EXPECT_TRUE(endpoint.signal_test(counter_addr, 7, L3L2OrchWaitCmp::GE, &result)) << endpoint.error().message;
    EXPECT_TRUE(result.matched);
    EXPECT_TRUE(endpoint.signal_test(counter_addr, 8, L3L2OrchWaitCmp::LT, &result)) << endpoint.error().message;
    EXPECT_TRUE(result.matched);
    EXPECT_TRUE(endpoint.signal_test(counter_addr, 7, L3L2OrchWaitCmp::LE, &result)) << endpoint.error().message;
    EXPECT_TRUE(result.matched);

    EXPECT_TRUE(endpoint.signal_test(counter_addr, 8, L3L2OrchWaitCmp::EQ, &result)) << endpoint.error().message;
    EXPECT_FALSE(result.matched);
    EXPECT_EQ(result.observed, 7);
    EXPECT_EQ(endpoint.error().kind, L3L2EndpointErrorKind::NONE);
}

TEST(L3L2OrchEndpointTest, SignalWaitTimeoutCarriesStructuredMetadata) {
    RegionStorage storage{};
    L3L2OrchEndpoint endpoint(make_desc(&storage));
    uint64_t counter_addr = 0;
    ASSERT_TRUE(endpoint.counter_addr(0, &counter_addr)) << endpoint.error().message;
    int32_t observed = 0;

    EXPECT_FALSE(endpoint.signal_wait(counter_addr, 1, L3L2OrchWaitCmp::EQ, 1, &observed));

    EXPECT_EQ(endpoint.error().kind, L3L2EndpointErrorKind::SIGNAL_TIMEOUT);
    EXPECT_STREQ(endpoint.error().op, "signal_wait");
    EXPECT_EQ(endpoint.error().region_id, 17u);
    EXPECT_EQ(endpoint.error().counter_addr, counter_addr);
    EXPECT_EQ(endpoint.error().counter_operand, 1);
    EXPECT_EQ(endpoint.error().observed_counter, 0);
    EXPECT_EQ(observed, 0);
}

TEST(L3L2OrchEndpointTest, SignalWaitDoesNotTreatGreaterObservedValueAsProtocolError) {
    RegionStorage storage{};
    L3L2OrchEndpoint endpoint(make_desc(&storage));
    uint64_t counter_addr = 0;
    ASSERT_TRUE(endpoint.counter_addr(0, &counter_addr)) << endpoint.error().message;
    storage.counters[0] = 9;
    int32_t observed = 0;

    EXPECT_TRUE(endpoint.signal_wait(counter_addr, 8, L3L2OrchWaitCmp::GE, 1'000'000, &observed))
        << endpoint.error().message;

    EXPECT_EQ(observed, 9);
    EXPECT_EQ(endpoint.error().kind, L3L2EndpointErrorKind::NONE);
}

TEST(L3L2OrchEndpointTest, RejectsBadDescriptorScalars) {
    std::array<uint64_t, L3L2_ORCH_REGION_DESC_SCALAR_COUNT> scalars{};

    L3L2OrchEndpoint endpoint(scalars.data(), scalars.size());

    EXPECT_EQ(endpoint.error().kind, L3L2EndpointErrorKind::BAD_DESCRIPTOR);
    EXPECT_STREQ(endpoint.error().op, "init");
    EXPECT_NE(endpoint.error().message, nullptr);
}

}  // namespace
