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
#include <type_traits>

#include <gtest/gtest.h>

#include "aicpu/device_time.h"
#include "aicpu/worker_chip_orch_endpoint.h"
#include "common/worker_chip_orch_comm.h"

namespace {

struct RegionStorage {
    alignas(64) std::array<uint8_t, 128> payload{};
    alignas(64) std::array<int32_t, 32> counters{};
};

WorkerChipOrchRegionDesc make_desc(RegionStorage *storage) {
    return WorkerChipOrchRegionDesc{
        worker_chip_orch_comm::magic_version(),
        17,
        reinterpret_cast<uint64_t>(storage->payload.data()),
        storage->payload.size(),
        reinterpret_cast<uint64_t>(storage->counters.data()),
        storage->counters.size() * sizeof(int32_t),
    };
}

TEST(WorkerChipOrchEndpointTest, DecodesDescriptorScalarsAndCounterRange) {
    RegionStorage storage{};
    WorkerChipOrchRegionDesc desc = make_desc(&storage);
    std::array<uint64_t, WORKER_CHIP_ORCH_REGION_DESC_SCALAR_COUNT> scalars{};
    ASSERT_TRUE(worker_chip_orch_comm::encode_desc(desc, scalars.data(), scalars.size()));

    WorkerChipOrchEndpoint endpoint(scalars.data(), scalars.size());

    ASSERT_EQ(endpoint.error().kind, WorkerChipEndpointErrorKind::NONE) << endpoint.error().message;
    EXPECT_EQ(endpoint.descriptor().counter_base, desc.counter_base);
    EXPECT_EQ(endpoint.descriptor().counter_bytes, desc.counter_bytes);
    EXPECT_EQ(endpoint.view().payload_span().base, desc.payload_base);
    EXPECT_EQ(endpoint.view().counter_span().base, desc.counter_base);
    EXPECT_FALSE(endpoint.view().failed());

    uint64_t counter_addr = 0;
    ASSERT_TRUE(endpoint.counter_addr(8, counter_addr)) << endpoint.error().message;
    EXPECT_EQ(counter_addr, desc.counter_base + 8);
}

TEST(WorkerChipOrchEndpointTest, ConvertsCounterTicksToNanoseconds) {
    EXPECT_EQ(sys_cnt_ticks_to_ns(50'000'000, 50'000'000), 1'000'000'000);
    EXPECT_EQ(sys_cnt_elapsed_ns(10, 25, 1'000'000'000), 15u);
}

TEST(WorkerChipOrchEndpointTest, ErrorOperationStringsAndMessageCopyAreStable) {
    static_assert(std::is_standard_layout<WorkerChipEndpointError>::value, "error must be POD-like");
    static_assert(std::is_trivially_copyable<WorkerChipEndpointError>::value, "error must be fixed-size");
    EXPECT_EQ(sizeof(WorkerChipEndpointError::message), 256u);

    EXPECT_STREQ(worker_chip_endpoint_op_to_string(WorkerChipEndpointOp::INIT), "init");
    EXPECT_STREQ(worker_chip_endpoint_op_to_string(WorkerChipEndpointOp::COUNTER_ADDR), "counter_addr");
    EXPECT_STREQ(worker_chip_endpoint_op_to_string(WorkerChipEndpointOp::PAYLOAD_READ), "payload_read");
    EXPECT_STREQ(worker_chip_endpoint_op_to_string(WorkerChipEndpointOp::PAYLOAD_WRITE), "payload_write");
    EXPECT_STREQ(worker_chip_endpoint_op_to_string(WorkerChipEndpointOp::SIGNAL_NOTIFY), "signal_notify");
    EXPECT_STREQ(worker_chip_endpoint_op_to_string(WorkerChipEndpointOp::SIGNAL_TEST), "signal_test");
    EXPECT_STREQ(worker_chip_endpoint_op_to_string(WorkerChipEndpointOp::SIGNAL_WAIT), "signal_wait");
    EXPECT_STREQ(worker_chip_endpoint_op_to_string(static_cast<WorkerChipEndpointOp>(99)), "unknown");
}

TEST(WorkerChipOrchEndpointTest, PayloadWriteCopiesSmallMetadataIntoPayloadRange) {
    RegionStorage storage{};
    WorkerChipOrchEndpoint endpoint(make_desc(&storage));
    const uint32_t marker = 0xA5B6C7D8u;

    ASSERT_TRUE(endpoint.payload_write(12, &marker, sizeof(marker))) << endpoint.error().message;

    uint32_t observed = 0;
    std::memcpy(&observed, storage.payload.data() + 12, sizeof(observed));
    EXPECT_EQ(observed, marker);
}

TEST(WorkerChipOrchEndpointTest, PayloadReadViewSeesChangingHeaderAcrossRounds) {
    RegionStorage storage{};
    WorkerChipOrchEndpoint endpoint(make_desc(&storage));
    WorkerChipOrchPayloadView view{};
    ASSERT_TRUE(endpoint.payload_read(0, sizeof(uint32_t), view)) << endpoint.error().message;
    ASSERT_NE(view.gm_addr, 0u);
    auto *header = reinterpret_cast<uint32_t *>(static_cast<uintptr_t>(view.gm_addr));

    storage.payload[0] = 0x11;
    EXPECT_EQ(*header & 0xFFu, 0x11u);

    storage.payload[0] = 0x22;
    EXPECT_EQ(*header & 0xFFu, 0x22u);
}

TEST(WorkerChipOrchEndpointTest, PayloadBoundsErrorCarriesStructuredMetadata) {
    RegionStorage storage{};
    WorkerChipOrchEndpoint endpoint(make_desc(&storage));

    WorkerChipOrchPayloadView view{0xCAFE, 0xBEEF};

    EXPECT_FALSE(endpoint.payload_read(120, 16, view));
    EXPECT_EQ(view.gm_addr, 0u);
    EXPECT_EQ(view.nbytes, 0u);
    EXPECT_EQ(endpoint.error().kind, WorkerChipEndpointErrorKind::OUT_OF_BOUNDS);
    EXPECT_EQ(endpoint.error().op, WorkerChipEndpointOp::PAYLOAD_READ);
    EXPECT_EQ(endpoint.error().region_id, 17u);
    EXPECT_EQ(endpoint.error().counter_addr, 0u);
    EXPECT_STRNE(endpoint.error().message, "");
}

TEST(WorkerChipOrchEndpointTest, CounterAddrRejectsBadOffsets) {
    RegionStorage storage{};
    WorkerChipOrchEndpoint endpoint(make_desc(&storage));
    uint64_t counter_addr = 0xCAFE;

    EXPECT_FALSE(endpoint.counter_addr(2, counter_addr));

    EXPECT_EQ(counter_addr, 0u);
    EXPECT_EQ(endpoint.error().kind, WorkerChipEndpointErrorKind::OUT_OF_BOUNDS);
    EXPECT_EQ(endpoint.error().op, WorkerChipEndpointOp::COUNTER_ADDR);
    EXPECT_EQ(endpoint.error().region_id, 17u);
    EXPECT_EQ(endpoint.error().counter_addr, make_desc(&storage).counter_base + 2);
}

TEST(WorkerChipOrchEndpointTest, SignalNotifySetAndAddUpdateCounters) {
    RegionStorage storage{};
    WorkerChipOrchEndpoint endpoint(make_desc(&storage));
    uint64_t counter_addr = 0;
    ASSERT_TRUE(endpoint.counter_addr(0, counter_addr)) << endpoint.error().message;

    EXPECT_TRUE(endpoint.signal_notify(counter_addr, 5, WorkerChipOrchNotifyOp::Set)) << endpoint.error().message;
    EXPECT_EQ(storage.counters[0], 5);

    EXPECT_TRUE(endpoint.signal_notify(counter_addr, -2, WorkerChipOrchNotifyOp::Add)) << endpoint.error().message;
    EXPECT_EQ(storage.counters[0], 3);
}

TEST(WorkerChipOrchEndpointTest, SignalTestCoversAllComparisonsAndMismatchIsNotError) {
    RegionStorage storage{};
    WorkerChipOrchEndpoint endpoint(make_desc(&storage));
    uint64_t counter_addr = 0;
    ASSERT_TRUE(endpoint.counter_addr(4, counter_addr)) << endpoint.error().message;
    storage.counters[1] = 7;

    WorkerChipOrchSignalTestResult result{};
    EXPECT_TRUE(endpoint.signal_test(counter_addr, 7, WorkerChipOrchWaitCmp::EQ, result)) << endpoint.error().message;
    EXPECT_TRUE(result.matched);
    EXPECT_EQ(result.observed, 7);

    EXPECT_TRUE(endpoint.signal_test(counter_addr, 8, WorkerChipOrchWaitCmp::NE, result)) << endpoint.error().message;
    EXPECT_TRUE(result.matched);
    EXPECT_TRUE(endpoint.signal_test(counter_addr, 6, WorkerChipOrchWaitCmp::GT, result)) << endpoint.error().message;
    EXPECT_TRUE(result.matched);
    EXPECT_TRUE(endpoint.signal_test(counter_addr, 7, WorkerChipOrchWaitCmp::GE, result)) << endpoint.error().message;
    EXPECT_TRUE(result.matched);
    EXPECT_TRUE(endpoint.signal_test(counter_addr, 8, WorkerChipOrchWaitCmp::LT, result)) << endpoint.error().message;
    EXPECT_TRUE(result.matched);
    EXPECT_TRUE(endpoint.signal_test(counter_addr, 7, WorkerChipOrchWaitCmp::LE, result)) << endpoint.error().message;
    EXPECT_TRUE(result.matched);

    EXPECT_TRUE(endpoint.signal_test(counter_addr, 8, WorkerChipOrchWaitCmp::EQ, result)) << endpoint.error().message;
    EXPECT_FALSE(result.matched);
    EXPECT_EQ(result.observed, 7);
    EXPECT_EQ(endpoint.error().kind, WorkerChipEndpointErrorKind::NONE);
}

TEST(WorkerChipOrchEndpointTest, SignalWaitTimeoutCarriesStructuredMetadata) {
    RegionStorage storage{};
    WorkerChipOrchEndpoint endpoint(make_desc(&storage));
    uint64_t counter_addr = 0;
    ASSERT_TRUE(endpoint.counter_addr(0, counter_addr)) << endpoint.error().message;
    int32_t observed = 0;

    EXPECT_FALSE(endpoint.signal_wait(counter_addr, 1, WorkerChipOrchWaitCmp::EQ, 1, observed));

    EXPECT_EQ(endpoint.error().kind, WorkerChipEndpointErrorKind::SIGNAL_TIMEOUT);
    EXPECT_EQ(endpoint.error().op, WorkerChipEndpointOp::SIGNAL_WAIT);
    EXPECT_EQ(endpoint.error().region_id, 17u);
    EXPECT_EQ(endpoint.error().counter_addr, counter_addr);
    EXPECT_EQ(endpoint.error().counter_operand, 1);
    EXPECT_EQ(endpoint.error().observed_counter, 0);
    EXPECT_EQ(observed, 0);
    EXPECT_FALSE(endpoint.view().failed());

    WorkerChipOrchPayloadView view{};
    EXPECT_FALSE(endpoint.payload_read(0, sizeof(uint32_t), view));
    EXPECT_EQ(endpoint.error().kind, WorkerChipEndpointErrorKind::SIGNAL_TIMEOUT);
}

TEST(WorkerChipOrchEndpointTest, SignalWaitDoesNotTreatGreaterObservedValueAsProtocolError) {
    RegionStorage storage{};
    WorkerChipOrchEndpoint endpoint(make_desc(&storage));
    uint64_t counter_addr = 0;
    ASSERT_TRUE(endpoint.counter_addr(0, counter_addr)) << endpoint.error().message;
    storage.counters[0] = 9;
    int32_t observed = 0;

    EXPECT_TRUE(endpoint.signal_wait(counter_addr, 8, WorkerChipOrchWaitCmp::GE, 1'000'000, observed))
        << endpoint.error().message;

    EXPECT_EQ(observed, 9);
    EXPECT_EQ(endpoint.error().kind, WorkerChipEndpointErrorKind::NONE);
}

TEST(WorkerChipOrchEndpointTest, RejectsBadDescriptorScalars) {
    std::array<uint64_t, WORKER_CHIP_ORCH_REGION_DESC_SCALAR_COUNT> scalars{};

    WorkerChipOrchEndpoint endpoint(scalars.data(), scalars.size());

    EXPECT_EQ(endpoint.error().kind, WorkerChipEndpointErrorKind::BAD_DESCRIPTOR);
    EXPECT_EQ(endpoint.error().op, WorkerChipEndpointOp::INIT);
    EXPECT_STREQ(endpoint.error().message, "invalid compatibility descriptor");
}

TEST(WorkerChipOrchEndpointTest, RejectsInvalidIndependentSpansWithFixedMessage) {
    RegionStorage storage{};
    WorkerChipOrchRegionDesc desc = make_desc(&storage);
    desc.payload_bytes = 0;

    WorkerChipOrchEndpoint endpoint(desc);

    EXPECT_EQ(endpoint.error().kind, WorkerChipEndpointErrorKind::BAD_DESCRIPTOR);
    EXPECT_STREQ(endpoint.error().message, "invalid independent local spans");
}

TEST(WorkerChipOrchEndpointTest, UnknownNotifyAndWaitStaySignalProtocolWithoutEnteringView) {
    RegionStorage storage{};
    WorkerChipOrchEndpoint endpoint(make_desc(&storage));
    uint64_t counter_addr = 0;
    ASSERT_TRUE(endpoint.counter_addr(0, counter_addr)) << endpoint.error().message;

    EXPECT_FALSE(endpoint.signal_notify(counter_addr, 1, static_cast<WorkerChipOrchNotifyOp>(99)));
    EXPECT_EQ(endpoint.error().kind, WorkerChipEndpointErrorKind::SIGNAL_PROTOCOL);
    EXPECT_FALSE(endpoint.view().failed());

    WorkerChipOrchEndpoint second(make_desc(&storage));
    WorkerChipOrchSignalTestResult result{};
    EXPECT_FALSE(second.signal_test(counter_addr, 1, static_cast<WorkerChipOrchWaitCmp>(99), result));
    EXPECT_EQ(second.error().kind, WorkerChipEndpointErrorKind::SIGNAL_PROTOCOL);
    EXPECT_FALSE(second.view().failed());
}

struct FailingCopyOps : LocalMemoryOps {
    bool copy_bytes(void *, const void *, size_t) const { return false; }
};

struct FailingInvalidateOps : LocalMemoryOps {
    bool invalidate(const void *, size_t) const { return false; }
};

struct FailingLoadOps : LocalMemoryOps {
    bool load_i32(uint64_t, int32_t &) const { return false; }
};

TEST(WorkerChipOrchEndpointTest, IssuedPayloadFailuresMapToLocalOperation) {
    RegionStorage storage{};
    WorkerChipOrchEndpointImpl<FailingInvalidateOps> reader(make_desc(&storage));
    WorkerChipOrchPayloadView view{};
    EXPECT_FALSE(reader.payload_read(0, 4, view));
    EXPECT_EQ(reader.error().kind, WorkerChipEndpointErrorKind::LOCAL_OPERATION);

    WorkerChipOrchEndpointImpl<FailingCopyOps> writer(make_desc(&storage));
    const uint32_t marker = 1;
    EXPECT_FALSE(writer.payload_write(0, &marker, sizeof(marker)));
    EXPECT_EQ(writer.error().kind, WorkerChipEndpointErrorKind::LOCAL_OPERATION);
}

TEST(WorkerChipOrchEndpointTest, IssuedCounterFailuresMapToLocalOperation) {
    RegionStorage storage{};
    uint64_t counter_addr = 0;
    {
        WorkerChipOrchEndpoint probe(make_desc(&storage));
        ASSERT_TRUE(probe.counter_addr(0, counter_addr)) << probe.error().message;
    }

    WorkerChipOrchEndpointImpl<FailingLoadOps> notify_ep(make_desc(&storage));
    EXPECT_FALSE(notify_ep.signal_notify(counter_addr, 1, WorkerChipOrchNotifyOp::Add));
    EXPECT_EQ(notify_ep.error().kind, WorkerChipEndpointErrorKind::LOCAL_OPERATION);

    WorkerChipOrchEndpointImpl<FailingLoadOps> test_ep(make_desc(&storage));
    WorkerChipOrchSignalTestResult result{};
    EXPECT_FALSE(test_ep.signal_test(counter_addr, 1, WorkerChipOrchWaitCmp::EQ, result));
    EXPECT_EQ(test_ep.error().kind, WorkerChipEndpointErrorKind::LOCAL_OPERATION);

    WorkerChipOrchEndpointImpl<FailingLoadOps> wait_ep(make_desc(&storage));
    int32_t observed = 0;
    EXPECT_FALSE(wait_ep.signal_wait(counter_addr, 1, WorkerChipOrchWaitCmp::EQ, 1, observed));
    EXPECT_EQ(wait_ep.error().kind, WorkerChipEndpointErrorKind::LOCAL_OPERATION);
}

TEST(WorkerChipOrchEndpointTest, EnumConversionCoversSetAddAndAllComparisons) {
    RegionNotifyOp notify{};
    ASSERT_TRUE(worker_chip_notify_op_to_region(WorkerChipOrchNotifyOp::Set, notify));
    EXPECT_EQ(notify, RegionNotifyOp::Set);
    ASSERT_TRUE(worker_chip_notify_op_to_region(WorkerChipOrchNotifyOp::Add, notify));
    EXPECT_EQ(notify, RegionNotifyOp::Add);
    EXPECT_FALSE(worker_chip_notify_op_to_region(static_cast<WorkerChipOrchNotifyOp>(3), notify));

    RegionWaitCmp cmp{};
    ASSERT_TRUE(worker_chip_wait_cmp_to_region(WorkerChipOrchWaitCmp::EQ, cmp));
    EXPECT_EQ(cmp, RegionWaitCmp::EQ);
    ASSERT_TRUE(worker_chip_wait_cmp_to_region(WorkerChipOrchWaitCmp::NE, cmp));
    EXPECT_EQ(cmp, RegionWaitCmp::NE);
    ASSERT_TRUE(worker_chip_wait_cmp_to_region(WorkerChipOrchWaitCmp::GT, cmp));
    EXPECT_EQ(cmp, RegionWaitCmp::GT);
    ASSERT_TRUE(worker_chip_wait_cmp_to_region(WorkerChipOrchWaitCmp::GE, cmp));
    EXPECT_EQ(cmp, RegionWaitCmp::GE);
    ASSERT_TRUE(worker_chip_wait_cmp_to_region(WorkerChipOrchWaitCmp::LT, cmp));
    EXPECT_EQ(cmp, RegionWaitCmp::LT);
    ASSERT_TRUE(worker_chip_wait_cmp_to_region(WorkerChipOrchWaitCmp::LE, cmp));
    EXPECT_EQ(cmp, RegionWaitCmp::LE);
    EXPECT_FALSE(worker_chip_wait_cmp_to_region(static_cast<WorkerChipOrchWaitCmp>(9), cmp));
    EXPECT_EQ(
        worker_chip_map_view_error_kind(RegionViewErrorKind::ISSUED_FAILURE),
        WorkerChipEndpointErrorKind::LOCAL_OPERATION
    );
    EXPECT_EQ(
        worker_chip_map_view_error_kind(RegionViewErrorKind::INVALID_ENUM), WorkerChipEndpointErrorKind::SIGNAL_PROTOCOL
    );
}

TEST(WorkerChipOrchEndpointTest, RejectsAbiMajor2AndOldContiguousMagic) {
    RegionStorage storage{};
    WorkerChipOrchRegionDesc desc = make_desc(&storage);
    desc.magic_version = worker_chip_orch_comm_pack_magic_version(WORKER_CHIP_ORCH_COMM_MAGIC, 2, 0);
    std::array<uint64_t, WORKER_CHIP_ORCH_REGION_DESC_SCALAR_COUNT> scalars{
        desc.magic_version, desc.region_id,    desc.payload_base,
        desc.payload_bytes, desc.counter_base, desc.counter_bytes,
    };

    WorkerChipOrchEndpoint endpoint(scalars.data(), scalars.size());

    EXPECT_EQ(endpoint.error().kind, WorkerChipEndpointErrorKind::BAD_DESCRIPTOR);
    EXPECT_EQ(endpoint.error().op, WorkerChipEndpointOp::INIT);
    EXPECT_TRUE(endpoint.view().failed());
    EXPECT_STREQ(endpoint.error().message, "invalid compatibility descriptor");
}

}  // namespace
