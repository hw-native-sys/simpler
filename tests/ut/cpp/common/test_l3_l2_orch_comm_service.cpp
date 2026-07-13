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

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "common/l3_l2_orch_comm.h"
#include "host/l3_l2_orch_comm_service.h"

namespace {

class FakeBackend : public L3L2OrchCommBackend {
public:
    ~FakeBackend() override {
        for (void *p : live_) {
            std::free(p);
        }
    }

    void *l3_l2_allocate_region_bytes(uint64_t bytes) override {
        void *p = nullptr;
        if (posix_memalign(&p, 64, static_cast<size_t>(bytes)) != 0) {
            return nullptr;
        }
        std::memset(p, 0, static_cast<size_t>(bytes));
        live_.push_back(p);
        return p;
    }

    void l3_l2_free_region_bytes(void *ptr) override {
        for (auto it = live_.begin(); it != live_.end(); ++it) {
            if (*it == ptr) {
                std::free(ptr);
                live_.erase(it);
                return;
            }
        }
    }

    int l3_l2_copy_to_device(void *dev_ptr, const void *host_ptr, uint64_t bytes) override {
        if (fail_copy_) {
            return -1;
        }
        std::memcpy(dev_ptr, host_ptr, static_cast<size_t>(bytes));
        return 0;
    }

    int l3_l2_copy_from_device(void *host_ptr, const void *dev_ptr, uint64_t bytes) override {
        if (fail_copy_) {
            return -1;
        }
        std::memcpy(host_ptr, dev_ptr, static_cast<size_t>(bytes));
        return 0;
    }

    std::thread l3_l2_create_service_thread(std::function<void()> fn) override { return std::thread(std::move(fn)); }

    bool fail_copy_{false};

private:
    std::vector<void *> live_;
};

struct ServiceFixture : public ::testing::Test {
    FakeBackend backend;
    L3L2OrchCommControlBlock control{};
    L3L2OrchCommService service;
    L3L2OrchCommClient client;

    void SetUp() override {
        ASSERT_EQ(service.start(&backend, &control, sizeof(control)), 0);
        client.attach(&control, sizeof(control));
    }

    void TearDown() override { service.stop(); }

    L3L2OrchRegionDesc alloc_region(uint64_t payload_bytes = 128, uint64_t counter_bytes = 128) {
        L3L2OrchCommRequest req{};
        req.cmd = static_cast<uint32_t>(L3L2OrchCommCmd::ALLOC_REGION);
        req.payload_bytes = payload_bytes;
        req.counter_bytes = counter_bytes;
        L3L2OrchCommResponse resp = submit(req);
        EXPECT_EQ(resp.status, 0) << resp.message;
        EXPECT_EQ(l3_l2_orch_comm::validate_desc(resp.desc), L3L2OrchCommValidationError::OK);
        return resp.desc;
    }

    L3L2OrchCommResponse submit(const L3L2OrchCommRequest &req, uint64_t timeout_ns = 1000000000ULL) {
        L3L2OrchCommResponse resp{};
        int rc = client.submit(req, &resp, timeout_ns);
        EXPECT_EQ(rc, 0) << "client timed out";
        return resp;
    }

    L3L2OrchCommResponse
    notify(const L3L2OrchRegionDesc &desc, uint64_t counter_addr, int32_t value, L3L2OrchNotifyOp op) {
        L3L2OrchCommRequest req{};
        req.cmd = static_cast<uint32_t>(L3L2OrchCommCmd::SIGNAL_NOTIFY);
        req.region_id = desc.region_id;
        req.counter_addr = counter_addr;
        req.counter_operand = value;
        req.op = static_cast<uint32_t>(op);
        return submit(req);
    }

    L3L2OrchCommResponse
    test(const L3L2OrchRegionDesc &desc, uint64_t counter_addr, int32_t cmp_value, L3L2OrchWaitCmp cmp) {
        L3L2OrchCommRequest req{};
        req.cmd = static_cast<uint32_t>(L3L2OrchCommCmd::SIGNAL_TEST);
        req.region_id = desc.region_id;
        req.counter_addr = counter_addr;
        req.counter_operand = cmp_value;
        req.op = static_cast<uint32_t>(cmp);
        return submit(req);
    }

    L3L2OrchCommResponse wait(
        const L3L2OrchRegionDesc &desc, uint64_t counter_addr, int32_t cmp_value, L3L2OrchWaitCmp cmp,
        uint64_t timeout_ns
    ) {
        L3L2OrchCommRequest req{};
        req.cmd = static_cast<uint32_t>(L3L2OrchCommCmd::SIGNAL_WAIT);
        req.region_id = desc.region_id;
        req.counter_addr = counter_addr;
        req.counter_operand = cmp_value;
        req.op = static_cast<uint32_t>(cmp);
        req.timeout_ns = timeout_ns;
        return submit(req);
    }
};

TEST_F(ServiceFixture, AllocRegionReturnsDescriptorAndInitializesCounters) {
    L3L2OrchRegionDesc desc = alloc_region();

    EXPECT_EQ(desc.counter_bytes, 128u);
    EXPECT_EQ(desc.counter_base % L3L2_ORCH_COMM_COUNTER_BASE_ALIGNMENT, 0u);

    L3L2OrchCommResponse first = test(desc, desc.counter_base, 0, L3L2OrchWaitCmp::EQ);
    EXPECT_EQ(first.status, 0) << first.message;
    EXPECT_EQ(first.observed_counter, 0);
    EXPECT_EQ(first.matched, 1u);

    uint64_t last_addr = desc.counter_base + desc.counter_bytes - sizeof(int32_t);
    L3L2OrchCommResponse last = test(desc, last_addr, 0, L3L2OrchWaitCmp::EQ);
    EXPECT_EQ(last.status, 0) << last.message;
    EXPECT_EQ(last.observed_counter, 0);
    EXPECT_EQ(last.matched, 1u);
}

TEST_F(ServiceFixture, PayloadWriteAndReadRoundTripThroughService) {
    L3L2OrchRegionDesc desc = alloc_region();
    const uint8_t src[8] = {1, 3, 5, 7, 9, 11, 13, 15};
    uint8_t dst[8] = {};

    L3L2OrchCommRequest write_req{};
    write_req.cmd = static_cast<uint32_t>(L3L2OrchCommCmd::PAYLOAD_WRITE);
    write_req.region_id = desc.region_id;
    write_req.payload_offset = 16;
    write_req.host_ptr = reinterpret_cast<uint64_t>(src);
    write_req.payload_bytes = sizeof(src);
    EXPECT_EQ(submit(write_req).status, 0);

    L3L2OrchCommRequest read_req{};
    read_req.cmd = static_cast<uint32_t>(L3L2OrchCommCmd::PAYLOAD_READ);
    read_req.region_id = desc.region_id;
    read_req.payload_offset = 16;
    read_req.host_ptr = reinterpret_cast<uint64_t>(dst);
    read_req.payload_bytes = sizeof(dst);
    EXPECT_EQ(submit(read_req).status, 0);

    EXPECT_EQ(std::memcmp(src, dst, sizeof(src)), 0);
}

TEST_F(ServiceFixture, RejectsNonzeroReserved0BeforeCommandDispatch) {
    L3L2OrchCommRequest req{};
    req.cmd = static_cast<uint32_t>(L3L2OrchCommCmd::ALLOC_REGION);
    req.payload_bytes = 128;
    req.counter_bytes = 128;
    req.reserved0 = 1;

    L3L2OrchCommResponse resp = submit(req);

    EXPECT_NE(resp.status, 0);
    EXPECT_EQ(resp.error_kind, 1u);
    EXPECT_EQ(resp.desc.region_id, 0u);
    EXPECT_NE(std::strstr(resp.message, "reserved0"), nullptr);
}

TEST_F(ServiceFixture, PayloadCopyFailurePoisonsOnlyAffectedRegion) {
    L3L2OrchRegionDesc first = alloc_region();
    L3L2OrchRegionDesc second = alloc_region();
    const uint8_t first_src[4] = {2, 4, 6, 8};
    const uint8_t second_src[4] = {1, 3, 5, 7};
    uint8_t second_dst[4] = {};

    L3L2OrchCommRequest write_req{};
    write_req.cmd = static_cast<uint32_t>(L3L2OrchCommCmd::PAYLOAD_WRITE);
    write_req.region_id = first.region_id;
    write_req.host_ptr = reinterpret_cast<uint64_t>(first_src);
    write_req.payload_bytes = sizeof(first_src);

    backend.fail_copy_ = true;
    L3L2OrchCommResponse failed = submit(write_req);
    EXPECT_NE(failed.status, 0);
    EXPECT_EQ(failed.error_kind, 6u);
    EXPECT_EQ(failed.region_id, first.region_id);

    backend.fail_copy_ = false;
    L3L2OrchCommResponse poisoned = submit(write_req);
    EXPECT_NE(poisoned.status, 0);
    EXPECT_EQ(poisoned.error_kind, 4u);
    EXPECT_EQ(poisoned.region_id, first.region_id);

    write_req.region_id = second.region_id;
    write_req.host_ptr = reinterpret_cast<uint64_t>(second_src);
    write_req.payload_bytes = sizeof(second_src);
    EXPECT_EQ(submit(write_req).status, 0);

    L3L2OrchCommRequest read_req{};
    read_req.cmd = static_cast<uint32_t>(L3L2OrchCommCmd::PAYLOAD_READ);
    read_req.region_id = second.region_id;
    read_req.host_ptr = reinterpret_cast<uint64_t>(second_dst);
    read_req.payload_bytes = sizeof(second_dst);
    EXPECT_EQ(submit(read_req).status, 0);
    EXPECT_EQ(std::memcmp(second_src, second_dst, sizeof(second_src)), 0);
}

TEST_F(ServiceFixture, SignalNotifySetAndAddUpdateCounter) {
    L3L2OrchRegionDesc desc = alloc_region();
    uint64_t counter_addr = desc.counter_base + 64;

    EXPECT_EQ(notify(desc, counter_addr, 3, L3L2OrchNotifyOp::Set).status, 0);
    L3L2OrchCommResponse set_resp = test(desc, counter_addr, 3, L3L2OrchWaitCmp::EQ);
    EXPECT_EQ(set_resp.status, 0) << set_resp.message;
    EXPECT_EQ(set_resp.observed_counter, 3);
    EXPECT_EQ(set_resp.matched, 1u);

    EXPECT_EQ(notify(desc, counter_addr, -2, L3L2OrchNotifyOp::Add).status, 0);
    L3L2OrchCommResponse add_resp = test(desc, counter_addr, 1, L3L2OrchWaitCmp::EQ);
    EXPECT_EQ(add_resp.status, 0) << add_resp.message;
    EXPECT_EQ(add_resp.observed_counter, 1);
    EXPECT_EQ(add_resp.matched, 1u);
}

TEST_F(ServiceFixture, SignalTestMismatchReturnsObservedAndKeepsRegionUsable) {
    L3L2OrchRegionDesc desc = alloc_region();
    uint64_t counter_addr = desc.counter_base;

    EXPECT_EQ(notify(desc, counter_addr, 4, L3L2OrchNotifyOp::Set).status, 0);
    L3L2OrchCommResponse test_resp = test(desc, counter_addr, 5, L3L2OrchWaitCmp::GE);
    EXPECT_EQ(test_resp.status, 0) << test_resp.message;
    EXPECT_EQ(test_resp.observed_counter, 4);
    EXPECT_EQ(test_resp.matched, 0u);

    uint8_t byte = 1;
    L3L2OrchCommRequest write_req{};
    write_req.cmd = static_cast<uint32_t>(L3L2OrchCommCmd::PAYLOAD_WRITE);
    write_req.region_id = desc.region_id;
    write_req.host_ptr = reinterpret_cast<uint64_t>(&byte);
    write_req.payload_bytes = sizeof(byte);
    EXPECT_EQ(submit(write_req).status, 0);
}

TEST_F(ServiceFixture, SignalTestCoversAllWaitCmpValues) {
    L3L2OrchRegionDesc desc = alloc_region();
    uint64_t counter_addr = desc.counter_base;

    EXPECT_EQ(notify(desc, counter_addr, 5, L3L2OrchNotifyOp::Set).status, 0);

    EXPECT_EQ(test(desc, counter_addr, 5, L3L2OrchWaitCmp::EQ).matched, 1u);
    EXPECT_EQ(test(desc, counter_addr, 6, L3L2OrchWaitCmp::NE).matched, 1u);
    EXPECT_EQ(test(desc, counter_addr, 4, L3L2OrchWaitCmp::GT).matched, 1u);
    EXPECT_EQ(test(desc, counter_addr, 5, L3L2OrchWaitCmp::GE).matched, 1u);
    EXPECT_EQ(test(desc, counter_addr, 6, L3L2OrchWaitCmp::LT).matched, 1u);
    EXPECT_EQ(test(desc, counter_addr, 5, L3L2OrchWaitCmp::LE).matched, 1u);
    EXPECT_EQ(test(desc, counter_addr, 4, L3L2OrchWaitCmp::LT).matched, 0u);
}

TEST_F(ServiceFixture, SignalWaitPollsUntilMatchAndReturnsObserved) {
    L3L2OrchRegionDesc desc = alloc_region();
    uint64_t counter_addr = desc.counter_base + 64;

    EXPECT_EQ(notify(desc, counter_addr, 7, L3L2OrchNotifyOp::Set).status, 0);
    L3L2OrchCommResponse wait_resp = wait(desc, counter_addr, 5, L3L2OrchWaitCmp::GE, 100000000);
    EXPECT_EQ(wait_resp.status, 0) << wait_resp.message;
    EXPECT_EQ(wait_resp.observed_counter, 7);
    EXPECT_EQ(wait_resp.matched, 1u);
}

TEST_F(ServiceFixture, SignalWaitClampsExtremeTimeoutUntilCounterMatches) {
    L3L2OrchRegionDesc desc = alloc_region();
    uint64_t counter_addr = desc.counter_base;
    L3L2OrchCommResponse wait_resp{};

    std::thread waiter([&]() {
        wait_resp = wait(desc, counter_addr, 1, L3L2OrchWaitCmp::EQ, std::numeric_limits<uint64_t>::max());
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    *reinterpret_cast<int32_t *>(counter_addr) = 1;
    waiter.join();

    EXPECT_EQ(wait_resp.status, 0) << wait_resp.message;
    EXPECT_EQ(wait_resp.observed_counter, 1);
    EXPECT_EQ(wait_resp.matched, 1u);
}

TEST_F(ServiceFixture, SignalWaitTimeoutReturnsObservedAndDoesNotPoisonRegion) {
    L3L2OrchRegionDesc desc = alloc_region();
    uint64_t counter_addr = desc.counter_base;

    EXPECT_EQ(notify(desc, counter_addr, 2, L3L2OrchNotifyOp::Set).status, 0);
    L3L2OrchCommResponse wait_resp = wait(desc, counter_addr, 3, L3L2OrchWaitCmp::GE, 1000000);
    EXPECT_NE(wait_resp.status, 0);
    EXPECT_EQ(wait_resp.observed_counter, 2);
    EXPECT_EQ(wait_resp.matched, 0u);

    EXPECT_EQ(notify(desc, counter_addr, 3, L3L2OrchNotifyOp::Set).status, 0);
    L3L2OrchCommResponse test_resp = test(desc, counter_addr, 3, L3L2OrchWaitCmp::EQ);
    EXPECT_EQ(test_resp.status, 0) << test_resp.message;
    EXPECT_EQ(test_resp.matched, 1u);
}

TEST_F(ServiceFixture, SignalOperationsRejectInvalidCounterAddressWithoutPoisoningRegion) {
    L3L2OrchRegionDesc desc = alloc_region();

    L3L2OrchCommResponse unaligned = notify(desc, desc.counter_base + 2, 1, L3L2OrchNotifyOp::Set);
    EXPECT_NE(unaligned.status, 0);

    L3L2OrchCommResponse out_of_range = test(desc, desc.counter_base + desc.counter_bytes, 0, L3L2OrchWaitCmp::EQ);
    EXPECT_NE(out_of_range.status, 0);

    EXPECT_EQ(notify(desc, desc.counter_base, 1, L3L2OrchNotifyOp::Set).status, 0);
    EXPECT_EQ(test(desc, desc.counter_base, 1, L3L2OrchWaitCmp::EQ).matched, 1u);
}

TEST_F(ServiceFixture, MultipleRegionsKeepPayloadSignalsAndPoisonSeparate) {
    L3L2OrchRegionDesc first = alloc_region();
    L3L2OrchRegionDesc second = alloc_region();
    const uint8_t first_src[4] = {2, 4, 6, 8};
    const uint8_t second_src[4] = {1, 3, 5, 7};
    uint8_t first_dst[4] = {};
    uint8_t second_dst[4] = {};

    L3L2OrchCommRequest write_req{};
    write_req.cmd = static_cast<uint32_t>(L3L2OrchCommCmd::PAYLOAD_WRITE);
    write_req.region_id = first.region_id;
    write_req.host_ptr = reinterpret_cast<uint64_t>(first_src);
    write_req.payload_bytes = sizeof(first_src);
    EXPECT_EQ(submit(write_req).status, 0);

    write_req.region_id = second.region_id;
    write_req.host_ptr = reinterpret_cast<uint64_t>(second_src);
    write_req.payload_bytes = sizeof(second_src);
    EXPECT_EQ(submit(write_req).status, 0);

    L3L2OrchCommRequest read_req{};
    read_req.cmd = static_cast<uint32_t>(L3L2OrchCommCmd::PAYLOAD_READ);
    read_req.region_id = first.region_id;
    read_req.host_ptr = reinterpret_cast<uint64_t>(first_dst);
    read_req.payload_bytes = sizeof(first_dst);
    EXPECT_EQ(submit(read_req).status, 0);

    read_req.region_id = second.region_id;
    read_req.host_ptr = reinterpret_cast<uint64_t>(second_dst);
    read_req.payload_bytes = sizeof(second_dst);
    EXPECT_EQ(submit(read_req).status, 0);

    EXPECT_EQ(std::memcmp(first_src, first_dst, sizeof(first_src)), 0);
    EXPECT_EQ(std::memcmp(second_src, second_dst, sizeof(second_src)), 0);

    EXPECT_EQ(notify(first, first.counter_base, 3, L3L2OrchNotifyOp::Set).status, 0);
    EXPECT_EQ(notify(second, second.counter_base, 7, L3L2OrchNotifyOp::Set).status, 0);
    EXPECT_EQ(wait(first, first.counter_base, 3, L3L2OrchWaitCmp::EQ, 100000000).status, 0);
    EXPECT_EQ(wait(second, second.counter_base, 7, L3L2OrchWaitCmp::EQ, 100000000).status, 0);

    L3L2OrchCommResponse first_timeout = wait(first, first.counter_base + 64, 1, L3L2OrchWaitCmp::EQ, 1000000);
    EXPECT_NE(first_timeout.status, 0);
    EXPECT_EQ(first_timeout.region_id, first.region_id);

    read_req.region_id = first.region_id;
    read_req.host_ptr = reinterpret_cast<uint64_t>(first_dst);
    EXPECT_EQ(submit(read_req).status, 0);

    read_req.region_id = second.region_id;
    read_req.host_ptr = reinterpret_cast<uint64_t>(second_dst);
    EXPECT_EQ(submit(read_req).status, 0);
    EXPECT_EQ(std::memcmp(second_src, second_dst, sizeof(second_src)), 0);
}

TEST_F(ServiceFixture, FreeRegionIsIdempotent) {
    L3L2OrchRegionDesc desc = alloc_region();

    L3L2OrchCommRequest free_req{};
    free_req.cmd = static_cast<uint32_t>(L3L2OrchCommCmd::FREE_REGION);
    free_req.region_id = desc.region_id;
    EXPECT_EQ(submit(free_req).status, 0);
    EXPECT_EQ(submit(free_req).status, 0);
}

TEST(L3L2OrchCommClientTest, SubmitClampsExtremeTimeoutWhileWaitingForIdleAndDone) {
    L3L2OrchCommControlBlock control{};
    control.state.store(static_cast<uint32_t>(L3L2OrchCommControlState::RUNNING), std::memory_order_release);

    L3L2OrchCommClient client;
    ASSERT_EQ(client.attach(&control, sizeof(control)), 0);

    L3L2OrchCommRequest request{};
    request.cmd = static_cast<uint32_t>(L3L2OrchCommCmd::ALLOC_REGION);
    request.payload_bytes = 64;
    request.counter_bytes = 128;
    L3L2OrchCommResponse response{};
    int rc = -1;

    std::thread submitter([&]() {
        rc = client.submit(request, &response, std::numeric_limits<uint64_t>::max());
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    control.state.store(static_cast<uint32_t>(L3L2OrchCommControlState::IDLE), std::memory_order_release);

    const auto ready_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (control.state.load(std::memory_order_acquire) != static_cast<uint32_t>(L3L2OrchCommControlState::READY) &&
           std::chrono::steady_clock::now() < ready_deadline) {
        std::this_thread::sleep_for(std::chrono::nanoseconds(50000));
    }
    if (control.state.load(std::memory_order_acquire) != static_cast<uint32_t>(L3L2OrchCommControlState::READY)) {
        submitter.join();
        ADD_FAILURE() << "client returned before the control block became IDLE";
        EXPECT_EQ(rc, 0);
        return;
    }
    control.response.status = 0;
    control.state.store(static_cast<uint32_t>(L3L2OrchCommControlState::DONE), std::memory_order_release);

    submitter.join();
    EXPECT_EQ(rc, 0);
    EXPECT_EQ(response.status, 0);
    EXPECT_EQ(control.state.load(std::memory_order_acquire), static_cast<uint32_t>(L3L2OrchCommControlState::IDLE));
}

}  // namespace
