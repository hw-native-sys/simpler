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

    L3L2OrchRegionDesc alloc_region(uint64_t payload_bytes = 128) {
        L3L2OrchCommRequest req{};
        req.cmd = static_cast<uint32_t>(L3L2OrchCommCmd::ALLOC_REGION);
        req.nbytes = payload_bytes;
        L3L2OrchCommResponse resp = submit(req);
        EXPECT_EQ(resp.status, 0) << resp.message;
        EXPECT_EQ(l3_l2_orch_comm_validate_desc(resp.desc), L3L2OrchCommValidationError::OK);
        return resp.desc;
    }

    L3L2OrchCommResponse submit(const L3L2OrchCommRequest &req, uint64_t timeout_ns = 1000000000ULL) {
        L3L2OrchCommResponse resp{};
        int rc = client.submit(req, &resp, timeout_ns);
        EXPECT_EQ(rc, 0) << "client timed out";
        return resp;
    }
};

TEST_F(ServiceFixture, AllocRegionReturnsDescriptorAndInitializesSignals) {
    L3L2OrchRegionDesc desc = alloc_region();

    L3L2OrchCommRequest wait_req{};
    wait_req.cmd = static_cast<uint32_t>(L3L2OrchCommCmd::SIGNAL_WAIT);
    wait_req.region_id = desc.region_id;
    wait_req.signal_slot = static_cast<uint64_t>(L3L2OrchCommSignalSlot::L2_TO_L3);
    wait_req.seq = 1;
    wait_req.timeout_ns = 1000000;

    L3L2OrchCommResponse resp = submit(wait_req);
    EXPECT_NE(resp.status, 0);
    EXPECT_EQ(resp.region_id, desc.region_id);
    EXPECT_EQ(resp.observed_signal, 0u);
}

TEST_F(ServiceFixture, PayloadWriteAndReadRoundTripThroughService) {
    L3L2OrchRegionDesc desc = alloc_region();
    const uint8_t src[8] = {1, 3, 5, 7, 9, 11, 13, 15};
    uint8_t dst[8] = {};

    L3L2OrchCommRequest write_req{};
    write_req.cmd = static_cast<uint32_t>(L3L2OrchCommCmd::PAYLOAD_WRITE);
    write_req.region_id = desc.region_id;
    write_req.offset = 16;
    write_req.host_ptr = reinterpret_cast<uint64_t>(src);
    write_req.nbytes = sizeof(src);
    EXPECT_EQ(submit(write_req).status, 0);

    L3L2OrchCommRequest read_req{};
    read_req.cmd = static_cast<uint32_t>(L3L2OrchCommCmd::PAYLOAD_READ);
    read_req.region_id = desc.region_id;
    read_req.offset = 16;
    read_req.host_ptr = reinterpret_cast<uint64_t>(dst);
    read_req.nbytes = sizeof(dst);
    EXPECT_EQ(submit(read_req).status, 0);

    EXPECT_EQ(std::memcmp(src, dst, sizeof(src)), 0);
}

TEST_F(ServiceFixture, NotifyThenWaitSucceedsForExactSequence) {
    L3L2OrchRegionDesc desc = alloc_region();

    L3L2OrchCommRequest notify_req{};
    notify_req.cmd = static_cast<uint32_t>(L3L2OrchCommCmd::SIGNAL_NOTIFY);
    notify_req.region_id = desc.region_id;
    notify_req.signal_slot = static_cast<uint64_t>(L3L2OrchCommSignalSlot::L2_TO_L3);
    notify_req.seq = 3;
    EXPECT_EQ(submit(notify_req).status, 0);

    L3L2OrchCommRequest wait_req{};
    wait_req.cmd = static_cast<uint32_t>(L3L2OrchCommCmd::SIGNAL_WAIT);
    wait_req.region_id = desc.region_id;
    wait_req.signal_slot = static_cast<uint64_t>(L3L2OrchCommSignalSlot::L2_TO_L3);
    wait_req.seq = 3;
    wait_req.timeout_ns = 100000000;
    EXPECT_EQ(submit(wait_req).status, 0);
}

TEST_F(ServiceFixture, SignalWaitGreaterObservedValuePoisonsRegion) {
    L3L2OrchRegionDesc desc = alloc_region();

    L3L2OrchCommRequest notify_req{};
    notify_req.cmd = static_cast<uint32_t>(L3L2OrchCommCmd::SIGNAL_NOTIFY);
    notify_req.region_id = desc.region_id;
    notify_req.signal_slot = static_cast<uint64_t>(L3L2OrchCommSignalSlot::L2_TO_L3);
    notify_req.seq = 5;
    EXPECT_EQ(submit(notify_req).status, 0);

    L3L2OrchCommRequest wait_req{};
    wait_req.cmd = static_cast<uint32_t>(L3L2OrchCommCmd::SIGNAL_WAIT);
    wait_req.region_id = desc.region_id;
    wait_req.signal_slot = static_cast<uint64_t>(L3L2OrchCommSignalSlot::L2_TO_L3);
    wait_req.seq = 4;
    wait_req.timeout_ns = 100000000;
    L3L2OrchCommResponse wait_resp = submit(wait_req);
    EXPECT_NE(wait_resp.status, 0);
    EXPECT_EQ(wait_resp.observed_signal, 5u);

    uint8_t byte = 1;
    L3L2OrchCommRequest write_req{};
    write_req.cmd = static_cast<uint32_t>(L3L2OrchCommCmd::PAYLOAD_WRITE);
    write_req.region_id = desc.region_id;
    write_req.host_ptr = reinterpret_cast<uint64_t>(&byte);
    write_req.nbytes = sizeof(byte);
    EXPECT_NE(submit(write_req).status, 0);
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
    write_req.nbytes = sizeof(first_src);
    EXPECT_EQ(submit(write_req).status, 0);

    write_req.region_id = second.region_id;
    write_req.host_ptr = reinterpret_cast<uint64_t>(second_src);
    write_req.nbytes = sizeof(second_src);
    EXPECT_EQ(submit(write_req).status, 0);

    L3L2OrchCommRequest read_req{};
    read_req.cmd = static_cast<uint32_t>(L3L2OrchCommCmd::PAYLOAD_READ);
    read_req.region_id = first.region_id;
    read_req.host_ptr = reinterpret_cast<uint64_t>(first_dst);
    read_req.nbytes = sizeof(first_dst);
    EXPECT_EQ(submit(read_req).status, 0);

    read_req.region_id = second.region_id;
    read_req.host_ptr = reinterpret_cast<uint64_t>(second_dst);
    read_req.nbytes = sizeof(second_dst);
    EXPECT_EQ(submit(read_req).status, 0);

    EXPECT_EQ(std::memcmp(first_src, first_dst, sizeof(first_src)), 0);
    EXPECT_EQ(std::memcmp(second_src, second_dst, sizeof(second_src)), 0);

    L3L2OrchCommRequest notify_req{};
    notify_req.cmd = static_cast<uint32_t>(L3L2OrchCommCmd::SIGNAL_NOTIFY);
    notify_req.signal_slot = static_cast<uint64_t>(L3L2OrchCommSignalSlot::L2_TO_L3);
    notify_req.region_id = first.region_id;
    notify_req.seq = 3;
    EXPECT_EQ(submit(notify_req).status, 0);
    notify_req.region_id = second.region_id;
    notify_req.seq = 7;
    EXPECT_EQ(submit(notify_req).status, 0);

    L3L2OrchCommRequest wait_req{};
    wait_req.cmd = static_cast<uint32_t>(L3L2OrchCommCmd::SIGNAL_WAIT);
    wait_req.signal_slot = static_cast<uint64_t>(L3L2OrchCommSignalSlot::L2_TO_L3);
    wait_req.timeout_ns = 100000000;
    wait_req.region_id = first.region_id;
    wait_req.seq = 3;
    EXPECT_EQ(submit(wait_req).status, 0);
    wait_req.region_id = second.region_id;
    wait_req.seq = 7;
    EXPECT_EQ(submit(wait_req).status, 0);

    notify_req.region_id = first.region_id;
    notify_req.seq = 9;
    EXPECT_EQ(submit(notify_req).status, 0);
    wait_req.region_id = first.region_id;
    wait_req.seq = 8;
    L3L2OrchCommResponse first_poison = submit(wait_req);
    EXPECT_NE(first_poison.status, 0);
    EXPECT_EQ(first_poison.region_id, first.region_id);

    read_req.region_id = first.region_id;
    read_req.host_ptr = reinterpret_cast<uint64_t>(first_dst);
    EXPECT_NE(submit(read_req).status, 0);

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

}  // namespace
