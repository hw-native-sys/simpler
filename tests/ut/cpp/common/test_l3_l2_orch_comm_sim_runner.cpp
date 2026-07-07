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

#include <cstdint>
#include <cstring>

#include <gtest/gtest.h>

#include "common/l3_l2_orch_comm.h"
#include "device_runner_base.h"
#include "host/l3_l2_orch_comm_service.h"

namespace {

class TestSimRunner : public SimDeviceRunnerBase {
public:
    int run(Runtime &, const CallConfig &) override { return 0; }
    int finalize() override {
        l3_l2_orch_comm_shutdown();
        mem_alloc_.finalize();
        return 0;
    }

private:
    int ensure_binaries_loaded() override { return 0; }
    int invoke_device_register(const RegisterCallableArgs &) override { return 0; }
};

L3L2OrchCommResponse
submit(L3L2OrchCommClient &client, const L3L2OrchCommRequest &request, uint64_t timeout_ns = 1000000000ULL) {
    L3L2OrchCommResponse response{};
    EXPECT_EQ(client.submit(request, &response, timeout_ns), 0);
    return response;
}

TEST(L3L2OrchCommSimRunnerTest, RunnerOwnedServiceHandlesPayloadSignalAndFree) {
    TestSimRunner runner;
    L3L2OrchCommControlBlock control{};
    ASSERT_EQ(runner.l3_l2_orch_comm_init(&control, sizeof(control)), 0);

    L3L2OrchCommClient client;
    ASSERT_EQ(client.attach(&control, sizeof(control)), 0);

    L3L2OrchCommRequest alloc{};
    alloc.cmd = static_cast<uint32_t>(L3L2OrchCommCmd::ALLOC_REGION);
    alloc.payload_bytes = 64;
    alloc.counter_bytes = 128;
    L3L2OrchCommResponse alloc_resp = submit(client, alloc);
    ASSERT_EQ(alloc_resp.status, 0) << alloc_resp.message;
    ASSERT_EQ(l3_l2_orch_comm_validate_desc(alloc_resp.desc), L3L2OrchCommValidationError::OK);

    const uint8_t src[4] = {2, 4, 6, 8};
    uint8_t dst[4] = {};
    L3L2OrchCommRequest write{};
    write.cmd = static_cast<uint32_t>(L3L2OrchCommCmd::PAYLOAD_WRITE);
    write.region_id = alloc_resp.desc.region_id;
    write.payload_offset = 12;
    write.host_ptr = reinterpret_cast<uint64_t>(src);
    write.payload_bytes = sizeof(src);
    EXPECT_EQ(submit(client, write).status, 0);

    L3L2OrchCommRequest read{};
    read.cmd = static_cast<uint32_t>(L3L2OrchCommCmd::PAYLOAD_READ);
    read.region_id = alloc_resp.desc.region_id;
    read.payload_offset = 12;
    read.host_ptr = reinterpret_cast<uint64_t>(dst);
    read.payload_bytes = sizeof(dst);
    EXPECT_EQ(submit(client, read).status, 0);
    EXPECT_EQ(std::memcmp(src, dst, sizeof(src)), 0);

    L3L2OrchCommRequest notify{};
    notify.cmd = static_cast<uint32_t>(L3L2OrchCommCmd::SIGNAL_NOTIFY);
    notify.region_id = alloc_resp.desc.region_id;
    notify.counter_addr = alloc_resp.desc.counter_base;
    notify.counter_operand = 1;
    notify.op = static_cast<uint32_t>(L3L2OrchNotifyOp::Set);
    EXPECT_EQ(submit(client, notify).status, 0);

    L3L2OrchCommRequest wait{};
    wait.cmd = static_cast<uint32_t>(L3L2OrchCommCmd::SIGNAL_WAIT);
    wait.region_id = alloc_resp.desc.region_id;
    wait.counter_addr = alloc_resp.desc.counter_base;
    wait.counter_operand = 1;
    wait.op = static_cast<uint32_t>(L3L2OrchWaitCmp::EQ);
    wait.timeout_ns = 100000000;
    L3L2OrchCommResponse wait_resp = submit(client, wait);
    EXPECT_EQ(wait_resp.status, 0);
    EXPECT_EQ(wait_resp.observed_counter, 1);

    L3L2OrchCommRequest free_req{};
    free_req.cmd = static_cast<uint32_t>(L3L2OrchCommCmd::FREE_REGION);
    free_req.region_id = alloc_resp.desc.region_id;
    EXPECT_EQ(submit(client, free_req).status, 0);

    EXPECT_EQ(runner.l3_l2_orch_comm_shutdown(), 0);
}

}  // namespace
