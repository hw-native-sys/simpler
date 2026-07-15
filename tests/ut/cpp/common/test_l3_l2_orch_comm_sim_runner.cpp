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
#include <vector>

#include <gtest/gtest.h>

#include "common/l3_l2_orch_comm.h"
#include "device_runner_base.h"
#include "host/l3_l2_orch_comm_service.h"

namespace {

class TestSimRunner : public SimDeviceRunnerBase {
public:
    ~TestSimRunner() override { finalize(); }

    int run(Runtime &, const CallConfig &) override { return 0; }
    int finalize() override {
        if (finalized_) return 0;
        finalized_ = true;
        l3_l2_orch_comm_shutdown();
        release_callable_state();
        return mem_alloc_.finalize();
    }

    size_t chip_callable_buffer_count() const { return chip_callable_buffers_.size(); }
    int chip_callable_buffer_refcount(uint64_t hash) const {
        auto it = chip_callable_buffers_.find(hash);
        return it == chip_callable_buffers_.end() ? 0 : it->second.refcount;
    }

private:
    int ensure_binaries_loaded() override { return 0; }
    int invoke_device_register(const RegisterCallableArgs &) override { return 0; }

    bool finalized_{false};
};

std::vector<uint8_t> build_zero_child_chip_callable() {
    const uint8_t fake_orch_so[] = {0x7f, 'E', 'L', 'F', 0xaa, 0xbb};
    return make_callable<CoreCallable, CHIP_MAX_TENSOR_ARGS, 1024>(
        nullptr, 0, "orch_fn", fake_orch_so, sizeof(fake_orch_so), nullptr, nullptr, 0, "cfg_name"
    );
}

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
    ASSERT_EQ(l3_l2_orch_comm::validate_desc(alloc_resp.desc), L3L2OrchCommValidationError::OK);

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

TEST(ChipCallableBufferLifetimeSimTest, IdenticalRegistrationsReleaseAfterLastUnregister) {
    TestSimRunner runner;
    auto chip_buf = build_zero_child_chip_callable();
    const auto *callable = reinterpret_cast<const ChipCallable *>(chip_buf.data());
    const ChipCallableLayout layout = compute_chip_callable_layout(callable);
    ASSERT_NE(layout.content_hash, 0u);

    const uint64_t first_dev = runner.upload_chip_callable_buffer(callable);
    ASSERT_NE(first_dev, 0u);
    ASSERT_EQ(
        runner.record_device_orch_callable(
            1, layout.content_hash, first_dev, callable->binary_data(), callable->binary_size(), callable->func_name(),
            callable->config_name(), {}, {}
        ),
        0
    );
    EXPECT_EQ(runner.chip_callable_buffer_count(), 1u);
    EXPECT_EQ(runner.chip_callable_buffer_refcount(layout.content_hash), 1);

    const uint64_t second_dev = runner.upload_chip_callable_buffer(callable);
    ASSERT_EQ(second_dev, first_dev);
    ASSERT_EQ(
        runner.record_device_orch_callable(
            2, layout.content_hash, second_dev, callable->binary_data(), callable->binary_size(), callable->func_name(),
            callable->config_name(), {}, {}
        ),
        0
    );
    EXPECT_EQ(runner.chip_callable_buffer_count(), 1u);
    EXPECT_EQ(runner.chip_callable_buffer_refcount(layout.content_hash), 2);

    EXPECT_EQ(runner.unregister_callable(1), 0);
    EXPECT_EQ(runner.chip_callable_buffer_count(), 1u);
    EXPECT_EQ(runner.chip_callable_buffer_refcount(layout.content_hash), 1);

    EXPECT_EQ(runner.unregister_callable(2), 0);
    EXPECT_EQ(runner.chip_callable_buffer_count(), 0u);
    EXPECT_EQ(runner.chip_callable_buffer_refcount(layout.content_hash), 0);
    EXPECT_EQ(runner.finalize(), 0);
}

}  // namespace
