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

#include <cstdlib>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "common/l3_l2_orch_comm.h"
#include "device_runner_base.h"
#include "host/l3_l2_orch_comm_service.h"
#include "pto_runtime_c_api.h"

namespace {

class UnsupportedL3L2Runner : public DeviceRunnerBase {
public:
    int run(Runtime &, const CallConfig &) override { return 0; }
    int finalize() override { return 0; }
    bool l3_l2_orch_comm_supported() const override { return false; }
};

class ChipCallableTestOnboardRunner : public DeviceRunnerBase {
public:
    ~ChipCallableTestOnboardRunner() override { finalize(); }

    int run(Runtime &, const CallConfig &) override { return 0; }
    int finalize() override {
        if (finalized_) return 0;
        finalized_ = true;
        return finalize_common();
    }
    bool l3_l2_orch_comm_supported() const override { return false; }

    int prepare_for_upload(int device_id) {
        int rc = attach_current_thread(device_id);
        if (rc != 0) return rc;
        return rtStreamCreate(&stream_aicpu_, 0);
    }

    size_t chip_callable_buffer_count() const { return chip_callable_buffers_.size(); }
    int chip_callable_buffer_refcount(uint64_t hash) const {
        auto it = chip_callable_buffers_.find(hash);
        return it == chip_callable_buffers_.end() ? 0 : it->second.refcount;
    }

private:
    bool finalized_{false};
};

std::vector<int> read_ctest_devices() {
    std::vector<int> ids;
    const char *count_str = std::getenv("CTEST_RESOURCE_GROUP_COUNT");
    if (count_str == nullptr) return ids;

    int group_count = std::atoi(count_str);
    for (int group = 0; group < group_count; ++group) {
        std::string var = "CTEST_RESOURCE_GROUP_" + std::to_string(group) + "_NPUS";
        const char *value = std::getenv(var.c_str());
        if (value == nullptr) continue;

        std::string resources(value);
        size_t pos = 0;
        while ((pos = resources.find("id:", pos)) != std::string::npos) {
            pos += 3;
            ids.push_back(std::atoi(resources.c_str() + pos));
        }
    }
    return ids;
}

std::vector<uint8_t> build_zero_child_chip_callable() {
    const uint8_t fake_orch_so[] = {0x7f, 'E', 'L', 'F', 0xaa, 0xbb};
    return make_callable<CoreCallable, CHIP_MAX_TENSOR_ARGS, 1024>(
        nullptr, 0, "orch_fn", fake_orch_so, sizeof(fake_orch_so), nullptr, nullptr, 0, "cfg_name"
    );
}

TEST(L3L2OrchCommOnboardRunnerTest, UnsupportedRunnerInitFailsAndShutdownIsNoop) {
    UnsupportedL3L2Runner runner;
    L3L2OrchCommControlBlock control{};

    EXPECT_EQ(runner.l3_l2_orch_comm_init(&control, sizeof(control)), PTO_RUNTIME_ERR_UNSUPPORTED);
    EXPECT_EQ(runner.l3_l2_orch_comm_shutdown(), 0);
}

TEST(ChipCallableBufferLifetimeOnboardTest, IdenticalRegistrationsReleaseAfterLastUnregister) {
    auto devices = read_ctest_devices();
    ASSERT_FALSE(devices.empty()) << "need one NPU device; run with --resource-spec-file";

    ChipCallableTestOnboardRunner runner;
    ASSERT_EQ(runner.prepare_for_upload(devices.front()), 0);

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
