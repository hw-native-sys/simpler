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
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "chip_callable_layout.h"
#include "host/callable_diagnostics.h"

namespace {

class CallableDiagnosticsTest : public ::testing::Test {
protected:
    void SetUp() override {
        logger_.set_level(simpler::log::LogLevel::TIMING);
        ASSERT_TRUE(logger_.flush(1000));
        testing::internal::CaptureStderr();
    }
    void TearDown() override {
        if (!collected_) {
            logger_.flush(1000);
            testing::internal::GetCapturedStderr();
        }
        logger_.set_level(simpler::log::LogLevel::NUL);
    }
    std::string collect() {
        EXPECT_TRUE(logger_.flush(1000));
        collected_ = true;
        return testing::internal::GetCapturedStderr();
    }
    static std::vector<uint8_t> callable() {
        const uint8_t first[] = {1, 2, 3, 4};
        const uint8_t second[124] = {5, 6, 7, 8};
        std::vector<uint8_t> children[] = {
            make_callable<CORE_MAX_TENSOR_ARGS>(nullptr, 0, first, sizeof(first)),
            make_callable<CORE_MAX_TENSOR_ARGS>(nullptr, 0, second, sizeof(second)),
        };
        const int32_t ids[] = {17, 3};
        const uint8_t orch[] = {0x7f, 'E', 'L', 'F'};
        return make_callable<CoreCallable, CHIP_MAX_TENSOR_ARGS, 1024>(
            nullptr, 0, "orch", orch, sizeof(orch), ids, children, 2, ""
        );
    }
    HostLogger &logger_ = HostLogger::get_instance();
    bool collected_{false};
};

TEST_F(CallableDiagnosticsTest, DefaultThresholdReportsPatchedEntryRangesAndSparseFunctionIds) {
    auto caller = callable();
    const auto original = caller;
    const auto *source = reinterpret_cast<const ChipCallable *>(caller.data());
    auto image_bytes = caller;
    const auto layout = compute_chip_callable_layout(source);
    patch_chip_callable_scratch_for_device(source, layout, 0x100000, image_bytes.data());
    auto &image = *reinterpret_cast<ChipCallable *>(image_bytes.data());
    // The record describes the supplied publication, even when its entry
    // differs from an address recomputed from the unresolved caller object.
    const_cast<CoreCallable &>(image.child(0)).set_resolved_addr(0x200000);
    const_cast<CoreCallable &>(image.child(1)).set_resolved_addr(0x300000);

    log_callable_image(4, &image, 0x1234, 0x100000, image_bytes.size(), image);
    const auto output = collect();
    EXPECT_NE(output.find("Callable image: device=4"), std::string::npos);
    EXPECT_NE(output.find("chip_hash=0x1234 chip_dev=0x100000"), std::string::npos);
    EXPECT_NE(output.find("children=2"), std::string::npos);
    EXPECT_NE(
        output.find("func_id=17 code_begin=0x200000 code_end_exclusive=0x200004 code_bytes=4"), std::string::npos
    );
    EXPECT_NE(
        output.find("func_id=3 code_begin=0x300000 code_end_exclusive=0x30007c code_bytes=124"), std::string::npos
    );
    EXPECT_NE(output.find("code_fnv1a64="), std::string::npos);
    EXPECT_EQ(caller, original);
}

TEST_F(CallableDiagnosticsTest, CodeFingerprintChangesWithBytesAtTheSameAddress) {
    auto bytes = callable();
    auto &image = *reinterpret_cast<ChipCallable *>(bytes.data());
    const_cast<CoreCallable &>(image.child(0)).set_resolved_addr(0x200000);
    log_callable_image(0, &image, 0x1234, 0x100000, bytes.size(), image);
    auto *code = static_cast<uint8_t *>(const_cast<void *>(image.child(0).binary_data()));
    code[0] ^= 0xff;
    log_callable_image(0, &image, 0x1234, 0x100000, bytes.size(), image);
    const auto output = collect();
    const auto first = output.find("func_id=17");
    ASSERT_NE(first, std::string::npos);
    const auto second = output.find("func_id=17", first + 1);
    ASSERT_NE(second, std::string::npos);
    const auto hash_at = [&](size_t start) {
        const auto at = output.find("code_fnv1a64=", start);
        return output.substr(at, output.find('\n', at) - at);
    };
    EXPECT_NE(hash_at(first), hash_at(second));
}

TEST_F(CallableDiagnosticsTest, ExplicitWarningThresholdSuppressesPublicationRecords) {
    auto bytes = callable();
    logger_.set_level(simpler::log::LogLevel::WARN);
    log_callable_image(
        0, bytes.data(), 1, 0x100000, bytes.size(), *reinterpret_cast<const ChipCallable *>(bytes.data())
    );
    EXPECT_EQ(collect(), "");
}

}  // namespace
