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

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <unistd.h>
#include <vector>

#include "affinity_allowed_file.h"

namespace {

constexpr const char *kPlanEnv = "SIMPLER_AICPU_AFFINITY_PLAN";
constexpr uint64_t kOccupy = UINT64_C(0x1f8);  // CPUs 3..8.

class PlanPathFixture : public ::testing::Test {
protected:
    void SetUp() override {
        const char *old = std::getenv(kPlanEnv);
        if (old != nullptr) {
            had_old_env_ = true;
            old_env_ = old;
        }
        directory_ = std::filesystem::temp_directory_path() /
                     ("simpler-affinity-side-" + std::to_string(static_cast<long long>(getpid())) + "-" +
                      std::to_string(sequence_++));
        std::filesystem::create_directories(directory_);
        base_ = directory_ / "plan.json";
        ASSERT_EQ(setenv(kPlanEnv, base_.c_str(), 1), 0);
    }

    void TearDown() override {
        if (had_old_env_) {
            (void)setenv(kPlanEnv, old_env_.c_str(), 1);
        } else {
            (void)unsetenv(kPlanEnv);
        }
        std::error_code ec;
        std::filesystem::remove_all(directory_, ec);
    }

    std::filesystem::path side_path(int32_t device_id = 0) const { return pto::a5::affinity_cpus_side_path(device_id); }

    void write_side(const std::string &text) const {
        std::ofstream output(side_path());
        ASSERT_TRUE(output.good());
        output << text;
        ASSERT_TRUE(output.good());
    }

    static std::string valid_side() {
        return "schema_version=3\n"
               "device_id=0\n"
               "soc=Ascend950PR_9599\n"
               "source=manual\n"
               "occupy_mask=0x1f8\n"
               "active_count=5\n"
               "cpus=3,4,5,6,7\n";
    }

    static std::string replacing(std::string text, const std::string &from, const std::string &to) {
        const size_t pos = text.find(from);
        EXPECT_NE(pos, std::string::npos);
        if (pos != std::string::npos) text.replace(pos, from.size(), to);
        return text;
    }

private:
    static inline int sequence_ = 0;
    bool had_old_env_{false};
    std::string old_env_;

protected:
    std::filesystem::path directory_;
    std::filesystem::path base_;
};

TEST_F(PlanPathFixture, DerivesPerDevicePathFromBaseTemplateOrExactSuffix) {
    EXPECT_EQ(side_path(2), directory_ / "plan.2.cpus");

    const std::string templated = (directory_ / "custom.{device}.json").string();
    ASSERT_EQ(setenv(kPlanEnv, templated.c_str(), 1), 0);
    EXPECT_EQ(side_path(3), directory_ / "custom.3.cpus");

    const std::string exact = (directory_ / "custom.4.json").string();
    ASSERT_EQ(setenv(kPlanEnv, exact.c_str(), 1), 0);
    EXPECT_EQ(side_path(4), directory_ / "custom.4.cpus");

    const std::string alternate_suffix = (directory_ / "custom.5.plan").string();
    ASSERT_EQ(setenv(kPlanEnv, alternate_suffix.c_str(), 1), 0);
    EXPECT_EQ(side_path(5), directory_ / "custom.5.cpus");
}

TEST(A5AffinityAllowedFile, BuildsExactContiguousWidthWithoutClamping) {
    std::vector<int32_t> allowed;
    ASSERT_TRUE(pto::a5::build_occupy_contiguous_allowed(kOccupy, 2, allowed));
    EXPECT_EQ(allowed, (std::vector<int32_t>{3, 4}));
    ASSERT_TRUE(pto::a5::build_occupy_contiguous_allowed(kOccupy, 4, allowed));
    EXPECT_EQ(allowed, (std::vector<int32_t>{3, 4, 5, 6}));
    ASSERT_TRUE(pto::a5::build_occupy_contiguous_allowed(kOccupy, 5, allowed));
    EXPECT_EQ(allowed, (std::vector<int32_t>{3, 4, 5, 6, 7}));
    EXPECT_FALSE(pto::a5::build_occupy_contiguous_allowed(UINT64_C(0x18), 3, allowed));
    EXPECT_TRUE(allowed.empty());
}

TEST(A5AffinityAllowedFile, HonorsExplicitActiveCountAndUsesRttOnlyForFourSchedulers) {
    int32_t active = 0;
    bool use_rtt = false;
    for (int32_t requested : {2, 3, 4}) {
        ASSERT_TRUE(pto::a5::resolve_aicpu_active_count(requested, 6, active, use_rtt));
        EXPECT_EQ(active, requested);
        EXPECT_FALSE(use_rtt);
    }
    ASSERT_TRUE(pto::a5::resolve_aicpu_active_count(5, 6, active, use_rtt));
    EXPECT_EQ(active, 5);
    EXPECT_TRUE(use_rtt);

    ASSERT_TRUE(pto::a5::resolve_aicpu_active_count(0, 6, active, use_rtt));
    EXPECT_EQ(active, 5);
    EXPECT_TRUE(use_rtt);
    ASSERT_TRUE(pto::a5::resolve_aicpu_active_count(0, 3, active, use_rtt));
    EXPECT_EQ(active, 3);
    EXPECT_FALSE(use_rtt);

    EXPECT_FALSE(pto::a5::resolve_aicpu_active_count(5, 4, active, use_rtt));
    EXPECT_FALSE(pto::a5::resolve_aicpu_active_count(1, 6, active, use_rtt));
    EXPECT_FALSE(pto::a5::resolve_aicpu_active_count(6, 6, active, use_rtt));
    EXPECT_FALSE(pto::a5::resolve_aicpu_active_count(0, 15, active, use_rtt));
}

TEST_F(PlanPathFixture, LoadsOnlyTheExactHardwareAndSchemaContract) {
    write_side(valid_side());
    std::vector<int32_t> allowed;
    std::string source;
    ASSERT_TRUE(pto::a5::load_affinity_cpus_side_file("Ascend950PR_9599", 0, kOccupy, allowed, source));
    EXPECT_EQ(allowed, (std::vector<int32_t>{3, 4, 5, 6, 7}));
    EXPECT_EQ(source, "manual");
}

TEST_F(PlanPathFixture, RejectsMalformedStaleAndUnsafePlans) {
    const std::vector<std::string> invalid = {
        replacing(valid_side(), "schema_version=3", "schema_version=2"),
        replacing(valid_side(), "device_id=0", "device_id=1"),
        replacing(valid_side(), "soc=Ascend950PR_9599", "soc=Ascend950PR_other"),
        replacing(valid_side(), "source=manual", "source=unknown"),
        replacing(valid_side(), "occupy_mask=0x1f8", "occupy_mask=0xf8"),
        replacing(valid_side(), "active_count=5", "active_count=4"),
        replacing(valid_side(), "cpus=3,4,5,6,7", "cpus=3,4,5,6"),
        replacing(valid_side(), "cpus=3,4,5,6,7", "cpus=3,4,5,6,9"),
        replacing(valid_side(), "cpus=3,4,5,6,7", "cpus=3,4,5,6,6"),
        replacing(valid_side(), "cpus=3,4,5,6,7", "cpus=3,,4,5,6,7"),
        replacing(valid_side(), "cpus=3,4,5,6,7", "cpus=3,4,5,6,7,"),
        valid_side() + "unknown=value\n",
    };
    for (const auto &text : invalid) {
        write_side(text);
        std::vector<int32_t> allowed;
        std::string source;
        EXPECT_FALSE(pto::a5::load_affinity_cpus_side_file("Ascend950PR_9599", 0, kOccupy, allowed, source));
    }
}

}  // namespace
