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

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>

#include "tensormap_and_ringbuffer/kernel_clear_plan.h"

namespace {
using namespace simpler::tmr;

const TmrKernelClearBinding kBinding{11, {0x10000, 64}, {0x20000, 192}, 3};

void expect_equal(const TmrKernelClearPlan &actual, const TmrKernelClearPlan &expected) {
    EXPECT_EQ(actual.context_generation, expected.context_generation);
    for (size_t i = 0; i < actual.regions.size(); ++i) {
        EXPECT_EQ(actual.regions[i].address, expected.regions[i].address);
        EXPECT_EQ(actual.regions[i].bytes, expected.regions[i].bytes);
    }
}

TEST(TmrKernelClearPlan, BuildsOnlyDynamicRegions) {
    TmrKernelClearPlan plan;
    ASSERT_TRUE(build_tmr_kernel_clear_plan(kBinding, &plan));
    EXPECT_EQ(plan.context_generation, 11u);
    EXPECT_EQ(plan.regions[0].address, kBinding.control.address);
    EXPECT_EQ(plan.regions[0].bytes, sizeof(TmrLaunchControl));
    EXPECT_EQ(plan.regions[1].address, kBinding.reports.address);
    EXPECT_EQ(plan.regions[1].bytes, 3u * sizeof(TmrCoreReport));

    auto adjacent = kBinding;
    adjacent.reports.address = adjacent.control.address + adjacent.control.bytes;
    EXPECT_TRUE(build_tmr_kernel_clear_plan(adjacent, &plan));
    adjacent.control.address = adjacent.reports.address + adjacent.reports.bytes;
    EXPECT_TRUE(build_tmr_kernel_clear_plan(adjacent, &plan));
    EXPECT_FALSE(build_tmr_kernel_clear_plan(kBinding, nullptr));
}

TEST(TmrKernelClearPlan, InvalidBindingPreservesOutput) {
    TmrKernelClearPlan output;
    ASSERT_TRUE(build_tmr_kernel_clear_plan(kBinding, &output));
    const auto original = output;
    auto reject = [&](const TmrKernelClearBinding &binding) {
        EXPECT_FALSE(build_tmr_kernel_clear_plan(binding, &output));
        expect_equal(output, original);
    };
    auto invalid = kBinding;
    invalid.context_generation = 0;
    reject(invalid);
    for (int32_t count : {0, -1, std::numeric_limits<int32_t>::min()}) {
        invalid = kBinding;
        invalid.worker_count = count;
        reject(invalid);
    }
    for (bool control : {false, true}) {
        for (uint64_t address : {uint64_t{0}, uint64_t{1}, std::numeric_limits<uint64_t>::max() - 63}) {
            invalid = kBinding;
            (control ? invalid.control : invalid.reports).address = address;
            reject(invalid);
        }
        for (uint64_t bytes : {uint64_t{0}, uint64_t{1}, uint64_t{4096}, std::numeric_limits<uint64_t>::max()}) {
            invalid = kBinding;
            (control ? invalid.control : invalid.reports).bytes = bytes;
            reject(invalid);
        }
    }
    invalid = kBinding;
    invalid.reports.address = invalid.control.address;
    reject(invalid);
    invalid = kBinding;
    invalid.control.address = invalid.reports.address + 64;
    reject(invalid);
    invalid = kBinding;
    invalid.worker_count = 2;
    reject(invalid);
}

struct alignas(64) ClearFixture {
    std::array<uint8_t, 64> before;
    TmrLaunchControl control;
    std::array<uint8_t, 64> between;
    std::array<TmrCoreReport, 3> reports;
    std::array<uint8_t, 64> after;
};

TmrKernelClearBinding fixture_binding(ClearFixture &fixture) {
    return {
        17,
        {reinterpret_cast<uint64_t>(&fixture.control), sizeof(fixture.control)},
        {reinterpret_cast<uint64_t>(fixture.reports.data()), sizeof(fixture.reports)},
        static_cast<int32_t>(fixture.reports.size())
    };
}

// CPU memory models only the selected byte ranges; this does not exercise
// asynchronous CANN writes or device cache coherence.
void clear_region(const TmrClearRegion &region) {
    std::memset(reinterpret_cast<void *>(region.address), 0, static_cast<size_t>(region.bytes));
}

bool all_zero(const TmrClearRegion &region) {
    const auto *begin = reinterpret_cast<const uint8_t *>(region.address);
    return std::all_of(begin, begin + region.bytes, [](uint8_t value) {
        return value == 0;
    });
}

TEST(TmrKernelClearPlan, ClearingPreservesSurroundingStaticBytes) {
    ClearFixture fixture;
    std::memset(&fixture, 0xa5, sizeof(fixture));
    const auto binding = fixture_binding(fixture);
    TmrKernelClearPlan plan;
    ASSERT_TRUE(build_tmr_kernel_clear_plan(binding, &plan));
    for (const auto &region : plan.regions)
        clear_region(region);
    for (const auto &region : plan.regions)
        EXPECT_TRUE(all_zero(region));
    for (const auto *guard : {&fixture.before, &fixture.between, &fixture.after}) {
        EXPECT_TRUE(std::all_of(guard->begin(), guard->end(), [](uint8_t value) {
            return value == 0xa5;
        }));
    }
}

TEST(TmrKernelClearPlan, SkippingEitherClearLeavesObservablePriorRoundState) {
    ClearFixture fixture{};
    const auto binding = fixture_binding(fixture);
    TmrKernelClearPlan plan;
    ASSERT_TRUE(build_tmr_kernel_clear_plan(binding, &plan));
    for (size_t skipped = 0; skipped < plan.regions.size(); ++skipped) {
        std::memset(&fixture, 0, sizeof(fixture));
        fixture.control.completion = static_cast<uint32_t>(TmrCompletion::Complete);
        fixture.control.round_epoch = 19;
        fixture.reports[0].aicore_done = 1;
        fixture.reports[1].task = uint64_t{73};
        fixture.reports[2].report_epoch = uint64_t{19};
        clear_region(plan.regions[1 - skipped]);
        EXPECT_FALSE(all_zero(plan.regions[skipped]));
        EXPECT_TRUE(all_zero(plan.regions[1 - skipped]));
        if (skipped == 0) {
            EXPECT_EQ(fixture.control.round_epoch, 19u);
            EXPECT_EQ(fixture.control.completion, static_cast<uint32_t>(TmrCompletion::Complete));
        } else {
            EXPECT_EQ(fixture.reports[0].aicore_done, 1u);
            EXPECT_EQ(fixture.reports[1].task, uint64_t{73});
            EXPECT_EQ(fixture.reports[2].report_epoch, uint64_t{19});
        }
    }
}

}  // namespace
