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
#include <array>
#include "tensormap_and_ringbuffer/kernel_round_storage.h"

namespace {
const void *invalidated = nullptr;
size_t invalidated_bytes = 0;
const void *flushed = nullptr;
}  // namespace
namespace aicpu_cache_maintenance {
void invalidate_range_impl(const void *address, size_t bytes) {
    invalidated = address;
    invalidated_bytes = bytes;
}
void flush_range_impl(const void *address, size_t) { flushed = address; }
}  // namespace aicpu_cache_maintenance

TEST(TmrKernelRoundStorage, BorrowsReportsAndPublishesFinalStatusWithoutCoreAcknowledgments) {
    using namespace simpler::tmr;
    TmrLaunchControl control{};
    std::array<TmrCoreReport, 3> reports{};
    KernelRoundStorage storage;
    ASSERT_TRUE(storage.attach({&control, reports.data(), 3, 11}));
    EXPECT_EQ(storage.reports(), reports.data());
    EXPECT_EQ(invalidated, reports.data());
    EXPECT_EQ(invalidated_bytes, sizeof(reports));
    storage.publish_status(-19, 0);
    EXPECT_EQ(control.runtime_status, -19);
    EXPECT_EQ(control.round_epoch, 11u);
    EXPECT_EQ(control.completion, 1u);
    EXPECT_EQ(flushed, &control);
    for (const auto &report : reports)
        EXPECT_EQ(report.aicore_done, 0u);
}

TEST(TmrKernelRoundStorage, RejectsOverlappingAndUnalignedStorage) {
    using namespace simpler::tmr;
    TmrLaunchControl control{};
    std::array<TmrCoreReport, 3> reports{};
    KernelRoundStorage storage;
    EXPECT_FALSE(storage.attach({&control, reinterpret_cast<TmrCoreReport *>(&control), 1, 1}));
    EXPECT_FALSE(storage.attach({&control, reports.data(), 3, 0}));
    EXPECT_FALSE(storage.attach({nullptr, reports.data(), 3, 1}));
    EXPECT_FALSE(storage.attach({&control, reports.data(), 0, 1}));
}
