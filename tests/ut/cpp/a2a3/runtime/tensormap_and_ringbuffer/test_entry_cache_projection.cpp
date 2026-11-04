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
#include "common/kernel_args.h"
#include "aicpu/platform_entry_args.h"
#include "runtime.h"

extern "C" int simpler_aicpu_exec(void *);
namespace {
PlatformEntryArgs received;
int calls = 0;
}  // namespace
extern "C" int aicpu_execute(Runtime *) {
    received = get_platform_entry_args(0);
    ++calls;
    return 0;
}
TEST(EntryCacheProjection, ActualKernelEntryPreservesEveryPolicy) {
    Runtime runtime;
    runtime.set_aicpu_allowed_cpu_count(1);
    runtime.set_aicpu_launch_count(1);
    KernelArgs args{};
    args.runtime_args = &runtime;
    args.entry_args_source = static_cast<uint32_t>(EntryArgsSource::LaunchEnvelope);
    args.entry_args_offset = LAUNCH_ENVELOPE_HEADER_BYTES;
    args.entry_tensor_count = 4;
    args.entry_scalar_count = 2;
    for (auto policy : {EntryCacheMaintenance::Full, EntryCacheMaintenance::FullFreshen, EntryCacheMaintenance::Live}) {
        args.entry_cache_maintenance = static_cast<uint32_t>(policy);
        const int before = calls;
        ASSERT_EQ(simpler_aicpu_exec(&args), 0);
        ASSERT_EQ(calls, before + 1);
        EXPECT_EQ(received.args_base, &args);
        EXPECT_EQ(received.cache_maintenance, args.entry_cache_maintenance);
        EXPECT_EQ(received.offset, args.entry_args_offset);
        EXPECT_EQ(received.tensor_count, 4u);
        EXPECT_EQ(received.scalar_count, 2u);
    }
    set_platform_entry_args(0, {});
}
