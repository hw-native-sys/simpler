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
/**
 * The func_id -> CoreCallable dispatch table holds exactly the active
 * callable's mappings.
 *
 * Every entry is an address inside one callable's retained ChipCallable buffer,
 * which unregistering that callable frees for the allocator to hand out again.
 * The scheduler dereferences the entry to read `CoreCallable::resolved_addr()`
 * and the AICore calls what it finds, so an entry that outlives its buffer is
 * executed as code — it surfaces as an AICore UB out-of-bounds with a program
 * counter in a data region (issue #1489).
 *
 * `bind_callable_to_runtime` therefore clears the table before replaying the
 * active callable's addresses. These tests pin that contract on the Runtime
 * itself, so it holds without a device.
 */

#include <gtest/gtest.h>

#include <cstdint>

#include "host_build_graph/runtime.h"

namespace {

// Stand-ins for two callables' CoreCallable addresses. Nothing dereferences
// them; only the table's bookkeeping is under test.
constexpr uint64_t kCallableAChild0 = 0x7000'0000'1000ull;
constexpr uint64_t kCallableAChild1 = 0x7000'0000'2000ull;
constexpr uint64_t kCallableBChild0 = 0x7000'0000'9000ull;

TEST(DispatchTable, StartsEmpty) {
    Runtime runtime;
    for (int func_id = 0; func_id < RUNTIME_MAX_FUNC_ID; func_id++) {
        ASSERT_EQ(runtime.get_function_bin_addr(func_id), 0u) << "func_id=" << func_id;
    }
}

TEST(DispatchTable, ReplayPublishesAnAddress) {
    Runtime runtime;
    runtime.replay_function_bin_addr(0, kCallableAChild0);
    EXPECT_EQ(runtime.get_function_bin_addr(0), kCallableAChild0);
}

TEST(DispatchTable, ClearDropsEveryMapping) {
    Runtime runtime;
    runtime.replay_function_bin_addr(0, kCallableAChild0);
    runtime.replay_function_bin_addr(1, kCallableAChild1);

    runtime.clear_function_bin_addrs();

    EXPECT_EQ(runtime.get_function_bin_addr(0), 0u);
    EXPECT_EQ(runtime.get_function_bin_addr(1), 0u);
}

// The regression: callable A defines two children, callable B only one. B's
// bind must not leave A's second address behind, because A's ChipCallable
// buffer is freed when A unregisters.
TEST(DispatchTable, ANarrowerCallableLeavesNoStaleEntry) {
    Runtime runtime;

    // bind(A) — func_ids {0, 1}
    runtime.clear_function_bin_addrs();
    runtime.replay_function_bin_addr(0, kCallableAChild0);
    runtime.replay_function_bin_addr(1, kCallableAChild1);
    ASSERT_EQ(runtime.get_function_bin_addr(1), kCallableAChild1);

    // A unregisters, freeing its buffer; bind(B) — func_id {0} only
    runtime.clear_function_bin_addrs();
    runtime.replay_function_bin_addr(0, kCallableBChild0);

    EXPECT_EQ(runtime.get_function_bin_addr(0), kCallableBChild0);
    EXPECT_EQ(runtime.get_function_bin_addr(1), 0u)
        << "func_id 1 still points into the ChipCallable buffer that unregistering A freed";
}

// A run whose graph names a func_id the active callable does not define must
// resolve to zero, which the AICore's `function_bin_addr == 0` check rejects,
// rather than to some earlier callable's address.
TEST(DispatchTable, AnUndefinedFuncIdResolvesToZero) {
    Runtime runtime;
    runtime.clear_function_bin_addrs();
    runtime.replay_function_bin_addr(0, kCallableAChild0);
    runtime.replay_function_bin_addr(1, kCallableAChild1);

    runtime.clear_function_bin_addrs();
    runtime.replay_function_bin_addr(7, kCallableBChild0);

    for (int func_id = 0; func_id < RUNTIME_MAX_FUNC_ID; func_id++) {
        if (func_id == 7) continue;
        ASSERT_EQ(runtime.get_function_bin_addr(func_id), 0u) << "func_id=" << func_id;
    }
}

TEST(DispatchTable, OutOfRangeFuncIdsAreRejectedNotWritten) {
    Runtime runtime;
    runtime.replay_function_bin_addr(-1, kCallableAChild0);
    runtime.replay_function_bin_addr(RUNTIME_MAX_FUNC_ID, kCallableAChild0);

    EXPECT_EQ(runtime.get_function_bin_addr(-1), 0u);
    EXPECT_EQ(runtime.get_function_bin_addr(RUNTIME_MAX_FUNC_ID), 0u);
}

}  // namespace
