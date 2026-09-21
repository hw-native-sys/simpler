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
 * The func_id -> CoreCallable dispatch table a Runtime resolves through holds
 * exactly the active callable's mappings.
 *
 * Every entry is an address inside one callable's retained registration block,
 * which unregistering that callable frees for the allocator to hand out again.
 * The scheduler dereferences the entry to read `CoreCallable::resolved_addr()`
 * and the AICore calls what it finds, so an entry that outlives its block is
 * executed as code — it surfaces as an AICore UB out-of-bounds with a program
 * counter in a data region (issue #1489).
 *
 * The table is owned by that registration and a bind installs one reference to
 * it, so what these tests pin is that the reference names one callable's table
 * and its length: a func_id the active callable does not define resolves to
 * zero whatever an earlier callable published there. They hold on the Runtime
 * itself, without a device.
 */

#include <gtest/gtest.h>

#include <cstdint>

#include "host_build_graph/runtime.h"

namespace {

// Stand-ins for two callables' registration-owned tables. Nothing dereferences
// the entries; only the reference's bookkeeping is under test. The device
// addresses are distinct from the host views so a lookup that reached for the
// device address instead would read the wrong memory.
constexpr uint64_t kCallableAChild0 = 0x7000'0000'1000ull;
constexpr uint64_t kCallableAChild1 = 0x7000'0000'2000ull;
constexpr uint64_t kCallableBChild0 = 0x7000'0000'9000ull;
constexpr uint64_t kTableADev = 0x7100'0000'0000ull;
constexpr uint64_t kTableBDev = 0x7100'0000'8000ull;

// Callable A: func_ids {0, 1}. Callable B: func_id {0} only, and the wider
// variant {0..7} used for the sparse-hole case.
const uint64_t kTableA[] = {kCallableAChild0, kCallableAChild1};
const uint64_t kTableB[] = {kCallableBChild0};
const uint64_t kTableBSparse[] = {0, 0, 0, 0, 0, 0, 0, kCallableBChild0};

TEST(DispatchTable, StartsEmpty) {
    Runtime runtime;
    EXPECT_EQ(runtime.callable_table_len(), 0u);
    for (int func_id = 0; func_id < RUNTIME_MAX_FUNC_ID; func_id++) {
        ASSERT_EQ(runtime.get_function_bin_addr(func_id), 0u) << "func_id=" << func_id;
    }
}

TEST(DispatchTable, BindPublishesTheCallablesAddresses) {
    Runtime runtime;
    runtime.set_callable_tables(kTableA, kTableADev, 0, 2);
    EXPECT_EQ(runtime.get_function_bin_addr(0), kCallableAChild0);
    EXPECT_EQ(runtime.get_function_bin_addr(1), kCallableAChild1);
    EXPECT_EQ(runtime.dev.callable_table_addr_, kTableADev);
    EXPECT_EQ(runtime.dev.callable_table_len_, 2u);
}

TEST(DispatchTable, ClearDropsEveryMapping) {
    Runtime runtime;
    runtime.set_callable_tables(kTableA, kTableADev, 0, 2);

    runtime.clear_callable_tables();

    EXPECT_EQ(runtime.get_function_bin_addr(0), 0u);
    EXPECT_EQ(runtime.get_function_bin_addr(1), 0u);
    EXPECT_EQ(runtime.dev.callable_table_addr_, 0u);
    EXPECT_EQ(runtime.dev.callable_table_len_, 0u);
}

// The regression: callable A defines two children, callable B only one. B's
// bind must not leave A's second address reachable, because A's registration
// block is freed when A unregisters.
TEST(DispatchTable, ANarrowerCallableLeavesNoStaleEntry) {
    Runtime runtime;

    runtime.set_callable_tables(kTableA, kTableADev, 0, 2);
    ASSERT_EQ(runtime.get_function_bin_addr(1), kCallableAChild1);

    // A unregisters, freeing its block; bind(B) names B's own table.
    runtime.set_callable_tables(kTableB, kTableBDev, 0, 1);

    EXPECT_EQ(runtime.get_function_bin_addr(0), kCallableBChild0);
    EXPECT_EQ(runtime.get_function_bin_addr(1), 0u)
        << "func_id 1 still resolves into the registration block that unregistering A freed";
}

// A run whose graph names a func_id the active callable does not define must
// resolve to zero, which the AICore's `function_bin_addr == 0` check rejects,
// rather than to some earlier callable's address. Holes inside the bound
// length and indices past it must answer alike.
TEST(DispatchTable, AnUndefinedFuncIdResolvesToZero) {
    Runtime runtime;
    runtime.set_callable_tables(kTableA, kTableADev, 0, 2);

    runtime.set_callable_tables(kTableBSparse, kTableBDev, 0, 8);

    for (int func_id = 0; func_id < RUNTIME_MAX_FUNC_ID; func_id++) {
        if (func_id == 7) continue;
        ASSERT_EQ(runtime.get_function_bin_addr(func_id), 0u) << "func_id=" << func_id;
    }
    EXPECT_EQ(runtime.get_function_bin_addr(7), kCallableBChild0);
}

TEST(DispatchTable, OutOfRangeFuncIdsResolveToZero) {
    Runtime runtime;
    runtime.set_callable_tables(kTableA, kTableADev, 0, 2);

    EXPECT_EQ(runtime.get_function_bin_addr(-1), 0u);
    EXPECT_EQ(runtime.get_function_bin_addr(2), 0u);
    EXPECT_EQ(runtime.get_function_bin_addr(RUNTIME_MAX_FUNC_ID), 0u);
}

// An empty table is not a reference: a callable with no children must leave the
// descriptor naming nothing, so the device guard short-circuits instead of
// dereferencing an address no table lives at.
TEST(DispatchTable, AnEmptyTableIsRefusedRatherThanPublished) {
    Runtime runtime;
    runtime.set_callable_tables(kTableA, kTableADev, 0, 2);

    runtime.set_callable_tables(nullptr, 0, 0, 0);

    EXPECT_EQ(runtime.dev.callable_table_addr_, 0u);
    EXPECT_EQ(runtime.dev.callable_table_len_, 0u);
    EXPECT_EQ(runtime.get_function_bin_addr(0), 0u);
}

// The a5 host_build_graph AICore scheduler dispatches from the entry view, whose
// address the bind carries beside the object one and which the descriptor never
// holds — it reaches the device through the scheduler worker context.
TEST(DispatchTable, TheEntryViewIsHeldSeparatelyFromTheObjectView) {
    Runtime runtime;
    EXPECT_EQ(runtime.callable_entry_table_addr(), 0u);

    runtime.set_callable_tables(kTableA, kTableADev, kTableBDev, 2);
    EXPECT_EQ(runtime.callable_entry_table_addr(), kTableBDev);
    EXPECT_EQ(runtime.dev.callable_table_addr_, kTableADev);

    runtime.clear_callable_tables();
    EXPECT_EQ(runtime.callable_entry_table_addr(), 0u);
}

}  // namespace
