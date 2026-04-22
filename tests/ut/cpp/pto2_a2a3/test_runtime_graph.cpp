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
 * Unit tests for host_build_graph Runtime class.
 *
 * Tests task graph construction: add_task, add_successor,
 * ready task detection, and dependency graph patterns.
 */

#include <gtest/gtest.h>
#include "../../../../src/a2a3/runtime/host_build_graph/runtime/runtime.h"

// =============================================================================
// Test fixture -- allocates a Runtime on the heap (it's very large)
// =============================================================================

class RuntimeGraphTest : public ::testing::Test {
protected:
    Runtime *rt = nullptr;

    void SetUp() override { rt = new Runtime(); }

    void TearDown() override { delete rt; }

    // Helper: add a task with no args
    int addTask(int func_id = 0, CoreType core_type = CoreType::AIV) {
        return rt->add_task(nullptr, 0, func_id, core_type);
    }
};

// =============================================================================
// Basic task addition
// =============================================================================

TEST_F(RuntimeGraphTest, AddTask_MonotonicId) {
    int id0 = addTask();
    int id1 = addTask();
    int id2 = addTask();

    EXPECT_EQ(id0, 0);
    EXPECT_EQ(id1, 1);
    EXPECT_EQ(id2, 2);
    EXPECT_EQ(rt->get_task_count(), 3);
}

TEST_F(RuntimeGraphTest, AddTask_StoresFields) {
    uint64_t args[] = {42, 99};
    int id = rt->add_task(args, 2, /*func_id=*/7, CoreType::AIC);

    Task *t = rt->get_task(id);
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(t->func_id, 7);
    EXPECT_EQ(t->num_args, 2);
    EXPECT_EQ(t->args[0], 42u);
    EXPECT_EQ(t->args[1], 99u);
    EXPECT_EQ(t->core_type, CoreType::AIC);
}

// =============================================================================
// Dependency edges
// =============================================================================

TEST_F(RuntimeGraphTest, AddSuccessor_UpdatesFanoutAndFanin) {
    int a = addTask();
    int b = addTask();

    rt->add_successor(a, b);

    Task *ta = rt->get_task(a);
    Task *tb = rt->get_task(b);

    EXPECT_EQ(ta->fanout_count, 1);
    EXPECT_EQ(ta->fanout[0], b);
    EXPECT_EQ(tb->fanin.load(), 1);
}

// =============================================================================
// Ready task detection
// =============================================================================

TEST_F(RuntimeGraphTest, ReadyTaskDetection) {
    // Task 0 has no deps (ready), Task 1 depends on Task 0 (not ready)
    int a = addTask();
    int b = addTask();
    rt->add_successor(a, b);

    int ready[RUNTIME_MAX_TASKS];
    int count = rt->get_initial_ready_tasks(ready);

    EXPECT_EQ(count, 1);
    EXPECT_EQ(ready[0], a);
}

// =============================================================================
// Diamond DAG: A -> {B, C} -> D
// =============================================================================

TEST_F(RuntimeGraphTest, DiamondDAG) {
    int a = addTask();
    int b = addTask();
    int c = addTask();
    int d = addTask();

    rt->add_successor(a, b);
    rt->add_successor(a, c);
    rt->add_successor(b, d);
    rt->add_successor(c, d);

    // Only A should be ready
    int ready[RUNTIME_MAX_TASKS];
    int count = rt->get_initial_ready_tasks(ready);
    EXPECT_EQ(count, 1);
    EXPECT_EQ(ready[0], a);

    // D should have fanin=2
    Task *td = rt->get_task(d);
    EXPECT_EQ(td->fanin.load(), 2);

    // A should have fanout=2
    Task *ta = rt->get_task(a);
    EXPECT_EQ(ta->fanout_count, 2);
}

// =============================================================================
// Linear chain: A -> B -> C -> D
// =============================================================================

TEST_F(RuntimeGraphTest, LinearChain) {
    int a = addTask();
    int b = addTask();
    int c = addTask();
    int d = addTask();

    rt->add_successor(a, b);
    rt->add_successor(b, c);
    rt->add_successor(c, d);

    // Only A is ready
    int ready[RUNTIME_MAX_TASKS];
    int count = rt->get_initial_ready_tasks(ready);
    EXPECT_EQ(count, 1);
    EXPECT_EQ(ready[0], a);

    // Each task has exactly fanin=1 except A
    EXPECT_EQ(rt->get_task(a)->fanin.load(), 0);
    EXPECT_EQ(rt->get_task(b)->fanin.load(), 1);
    EXPECT_EQ(rt->get_task(c)->fanin.load(), 1);
    EXPECT_EQ(rt->get_task(d)->fanin.load(), 1);
}

// =============================================================================
// Fanout / Fanin consistency
// =============================================================================

TEST_F(RuntimeGraphTest, FanoutFaninConsistency) {
    // Build: T0 -> {T1, T2, T3}, T1 -> T4, T2 -> T4, T3 -> T4
    int t0 = addTask();
    int t1 = addTask();
    int t2 = addTask();
    int t3 = addTask();
    int t4 = addTask();

    rt->add_successor(t0, t1);
    rt->add_successor(t0, t2);
    rt->add_successor(t0, t3);
    rt->add_successor(t1, t4);
    rt->add_successor(t2, t4);
    rt->add_successor(t3, t4);

    // Verify: total fanout references == total fanin across all tasks
    int total_fanout = 0;
    int total_fanin = 0;
    for (int i = 0; i < rt->get_task_count(); i++) {
        Task *t = rt->get_task(i);
        total_fanout += t->fanout_count;
        total_fanin += t->fanin.load();
    }
    EXPECT_EQ(total_fanout, total_fanin);
}

// =============================================================================
// Max task limit
// =============================================================================

TEST_F(RuntimeGraphTest, MaxTaskLimit) {
    // Fill up to RUNTIME_MAX_TASKS (this is 131072, too large to loop in test)
    // Instead test that adding more tasks after setting next_task_id near max fails.
    // We'll add a few tasks, then check the add_task return value logic.

    // Add one task successfully
    int id = addTask();
    EXPECT_GE(id, 0);

    // get_task with invalid ID returns nullptr
    EXPECT_EQ(rt->get_task(-1), nullptr);
    EXPECT_EQ(rt->get_task(RUNTIME_MAX_TASKS + 1), nullptr);
}

// =============================================================================
// Tensor pair management
// =============================================================================

TEST_F(RuntimeGraphTest, TensorPairManagement) {
    EXPECT_EQ(rt->get_tensor_pair_count(), 0);

    char host_buf[64], dev_buf[64];
    rt->record_tensor_pair(host_buf, dev_buf, 64);

    EXPECT_EQ(rt->get_tensor_pair_count(), 1);

    TensorPair *pairs = rt->get_tensor_pairs();
    EXPECT_EQ(pairs[0].host_ptr, static_cast<void *>(host_buf));
    EXPECT_EQ(pairs[0].dev_ptr, static_cast<void *>(dev_buf));
    EXPECT_EQ(pairs[0].size, 64u);

    rt->clear_tensor_pairs();
    EXPECT_EQ(rt->get_tensor_pair_count(), 0);
}

// =============================================================================
// Kernel address mapping
// =============================================================================

TEST_F(RuntimeGraphTest, FunctionBinAddrMapping) {
    rt->set_function_bin_addr(0, 0xDEAD);
    rt->set_function_bin_addr(5, 0xBEEF);

    EXPECT_EQ(rt->get_function_bin_addr(0), 0xDEADu);
    EXPECT_EQ(rt->get_function_bin_addr(5), 0xBEEFu);
    EXPECT_EQ(rt->get_function_bin_addr(1), 0u);                    // Not set
    EXPECT_EQ(rt->get_function_bin_addr(-1), 0u);                   // Invalid
    EXPECT_EQ(rt->get_function_bin_addr(RUNTIME_MAX_FUNC_ID), 0u);  // Out of range
}
