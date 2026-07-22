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

#include <cstdint>

#include "pto_orchestration_api.h"

namespace {

PTO2Runtime *g_runtime = nullptr;
uint64_t g_graph_key = 0;
int32_t g_tensor_count = 0;
int32_t g_scalar_count = 0;
TensorArgType g_tensor_types[PTO2_GRAPH_MAX_TENSOR_ARGS]{};
uint64_t g_scalars[PTO2_GRAPH_MAX_SCALAR_ARGS]{};
uint16_t g_scalar_source = PTO2_TASK_ARG_STATIC;
int32_t g_build_count = 0;
int32_t g_end_count = 0;
int32_t g_begin_count = 0;
PTO2GraphScopeResult g_scope_result{.execute_block = true, .recording = true};

bool is_fatal(PTO2Runtime *) { return false; }

PTO2GraphScopeResult graph_begin(PTO2Runtime *, uint64_t graph_key, const L2TaskArgs &args) {
    g_begin_count++;
    g_graph_key = graph_key;
    g_tensor_count = args.tensor_count();
    g_scalar_count = args.scalar_count();
    for (int32_t i = 0; i < g_tensor_count; ++i)
        g_tensor_types[i] = args.tag(i);
    for (int32_t i = 0; i < g_scalar_count; ++i)
        g_scalars[i] = args.scalar(i);
    return g_scope_result;
}

void graph_end(PTO2Runtime *) { g_end_count++; }

void graph_definition(const L2TaskArgs &args) {
    g_build_count++;
    L0TaskArgs node_args;
    node_args.add_scalar(args.scalar(0));
    g_scalar_source = node_args.scalar_source_index(0);
}

void empty_definition(const L2TaskArgs &) { g_build_count++; }

}  // namespace

extern "C" PTO2Runtime *framework_current_runtime() { return g_runtime; }

extern "C" void framework_bind_runtime(PTO2Runtime *rt) { g_runtime = rt; }

TEST(ReplayGraphApi, UsesFunctionIdentityAndL2TaskArgs) {
    PTO2RuntimeOps ops{};
    ops.is_fatal = is_fatal;
    ops.graph_begin = graph_begin;
    ops.graph_end = graph_end;
    PTO2Runtime runtime{.ops = &ops};
    framework_bind_runtime(&runtime);

    uint32_t shape[] = {4};
    uint16_t input_storage[4]{};
    uint16_t output_storage[4]{};
    Tensor input = make_tensor_external(input_storage, shape, 1, DataType::FLOAT16);
    Tensor output = make_tensor_external(output_storage, shape, 1, DataType::FLOAT16);
    L2TaskArgs args;
    args.add_input(input);
    args.add_output(output);
    args.add_scalar(uint64_t{7});

    g_build_count = 0;
    g_end_count = 0;
    g_begin_count = 0;
    g_scope_result = PTO2GraphScopeResult{.execute_block = true, .recording = true};
    rt_submit_graph(&graph_definition, args);

    EXPECT_EQ(g_build_count, 1);
    EXPECT_EQ(g_end_count, 1);
    ASSERT_EQ(g_tensor_count, 2);
    ASSERT_EQ(g_scalar_count, 1);
    EXPECT_EQ(g_tensor_types[0], TensorArgType::INPUT);
    EXPECT_EQ(g_tensor_types[1], TensorArgType::OUTPUT_EXISTING);
    EXPECT_EQ(g_scalars[0], 7u);
    EXPECT_EQ(g_scalar_source, 0u);

    uint64_t pointer_key = g_graph_key;
    g_scope_result = PTO2GraphScopeResult{.execute_block = false, .recording = false};
    rt_submit_graph(&graph_definition, args);
    EXPECT_EQ(g_build_count, 1);
    EXPECT_EQ(g_end_count, 1);

    rt_submit_graph(uint64_t{0x1234}, &graph_definition, args);
    EXPECT_NE(g_graph_key, pointer_key);

    framework_bind_runtime(nullptr);
}

TEST(ReplayGraphApi, AllocationOutputFallsBackWithoutReadingTensorCreateInfoAsTensor) {
    uint32_t shape[] = {4};
    TensorCreateInfo create_info(shape, 1, DataType::FLOAT16);
    L2TaskArgs args;
    args.add_output(create_info);

    PTO2RuntimeOps ops{};
    ops.is_fatal = is_fatal;
    ops.graph_begin = graph_begin;
    ops.graph_end = graph_end;
    PTO2Runtime runtime{.ops = &ops};
    framework_bind_runtime(&runtime);
    g_begin_count = 0;
    g_build_count = 0;

    rt_submit_graph(&empty_definition, args);

    EXPECT_EQ(g_begin_count, 0);
    EXPECT_EQ(g_build_count, 1);
    framework_bind_runtime(nullptr);
}
