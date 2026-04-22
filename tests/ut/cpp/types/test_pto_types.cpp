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
 * Unit tests for Arg and TaskOutputTensors from pto_types.h.
 *
 * Tests argument ordering enforcement, tensor/scalar storage,
 * error propagation, add_scalars_i32 zero-extension, copy_scalars_from,
 * and TaskOutputTensors materialization.
 */

#include <cstdint>
#include <cstring>

#include <gtest/gtest.h>

#include "common.h"
#include "pto_orchestration_api.h"
#include "pto_types.h"

// =============================================================================
// Helpers
// =============================================================================

static Tensor make_test_tensor(void *buf) {
    uint32_t shapes[2] = {4, 8};
    return make_tensor_external(buf, shapes, 2, DataType::FLOAT32);
}

// =============================================================================
// TaskOutputTensors
// =============================================================================

TEST(TaskOutputTensors, InitialState) {
    TaskOutputTensors out;
    EXPECT_TRUE(out.empty());
    EXPECT_EQ(out.size(), 0u);
}

TEST(TaskOutputTensors, MaterializeAddsOne) {
    float buf[4] = {};
    Tensor t = make_test_tensor(buf);

    TaskOutputTensors out;
    out.materialize_output(t);

    EXPECT_FALSE(out.empty());
    EXPECT_EQ(out.size(), 1u);
}

TEST(TaskOutputTensors, GetRefReturnsCorrectTensor) {
    float buf0[4] = {};
    float buf1[4] = {};
    Tensor t0 = make_test_tensor(buf0);
    Tensor t1 = make_test_tensor(buf1);

    TaskOutputTensors out;
    out.materialize_output(t0);
    out.materialize_output(t1);

    EXPECT_EQ(&out.get_ref(0), &t0);
    EXPECT_EQ(&out.get_ref(1), &t1);
    EXPECT_EQ(out.size(), 2u);
}

TEST(TaskOutputTensors, GetRefOutOfRangeThrows) {
    TaskOutputTensors out;
    EXPECT_THROW(out.get_ref(0), AssertionError);
}

TEST(TaskOutputTensors, MaxOutputsFill) {
    float bufs[PTO2_MAX_OUTPUTS] = {};
    std::vector<Tensor> tensors;
    tensors.reserve(PTO2_MAX_OUTPUTS);

    TaskOutputTensors out;
    for (int i = 0; i < PTO2_MAX_OUTPUTS; i++) {
        tensors.push_back(make_test_tensor(&bufs[i]));
        out.materialize_output(tensors.back());
    }
    EXPECT_EQ(out.size(), static_cast<uint32_t>(PTO2_MAX_OUTPUTS));
}

// =============================================================================
// Arg -- initial state
// =============================================================================

TEST(Arg, DefaultState) {
    Arg a;
    EXPECT_FALSE(a.has_error);
    EXPECT_EQ(a.error_msg, nullptr);
    EXPECT_EQ(a.tensor_count(), 0);
    EXPECT_EQ(a.scalar_count(), 0);
}

// =============================================================================
// Arg -- add_input / add_output / add_inout
// =============================================================================

TEST(Arg, AddInput) {
    float buf[4] = {};
    Tensor t = make_test_tensor(buf);
    Arg a;
    a.add_input(t);
    EXPECT_EQ(a.tensor_count(), 1);
    EXPECT_EQ(a.tag(0), TensorArgType::INPUT);
    EXPECT_EQ(a.tensor(0).ptr, &t);
    EXPECT_FALSE(a.has_error);
}

TEST(Arg, AddOutput) {
    uint32_t shapes[2] = {4, 8};
    TensorCreateInfo ci(shapes, 2, DataType::FLOAT32);
    Arg a;
    a.add_output(ci);
    EXPECT_EQ(a.tensor_count(), 1);
    EXPECT_EQ(a.tag(0), TensorArgType::OUTPUT);
    EXPECT_EQ(a.tensor(0).create_info, &ci);
    EXPECT_FALSE(a.has_error);
}

TEST(Arg, AddInout) {
    float buf[4] = {};
    Tensor t = make_test_tensor(buf);
    Arg a;
    a.add_inout(t);
    EXPECT_EQ(a.tensor_count(), 1);
    EXPECT_EQ(a.tag(0), TensorArgType::INOUT);
    EXPECT_EQ(a.tensor(0).ptr, &t);
    EXPECT_FALSE(a.has_error);
}

TEST(Arg, MixedInputOutputInout) {
    float buf_in[4] = {}, buf_inout[4] = {};
    Tensor tin = make_test_tensor(buf_in);
    Tensor tinout = make_test_tensor(buf_inout);
    uint32_t shapes_in[2] = {4, 8};
    TensorCreateInfo ci(shapes_in, 1, DataType::FLOAT32);

    Arg a;
    a.add_input(tin);
    a.add_output(ci);
    a.add_inout(tinout);

    EXPECT_EQ(a.tensor_count(), 3);
    EXPECT_EQ(a.tag(0), TensorArgType::INPUT);
    EXPECT_EQ(a.tag(1), TensorArgType::OUTPUT);
    EXPECT_EQ(a.tag(2), TensorArgType::INOUT);
    EXPECT_FALSE(a.has_error);
}

// =============================================================================
// Arg -- ordering enforcement: tensor after scalar sets error
// =============================================================================

TEST(Arg, TensorAfterScalarSetsError) {
    float buf[4] = {};
    Tensor t = make_test_tensor(buf);
    Arg a;
    a.add_scalar(uint64_t(42));
    a.add_input(t);  // invalid: tensor after scalar
    EXPECT_TRUE(a.has_error);
    EXPECT_NE(a.error_msg, nullptr);
    // The scalar was recorded, the tensor was not
    EXPECT_EQ(a.tensor_count(), 0);
    EXPECT_EQ(a.scalar_count(), 1);
}

TEST(Arg, OutputAfterScalarSetsError) {
    uint32_t shapes_in[2] = {4, 8};
    TensorCreateInfo ci(shapes_in, 1, DataType::FLOAT32);
    Arg a;
    a.add_scalar(uint64_t(1));
    a.add_output(ci);
    EXPECT_TRUE(a.has_error);
    EXPECT_EQ(a.tensor_count(), 0);
}

TEST(Arg, InoutAfterScalarSetsError) {
    float buf[4] = {};
    Tensor t = make_test_tensor(buf);
    Arg a;
    a.add_scalar(uint64_t(1));
    a.add_inout(t);
    EXPECT_TRUE(a.has_error);
    EXPECT_EQ(a.tensor_count(), 0);
}

// =============================================================================
// Arg -- capacity limits
// =============================================================================

TEST(Arg, TensorCapacityExceeded) {
    Arg a;
    for (int i = 0; i < MAX_TENSOR_ARGS; i++) {
        float dummy = 0.0f;
        Tensor t = make_test_tensor(&dummy);
        a.add_input(t);
        ASSERT_FALSE(a.has_error) << "Failed at tensor " << i;
    }
    // One more should trigger the error
    float extra = 0.0f;
    Tensor t_extra = make_test_tensor(&extra);
    a.add_input(t_extra);
    EXPECT_TRUE(a.has_error);
    EXPECT_EQ(a.tensor_count(), MAX_TENSOR_ARGS);
}

TEST(Arg, ScalarCapacityExceeded) {
    Arg a;
    for (int i = 0; i < MAX_SCALAR_ARGS; i++) {
        a.add_scalar(static_cast<uint64_t>(i));
        ASSERT_FALSE(a.has_error) << "Failed at scalar " << i;
    }
    a.add_scalar(uint64_t(999));
    EXPECT_TRUE(a.has_error);
    EXPECT_EQ(a.scalar_count(), MAX_SCALAR_ARGS);
}

// =============================================================================
// Arg -- add_scalar with various types
// =============================================================================

TEST(Arg, AddScalarUint64) {
    Arg a;
    a.add_scalar(uint64_t(0xDEADBEEFCAFEBABEULL));
    EXPECT_EQ(a.scalar_count(), 1);
    EXPECT_EQ(a.scalar(0), 0xDEADBEEFCAFEBABEULL);
    EXPECT_FALSE(a.has_error);
}

TEST(Arg, AddScalarFloat) {
    Arg a;
    float v = 3.14f;
    a.add_scalar(v);
    EXPECT_EQ(a.scalar_count(), 1);
    EXPECT_EQ(a.scalar(0), to_u64(v));
    EXPECT_FALSE(a.has_error);
}

TEST(Arg, AddScalarInt32) {
    Arg a;
    int32_t v = -7;
    a.add_scalar(v);
    EXPECT_EQ(a.scalar_count(), 1);
    EXPECT_EQ(a.scalar(0), to_u64(v));
    EXPECT_FALSE(a.has_error);
}

// =============================================================================
// Arg -- add_scalars (batch uint64)
// =============================================================================

TEST(Arg, AddScalarsBatch) {
    Arg a;
    uint64_t vals[3] = {10, 20, 30};
    a.add_scalars(vals, 3);
    EXPECT_EQ(a.scalar_count(), 3);
    EXPECT_EQ(a.scalar(0), 10u);
    EXPECT_EQ(a.scalar(1), 20u);
    EXPECT_EQ(a.scalar(2), 30u);
    EXPECT_FALSE(a.has_error);
}

TEST(Arg, AddScalarsBatchOverCapacitySetsError) {
    Arg a;
    // Fill to capacity minus 1
    for (int i = 0; i < MAX_SCALAR_ARGS - 1; i++) {
        a.add_scalar(uint64_t(i));
    }
    // Batch of 3 would overflow by 2
    uint64_t vals[3] = {1, 2, 3};
    a.add_scalars(vals, 3);
    EXPECT_TRUE(a.has_error);
}

// =============================================================================
// Arg -- add_scalars_i32 (zero-extension)
// =============================================================================

TEST(Arg, AddScalarsI32ZeroExtends) {
    Arg a;
    int32_t vals[4] = {0, 1, -1, 0x7FFFFFFF};
    a.add_scalars_i32(vals, 4);
    EXPECT_EQ(a.scalar_count(), 4);
    EXPECT_EQ(a.scalar(0), uint64_t(0));
    EXPECT_EQ(a.scalar(1), uint64_t(1));
    // -1 as int32 is 0xFFFFFFFF; zero-extended to uint64 is 0x00000000FFFFFFFF
    EXPECT_EQ(a.scalar(2), uint64_t(0x00000000FFFFFFFFull));
    EXPECT_EQ(a.scalar(3), uint64_t(0x000000007FFFFFFFull));
    EXPECT_FALSE(a.has_error);
}

TEST(Arg, AddScalarsI32NegativeValues) {
    Arg a;
    int32_t vals[2] = {-1, -2};
    a.add_scalars_i32(vals, 2);
    // -1 -> 0xFFFFFFFF zero-extended -> 0x00000000FFFFFFFF
    // -2 -> 0xFFFFFFFE zero-extended -> 0x00000000FFFFFFFE
    EXPECT_EQ(a.scalar(0), uint64_t(0xFFFFFFFFull));
    EXPECT_EQ(a.scalar(1), uint64_t(0xFFFFFFFEull));
}

TEST(Arg, AddScalarsI32SingleElement) {
    Arg a;
    int32_t v = 42;
    a.add_scalars_i32(&v, 1);
    EXPECT_EQ(a.scalar_count(), 1);
    EXPECT_EQ(a.scalar(0), uint64_t(42));
}

TEST(Arg, AddScalarsI32OverCapacitySetsError) {
    Arg a;
    for (int i = 0; i < MAX_SCALAR_ARGS - 1; i++) {
        a.add_scalar(uint64_t(i));
    }
    int32_t vals[3] = {1, 2, 3};
    a.add_scalars_i32(vals, 3);
    EXPECT_TRUE(a.has_error);
}

// =============================================================================
// Arg -- copy_scalars_from
// =============================================================================

TEST(Arg, CopyScalarsFrom) {
    Arg src;
    src.add_scalar(uint64_t(10));
    src.add_scalar(uint64_t(20));
    src.add_scalar(uint64_t(30));

    Arg dst;
    dst.copy_scalars_from(src, 1, 2);  // copy scalars[1..2] = {20, 30}
    EXPECT_EQ(dst.scalar_count(), 2);
    EXPECT_EQ(dst.scalar(0), uint64_t(20));
    EXPECT_EQ(dst.scalar(1), uint64_t(30));
    EXPECT_FALSE(dst.has_error);
}

TEST(Arg, CopyScalarsFromOutOfBoundsSetsError) {
    Arg src;
    src.add_scalar(uint64_t(1));

    Arg dst;
    dst.copy_scalars_from(src, 0, 5);  // only 1 scalar available, request 5
    EXPECT_TRUE(dst.has_error);
}

TEST(Arg, CopyScalarsFromFull) {
    Arg src;
    for (int i = 0; i < MAX_SCALAR_ARGS; i++) {
        src.add_scalar(static_cast<uint64_t>(i));
    }
    Arg dst;
    for (int i = 0; i < MAX_SCALAR_ARGS - 1; i++) {
        dst.add_scalar(uint64_t(0));
    }
    // dst has MAX-1 scalars; copying 2 from src would overflow
    dst.copy_scalars_from(src, 0, 2);
    EXPECT_TRUE(dst.has_error);
}

// =============================================================================
// Arg -- reset clears all state
// =============================================================================

TEST(Arg, ResetClearsAll) {
    float buf[4] = {};
    Tensor t = make_test_tensor(buf);
    Arg a;
    a.add_input(t);
    a.add_scalar(uint64_t(99));
    a.set_error("deliberate error");

    a.reset();
    EXPECT_EQ(a.tensor_count(), 0);
    EXPECT_EQ(a.scalar_count(), 0);
    EXPECT_FALSE(a.has_error);
    EXPECT_EQ(a.error_msg, nullptr);
}

// =============================================================================
// Arg -- set_error is idempotent (first error wins)
// =============================================================================

TEST(Arg, SetErrorFirstWins) {
    Arg a;
    a.set_error("first");
    a.set_error("second");
    EXPECT_STREQ(a.error_msg, "first");
}

// =============================================================================
// Arg -- launch_spec default
// =============================================================================

TEST(Arg, LaunchSpecDefaultBlockNum) {
    Arg a;
    EXPECT_EQ(a.launch_spec.block_num(), 1);
}
