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
 * Unit tests for PTO2DispatchPayload and SPMD context structures.
 *
 * Tests layout constants, alignment, static_assert consistency, and the
 * get_block_idx / get_block_num / get_sub_block_id intrinsic accessors.
 */

#include <cstdint>

#include <gtest/gtest.h>

#include "intrinsic.h"
#include "pto2_dispatch_payload.h"
#include "pto_types.h"

// =============================================================================
// Compile-time constant consistency
// =============================================================================

TEST(DispatchPayloadConstants, LocalContextIndex) {
    // SPMD_LOCAL_CONTEXT_INDEX must equal MAX_TENSOR_ARGS + MAX_SCALAR_ARGS
    EXPECT_EQ(SPMD_LOCAL_CONTEXT_INDEX, MAX_TENSOR_ARGS + MAX_SCALAR_ARGS);
}

TEST(DispatchPayloadConstants, GlobalContextIndex) {
    EXPECT_EQ(SPMD_GLOBAL_CONTEXT_INDEX, SPMD_LOCAL_CONTEXT_INDEX + 1);
}

TEST(DispatchPayloadConstants, ExtParamsCount) { EXPECT_EQ(PTO2_EXT_PARAMS_COUNT, 2); }

TEST(DispatchPayloadConstants, DispatchMaxArgs) {
    EXPECT_EQ(PTO2_DISPATCH_MAX_ARGS, MAX_TENSOR_ARGS + MAX_SCALAR_ARGS + PTO2_EXT_PARAMS_COUNT);
}

// =============================================================================
// PTO2DispatchPayload layout and alignment
// =============================================================================

// ABI contract: alignment must match device dispatch requirements.
TEST(DispatchPayloadLayout, IsAlignedTo64Bytes) { EXPECT_EQ(alignof(PTO2DispatchPayload), 64u); }

TEST(DispatchPayloadLayout, ArgsArrayHasCorrectSize) {
    PTO2DispatchPayload p{};
    EXPECT_EQ(sizeof(p.args) / sizeof(p.args[0]), static_cast<size_t>(PTO2_DISPATCH_MAX_ARGS));
}

// ABI contract: element size must match shared memory layout.
TEST(DispatchPayloadLayout, ArgElementIs8Bytes) {
    PTO2DispatchPayload p{};
    EXPECT_EQ(sizeof(p.args[0]), 8u);
}

// =============================================================================
// LocalContext
// =============================================================================

TEST(LocalContext, FieldsReadWrite) {
    LocalContext lctx{3, 8};
    EXPECT_EQ(lctx.block_idx, 3);
    EXPECT_EQ(lctx.block_num, 8);
}

TEST(LocalContext, DefaultZero) {
    LocalContext lctx{};
    EXPECT_EQ(lctx.block_idx, 0);
    EXPECT_EQ(lctx.block_num, 0);
}

// =============================================================================
// GlobalContext
// =============================================================================

TEST(GlobalContext, FieldReadWrite) {
    GlobalContext gctx{1};
    EXPECT_EQ(gctx.sub_block_id, 1);
}

// =============================================================================
// Intrinsic accessor functions
// =============================================================================

// Build a minimal args[] array with context pointers at the correct indices.
struct IntrinsicTestSetup {
    static constexpr int kArgsLen = SPMD_GLOBAL_CONTEXT_INDEX + 1;
    LocalContext lctx;
    GlobalContext gctx;
    uint64_t args[kArgsLen];

    IntrinsicTestSetup(int block_idx, int block_num, int sub_block_id) :
        lctx{block_idx, block_num},
        gctx{sub_block_id} {
        for (auto &a : args)
            a = 0;
        args[SPMD_LOCAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&lctx);
        args[SPMD_GLOBAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&gctx);
    }

    int64_t *raw() { return reinterpret_cast<int64_t *>(args); }
};

TEST(IntrinsicAccessors, GetBlockIdx) {
    IntrinsicTestSetup s(5, 10, 0);
    EXPECT_EQ(get_block_idx(s.raw()), 5);
}

TEST(IntrinsicAccessors, GetBlockNum) {
    IntrinsicTestSetup s(0, 7, 0);
    EXPECT_EQ(get_block_num(s.raw()), 7);
}

TEST(IntrinsicAccessors, GetSubBlockId_AIV0) {
    IntrinsicTestSetup s(0, 1, 0);
    EXPECT_EQ(get_sub_block_id(s.raw()), 0);
}

TEST(IntrinsicAccessors, GetSubBlockId_AIV1) {
    IntrinsicTestSetup s(0, 1, 1);
    EXPECT_EQ(get_sub_block_id(s.raw()), 1);
}

TEST(IntrinsicAccessors, BlockIdxAndNumIndependent) {
    // Changing block_idx must not affect block_num and vice versa
    IntrinsicTestSetup s(2, 4, 0);
    EXPECT_EQ(get_block_idx(s.raw()), 2);
    EXPECT_EQ(get_block_num(s.raw()), 4);

    s.lctx.block_idx = 3;
    EXPECT_EQ(get_block_idx(s.raw()), 3);
    EXPECT_EQ(get_block_num(s.raw()), 4);
}

TEST(IntrinsicAccessors, ContextPointersAreAtCorrectSlots) {
    IntrinsicTestSetup s(1, 2, 0);
    // The value at SPMD_LOCAL_CONTEXT_INDEX must point to lctx
    auto lctx_ptr = reinterpret_cast<LocalContext *>(static_cast<uint64_t>(s.args[SPMD_LOCAL_CONTEXT_INDEX]));
    EXPECT_EQ(lctx_ptr, &s.lctx);

    auto gctx_ptr = reinterpret_cast<GlobalContext *>(static_cast<uint64_t>(s.args[SPMD_GLOBAL_CONTEXT_INDEX]));
    EXPECT_EQ(gctx_ptr, &s.gctx);
}
