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
 * Unit tests for pto_submit_types.h
 *
 * Tests submit contract types: subtask masks, resource shapes,
 * active mask derivation, and launch spec.
 */

#include <gtest/gtest.h>

#include "pto_submit_types.h"

// =============================================================================
// pto2_subtask_active
// =============================================================================

TEST(SubtaskActive, AICMaskActivatesAICSlot) {
    EXPECT_TRUE(pto2_subtask_active(PTO2_SUBTASK_MASK_AIC, PTO2SubtaskSlot::AIC));
    EXPECT_FALSE(pto2_subtask_active(PTO2_SUBTASK_MASK_AIC, PTO2SubtaskSlot::AIV0));
    EXPECT_FALSE(pto2_subtask_active(PTO2_SUBTASK_MASK_AIC, PTO2SubtaskSlot::AIV1));
}

TEST(SubtaskActive, AIV0MaskActivatesAIV0Slot) {
    EXPECT_FALSE(pto2_subtask_active(PTO2_SUBTASK_MASK_AIV0, PTO2SubtaskSlot::AIC));
    EXPECT_TRUE(pto2_subtask_active(PTO2_SUBTASK_MASK_AIV0, PTO2SubtaskSlot::AIV0));
    EXPECT_FALSE(pto2_subtask_active(PTO2_SUBTASK_MASK_AIV0, PTO2SubtaskSlot::AIV1));
}

TEST(SubtaskActive, AIV1MaskActivatesAIV1Slot) {
    EXPECT_FALSE(pto2_subtask_active(PTO2_SUBTASK_MASK_AIV1, PTO2SubtaskSlot::AIC));
    EXPECT_FALSE(pto2_subtask_active(PTO2_SUBTASK_MASK_AIV1, PTO2SubtaskSlot::AIV0));
    EXPECT_TRUE(pto2_subtask_active(PTO2_SUBTASK_MASK_AIV1, PTO2SubtaskSlot::AIV1));
}

TEST(SubtaskActive, CombinedMask) {
    uint8_t mask = PTO2_SUBTASK_MASK_AIC | PTO2_SUBTASK_MASK_AIV1;
    EXPECT_TRUE(pto2_subtask_active(mask, PTO2SubtaskSlot::AIC));
    EXPECT_FALSE(pto2_subtask_active(mask, PTO2SubtaskSlot::AIV0));
    EXPECT_TRUE(pto2_subtask_active(mask, PTO2SubtaskSlot::AIV1));
}

TEST(SubtaskActive, AllActive) {
    uint8_t mask = PTO2_SUBTASK_MASK_AIC | PTO2_SUBTASK_MASK_AIV0 | PTO2_SUBTASK_MASK_AIV1;
    EXPECT_TRUE(pto2_subtask_active(mask, PTO2SubtaskSlot::AIC));
    EXPECT_TRUE(pto2_subtask_active(mask, PTO2SubtaskSlot::AIV0));
    EXPECT_TRUE(pto2_subtask_active(mask, PTO2SubtaskSlot::AIV1));
}

// =============================================================================
// pto2_active_mask_to_shape
// =============================================================================

TEST(ActiveMaskToShape, SingleAIC) {
    EXPECT_EQ(pto2_active_mask_to_shape(PTO2_SUBTASK_MASK_AIC), PTO2ResourceShape::AIC);
}

TEST(ActiveMaskToShape, SingleAIV0) {
    EXPECT_EQ(pto2_active_mask_to_shape(PTO2_SUBTASK_MASK_AIV0), PTO2ResourceShape::AIV);
}

TEST(ActiveMaskToShape, SingleAIV1) {
    EXPECT_EQ(pto2_active_mask_to_shape(PTO2_SUBTASK_MASK_AIV1), PTO2ResourceShape::AIV);
}

TEST(ActiveMaskToShape, TwoActiveBecomesMIX) {
    uint8_t mask = PTO2_SUBTASK_MASK_AIC | PTO2_SUBTASK_MASK_AIV0;
    EXPECT_EQ(pto2_active_mask_to_shape(mask), PTO2ResourceShape::MIX);
}

TEST(ActiveMaskToShape, AllThreeBecomesMIX) {
    uint8_t mask = PTO2_SUBTASK_MASK_AIC | PTO2_SUBTASK_MASK_AIV0 | PTO2_SUBTASK_MASK_AIV1;
    EXPECT_EQ(pto2_active_mask_to_shape(mask), PTO2ResourceShape::MIX);
}

// =============================================================================
// pto2_mixed_kernels_to_active_mask
// =============================================================================

TEST(MixedKernelsToMask, AllInvalid) {
    MixedKernels mk;
    EXPECT_EQ(pto2_mixed_kernels_to_active_mask(mk), 0);
}

TEST(MixedKernelsToMask, AICOnly) {
    MixedKernels mk;
    mk.aic_kernel_id = 42;
    EXPECT_EQ(pto2_mixed_kernels_to_active_mask(mk), PTO2_SUBTASK_MASK_AIC);
}

TEST(MixedKernelsToMask, AIV0Only) {
    MixedKernels mk;
    mk.aiv0_kernel_id = 7;
    EXPECT_EQ(pto2_mixed_kernels_to_active_mask(mk), PTO2_SUBTASK_MASK_AIV0);
}

TEST(MixedKernelsToMask, AllValid) {
    MixedKernels mk;
    mk.aic_kernel_id = 1;
    mk.aiv0_kernel_id = 2;
    mk.aiv1_kernel_id = 3;
    uint8_t expected = PTO2_SUBTASK_MASK_AIC | PTO2_SUBTASK_MASK_AIV0 | PTO2_SUBTASK_MASK_AIV1;
    EXPECT_EQ(pto2_mixed_kernels_to_active_mask(mk), expected);
}

// =============================================================================
// MixedKernels defaults
// =============================================================================

TEST(MixedKernels, DefaultsAreInvalid) {
    MixedKernels mk;
    EXPECT_EQ(mk.aic_kernel_id, INVALID_KERNEL_ID);
    EXPECT_EQ(mk.aiv0_kernel_id, INVALID_KERNEL_ID);
    EXPECT_EQ(mk.aiv1_kernel_id, INVALID_KERNEL_ID);
}

// =============================================================================
// PTO2LaunchSpec
// =============================================================================

TEST(LaunchSpec, DefaultBlockNumIsOne) {
    PTO2LaunchSpec spec;
    EXPECT_EQ(spec.block_num(), 1);
}

TEST(LaunchSpec, SetAndGet) {
    PTO2LaunchSpec spec;
    spec.set_block_num(4);
    EXPECT_EQ(spec.block_num(), 4);
}

// =============================================================================
// Constants
// =============================================================================

TEST(Constants, SubtaskSlotCount) { EXPECT_EQ(PTO2_SUBTASK_SLOT_COUNT, 3); }

TEST(Constants, NumResourceShapes) { EXPECT_EQ(PTO2_NUM_RESOURCE_SHAPES, 3); }

TEST(Constants, InvalidKernelId) { EXPECT_EQ(INVALID_KERNEL_ID, -1); }
