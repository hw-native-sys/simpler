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
 * PTO Submit Types - Shared submit-contract definitions
 *
 * Header-only definitions shared by orchestration-facing and runtime-facing
 * headers. Keeps orchestration slim (no dependency on pto_runtime2_types.h).
 */

#pragma once

#include <stdint.h>

inline constexpr int32_t INVALID_KERNEL_ID = -1;

/**
 * Subtask slot count: AIC, AIV0, AIV1
 */
inline constexpr int32_t PTO2_SUBTASK_SLOT_COUNT = 3;

/**
 * Subtask slot indices
 */
enum class PTO2SubtaskSlot : uint8_t {
    AIC = 0,
    AIV0 = 1,
    AIV1 = 2,
};

/**
 * Subtask mask bits (for active_mask)
 */
inline constexpr uint8_t PTO2_SUBTASK_MASK_AIC = (1u << 0);         // 0x1
inline constexpr uint8_t PTO2_SUBTASK_MASK_AIV0 = (1u << 1);        // 0x2
inline constexpr uint8_t PTO2_SUBTASK_MASK_AIV1 = (1u << 2);        // 0x4
inline constexpr uint8_t PTO2_SUBTASK_FLAG_SYNC_START = (1u << 3);  // 0x8: all blocks must launch atomically

/**
 * Test whether a subtask slot is active in a given mask
 */
static inline bool pto2_subtask_active(uint8_t mask, PTO2SubtaskSlot slot) {
    return (mask & (1u << static_cast<uint8_t>(slot))) != 0;
}

/**
 * Extract only the core bits from active_mask (strips flag bits).
 */
static inline uint8_t pto2_core_mask(uint8_t active_mask) { return active_mask & 0x07u; }

/**
 * Check whether a task requires all blocks to be launched atomically.
 */
static inline bool pto2_requires_sync_start(uint8_t active_mask) {
    return (active_mask & PTO2_SUBTASK_FLAG_SYNC_START) != 0;
}

/**
 * Mixed-task submit contract.
 *
 * Each field holds either a valid kernel ID or INVALID_KERNEL_ID (inactive).
 * At least one slot must be valid.
 */
struct MixedKernels {
    int32_t aic_kernel_id{INVALID_KERNEL_ID};
    int32_t aiv0_kernel_id{INVALID_KERNEL_ID};
    int32_t aiv1_kernel_id{INVALID_KERNEL_ID};
};

/**
 * Resource shape — classifies a MixedKernels into one of 3 scheduling buckets.
 *
 * Multi-subtask tasks (2+ active slots) are all scheduled as MIX, which
 * requires a fully-idle cluster (1 AIC + 2 AIV).  The actual cores used
 * are determined at dispatch time by active_mask — unused cores in the
 * cluster remain idle and available for single-core tasks.
 */
enum class PTO2ResourceShape : uint8_t {
    AIC = 0,  // Single AIC
    AIV = 1,  // Single AIV
    MIX = 2,  // Full cluster (dispatch uses active_mask)
};

inline constexpr int32_t PTO2_NUM_RESOURCE_SHAPES = 3;

/**
 * Derive resource shape from active_mask.
 * Caller must ensure active_mask is valid (at least one bit set).
 */
static inline PTO2ResourceShape pto2_active_mask_to_shape(uint8_t active_mask) {
    uint8_t core_mask = pto2_core_mask(active_mask);
    int bit_count = __builtin_popcount(core_mask);
    if (bit_count >= 2) return PTO2ResourceShape::MIX;
    if (core_mask & PTO2_SUBTASK_MASK_AIC) return PTO2ResourceShape::AIC;
    return PTO2ResourceShape::AIV;
}

/**
 * Compute active_mask from MixedKernels.
 */
static inline uint8_t pto2_mixed_kernels_to_active_mask(const MixedKernels &mk) {
    uint8_t mask = 0;
    if (mk.aic_kernel_id != INVALID_KERNEL_ID) mask |= PTO2_SUBTASK_MASK_AIC;
    if (mk.aiv0_kernel_id != INVALID_KERNEL_ID) mask |= PTO2_SUBTASK_MASK_AIV0;
    if (mk.aiv1_kernel_id != INVALID_KERNEL_ID) mask |= PTO2_SUBTASK_MASK_AIV1;
    return mask;
}

/**
 * SPMD launch parameters carried inside Arg.
 *
 * Controls how many logical blocks (SPMD dimension) a single task
 * is expanded into at dispatch time.  Each block receives a unique
 * block_idx in [0, block_num) via the per-dispatch LocalContext.
 */
class PTO2LaunchSpec {
public:
    constexpr PTO2LaunchSpec() = default;

    int16_t block_num() const { return block_num_; }
    void set_block_num(int16_t n) { block_num_ = n; }

    bool require_sync_start() const { return require_sync_start_; }
    void set_require_sync_start(bool v) { require_sync_start_ = v; }

private:
    int16_t block_num_{1};
    bool require_sync_start_{false};
};
