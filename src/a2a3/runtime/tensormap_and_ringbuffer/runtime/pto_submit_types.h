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
 * Subtask mask bits (for ActiveMask)
 */
inline constexpr uint8_t PTO2_SUBTASK_MASK_AIC = (1u << 0);         // 0x1
inline constexpr uint8_t PTO2_SUBTASK_MASK_AIV0 = (1u << 1);        // 0x2
inline constexpr uint8_t PTO2_SUBTASK_MASK_AIV1 = (1u << 2);        // 0x4
inline constexpr uint8_t PTO2_SUBTASK_FLAG_SYNC_START = (1u << 3);  // 0x8: all blocks must launch atomically

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
 * Bitmask of active subtask slots + flags, sizeof == 1.
 */
class ActiveMask {
public:
    constexpr ActiveMask() = default;
    constexpr explicit ActiveMask(uint8_t raw) :
        raw_(raw) {}

    uint8_t raw() const { return raw_; }

    bool subtask_active(PTO2SubtaskSlot slot) const { return (raw_ & (1u << static_cast<uint8_t>(slot))) != 0; }

    uint8_t core_mask() const { return raw_ & 0x07u; }

    bool requires_sync_start() const { return (raw_ & PTO2_SUBTASK_FLAG_SYNC_START) != 0; }

    PTO2ResourceShape to_shape() const {
        uint8_t cmask = core_mask();
        int bit_count = __builtin_popcount(cmask);
        if (bit_count >= 2) return PTO2ResourceShape::MIX;
        if (cmask & PTO2_SUBTASK_MASK_AIC) return PTO2ResourceShape::AIC;
        return PTO2ResourceShape::AIV;
    }

    void set_sync_start() { raw_ |= PTO2_SUBTASK_FLAG_SYNC_START; }

    bool operator==(ActiveMask other) const { return raw_ == other.raw_; }
    bool operator!=(ActiveMask other) const { return raw_ != other.raw_; }

    ActiveMask operator|(ActiveMask other) const { return ActiveMask(raw_ | other.raw_); }
    ActiveMask &operator|=(ActiveMask other) {
        raw_ |= other.raw_;
        return *this;
    }

    ActiveMask operator&(uint8_t mask) const { return ActiveMask(raw_ & mask); }

    bool has_mask(uint8_t mask) const { return (raw_ & mask) != 0; }

    explicit operator bool() const { return raw_ != 0; }

private:
    uint8_t raw_{0};
};

static_assert(sizeof(ActiveMask) == 1, "ActiveMask must be exactly 1 byte");

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

    ActiveMask to_active_mask() const {
        uint8_t mask = 0;
        if (aic_kernel_id != INVALID_KERNEL_ID) mask |= PTO2_SUBTASK_MASK_AIC;
        if (aiv0_kernel_id != INVALID_KERNEL_ID) mask |= PTO2_SUBTASK_MASK_AIV0;
        if (aiv1_kernel_id != INVALID_KERNEL_ID) mask |= PTO2_SUBTASK_MASK_AIV1;
        return ActiveMask(mask);
    }
};

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
