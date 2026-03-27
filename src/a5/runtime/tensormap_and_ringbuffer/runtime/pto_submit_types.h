/**
 * PTO Submit Types - Shared submit-contract definitions
 *
 * Header-only definitions shared by orchestration-facing and runtime-facing
 * headers. Keeps orchestration slim (no dependency on pto_runtime2_types.h).
 */

#ifndef PTO_SUBMIT_TYPES_H
#define PTO_SUBMIT_TYPES_H

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
 * Subtask mask bits (for active_mask / subtask_done_mask)
 */
inline constexpr uint8_t PTO2_SUBTASK_MASK_AIC = (1u << 0);   // 0x1
inline constexpr uint8_t PTO2_SUBTASK_MASK_AIV0 = (1u << 1);  // 0x2
inline constexpr uint8_t PTO2_SUBTASK_MASK_AIV1 = (1u << 2);  // 0x4

/**
 * Test whether a subtask slot is active in a given mask
 */
static inline bool pto2_subtask_active(uint8_t mask, PTO2SubtaskSlot slot) {
    return (mask & (1u << static_cast<uint8_t>(slot))) != 0;
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
 * Resource shape — classifies a MixedKernels into one of 5 queue buckets.
 */
enum class PTO2ResourceShape : uint8_t {
    AIC_ONLY = 0,    // AIC only
    AIV_X1 = 1,      // One AIV slot
    AIV_X2 = 2,      // Both AIV slots
    AIC_AIV_X1 = 3,  // AIC + one AIV
    AIC_AIV_X2 = 4,  // AIC + both AIV
};

inline constexpr int32_t PTO2_NUM_RESOURCE_SHAPES = 5;

/**
 * Derive resource shape from active_mask.
 * Caller must ensure active_mask is valid (at least one bit set).
 */
static inline PTO2ResourceShape pto2_active_mask_to_shape(uint8_t active_mask) {
    bool has_aic = (active_mask & PTO2_SUBTASK_MASK_AIC) != 0;
    int aiv_count = ((active_mask & PTO2_SUBTASK_MASK_AIV0) != 0) + ((active_mask & PTO2_SUBTASK_MASK_AIV1) != 0);

    if (has_aic) {
        if (aiv_count == 0) return PTO2ResourceShape::AIC_ONLY;
        if (aiv_count == 1) return PTO2ResourceShape::AIC_AIV_X1;
        return PTO2ResourceShape::AIC_AIV_X2;
    }
    if (aiv_count == 1) return PTO2ResourceShape::AIV_X1;
    return PTO2ResourceShape::AIV_X2;
}

/**
 * Compute active_mask from MixedKernels.
 */
static inline uint8_t pto2_mixed_kernels_to_active_mask(const MixedKernels& mk) {
    uint8_t mask = 0;
    if (mk.aic_kernel_id != INVALID_KERNEL_ID) mask |= PTO2_SUBTASK_MASK_AIC;
    if (mk.aiv0_kernel_id != INVALID_KERNEL_ID) mask |= PTO2_SUBTASK_MASK_AIV0;
    if (mk.aiv1_kernel_id != INVALID_KERNEL_ID) mask |= PTO2_SUBTASK_MASK_AIV1;
    return mask;
}

#endif  // PTO_SUBMIT_TYPES_H
