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
inline constexpr uint8_t PTO2_SUBTASK_MASK_AIC = (1u << 0);   // 0x1
inline constexpr uint8_t PTO2_SUBTASK_MASK_AIV0 = (1u << 1);  // 0x2
inline constexpr uint8_t PTO2_SUBTASK_MASK_AIV1 = (1u << 2);  // 0x4

// Dispatch-predicate comparison operator. The scheduler evaluates the predicate
// at the dispatch point — the task is ready (fanin satisfied), so the predicate
// address' producer has completed and the value read is current, without the
// wait_for_tensor_ready() stall that get_tensor_data() pays in orchestration.
// PASS => dispatch normally; FAIL => retire inline via the dep-only path.
enum class PredicateOp : uint8_t { NONE = 0, EQ, NE, GT, LT, GE, LE };

// Resolved dispatch predicate stored on a task's payload (AICPU-side only): the
// absolute GM address of the predicate element + the comparison. op == NONE
// means "no predicate — always dispatch". Populated at submit from an
// L0TaskPredicate; evaluated by the scheduler at the dispatch point via pass().
// Layout is 18 bytes (8-aligned).
struct DispatchPredicate {
    uint64_t addr{0};      // absolute GM address of the predicate element (0 when op == NONE)
    int64_t target{0};     // value compared against
    uint8_t elem_size{0};  // width of the predicate element in bytes (1/2/4/8)
    PredicateOp op{PredicateOp::NONE};

    // true => dispatch, false => retire inline. Reads elem_size bytes at addr and
    // sign-extends to 64 bits before comparing to target. Safe to call only when
    // the owning task is ready (its producer has written the value).
    bool pass() const {
        if (op == PredicateOp::NONE) return true;
        int64_t v = 0;
        __builtin_memcpy(&v, reinterpret_cast<const void *>(addr), elem_size);
        uint32_t bits = static_cast<uint32_t>(elem_size) * 8u;
        if (bits < 64u) {
            int64_t shift = static_cast<int64_t>(64u - bits);
            v = (v << shift) >> shift;
        }
        switch (op) {
        case PredicateOp::EQ:
            return v == target;
        case PredicateOp::NE:
            return v != target;
        case PredicateOp::GT:
            return v > target;
        case PredicateOp::LT:
            return v < target;
        case PredicateOp::GE:
            return v >= target;
        case PredicateOp::LE:
            return v <= target;
        default:
            return true;
        }
    }
};

/**
 * Resource shape — classifies a MixedKernels into one of 3 scheduling buckets.
 *
 * Multi-subtask tasks (2+ active slots) are all scheduled as MIX. Dispatch
 * chooses one cluster, then places each core named by active_mask by its own
 * state: idle cores take their running slot, already-running cores take their
 * pending slot. A cluster is usable as long as no used core's pending slot is
 * occupied; unused cores in the cluster are ignored.
 *
 * DUMMY is a synthetic shape for dep-only tasks (no AICore dispatch). Tasks
 * with an empty core_mask route to a dedicated DUMMY ready queue and are
 * completed inline by the scheduler dispatch loop, bypassing core allocation.
 */
enum class PTO2ResourceShape : uint8_t {
    AIC = 0,    // Single AIC
    AIV = 1,    // Single AIV
    MIX = 2,    // Full cluster (dispatch uses active_mask)
    DUMMY = 3,  // Dependency-only (no AICore dispatch)
};

// Number of *dispatchable* resource shapes (AIC, AIV, MIX). DUMMY does not
// allocate a per-shape ready_queue entry / local buffer — it lives in a
// dedicated queue inside PTO2SchedulerState.
inline constexpr int32_t PTO2_NUM_RESOURCE_SHAPES = 3;

/**
 * Bitmask of active subtask slots (AIC/AIV0/AIV1), sizeof == 1.
 *
 * Pure subtask-slot mask: only the low 3 bits (core_mask) are meaningful, so it
 * can be OR/==-combined when building MIX clusters without dragging unrelated
 * flag bits along. Per-task scheduling flags live on TaskAttrs instead.
 */
class ActiveMask {
public:
    constexpr ActiveMask() = default;
    constexpr explicit ActiveMask(uint8_t raw) :
        raw_(raw) {}

    uint8_t raw() const { return raw_; }

    bool subtask_active(PTO2SubtaskSlot slot) const { return (raw_ & (1u << static_cast<uint8_t>(slot))) != 0; }

    uint8_t core_mask() const { return raw_ & 0x07u; }

    PTO2ResourceShape to_shape() const {
        uint8_t cmask = core_mask();
        if (cmask == 0) return PTO2ResourceShape::DUMMY;
        int bit_count = __builtin_popcount(cmask);
        if (bit_count >= 2) return PTO2ResourceShape::MIX;
        if (cmask & PTO2_SUBTASK_MASK_AIC) return PTO2ResourceShape::AIC;
        return PTO2ResourceShape::AIV;
    }

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
 * Per-task scheduling attributes, packed into one byte on PTO2TaskSlotState.
 *
 * Single home for the independent per-task flags: an early-dispatch hint, the
 * two dispatch-time predicates (sync_start / has_predicate), and the selective
 * timing tag. Consolidating them here keeps active_mask a pure subtask-slot mask
 * and lands the timing tag on the scheduler's hot slot_state cache line.
 *
 *   bit 0     allow_early_resolve
 *   bit 1     sync_start
 *   bit 2     has_predicate
 *   bit 3     is_timed          (0 => untagged; timing_tag ignored)
 *   bits 4-7  timing_tag (0..15)
 */
class TaskAttrs {
public:
    constexpr TaskAttrs() = default;

    bool allow_early_resolve() const { return (raw_ & BIT_EARLY_RESOLVE) != 0; }
    void set_early_resolve(bool v) {
        if (v) {
            raw_ |= BIT_EARLY_RESOLVE;
        } else {
            raw_ &= static_cast<uint8_t>(~BIT_EARLY_RESOLVE);
        }
    }

    bool requires_sync_start() const { return (raw_ & BIT_SYNC_START) != 0; }
    void set_sync_start() { raw_ |= BIT_SYNC_START; }

    bool has_predicate() const { return (raw_ & BIT_HAS_PREDICATE) != 0; }
    void set_predicate() { raw_ |= BIT_HAS_PREDICATE; }

    bool is_timed() const { return (raw_ & BIT_IS_TIMED) != 0; }

    // 0..15 timing tag when tagged, else -1 (matches TASK_TIMING_SLOT_NONE; the
    // equality is guarded by a static_assert in pto_types.h).
    int32_t timing_slot() const { return is_timed() ? static_cast<int32_t>(raw_ >> TIMING_TAG_SHIFT) : -1; }

    // slot < 0 => untagged; 0..15 => tagged. Arg::set_task_timing_slot already
    // range-checks 0..NUM_TASK_TIMING_SLOTS-1, so only the low 4 bits are stored.
    void set_timing_slot(int32_t slot) {
        raw_ &= static_cast<uint8_t>(~(BIT_IS_TIMED | TIMING_TAG_MASK));
        if (slot >= 0) {
            raw_ |= static_cast<uint8_t>(BIT_IS_TIMED | ((slot & 0x0F) << TIMING_TAG_SHIFT));
        }
    }

private:
    static constexpr uint8_t BIT_EARLY_RESOLVE = 1u << 0;
    static constexpr uint8_t BIT_SYNC_START = 1u << 1;
    static constexpr uint8_t BIT_HAS_PREDICATE = 1u << 2;
    static constexpr uint8_t BIT_IS_TIMED = 1u << 3;
    static constexpr uint8_t TIMING_TAG_SHIFT = 4;
    static constexpr uint8_t TIMING_TAG_MASK = 0xF0u;

    uint8_t raw_{0};
};

static_assert(sizeof(TaskAttrs) == 1, "TaskAttrs must be exactly 1 byte");

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
