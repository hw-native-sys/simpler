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
 * @file pto_fanin_builder.h
 * @brief Pointer-based fanin accumulator shared by orch (explicit_dep prefix)
 *        and wiring (compute_task_fanin lookup).
 *
 * Both stages append into the SAME payload.fanin_inline_slot_states[] /
 * spill pool, so we never copy the inline slots between stages.
 *
 * Decoupled from PTO2OrchestratorState: append_fanin_or_fail takes the
 * relevant SM ring header and an error-code atomic directly, so it works
 * from either thread.
 */

#ifndef SRC_A5_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_PTO_FANIN_BUILDER_H_
#define SRC_A5_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_PTO_FANIN_BUILDER_H_

#include <atomic>
#include <cstdint>

#include "pto_ring_buffer.h"     // PTO2FaninPool, for_each_fanin_storage, PTO2FaninSpillEntry
#include "pto_runtime2_types.h"  // PTO2TaskSlotState, PTO2_FANIN_INLINE_CAP
#include "pto_shared_memory.h"   // PTO2SharedMemoryRingHeader
#include "pto_types.h"           // PTO2_ERROR_NONE, PTO2_ERROR_DEP_POOL_OVERFLOW

struct PTO2FaninBuilder {
    PTO2FaninBuilder(PTO2TaskSlotState **inline_slots_buf, PTO2FaninPool &spill_pool_ref) :
        count(0),
        spill_start(0),
        inline_slots(inline_slots_buf),
        spill_pool(spill_pool_ref) {}
    int32_t count{0};
    int32_t spill_start{0};
    PTO2TaskSlotState **inline_slots;
    PTO2FaninPool &spill_pool;

    template <typename Fn>
    PTO2FaninForEachReturn<Fn> for_each(Fn &&fn) const {
        return for_each_fanin_storage(inline_slots, count, spill_start, spill_pool, static_cast<Fn &&>(fn));
    }

    bool contains(PTO2TaskSlotState *prod_state) const {
        bool found = false;
        for_each([&](PTO2TaskSlotState *slot_state) {
            if (slot_state == prod_state) {
                found = true;
                return false;
            }
            return true;
        });
        return found;
    }
};

// Append `prod_state` to the builder, deduping against entries already present.
// On dep_pool overflow, writes PTO2_ERROR_DEP_POOL_OVERFLOW into *err_code (if
// currently NONE) and returns false. Caller bails out with the same return
// value.
//
// `ring` is used for fanin_pool.ensure_space (spill path).
// `err_code` may be nullptr (skip fatal reporting).
inline bool append_fanin_or_fail(
    PTO2SharedMemoryRingHeader &ring, std::atomic<int32_t> *err_code, PTO2FaninBuilder *fanin_builder,
    PTO2TaskSlotState *prod_state
) {
    if (fanin_builder->contains(prod_state)) {
        return true;
    }

    if (fanin_builder->count < PTO2_FANIN_INLINE_CAP) {
        fanin_builder->inline_slots[fanin_builder->count++] = prod_state;
        return true;
    }

    PTO2FaninPool &fanin_pool = fanin_builder->spill_pool;
    auto mark_fatal = [&]() {
        if (err_code != nullptr) {
            int32_t expected = PTO2_ERROR_NONE;
            err_code->compare_exchange_strong(
                expected, PTO2_ERROR_DEP_POOL_OVERFLOW, std::memory_order_acq_rel, std::memory_order_acquire
            );
        }
    };
    if (!fanin_pool.ensure_space(ring, 1)) {
        mark_fatal();
        return false;
    }
    int32_t spill_idx = fanin_pool.top;
    PTO2FaninSpillEntry *entry = fanin_pool.alloc();
    if (entry == nullptr) {
        mark_fatal();
        return false;
    }
    if (fanin_builder->count == PTO2_FANIN_INLINE_CAP) {
        fanin_builder->spill_start = spill_idx;
    }
    entry->slot_state = prod_state;
    fanin_builder->count++;
    return true;
}

#endif  // SRC_A5_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_PTO_FANIN_BUILDER_H_
