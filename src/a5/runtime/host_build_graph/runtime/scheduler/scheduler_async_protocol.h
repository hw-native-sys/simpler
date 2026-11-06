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

#pragma once

#include "scheduler_async_layout.h"
#include "scheduler_local_storage.h"
#include "scheduler_protocol.h"

struct SchedulerAsyncProtocolClock {
    inline __aicore__ static uint64_t now() { return scheduler_cycles(); }
};

struct SchedulerAsyncProtocolLoad {
    inline __aicore__ static uint64_t word(__gm__ const uint8_t *address) {
#if defined(__CCE_AICORE__)
        return static_cast<uint64_t>(
            __builtin_cce_ld_dev(reinterpret_cast<__gm__ uint64_t *>(const_cast<__gm__ uint8_t *>(address)), 0)
        );
#else
        uint64_t result = 0;
        __builtin_memcpy(&result, address, sizeof(result));
        return result;
#endif
    }
};

// The release owner may publish the second line while CHECK reads the first.
template <typename Reader = SchedulerAsyncProtocolLoad>
inline __aicore__ void scheduler_async_writer_snapshot(
    __gm__ const SchedulerAsyncGroupWriter *source, SchedulerAsyncGroupWriter *destination
) {
    scheduler_reset_local(destination);
    for (uint32_t offset = 0; offset < offsetof(SchedulerAsyncGroupWriter, release); offset += sizeof(uint64_t)) {
        const uint64_t word = Reader::word(reinterpret_cast<__gm__ const uint8_t *>(source) + offset);
        __builtin_memcpy(reinterpret_cast<uint8_t *>(destination) + offset, &word, sizeof(word));
    }
}

inline __aicore__ void scheduler_async_protocol_store(__gm__ uint64_t *address, uint64_t value) {
#if defined(__CCE_AICORE__)
    st_dev(value, address, 0);
#else
    __atomic_store_n(address, value, __ATOMIC_RELEASE);
#endif
}

template <typename Clock = SchedulerAsyncProtocolClock, typename Local>
inline __aicore__ bool scheduler_async_protocol_point(
    __gm__ void *base, const Local *local, uint64_t epoch, __gm__ SchedulerProtocolStamp *stamp,
    __gm__ uint64_t *worker_index = nullptr
) {
    auto *layout = scheduler_protocol_layout_at(base, local);
    if (layout == nullptr) return true;
    if (layout->run_epoch != epoch) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
        return false;
    }
    if (scheduler_gm_query(stamp->sequence) != 0) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_DUPLICATE);
        return false;
    }
    SchedulerProtocolStamp next{};
    if (!scheduler_protocol_reserve_stamp(base, local, layout, &next, Clock::now())) return false;
    if (worker_index != nullptr) scheduler_async_protocol_store(worker_index, local->worker_id());
    scheduler_async_protocol_store(&stamp->cycles, next.cycles);
    scheduler_async_protocol_store(&stamp->sequence, next.sequence);
    scheduler_cache_barrier();
    return true;
}

template <typename Local>
inline __aicore__ bool scheduler_async_protocol_sealed(
    __gm__ void *base, const Local *local, const SchedulerAsyncGroupIdentity &identity,
    __gm__ SchedulerAsyncGroupWriter *writer
) {
    auto *layout = scheduler_protocol_layout_at(base, local);
    if (layout == nullptr) return true;
    auto *execution = scheduler_protocol_execution_at(
        base, local, layout, identity.task_ref, identity.block_idx, identity.subtask_slot
    );
    if (execution == nullptr) return false;
    const uint32_t slot = execution->pending_slot;
    if (execution->run_epoch != identity.run_epoch || execution->worker_index != local->worker_id() ||
        execution->task_ref != identity.task_ref || execution->block_idx != identity.block_idx ||
        execution->subtask_slot != identity.subtask_slot || execution->generation == 0 ||
        execution->entry.sequence == 0 || slot >= SCHEDULER_PENDING_SLOT_COUNT ||
        local->slots[slot].generation != execution->generation || local->slots[slot].task_ref != identity.task_ref ||
        local->slots[slot].block_idx != identity.block_idx ||
        local->slots[slot].subtask_slot != identity.subtask_slot ||
        ((execution->flags & SCHEDULER_PROTOCOL_EXECUTED) != 0 &&
         (execution->returned.sequence == 0 || execution->visible.sequence == 0))) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
        return false;
    }
    return scheduler_async_protocol_point(base, local, identity.run_epoch, &writer->sealed_at);
}
