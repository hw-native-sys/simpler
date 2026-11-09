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

#include "common/l0_profiling.h"
#include "common/memory_barrier.h"

// A wrapped 32-bit sample counter is unambiguous only when both range cursors
// are ordered within one ring length of the host's consumed cursor.
inline bool
l0_dfx_wrap_unambiguous(uint32_t consumed_samp_wrt, uint32_t start_samp_wrt, uint32_t end_samp_wrt, uint32_t ring_len) {
    const uint32_t start_distance = start_samp_wrt - consumed_samp_wrt;
    const uint32_t end_distance = end_samp_wrt - consumed_samp_wrt;
    return end_distance < ring_len && start_distance <= end_distance;
}

// Publish a sealed ring range to a per-thread ready queue. The host reads the
// ring directly at the published cursors; no device-side copy happens here.
// The caller has proven the range is stable (sampled on a previous checkpoint)
// and non-overlapping with the range already published.
inline int l0_dfx_publish_range(
    L0DataHeader *header, int thread_index, uint32_t core_index, uint32_t start_wptr, uint32_t start_samp_wrt,
    uint32_t end_wptr, uint32_t end_samp_wrt, bool wrap_unambiguous, uint32_t ring_len, uint32_t ring_capacity,
    uint32_t flags
) {
    if (header == nullptr || ring_len == 0 || ring_capacity < ring_len || core_index >= header->num_cores ||
        thread_index < 0 || thread_index >= static_cast<int>(header->num_threads)) {
        return -1;
    }
    auto *state = l0_dfx_state(header, core_index);
    const auto range =
        simpler::dfx::l0::plan_ring_range(start_wptr, start_samp_wrt, end_samp_wrt, end_wptr, ring_len, ring_capacity);
    if (range.invalid_pointer || range.pointer_mismatch || range.possible_overrun || range.bytes >= ring_len ||
        range.lost_bytes != 0 || (range.counter_wrapped && !wrap_unambiguous)) {
        state->capture_error = L0_DFX_CAPTURE_RANGE_ERROR;
        return -1;
    }
    uint32_t tail = header->queue_tails[thread_index];
    const uint32_t next_tail = (tail + 1) % L0_DFX_READY_QUEUE_SIZE;
    if (next_tail == header->queue_heads[thread_index]) return -1;  // ready queue full
    auto &entry = header->queues[thread_index][tail];
    entry.core_index = core_index;
    entry.start_wptr = start_wptr;
    entry.end_wptr = end_wptr;
    entry.start_samp_wrt = start_samp_wrt;
    entry.end_samp_wrt = end_samp_wrt;
    entry.sequence = static_cast<uint32_t>(state->next_range_sequence++);
    entry.flags = flags;
    wmb();
    header->queue_tails[thread_index] = next_tail;
    state->published_wptr = end_wptr;
    state->published_samp_wrt = end_samp_wrt;
    state->published_bytes += range.bytes;
    if (range.bytes != 0) ++state->batch_count;
    return 0;
}
